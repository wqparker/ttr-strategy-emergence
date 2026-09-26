"""Per-route statistics over a folder of game records (PLAN.md Phase 3, milestone 7).

    summary = analyze("runs/records")
    summary.values("claim_rate")                  # route id -> share of games claimed
    summary.values("claim_rate", agent="greedy")  # the same, for one agent's games
    summary.values("claim_rate", seat=0)          # the same, for seat 0 only
    summary.rows()                                # one dict per route, for pandas
    summary.player(seat=0)                        # seat 0's average results
    summary.player_rows()                         # one dict per player per game

Every statistic comes from the `claim_route` events in each replayed game's log,
so nothing here depends on Pygame. The viewer turns the values into colors
(viz/overlay.py); the strategy analysis (Phase 6) reads `rows()`.

Statistics, per route:

- claim_rate: share of games in which it was claimed.
- avg_turn: mean `game.turn` at the claim (main turns completed before it, as the
  log and the viewer count them). None if never claimed.
- contested: share of games in which it was closed because its double-route
  sibling was claimed (§6, 2–3 players). None for single routes.

A filter narrows each one to a player: `agent` (by name), `seat`, or both (that
agent in that seat). Rates then count only that player's claims (or closing
claims) over only the games that player sat in. A rate of 0 is a measurement
(never claimed in N games); None means there is nothing to measure. `ttr-sim`
batches always start with seat 0, so a seat is also a place in turn order.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ttr.board import Board, load_board
from ttr.game import Game
from ttr.record import GameRecord

# Statistic name -> label, in the order the viewer cycles through them.
STATS: Dict[str, str] = {
    "claim_rate": "claim rate",
    "avg_turn": "average turn claimed",
    "contested": "contested (double closed)",
}
UNKNOWN_AGENT = "unknown"


class AnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class Claim:
    agent: str
    seat: int
    turn: int


@dataclass
class RouteStats:
    claims: List[Claim] = field(default_factory=list)
    closed_by: List[Claim] = field(default_factory=list)  # the sibling claim that closed it


@dataclass(frozen=True)
class SeatGame:
    """One player's result in one game."""

    agent: str
    seat: int
    won: float  # 1, a share of a shared win, or 0
    total: int
    route_points: int
    ticket_points: int
    tickets_completed: int
    tickets_failed: int
    longest_bonus: bool
    claims: int  # routes claimed
    trains: int  # trains placed


@dataclass(frozen=True)
class PlayerSummary:
    """Averages over the SeatGames that match a filter (see Summary.player)."""

    games: int
    agents: Dict[str, int]  # agent name -> games, for the seats it covers
    win_rate: float
    total: float
    route_points: float
    ticket_points: float
    tickets_completed: float
    tickets_failed: float
    completion_rate: float  # completed / (completed + failed), over all its tickets
    longest_rate: float
    claims: float
    trains: float


def _matches(c, agent: Optional[str], seat: Optional[int]) -> bool:
    """A Claim or SeatGame belongs to the filtered player."""
    return (agent is None or c.agent == agent) and (seat is None or c.seat == seat)


@dataclass
class Summary:
    board: Board
    games: int = 0
    # Games each player sat in: by agent name, by seat, and by (agent, seat).
    games_by_agent: Counter = field(default_factory=Counter)
    games_by_seat: Counter = field(default_factory=Counter)
    games_by_agent_seat: Counter = field(default_factory=Counter)
    routes: Dict[int, RouteStats] = field(default_factory=dict)
    seat_games: List[SeatGame] = field(default_factory=list)  # finished games only

    def __post_init__(self) -> None:
        for route in self.board.routes:
            self.routes.setdefault(route.id, RouteStats())

    @property
    def agents(self) -> List[str]:
        return sorted(self.games_by_agent)

    @property
    def seats(self) -> List[int]:
        return sorted(self.games_by_seat)

    def add(self, game: Game, agents: Sequence[str] = ()) -> None:
        """Count one finished game. `agents` names each seat."""
        if game.board.name != self.board.name:
            raise AnalysisError(f"a {game.board.name!r} game in a {self.board.name!r} summary")
        names = [
            (agents[seat] if seat < len(agents) and agents[seat] else UNKNOWN_AGENT)
            for seat in range(game.num_players)
        ]
        self.games += 1
        for name in set(names):
            self.games_by_agent[name] += 1
        for seat, name in enumerate(names):
            self.games_by_seat[seat] += 1
            self.games_by_agent_seat[(name, seat)] += 1
        claims: Dict[int, Claim] = {}
        for event in game.log:
            if event.kind == "claim_route":
                rid = event.public["route"]
                claims[rid] = Claim(names[event.player], event.player, event.turn)
                self.routes[rid].claims.append(claims[rid])
        if game.num_players <= 3:  # §6: one claimed side closes the other
            for rid in game.route_owner:
                sibling = self.board.routes[rid].sibling
                if sibling is not None and sibling not in game.route_owner:
                    self.routes[sibling].closed_by.append(claims[rid])
        if game.result is not None:
            winners = game.result.winners
            for seat, (name, r) in enumerate(zip(names, game.result.players)):
                player = game.players[seat]
                self.seat_games.append(SeatGame(
                    agent=name,
                    seat=seat,
                    won=1 / len(winners) if seat in winners else 0.0,
                    total=r.total,
                    route_points=r.route_points,
                    ticket_points=r.ticket_points,
                    tickets_completed=r.tickets_completed,
                    tickets_failed=r.tickets_failed,
                    longest_bonus=r.longest_path_bonus,
                    claims=len(player.routes),
                    trains=self.board.trains_per_player - player.trains,
                ))

    # ------------------------------------------------------------ players

    def player(self, agent: Optional[str] = None, seat: Optional[int] = None) -> Optional[PlayerSummary]:
        """Average results of the filtered player over the finished games it
        sat in; None if there are none."""
        games = [g for g in self.seat_games if _matches(g, agent, seat)]
        if not games:
            return None

        def mean(attr: str) -> float:
            return statistics.mean(float(getattr(g, attr)) for g in games)

        done = sum(g.tickets_completed for g in games)
        held = done + sum(g.tickets_failed for g in games)

        return PlayerSummary(
            games=len(games),
            agents=dict(Counter(g.agent for g in games)),
            win_rate=mean("won"),
            total=mean("total"),
            route_points=mean("route_points"),
            ticket_points=mean("ticket_points"),
            tickets_completed=mean("tickets_completed"),
            tickets_failed=mean("tickets_failed"),
            completion_rate=done / held if held else 0.0,
            longest_rate=mean("longest_bonus"),
            claims=mean("claims"),
            trains=mean("trains"),
        )

    def player_rows(self) -> List[Dict[str, Any]]:
        """One row per player per finished game, for pandas."""
        return [asdict(g) for g in self.seat_games]

    # ------------------------------------------------------------- values

    def games_for(self, agent: Optional[str] = None, seat: Optional[int] = None) -> int:
        """Games the filtered player sat in (all games with no filter)."""
        if agent is None and seat is None:
            return self.games
        if seat is None:
            return self.games_by_agent.get(agent, 0)
        if agent is None:
            return self.games_by_seat.get(seat, 0)
        return self.games_by_agent_seat.get((agent, seat), 0)

    def value(self, route_id: int, stat: str, agent: Optional[str] = None,
              seat: Optional[int] = None) -> Optional[float]:
        """One route's statistic, or None where there is nothing to measure."""
        s = self.routes[route_id]
        games = self.games_for(agent, seat)
        if stat == "claim_rate":
            claims = sum(1 for c in s.claims if _matches(c, agent, seat))
            return claims / games if games else None
        if stat == "avg_turn":
            turns = [c.turn for c in s.claims if _matches(c, agent, seat)]
            return statistics.mean(turns) if turns else None
        if stat == "contested":
            if self.board.routes[route_id].sibling is None or not games:
                return None
            closed = sum(1 for c in s.closed_by if _matches(c, agent, seat))
            return closed / games
        raise AnalysisError(f"unknown statistic {stat!r}; choose from {sorted(STATS)}")

    def values(self, stat: str, agent: Optional[str] = None,
               seat: Optional[int] = None) -> Dict[int, Optional[float]]:
        return {rid: self.value(rid, stat, agent, seat) for rid in self.routes}

    def rows(self) -> List[Dict[str, Any]]:
        """One row per route: its board data, then every statistic overall, per
        agent (`claim_rate[greedy]`) and per seat (`claim_rate[P0]`). Ready for
        `pandas.DataFrame(rows)`."""
        out = []
        for route in self.board.routes:
            row: Dict[str, Any] = {
                "route": route.id,
                "a": route.a,
                "b": route.b,
                "length": route.length,
                "color": route.color.value if route.color else "gray",
                "double": route.sibling is not None,
                "games": self.games,
                "claims": len(self.routes[route.id].claims),
            }
            for stat in STATS:
                row[stat] = self.value(route.id, stat)
                for agent in self.agents:
                    row[f"{stat}[{agent}]"] = self.value(route.id, stat, agent)
                for seat in self.seats:
                    row[f"{stat}[P{seat}]"] = self.value(route.id, stat, seat=seat)
            out.append(row)
        return out


# ------------------------------------------------------------------ loading


def record_paths(path: Union[str, Path]) -> List[Path]:
    """A record file, or every `*.json` in a folder (sorted)."""
    path = Path(path)
    if path.is_dir():
        paths = sorted(path.glob("*.json"))
        if not paths:
            raise AnalysisError(f"no records (*.json) in {path}")
        return paths
    if path.exists():
        return [path]
    raise AnalysisError(f"no such file or folder: {path}")


def summarize(records: Iterable[GameRecord]) -> Summary:
    """Replay every record and count it. All records must use the same board."""
    summary: Optional[Summary] = None
    boards: Dict[str, Board] = {}
    for record in records:
        if record.board not in boards:
            boards[record.board] = load_board(record.board)
        board = boards[record.board]
        if summary is None:
            summary = Summary(board)
        elif board.name != summary.board.name:
            raise AnalysisError(
                f"records mix boards: {summary.board.name!r} and {board.name!r}"
            )
        summary.add(record.replay(board), record.agents)
    if summary is None:
        raise AnalysisError("no records to analyze")
    return summary


def analyze(path: Union[str, Path]) -> Summary:
    """Summarize a record file or a folder of them."""
    return summarize(GameRecord.load(p) for p in record_paths(path))
