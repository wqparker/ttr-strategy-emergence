"""What one viewer may see of a game state, as a plain view model.

The viewer is either all-seeing (`viewer=None`, every hand and ticket revealed)
or a seat, which sees its own cards and tickets plus what a careful player could
track of the others: hand size, ticket count, trains, score (public, RULES.md
§9 #13, #17) and the card-memory bounds from `ttr.memory` at the chosen level.

No Pygame here; the panels just draw what this produces, and the tests check the
hiding without a display.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ttr.board import Board
from ttr.cards import Color
from ttr.game import Game
from ttr.memory import CardMemory
from ttr.render import describe_event
from ttr.scoring import connected

RECENT_EVENTS = 3


@dataclass(frozen=True)
class TicketFact:
    id: int
    a: str
    b: str
    points: int
    done: bool  # already connected by the holder's claimed routes
    pending: bool = False  # drawn, not yet kept or returned


@dataclass
class SeatFacts:
    """One seat as this viewer sees it. The public fields are always filled."""

    seat: int
    name: str
    trains: int
    route_points: int
    hand_size: int
    ticket_count: int
    to_act: bool
    is_viewer: bool
    hand: Optional[Counter] = None  # exact, revealed seats only
    known: Optional[Counter] = None  # memory lower bounds (level >= 1)
    unknown: Optional[int] = None  # cards not accounted for by `known`
    tickets: Optional[List[TicketFact]] = None  # None = hidden

    @property
    def revealed(self) -> bool:
        return self.hand is not None


@dataclass
class TableFacts:
    market: List[Color]
    deck: int
    discard: int
    tickets_left: int
    turn: int
    phase: str
    current_player: int
    final_turns_remaining: Optional[int]
    game_over: bool
    winners: Tuple[int, ...] = ()
    unseen: Optional[Counter] = None  # memory level 2: deck + opponents' unknown


@dataclass
class ViewModel:
    seats: List[SeatFacts]
    table: TableFacts
    events: List[str] = field(default_factory=list)  # oldest first
    viewer: Optional[int] = None
    memory_level: int = 2

    @property
    def all_seeing(self) -> bool:
        return self.viewer is None


class Perspective:
    """Builds view models for one viewer. Keeps a card-memory tracker between
    calls and rebuilds it when the log shrinks, so replay scrubbing backwards is
    safe."""

    def __init__(self, viewer: Optional[int] = None, memory_level: int = 2) -> None:
        if memory_level not in (0, 1, 2):
            raise ValueError("memory level must be 0, 1 or 2")
        self.viewer = viewer
        self.memory_level = memory_level
        self._memory: Optional[CardMemory] = None
        self._log_len = 0

    @property
    def all_seeing(self) -> bool:
        return self.viewer is None

    def memory(self, game: Game):
        """The memory view for a seat viewer, or None when all-seeing."""
        if self.viewer is None:
            return None
        if self._memory is None or len(game.log) < self._log_len:
            self._memory = CardMemory(game.num_players, self.viewer, self.memory_level)
        self._log_len = len(game.log)
        return self._memory.view(game)

    def view(
        self,
        game: Game,
        names: Optional[Sequence[str]] = None,
        events: int = RECENT_EVENTS,
    ) -> ViewModel:
        memory = self.memory(game)
        seats = [self._seat(game, i, names, memory) for i in range(game.num_players)]
        table = TableFacts(
            market=list(game.market),
            deck=len(game.deck),
            discard=len(game.discard),
            tickets_left=len(game.ticket_deck),
            turn=game.turn,
            phase=game.phase.value,
            current_player=game.current_player,
            final_turns_remaining=game.final_turns_remaining,
            game_over=game.game_over,
            winners=tuple(game.result.winners) if game.result else (),
            unseen=Counter(memory.unseen) if memory and memory.unseen is not None else None,
        )
        return ViewModel(
            seats=seats,
            table=table,
            events=recent_events(game, events),
            viewer=self.viewer,
            memory_level=self.memory_level,
        )

    # ------------------------------------------------------------- one seat

    def _seat(self, game: Game, i: int, names, memory) -> SeatFacts:
        p = game.players[i]
        facts = SeatFacts(
            seat=i,
            name=(names[i] if names and i < len(names) else ""),
            trains=p.trains,
            route_points=p.route_points,
            hand_size=sum(p.hand.values()),
            ticket_count=len(p.tickets),
            to_act=(game.current_player == i and not game.game_over),
            is_viewer=(i == self.viewer),
        )
        if self.viewer is None or i == self.viewer:
            facts.hand = Counter(p.hand)
            facts.tickets = _tickets(game, i)
            return facts
        if memory is not None and memory.level >= 1:
            facts.known = Counter(memory.known.get(i, Counter()))
            facts.unknown = memory.unknown.get(i, facts.hand_size)
        return facts


def _tickets(game: Game, i: int) -> List[TicketFact]:
    p = game.players[i]
    mine = [game.board.routes[r] for r in p.routes]
    out = []
    for pending, ids in ((False, p.tickets), (True, p.pending_tickets)):
        for t in (game.board.tickets[x] for x in ids):
            out.append(
                TicketFact(
                    id=t.id,
                    a=t.a,
                    b=t.b,
                    points=t.points,
                    done=connected(mine, t.a, t.b),
                    pending=pending,
                )
            )
    return out


def recent_events(game: Game, count: int = RECENT_EVENTS) -> List[str]:
    """The last `count` describable log lines, oldest first. Only public fields
    are read, so this is safe in a player view."""
    lines: List[str] = []
    for event in reversed(game.log):
        if len(lines) >= count:
            break
        text = describe_event(game, event)
        if text:
            lines.append(text)
    return list(reversed(lines))


def city_code(board: Board, city: str) -> str:
    """Short city code from the board layout, for cramped panels."""
    pos = board.layout.get(city)
    return pos.code if pos else city[:3].upper()


def code_map(board: Board) -> Dict[str, str]:
    """City -> short code, for panels that have no room for full names."""
    return {c: city_code(board, c) for c in board.cities}
