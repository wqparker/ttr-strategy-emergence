"""Game records: a game is its setup plus the list of actions taken.

The engine is deterministic given its seed, so replaying the actions on a fresh
Game with the same setup reproduces the game exactly. Records are small JSON
files that drive the replay viewer and the analysis overlays (PLAN.md Phase 3).

    record = GameRecord.from_game(game, actions, agents=["greedy", "random"])
    record.save("runs/records/game.json")
    states = GameRecord.load("runs/records/game.json").replay_states()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from ttr.actions import Action, ClaimRoute, DrawBlind, DrawFaceUp, DrawTickets, KeepTickets, Pass, Pay
from ttr.board import Board, load_board
from ttr.cards import Color
from ttr.game import Game

FORMAT = "ttr-record"
VERSION = 1


class RecordError(ValueError):
    pass


# ------------------------------------------------------------ action codec


def action_to_dict(action: Action) -> Dict[str, Any]:
    if isinstance(action, DrawFaceUp):
        return {"type": "draw_face_up", "color": action.color.value}
    if isinstance(action, DrawBlind):
        return {"type": "draw_blind"}
    if isinstance(action, ClaimRoute):
        return {"type": "claim_route", "route": action.route_id}
    if isinstance(action, Pay):
        color = action.color.value if action.color is not None else None
        return {"type": "pay", "color": color, "locomotives": action.locomotives}
    if isinstance(action, DrawTickets):
        return {"type": "draw_tickets"}
    if isinstance(action, KeepTickets):
        return {"type": "keep_tickets", "tickets": sorted(action.ticket_ids)}
    if isinstance(action, Pass):
        return {"type": "pass"}
    raise TypeError(f"not an action: {action!r}")


def action_from_dict(d: Dict[str, Any]) -> Action:
    kind = d.get("type")
    if kind == "draw_face_up":
        return DrawFaceUp(Color(d["color"]))
    if kind == "draw_blind":
        return DrawBlind()
    if kind == "claim_route":
        return ClaimRoute(int(d["route"]))
    if kind == "pay":
        color = Color(d["color"]) if d["color"] is not None else None
        return Pay(color, int(d["locomotives"]))
    if kind == "draw_tickets":
        return DrawTickets()
    if kind == "keep_tickets":
        return KeepTickets(frozenset(int(t) for t in d["tickets"]))
    if kind == "pass":
        return Pass()
    raise RecordError(f"unknown action type {kind!r}")


# ------------------------------------------------------------------ record


@dataclass
class GameRecord:
    board: str  # bundled board name ("usa", "toy") or a path to a board JSON
    num_players: int
    seed: int
    first_player: int
    max_turns: Optional[int]
    actions: List[Action]
    agents: List[str] = field(default_factory=list)  # by seat, for display only
    # Summary of the recorded outcome; replay checks it to catch engine changes
    # that would make an old record replay differently.
    result: Optional[Dict[str, Any]] = None

    @classmethod
    def from_game(
        cls,
        game: Game,
        actions: Sequence[Action],
        agents: Sequence[str] = (),
        board: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> "GameRecord":
        """Record a game played from a fresh Game. `seed` must be the seed the
        game was created with (the Game doesn't keep it)."""
        if seed is None:
            raise RecordError("the game's seed is required to record it")
        return cls(
            board=board or game.board.name,
            num_players=game.num_players,
            seed=seed,
            first_player=game.first_player,
            max_turns=game.max_turns,
            actions=list(actions),
            agents=list(agents),
            result=_summarize(game) if game.game_over else None,
        )

    # --------------------------------------------------------------- replay

    def new_game(self, board: Optional[Board] = None) -> Game:
        return Game(
            board if board is not None else load_board(self.board),
            num_players=self.num_players,
            seed=self.seed,
            first_player=self.first_player,
            max_turns=self.max_turns,
        )

    def replay(self, board: Optional[Board] = None) -> Game:
        """The final state, after every recorded action."""
        game = self.new_game(board)
        for action in self.actions:
            game.step(action)
        self._check(game)
        return game

    def replay_states(self, board: Optional[Board] = None) -> List[Game]:
        """Every state: index 0 is the start, index i is after i actions."""
        game = self.new_game(board)
        states = [game.clone()]
        for action in self.actions:
            game.step(action)
            states.append(game.clone())
        self._check(game)
        return states

    def _check(self, game: Game) -> None:
        if self.result is not None and _summarize(game) != self.result:
            raise RecordError(
                "replay doesn't reproduce the recorded result; the record may predate "
                "an engine change"
            )

    # ----------------------------------------------------------------- JSON

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "board": self.board,
            "num_players": self.num_players,
            "seed": self.seed,
            "first_player": self.first_player,
            "max_turns": self.max_turns,
            "agents": self.agents,
            "result": self.result,
            "actions": [action_to_dict(a) for a in self.actions],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GameRecord":
        if d.get("format") != FORMAT:
            raise RecordError("not a game record")
        if d.get("version") != VERSION:
            raise RecordError(f"unsupported record version {d.get('version')!r}")
        return cls(
            board=d["board"],
            num_players=int(d["num_players"]),
            seed=int(d["seed"]),
            first_player=int(d["first_player"]),
            max_turns=d["max_turns"],
            actions=[action_from_dict(a) for a in d["actions"]],
            agents=list(d.get("agents", [])),
            result=d.get("result"),
        )

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # One action per line keeps files readable and diffs small.
        d = self.to_dict()
        actions = d.pop("actions")
        head = json.dumps(d, indent=2)[:-2]  # drop the closing "\n}"
        body = ",\n".join("    " + json.dumps(a) for a in actions)
        path.write_text(f'{head},\n  "actions": [\n{body}\n  ]\n}}\n', encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GameRecord":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _summarize(game: Game) -> Dict[str, Any]:
    assert game.result is not None
    r = game.result
    return {
        "winners": list(r.winners),
        "totals": [p.total for p in r.players],
        "truncated": r.truncated,
        "turns": game.turn,
    }
