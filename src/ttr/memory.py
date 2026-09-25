"""Card memory: exact bookkeeping of public card information from one seat's view.

PLAN.md "Card memory" defines the levels:

    0  snapshot only (market, discard, hand sizes: already on the table)
    1  + per opponent, a lower bound on the cards of each color they hold.
         A face-up take adds 1; paying for a route subtracts the amount paid,
         floored at 0 (paying 4 red with 2 known red means 2 came from blind draws).
    2  + the unseen pool: the 110-card deck minus the viewer's hand, the market,
         the discard pile and opponents' known cards. Each unseen card is equally
         likely to be in the deck or in an opponent's unknown cards.

Only public events are read (RULES.md §9 #17), so this is what a careful player
could track at the table. Used by the viewer's player perspective and by the env.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional

from ttr.cards import Color, standard_deck
from ttr.game import Game

FULL_DECK = Counter(standard_deck())


@dataclass
class MemoryView:
    level: int
    viewer: int
    known: Dict[int, Counter] = field(default_factory=dict)  # opponent -> color -> lower bound
    unknown: Dict[int, int] = field(default_factory=dict)  # opponent -> cards not accounted for
    unseen: Optional[Counter] = None  # level 2: color -> count, deck + opponents' unknown


class CardMemory:
    """Incremental tracker. Call `update(game)` whenever the game has advanced; it
    reads only log events it hasn't seen. A fresh tracker on any state catches up
    from the full log, so replays can build one at any step."""

    def __init__(self, num_players: int, viewer: int, level: int = 2) -> None:
        if level not in (0, 1, 2):
            raise ValueError("memory level must be 0, 1 or 2")
        self.num_players = num_players
        self.viewer = viewer
        self.level = level
        self._known: Dict[int, Counter] = {p: Counter() for p in range(num_players) if p != viewer}
        self._seen = 0  # log entries processed

    @classmethod
    def at(cls, game: Game, viewer: int, level: int = 2) -> "CardMemory":
        memory = cls(game.num_players, viewer, level)
        memory.update(game)
        return memory

    def update(self, game: Game) -> None:
        for event in game.log[self._seen:]:
            p = event.player
            if p is None or p == self.viewer:
                continue
            if event.kind == "draw_face_up":
                self._known[p][Color(event.public["color"])] += 1
            elif event.kind == "claim_route":
                known = self._known[p]
                for color, n in Counter(Color(c) for c in event.public["paid"]).items():
                    known[color] = max(0, known[color] - n)
                known += Counter()  # drop zero counts
                self._known[p] = known
        self._seen = len(game.log)

    def view(self, game: Game) -> MemoryView:
        """The memory at the current state, for the tracker's level."""
        self.update(game)
        result = MemoryView(level=self.level, viewer=self.viewer)
        if self.level == 0:
            return result
        for p, known in self._known.items():
            result.known[p] = Counter(known)
            result.unknown[p] = game.players[p].hand_size - sum(known.values())
        if self.level >= 2:
            unseen = Counter(FULL_DECK)
            unseen.subtract(game.players[self.viewer].hand)
            unseen.subtract(game.market)
            unseen.subtract(game.discard)
            for known in self._known.values():
                unseen.subtract(known)
            result.unseen = +unseen  # drop zero counts
        return result
