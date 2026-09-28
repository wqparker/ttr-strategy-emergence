"""Racer: the strategy every trained agent found against greedy, scripted
(PLAN.md, linear passes 2-8 and DQN pass 1). Keep the fewest, cheapest tickets
and draw no more; spend every train on long routes, which score the most points
per train; end the game before the opponent can finish its tickets.

- Tickets: keep the minimum, cheapest to connect first; never draw more.
- Claim the longest route it can pay for among routes of length >= `min_length`
  (any length once it has fewer trains than that, or in the final round),
  preferring one that touches its own routes (toward the longest-path bonus).
- Otherwise draw toward the long open route closest to affordable: its color from
  the market if on offer, else blind.
- Pay with the fewest Locomotives, and for a gray route the color it holds most of.

`min_length` 6 was the strongest against greedy (300 games each, 2026-09-28):
+11.7 margin and 64% wins, against -3.3 for 5 and -13.4 for 4. A 6-route scores
15 points for 6 trains, a 4-route 7 for 4.
"""

from __future__ import annotations

import random
from typing import List, Optional

from ttr.actions import Action, ClaimRoute, DrawBlind, DrawFaceUp, DrawTickets, KeepTickets, Pay
from ttr.agents.greedy import cheapest_path
from ttr.board import Route
from ttr.cards import TRAIN_COLORS, Color
from ttr.game import Game, Phase


class RacerAgent:
    name = "racer"

    def __init__(self, seed: Optional[int] = None, min_length: int = 6) -> None:
        self.rng = random.Random(seed)
        self.min_length = min_length

    def act(self, game: Game, player: int) -> Action:
        legal = game.legal_actions()
        if len(legal) == 1:
            return legal[0]
        if game.phase in (Phase.CHOOSE_INITIAL_TICKETS, Phase.KEEP_TICKETS):
            return self._choose_tickets(game, player, legal)
        if game.phase is Phase.CHOOSE_PAYMENT:
            hand = game.players[player].hand
            pays = [a for a in legal if isinstance(a, Pay)]
            return min(pays, key=lambda a: (a.locomotives, -(hand[a.color] if a.color is not None else 0)))
        if game.phase is Phase.DRAW_SECOND_CARD:
            return self._draw(game, player, legal)
        return self._main_action(game, player, legal)

    def _choose_tickets(self, game: Game, p: int, legal: List[Action]) -> Action:
        fewest = min(len(a.ticket_ids) for a in legal)
        offer = game.players[p].pending_tickets
        by_cost = sorted(offer, key=lambda t: cheapest_path(game, p, game.board.tickets[t].a,
                                                            game.board.tickets[t].b)[0])
        choice = KeepTickets(frozenset(by_cost[:fewest]))
        return choice if choice in legal else next(a for a in legal if len(a.ticket_ids) == fewest)

    def _shortest_allowed(self, game: Game, p: int) -> int:
        if game.final_turns_remaining is not None or game.players[p].trains < self.min_length:
            return 1
        return self.min_length

    def _main_action(self, game: Game, p: int, legal: List[Action]) -> Action:
        routes = game.board.routes
        shortest = self._shortest_allowed(game, p)
        claims = [a for a in legal if isinstance(a, ClaimRoute) and routes[a.route_id].length >= shortest]
        if claims:
            mine = {c for rid in game.players[p].routes for c in (routes[rid].a, routes[rid].b)}
            key = lambda a: (routes[a.route_id].length, routes[a.route_id].a in mine or routes[a.route_id].b in mine)
            best = max(key(a) for a in claims)
            return self.rng.choice([a for a in claims if key(a) == best])
        draw = self._draw(game, p, legal)
        if draw is not None:
            return draw
        others = [a for a in legal if not isinstance(a, DrawTickets)]
        return others[0] if others else legal[0]

    def _missing(self, game: Game, p: int, r: Route) -> int:
        """Cards still missing to claim `r` (Locomotives fill any gap)."""
        hand = game.players[p].hand
        have = hand[r.color] if r.color is not None else max(hand[c] for c in TRAIN_COLORS)
        return max(0, r.length - have - hand[Color.LOCOMOTIVE])

    def _goal(self, game: Game, p: int) -> Optional[Route]:
        """The open long route closest to affordable (longest on ties)."""
        trains = game.players[p].trains
        shortest = self._shortest_allowed(game, p)
        open_routes = [r for r in game.board.routes
                       if r.id not in game.route_owner and shortest <= r.length <= trains
                       and game.route_open_to(p, r)]
        if not open_routes:
            return None
        return min(open_routes, key=lambda r: (self._missing(game, p, r), -r.length))

    def _draw(self, game: Game, p: int, legal: List[Action]) -> Optional[Action]:
        goal = self._goal(game, p)
        if goal is not None:
            if goal.color is not None:
                wanted = [goal.color]
            else:  # gray: build up the color already held most
                hand = game.players[p].hand
                wanted = sorted(TRAIN_COLORS, key=lambda c: -hand[c])[:1]
            for color in wanted:
                if DrawFaceUp(color) in legal:
                    return DrawFaceUp(color)
        if DrawBlind() in legal:
            return DrawBlind()
        face_up = [a for a in legal if isinstance(a, DrawFaceUp) and a.color is not Color.LOCOMOTIVE]
        if face_up:
            return self.rng.choice(face_up)
        if DrawFaceUp(Color.LOCOMOTIVE) in legal:
            return DrawFaceUp(Color.LOCOMOTIVE)
        return None if game.phase is Phase.CHOOSE_ACTION else legal[0]
