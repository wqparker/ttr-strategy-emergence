"""Greedy ticket-chasing heuristic: the "reasonable human beginner" baseline.

Each turn it plans the cheapest path for every incomplete ticket (through its own
routes for free and open routes at their length). It then claims a planned route
if it can pay, otherwise draws the cards those routes need. Once all its tickets
are done, it draws more while it has trains to spare, or claims long routes for
points.

Per-bot difficulty option (PLAN.md "Agent design decisions"):
- avoid_final_ticket_draw: never draw tickets once the final round has started.
"""

from __future__ import annotations

import heapq
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ttr.actions import (
    Action,
    ClaimRoute,
    DrawBlind,
    DrawFaceUp,
    DrawTickets,
    KeepTickets,
    Pay,
)
from ttr.board import Board, Route, Ticket
from ttr.cards import Color
from ttr.game import Game, Phase
from ttr.scoring import connected

INF = float("inf")


def _adjacency(board: Board) -> Dict[str, List[Route]]:
    adj: Dict[str, List[Route]] = defaultdict(list)
    for r in board.routes:
        adj[r.a].append(r)
        adj[r.b].append(r)
    return adj


def cheapest_path(game: Game, p: int, a: str, b: str) -> Tuple[float, List[int]]:
    """Dijkstra from a to b. Own routes cost 0, open routes cost their length,
    anything else is impassable. Returns (cost, unclaimed route ids on the path)."""
    adj = _adjacency(game.board)
    dist: Dict[str, float] = {a: 0}
    via: Dict[str, Tuple[str, Route]] = {}
    heap = [(0, a)]
    while heap:
        d, city = heapq.heappop(heap)
        if city == b:
            break
        if d > dist.get(city, INF):
            continue
        for r in adj[city]:
            owner = game.route_owner.get(r.id)
            if owner == p:
                cost = 0
            elif game.route_open_to(p, r):
                cost = r.length
            else:
                continue
            nxt = r.b if r.a == city else r.a
            nd = d + cost
            if nd < dist.get(nxt, INF):
                dist[nxt] = nd
                via[nxt] = (city, r)
                heapq.heappush(heap, (nd, nxt))
    if b not in dist:
        return INF, []
    path: List[int] = []
    city = b
    while city != a:
        prev, r = via[city]
        if game.route_owner.get(r.id) != p:
            path.append(r.id)
        city = prev
    return dist[b], path


class GreedyAgent:
    name = "greedy"

    def __init__(
        self,
        seed: Optional[int] = None,
        avoid_final_ticket_draw: bool = True,
        draw_tickets_min_trains: int = 12,
        min_filler_route: int = 3,
    ) -> None:
        self.rng = random.Random(seed)
        self.avoid_final_ticket_draw = avoid_final_ticket_draw
        self.draw_tickets_min_trains = draw_tickets_min_trains
        self.min_filler_route = min_filler_route

    # ------------------------------------------------------------- dispatch

    def act(self, game: Game, player: int) -> Action:
        legal = game.legal_actions()
        if len(legal) == 1:
            return legal[0]
        if game.phase in (Phase.CHOOSE_INITIAL_TICKETS, Phase.KEEP_TICKETS):
            return self._choose_tickets(game, player, legal)
        if game.phase is Phase.CHOOSE_PAYMENT:
            return self._choose_payment(game, player, legal)
        if game.phase is Phase.DRAW_SECOND_CARD:
            return self._draw_card(game, player, legal)
        return self._main_action(game, player, legal)

    # ------------------------------------------------------------ planning

    def _open_tickets(self, game: Game, p: int) -> List[Ticket]:
        mine = [game.board.routes[r] for r in game.players[p].routes]
        return [
            t
            for t in (game.board.tickets[i] for i in game.players[p].tickets)
            if not connected(mine, t.a, t.b)
        ]

    def _targets(self, game: Game, p: int) -> Set[int]:
        """Unclaimed routes on the cheapest paths of reachable incomplete tickets,
        plus their open double-route siblings (either half will do)."""
        targets: Set[int] = set()
        for t in sorted(self._open_tickets(game, p), key=lambda t: -t.points):
            cost, path = cheapest_path(game, p, t.a, t.b)
            if cost == INF or cost > game.players[p].trains:
                continue  # unreachable or unaffordable; stop investing in it
            targets.update(path)
        for rid in list(targets):
            sib = game.board.routes[rid].sibling
            if sib is not None and game.route_open_to(p, game.board.routes[sib]):
                targets.add(sib)
        return targets

    def _color_needs(self, game: Game, p: int, targets: Set[int]) -> Counter:
        """Cards still missing, per color, to claim the colored target routes."""
        hand = game.players[p].hand
        needs: Counter = Counter()
        for rid in targets:
            r = game.board.routes[rid]
            if r.color is not None:
                needs[r.color] = max(needs[r.color], r.length - hand[r.color])
        return +needs

    # ------------------------------------------------------------- actions

    def _main_action(self, game: Game, p: int, legal: List[Action]) -> Action:
        claims = [a for a in legal if isinstance(a, ClaimRoute)]
        routes = game.board.routes
        final_round = game.final_turns_remaining is not None
        targets = self._targets(game, p)

        target_claims = [a for a in claims if a.route_id in targets]
        if target_claims:
            return max(target_claims, key=lambda a: routes[a.route_id].length)

        if final_round and claims:  # last chance to turn cards into points
            return max(claims, key=lambda a: routes[a.route_id].length)

        if not targets:
            can_draw_tickets = DrawTickets() in legal and not (
                final_round and self.avoid_final_ticket_draw
            )
            if can_draw_tickets and game.players[p].trains >= self.draw_tickets_min_trains:
                return DrawTickets()
            fillers = [a for a in claims if routes[a.route_id].length >= self.min_filler_route]
            if fillers:
                return max(fillers, key=lambda a: routes[a.route_id].length)

        draw = self._draw_card(game, p, legal, targets)
        if draw is not None:
            return draw
        if claims:
            return max(claims, key=lambda a: routes[a.route_id].length)
        non_ticket = [a for a in legal if not isinstance(a, DrawTickets)]
        if self.avoid_final_ticket_draw and final_round and non_ticket:
            return non_ticket[0]
        return legal[0]

    def _draw_card(
        self, game: Game, p: int, legal: List[Action], targets: Optional[Set[int]] = None
    ) -> Optional[Action]:
        if targets is None:
            targets = self._targets(game, p)
        needs = self._color_needs(game, p, targets)
        face_up = [
            a for a in legal if isinstance(a, DrawFaceUp) and a.color is not Color.LOCOMOTIVE
        ]
        useful = [a for a in face_up if needs[a.color] > 0]
        if useful:
            return max(useful, key=lambda a: needs[a.color])
        if DrawBlind() in legal:
            return DrawBlind()
        if face_up:
            return self.rng.choice(face_up)
        loco = DrawFaceUp(Color.LOCOMOTIVE)
        if loco in legal:
            return loco
        return None if game.phase is Phase.CHOOSE_ACTION else legal[0]

    def _choose_payment(self, game: Game, p: int, legal: List[Action]) -> Action:
        pending = game.pending_route
        targets = self._targets(game, p) - {pending}
        needs = self._color_needs(game, p, targets)
        hand = game.players[p].hand

        def key(pay: Pay):
            # Fewest Locomotives, then avoid colors other planned routes still need.
            conflict = needs[pay.color] if pay.color is not None else 0
            spare = hand[pay.color] if pay.color is not None else 0
            return (pay.locomotives, conflict, -spare)

        return min((a for a in legal if isinstance(a, Pay)), key=key)

    def _choose_tickets(self, game: Game, p: int, legal: List[Action]) -> Action:
        player = game.players[p]
        pending = [game.board.tickets[t] for t in player.pending_tickets]
        min_keep = min(len(a.ticket_ids) for a in legal)

        scored = []
        for t in pending:
            cost, _ = cheapest_path(game, p, t.a, t.b)
            scored.append((cost, -t.points, t))
        scored.sort(key=lambda x: (x[0], x[1]))

        # Trains already committed to incomplete tickets we hold.
        budget = player.trains - sum(
            min(cheapest_path(game, p, t.a, t.b)[0], player.trains)
            for t in self._open_tickets(game, p)
        )
        keep: List[int] = []
        for cost, _, t in scored:
            if len(keep) < min_keep or cost <= budget * 0.6:
                keep.append(t.id)
                budget -= cost if cost != INF else 0
        choice = KeepTickets(frozenset(keep))
        return choice if choice in legal else legal[0]
