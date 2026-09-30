"""Greedy ticket-chasing heuristic: the "reasonable human beginner" baseline.

Each turn it plans the cheapest path for every incomplete ticket (through its own
routes for free and open routes at their length). It then claims a planned route
if it can pay, otherwise draws the cards those routes need. Once all its tickets
are done, it draws more while it has trains to spare, or claims long routes for
points.

Per-bot difficulty option (PLAN.md "Agent design decisions"):
- avoid_final_ticket_draw: never draw tickets once the final round has started.

`WaryAgent` ("wary") is greedy that watches the tempo (`alert_trains`): once an
opponent is down to that many trains, or the final round has started, it draws no
more tickets, drops the tickets it can't finish in the turns likely left, and
turns its cards into the longest routes it can claim. Plain greedy never looks at
the opponent, which is what the racing strategy exploits (PLAN.md).

`CollectorAgent` ("collector") is greedy for tickets: 2-3 open at a time, topped
up as soon as fewer than 2 are far from done, the nearest one finished first.
"""

from __future__ import annotations

import heapq
import random
import weakref
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


_ADJACENCY: Dict[int, Tuple[Board, Dict[str, List[Route]]]] = {}


def _adjacency(board: Board) -> Dict[str, List[Route]]:
    """City -> routes touching it, built once per board object."""
    hit = _ADJACENCY.get(id(board))
    if hit is None or hit[0] is not board:
        adj: Dict[str, List[Route]] = defaultdict(list)
        for r in board.routes:
            adj[r.a].append(r)
            adj[r.b].append(r)
        hit = _ADJACENCY[id(board)] = (board, adj)
    return hit[1]


# game -> (route ownership it was computed for, {(p, a, b): (cost, path)}). A path
# depends only on who owns which route, so an answer holds until the next claim;
# bots, linear features and the observation ask the same questions many times a turn.
_PATHS: "weakref.WeakKeyDictionary[Game, Tuple[tuple, Dict[Tuple[int, str, str], Tuple[float, List[int]]]]]" = (
    weakref.WeakKeyDictionary())


def cheapest_path(game: Game, p: int, a: str, b: str) -> Tuple[float, List[int]]:
    """Dijkstra from a to b. Own routes cost 0, open routes cost their length,
    anything else is impassable. Returns (cost, unclaimed route ids on the path),
    remembered until the route ownership changes."""
    owners = tuple(game.route_owner.items())
    hit = _PATHS.get(game)
    if hit is None or hit[0] != owners:
        hit = _PATHS[game] = (owners, {})
    memo = hit[1]
    found = memo.get((p, a, b))
    if found is None:
        found = memo[p, a, b] = _dijkstra(game, p, a, b)
    return found[0], list(found[1])  # a fresh list: callers may change theirs


def _dijkstra(game: Game, p: int, a: str, b: str) -> Tuple[float, List[int]]:
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
        alert_trains: Optional[int] = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.avoid_final_ticket_draw = avoid_final_ticket_draw
        self.draw_tickets_min_trains = draw_tickets_min_trains
        self.min_filler_route = min_filler_route
        self.alert_trains = alert_trains  # None: never look at the opponents (plain greedy)

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

    def _alert(self, game: Game, p: int) -> bool:
        """The end is near: an opponent is down to `alert_trains`, or the final round is on."""
        if self.alert_trains is None:
            return False
        if game.final_turns_remaining is not None:
            return True
        return min(pl.trains for q, pl in enumerate(game.players) if q != p) <= self.alert_trains

    def _turns_left(self, game: Game, p: int) -> int:
        """A rough count of my turns before the game ends, once alerted: the
        opponent nearest the end places about 3 trains a turn until 2 are left,
        then everyone gets one more turn."""
        if game.final_turns_remaining is not None:
            return 1
        low = min(pl.trains for q, pl in enumerate(game.players) if q != p)
        return max(0, low - 2) // 3 + 1

    def _turns_needed(self, game: Game, p: int, path: List[int]) -> int:
        """Turns to claim `path`: one per route, plus two cards a turn for the
        cards still missing (Locomotives in hand fill any gap)."""
        hand = game.players[p].hand
        missing = 0
        for rid in path:
            r = game.board.routes[rid]
            have = hand[r.color] if r.color is not None else max(
                (n for c, n in hand.items() if c is not Color.LOCOMOTIVE), default=0)
            missing += max(0, r.length - have)
        missing = max(0, missing - hand[Color.LOCOMOTIVE])
        return len(path) + (missing + 1) // 2

    def _plans(self, game: Game, p: int) -> List[Tuple[float, Ticket, List[int]]]:
        """(trains still needed, ticket, unclaimed routes on its cheapest path) for
        each incomplete ticket still worth pursuing, highest points first. Alerted,
        only tickets that can still be finished in the turns likely left."""
        plans = []
        alert = self._alert(game, p)
        turns = self._turns_left(game, p) if alert else 0
        for t in sorted(self._open_tickets(game, p), key=lambda t: -t.points):
            cost, path = cheapest_path(game, p, t.a, t.b)
            if cost == INF or cost > game.players[p].trains:
                continue  # unreachable or unaffordable; stop investing in it
            if alert and self._turns_needed(game, p, path) > turns:
                continue  # too late for this one
            plans.append((cost, t, path))
        return plans

    def _targets(self, game: Game, p: int) -> Set[int]:
        """Unclaimed routes on the cheapest paths of the tickets in `_plans`, plus
        their open double-route siblings (either half will do)."""
        targets: Set[int] = set()
        for _, _, path in self._plans(game, p):
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
        alert = self._alert(game, p)
        targets = self._targets(game, p)

        target_claims = [a for a in claims if a.route_id in targets]
        if target_claims:
            return max(target_claims, key=lambda a: routes[a.route_id].length)

        if final_round and claims:  # last chance to turn cards into points
            return max(claims, key=lambda a: routes[a.route_id].length)

        if not targets:
            can_draw_tickets = DrawTickets() in legal and not alert and not (
                final_round and self.avoid_final_ticket_draw
            )
            if can_draw_tickets and game.players[p].trains >= self.draw_tickets_min_trains:
                return DrawTickets()
            shortest = 1 if alert else self.min_filler_route  # alerted: cash in any route
            fillers = [a for a in claims if routes[a.route_id].length >= shortest]
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


class WaryAgent(GreedyAgent):
    """Greedy that watches the tempo (module docstring): alerted once an opponent
    is down to `alert_trains` trains."""

    name = "wary"

    def __init__(self, seed: Optional[int] = None, alert_trains: int = 15, **options) -> None:
        super().__init__(seed, alert_trains=alert_trains, **options)


class CollectorAgent(WaryAgent):
    """Greedy for tickets: finish as many as it can. It keeps a pipeline of open
    tickets, drawing more whenever fewer than `min_open` are far from done (a ticket
    within `close_trains` of done doesn't count) and keeping up to `max_open` at a
    time, cheapest first. It claims for the ticket nearest completion first, so
    tickets get finished one at a time rather than all left half-built. Tempo-aware
    like wary: once alerted it draws no more tickets and drops hopeless ones."""

    name = "collector"

    def __init__(self, seed: Optional[int] = None, min_open: int = 2, max_open: int = 3, close_trains: int = 3,
                 draw_tickets_min_trains: int = 8, **options) -> None:
        super().__init__(seed, draw_tickets_min_trains=draw_tickets_min_trains, **options)
        self.min_open = min_open
        self.max_open = max_open
        self.close_trains = close_trains

    def _far(self, plans) -> int:
        return sum(1 for cost, _, _ in plans if cost > self.close_trains)

    def _main_action(self, game: Game, p: int, legal: List[Action]) -> Action:
        routes = game.board.routes
        plans = self._plans(game, p)
        need: Dict[int, float] = {}  # route -> trains still needed by the nearest-done ticket it serves
        for cost, _, path in plans:
            for rid in path:
                sib = routes[rid].sibling
                for r in (rid, sib) if sib is not None and game.route_open_to(p, routes[sib]) else (rid,):
                    need[r] = min(need.get(r, INF), cost)
        claims = [a for a in legal if isinstance(a, ClaimRoute) and a.route_id in need]
        if claims:
            return min(claims, key=lambda a: (need[a.route_id], -routes[a.route_id].length))
        if (DrawTickets() in legal and not self._alert(game, p)
                and game.players[p].trains >= self.draw_tickets_min_trains and self._far(plans) < self.min_open):
            return DrawTickets()
        return super()._main_action(game, p, legal)

    def _choose_tickets(self, game: Game, p: int, legal: List[Action]) -> Action:
        player = game.players[p]
        min_keep = min(len(a.ticket_ids) for a in legal)
        plans = self._plans(game, p)
        budget = player.trains - sum(min(cost, player.trains) for cost, _, _ in plans)
        offered = sorted(((cheapest_path(game, p, t.a, t.b)[0], -t.points, t)
                          for t in (game.board.tickets[i] for i in player.pending_tickets)),
                         key=lambda x: (x[0], x[1]))
        room = self.max_open - self._far(plans)
        keep: List[int] = []
        for cost, _, t in offered:
            if len(keep) < min_keep or (len(keep) < room and cost <= budget * 0.6):
                keep.append(t.id)
                budget -= cost if cost != INF else 0
        choice = KeepTickets(frozenset(keep))
        return choice if choice in legal else legal[0]
