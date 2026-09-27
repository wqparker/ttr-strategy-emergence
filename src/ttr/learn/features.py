"""Hand-made features of (state, action) for the linear Q-learning / SARSA agents
(PLAN.md "Methods to compare", tier A). First pass: a small set a player would
name when asked what matters, to be revised once the deep agents show which
features count.

Q(s, a) = w[type(a)] . phi(s, a). Every action type has its own weight vector
over the same state features plus features of that action, so the weights read
as "how much this agent likes each action type, and what makes one better":

    phi(s, a) = [bias, state features..., action features for type(a)...]

Everything is computed from the acting seat's view: its own hand and tickets,
the table, and route ownership (RULES.md §9 #17). Values are scaled to about
[0, 1] (the margin to [-1, 1]).

Planning quantities come from `cheapest_path` (ttr.agents.greedy): for each of
my incomplete tickets, the fewest trains still needed through my routes (free)
and routes open to me, and the unclaimed routes on that path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ttr.actions import (
    Action,
    ClaimRoute,
    DrawBlind,
    DrawFaceUp,
    DrawTickets,
    KeepTickets,
    Pass,
    Pay,
)
from ttr.agents.greedy import INF, cheapest_path
from ttr.cards import Color
from ttr.game import END_TRIGGER_TRAINS, Game, Phase
from ttr.scoring import connected

STATE_FEATURES = (
    "bias",
    "my_trains",  # trains left / 45
    "opp_trains",  # fewest trains any opponent has left / 45
    "final_round",  # the final round has started
    "hand_size",  # cards held / 30, capped at 1
    "open_tickets",  # incomplete tickets held / 5, capped
    "trains_needed",  # trains to finish every reachable incomplete ticket / 45, capped
    "tickets_lost",  # incomplete tickets that can't be finished (no path or too few trains) / 3, capped
    "route_margin",  # my route points - best opponent's / 100, in [-1, 1]
)

ACTION_FEATURES: Dict[str, Tuple[str, ...]] = {
    "claim": (
        "points",  # route points / 15
        "length",  # length / 6
        "on_ticket_path",  # on the cheapest path of one of my incomplete tickets
        "ticket_points",  # points of the tickets whose path uses it / 20, capped
        "completes_ticket",  # the last route one of those tickets needs
        "triggers_end",  # leaves me with 2 or fewer trains
    ),
    # The route's points are scored when it is paid for, so the payment step
    # carries the route's features too: Q(claim r) learns from max Q(pay) and
    # could not tell routes apart otherwise.
    "pay": (
        "locomotives",  # Locomotives spent / 6
        "needed_elsewhere",  # cards of this color my other path routes still need / 6
        "route_points",  # the route's points / 15
        "on_ticket_path",
        "ticket_points",
        "completes_ticket",
    ),
    "draw_color": (
        "needed",  # a color my path routes still need
        "need_amount",  # how many of it they still need / 6
        "second_draw",
    ),
    "draw_locomotive": (),
    "draw_blind": (
        "second_draw",
        "useful_face_up",  # a face-up card of a needed color was on offer instead
    ),
    "draw_tickets": (
        "spare_trains",  # trains left beyond what my incomplete tickets need / 45
    ),
    "keep": (
        "count",  # tickets kept / 3
        "points",  # their points / 30, capped
        "path_trains",  # sum of their cheapest paths / 45, capped (unreachable counts 45)
        "fits",  # that sum fits in the trains not already committed
        "unreachable",  # one of them has no path at all
        "initial",  # the opening choice (keep at least 2)
        "doomed_points",  # on my last turn: points of kept tickets not already connected / 30
    ),
    "pass": (),
}

ACTION_TYPES = tuple(ACTION_FEATURES)


def feature_names(kind: str) -> Tuple[str, ...]:
    return STATE_FEATURES + ACTION_FEATURES[kind]


def action_type(action: Action) -> str:
    if isinstance(action, ClaimRoute):
        return "claim"
    if isinstance(action, Pay):
        return "pay"
    if isinstance(action, DrawFaceUp):
        return "draw_locomotive" if action.color is Color.LOCOMOTIVE else "draw_color"
    if isinstance(action, DrawBlind):
        return "draw_blind"
    if isinstance(action, DrawTickets):
        return "draw_tickets"
    if isinstance(action, KeepTickets):
        return "keep"
    if isinstance(action, Pass):
        return "pass"
    raise TypeError(f"not an action: {action!r}")


@dataclass
class Context:
    """What the features of every legal action share, computed once per decision."""

    state: np.ndarray
    trains: int
    committed: int  # trains to finish my reachable incomplete tickets
    route_tickets: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)  # route -> [(ticket points, trains needed)]
    needs: Counter = field(default_factory=Counter)  # color -> cards my colored path routes still lack
    offer_costs: Dict[int, float] = field(default_factory=dict)  # ticket on offer -> cheapest path (INF if none)


def _cap(x: float) -> float:
    return min(x, 1.0)


def context(game: Game, p: int) -> Context:
    me = game.players[p]
    board = game.board
    full = board.trains_per_player
    mine = [board.routes[r] for r in me.routes]

    route_tickets: Dict[int, List[Tuple[int, int]]] = {}
    open_count = lost = committed = 0
    for tid in me.tickets:
        t = board.tickets[tid]
        if connected(mine, t.a, t.b):
            continue
        open_count += 1
        cost, path = cheapest_path(game, p, t.a, t.b)
        if cost == INF or cost > me.trains:
            lost += 1
            continue
        committed += int(cost)
        for rid in path:
            route_tickets.setdefault(rid, []).append((t.points, int(cost)))

    needs: Counter = Counter()
    for rid in route_tickets:
        r = board.routes[rid]
        if r.color is not None:
            needs[r.color] = max(needs[r.color], r.length - me.hand[r.color])
    needs = +needs

    opponents = [q for q in range(game.num_players) if q != p]
    best_opp = max(game.players[q].route_points for q in opponents)
    state = np.array([
        1.0,
        me.trains / full,
        min(game.players[q].trains for q in opponents) / full,
        float(game.final_turns_remaining is not None),
        _cap(me.hand_size / 30),
        _cap(open_count / 5),
        _cap(committed / full),
        _cap(lost / 3),
        max(-1.0, min(1.0, (me.route_points - best_opp) / 100)),
    ], dtype=np.float64)
    offer_costs = {}
    for tid in me.pending_tickets:
        t = board.tickets[tid]
        offer_costs[tid] = cheapest_path(game, p, t.a, t.b)[0]
    return Context(state=state, trains=me.trains, committed=committed, route_tickets=route_tickets,
                   needs=needs, offer_costs=offer_costs)


def action_features(game: Game, p: int, ctx: Context, action: Action) -> Tuple[str, np.ndarray]:
    """(action type, phi(s, a)) for one legal action."""
    kind = action_type(action)
    board = game.board
    full = board.trains_per_player
    second = float(game.phase is Phase.DRAW_SECOND_CARD)

    if kind == "claim":
        r = board.routes[action.route_id]
        served = ctx.route_tickets.get(r.id, [])
        extra = [
            r.points / 15,
            r.length / 6,
            float(bool(served)),
            _cap(sum(pts for pts, _ in served) / 20),
            float(any(need == r.length for _, need in served)),
            float(ctx.trains - r.length <= END_TRIGGER_TRAINS),
        ]
    elif kind == "pay":
        pending = game.pending_route
        needs: Counter = Counter()
        hand = game.players[p].hand
        for rid in ctx.route_tickets:
            r = board.routes[rid]
            if rid != pending and r.color is not None:
                needs[r.color] = max(needs[r.color], r.length - hand[r.color])
        conflict = needs[action.color] if action.color is not None else 0
        route = board.routes[pending]
        served = ctx.route_tickets.get(pending, [])
        extra = [
            action.locomotives / 6,
            max(conflict, 0) / 6,
            route.points / 15,
            float(bool(served)),
            _cap(sum(pts for pts, _ in served) / 20),
            float(any(need == route.length for _, need in served)),
        ]
    elif kind == "draw_color":
        need = ctx.needs[action.color]
        extra = [float(need > 0), _cap(need / 6), second]
    elif kind == "draw_blind":
        useful = any(c is not Color.LOCOMOTIVE and ctx.needs[c] > 0 for c in game.market)
        extra = [second, float(useful)]
    elif kind == "draw_tickets":
        extra = [max(0, ctx.trains - ctx.committed) / full]
    elif kind == "keep":
        costs = [ctx.offer_costs[tid] for tid in action.ticket_ids]
        points = sum(board.tickets[tid].points for tid in action.ticket_ids)
        total = sum(min(c, full) for c in costs)
        # Every turn in the final round is that player's last (§7), so a ticket
        # kept then can only score if it is already connected.
        last_turn = game.final_turns_remaining is not None
        doomed = sum(board.tickets[t].points for t, c in zip(action.ticket_ids, costs) if c > 0) if last_turn else 0
        extra = [
            len(costs) / 3,
            _cap(points / 30),
            _cap(total / full),
            float(total <= ctx.trains - ctx.committed),
            float(any(c == INF for c in costs)),
            float(game.phase is Phase.CHOOSE_INITIAL_TICKETS),
            _cap(doomed / 30),
        ]
    else:  # draw_locomotive, pass
        extra = []
    return kind, np.concatenate([ctx.state, np.asarray(extra, dtype=np.float64)])


def all_features(game: Game, p: int, actions: Optional[Sequence[Action]] = None
                 ) -> List[Tuple[str, np.ndarray]]:
    """(type, phi) for each legal action, in `game.legal_actions()` order."""
    if actions is None:
        actions = game.legal_actions()
    ctx = context(game, p)
    return [action_features(game, p, ctx, a) for a in actions]
