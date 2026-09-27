"""Hand-made features of (state, action) for the linear Q-learning / SARSA agents
(PLAN.md "Methods to compare", tier A).

Second pass: value + advantage. One value vector over the state features is
shared by every action; each action type adds an advantage over its own features:

    Q(s, a) = v . state(s)  +  w[type(a)] . psi(s, a)
    psi(s, a) = [bias, features of this action (and the state it matters in)...]

The shared value absorbs how good the position is, so the advantage weights only
have to rank the actions available in it. In the first pass every type carried
its own copy of the state features, which took nearly all the weight and left the
features that separate one claim from another near zero (PLAN.md).

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

ENDGAME_TRAINS = 6  # an opponent this close to the end trigger means the game may end soon

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
        "trains_after",  # trains I'd have left / 45
        "endgame",  # the final round started or an opponent is near the trigger
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
        "hand_size",
        "endgame",
    ),
    "draw_locomotive": (
        "hand_size",
        "endgame",
    ),
    "draw_blind": (
        "second_draw",
        "useful_face_up",  # a face-up card of a needed color was on offer instead
        "hand_size",
        "endgame",
    ),
    "draw_tickets": (
        "spare_trains",  # trains left beyond what my incomplete tickets need / 45
        "last_turn",  # the final round: nothing drawn now can be finished
        "open_tickets",
        "my_trains",
    ),
    "keep": (
        "count",  # tickets kept / 3
        "points",  # their points / 30, capped
        "path_trains",  # sum of their cheapest paths / 45, capped (unreachable counts 45)
        "fits",  # that sum fits in the trains not already committed
        "unreachable",  # one of them has no path at all
        "initial",  # the opening choice (keep at least 2)
        "doomed_points",  # on my last turn: points of kept tickets not already connected / 30
        "shared_path",  # share of their path trains already on my other tickets' paths
        "hardest_share",  # the costliest one's path / trains not yet committed, capped
        "points_per_train",  # their points / their path trains / 3, capped
    ),
    "pass": (),
}

ACTION_TYPES = tuple(ACTION_FEATURES)


def feature_names(kind: str) -> Tuple[str, ...]:
    """Names of psi(s, a) for an action type: its bias, then its features."""
    return ("bias",) + ACTION_FEATURES[kind]


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
    open_tickets: int
    endgame: bool
    route_tickets: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)  # route -> [(ticket points, trains needed)]
    needs: Counter = field(default_factory=Counter)  # color -> cards my colored path routes still lack
    offer_costs: Dict[int, float] = field(default_factory=dict)  # ticket on offer -> cheapest path (INF if none)
    offer_shared: Dict[int, int] = field(default_factory=dict)  # ticket on offer -> its path trains already on my paths


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
    opp_trains = min(game.players[q].trains for q in opponents)
    final_round = game.final_turns_remaining is not None
    state = np.array([
        1.0,
        me.trains / full,
        opp_trains / full,
        float(final_round),
        _cap(me.hand_size / 30),
        _cap(open_count / 5),
        _cap(committed / full),
        _cap(lost / 3),
        max(-1.0, min(1.0, (me.route_points - best_opp) / 100)),
    ], dtype=np.float64)

    offer_costs: Dict[int, float] = {}
    offer_shared: Dict[int, int] = {}
    for tid in me.pending_tickets:
        t = board.tickets[tid]
        cost, path = cheapest_path(game, p, t.a, t.b)
        offer_costs[tid] = cost
        offer_shared[tid] = sum(board.routes[r].length for r in path if r in route_tickets)
    return Context(state=state, trains=me.trains, committed=committed, open_tickets=open_count,
                   endgame=final_round or opp_trains <= ENDGAME_TRAINS, route_tickets=route_tickets,
                   needs=needs, offer_costs=offer_costs, offer_shared=offer_shared)


def action_features(game: Game, p: int, ctx: Context, action: Action) -> Tuple[str, np.ndarray]:
    """(action type, psi(s, a)) for one legal action."""
    kind = action_type(action)
    board = game.board
    full = board.trains_per_player
    second = float(game.phase is Phase.DRAW_SECOND_CARD)
    hand = _cap(game.players[p].hand_size / 30)
    endgame = float(ctx.endgame)

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
            (ctx.trains - r.length) / full,
            endgame,
        ]
    elif kind == "pay":
        pending = game.pending_route
        needs: Counter = Counter()
        cards = game.players[p].hand
        for rid in ctx.route_tickets:
            r = board.routes[rid]
            if rid != pending and r.color is not None:
                needs[r.color] = max(needs[r.color], r.length - cards[r.color])
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
        extra = [float(need > 0), _cap(need / 6), second, hand, endgame]
    elif kind == "draw_locomotive":
        extra = [hand, endgame]
    elif kind == "draw_blind":
        useful = any(c is not Color.LOCOMOTIVE and ctx.needs[c] > 0 for c in game.market)
        extra = [second, float(useful), hand, endgame]
    elif kind == "draw_tickets":
        extra = [
            max(0, ctx.trains - ctx.committed) / full,
            float(game.final_turns_remaining is not None),
            _cap(ctx.open_tickets / 5),
            ctx.trains / full,
        ]
    elif kind == "keep":
        ids = sorted(action.ticket_ids)
        costs = [ctx.offer_costs[tid] for tid in ids]
        points = sum(board.tickets[tid].points for tid in ids)
        total = sum(min(c, full) for c in costs)
        reachable = [(tid, c) for tid, c in zip(ids, costs) if c != INF]
        path_total = sum(c for _, c in reachable)
        shared = sum(ctx.offer_shared[tid] for tid, _ in reachable)
        budget = max(1, ctx.trains - ctx.committed)
        # Every turn in the final round is that player's last (§7), so a ticket
        # kept then can only score if it is already connected.
        last_turn = game.final_turns_remaining is not None
        doomed = sum(board.tickets[t].points for t, c in zip(ids, costs) if c > 0) if last_turn else 0
        extra = [
            len(costs) / 3,
            _cap(points / 30),
            _cap(total / full),
            float(total <= ctx.trains - ctx.committed),
            float(any(c == INF for c in costs)),
            float(game.phase is Phase.CHOOSE_INITIAL_TICKETS),
            _cap(doomed / 30),
            shared / path_total if path_total else 0.0,
            _cap(max(min(c, full) for c in costs) / budget),
            _cap(points / max(total, 1) / 3),
        ]
    else:  # pass
        extra = []
    return kind, np.asarray([1.0] + extra, dtype=np.float64)


def all_features(game: Game, p: int, actions: Optional[Sequence[Action]] = None
                 ) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """The state vector and (type, psi) for each legal action, in
    `game.legal_actions()` order."""
    if actions is None:
        actions = game.legal_actions()
    ctx = context(game, p)
    return ctx.state, [action_features(game, p, ctx, a) for a in actions]
