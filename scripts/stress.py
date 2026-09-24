"""Engine stress test: many random/greedy games across boards and player counts,
checking invariants after every step. Slower than the unit tests (~1 min).

    .venv/Scripts/python scripts/stress.py
"""
import random, sys, time
from ttr import Game, load_board, Color
from ttr.agents import GreedyAgent, RandomAgent
L = Color.LOCOMOTIVE
boards = {n: load_board(n) for n in ("usa", "toy")}
t0 = time.time(); games = 0; steps = 0; guard_hits = 0; passes = 0; trunc = 0
for seed in range(1500):
    bname = "usa" if seed % 3 else "toy"
    b = boards[bname]
    n = 2 + seed % 4
    mix = seed % 3  # 0 random, 1 greedy, 2 mixed
    agents = [RandomAgent(seed+i) if (mix == 0 or (mix == 2 and i % 2)) else GreedyAgent(seed+i) for i in range(n)]
    g = Game(b, num_players=n, seed=seed, max_turns=3000)
    while not g.game_over:
        p = g.current_player
        g.step(agents[p].act(g, p)); steps += 1
        tot = len(g.deck)+len(g.discard)+len(g.market)+sum(sum(pl.hand.values()) for pl in g.players)
        assert tot == 110, (seed, tot)
        assert all(v > 0 for pl in g.players for v in pl.hand.values())
        assert len(g.market) == 5 or not (g.deck or g.discard), (seed, "short market")
        if g.market.count(L) >= 3:
            assert not g._legal_market_possible(), seed
            guard_hits += 1
        for i, pl in enumerate(g.players):
            rs = [b.routes[r] for r in pl.routes]
            assert pl.trains == b.trains_per_player - sum(r.length for r in rs)
            assert pl.route_points == sum(r.points for r in rs)
        tk = sum(len(pl.tickets)+len(pl.pending_tickets) for pl in g.players) + len(g.ticket_deck) + len(getattr(g, "_initial_returns", []))
        assert tk == len(b.tickets), (seed, tk)
        if n <= 3:
            for r in b.routes:
                if r.sibling is not None:
                    assert not (r.id in g.route_owner and r.sibling in g.route_owner), seed
        for r in b.routes:
            if r.sibling is not None and r.id in g.route_owner and r.sibling in g.route_owner:
                assert g.route_owner[r.id] != g.route_owner[r.sibling]
    passes += sum(e.kind == "pass" for e in g.log)
    trunc += g.result.truncated
    games += 1
print(f"{games} games, {steps} steps, {time.time()-t0:.0f}s; market-guard states {guard_hits}; passes {passes}; truncated {trunc}")
