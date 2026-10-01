"""Instrumented games for the research paper (docs/paper/): what agents do, measured claim by claim.

    python docs/paper/measure.py anatomy      # one agent per method vs greedy and racer (2 players)
    python docs/paper/measure.py blocking     # PPO passes 3-5: blocking against chance, tempo, cross-play
    python docs/paper/measure.py blocking-self  # the same for pass 5's 4-player self-play arms
    python docs/paper/measure.py ticket-gap   # how far racing networks end from the tickets they drop
    python docs/paper/measure.py scripted     # scripted bots at 2-5 players: ticket planners vs racers
    (--workers N, --games N; results merge into docs/paper/measured.json, one key per measurement)

Per claim it records the route's length, whether the route (or, at 2-3 players, its
double-route sibling, closed by the claim) lies on the cheapest path of an incomplete
ticket of any opponent (true tickets: measurement only), and two chance baselines: the
share of the claimer's legal claims that would block (uniform), and the same among
legal claims of the chosen length (length-matched). Blocking means claiming such routes
more often than the length-matched baseline. Fresh seeds, never used in training.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ttr.actions import ClaimRoute
from ttr.agents.greedy import INF, cheapest_path
from ttr.agents.registry import make_agent
from ttr.board import load_board
from ttr.game import Game
from ttr.learn.metrics import game_metrics
from ttr.scoring import connected
from ttr.simulate import run_matches

OUT = Path(__file__).with_name("measured.json")
PPO = "ppo:runs/ppo"
LIN_BEST = "linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best"
# one representative per method (the strongest by the round robins), plus the bots
ANATOMY = {
    "greedy": "greedy", "racer": "racer", "wary": "wary", "collector": "collector",
    "linear (Q/SARSA)": LIN_BEST,
    "DQN": "dqn:runs/dqn/pass3/d3a_pool_s1.json@best",
    "PPO margin": f"{PPO}/pass2/p2b_pool_s3.json",
    "PPO score": f"{PPO}/pass3/p3b_score_s2.json",
}
# PPO arms by table size: (margin arm, score arm, seeds)
ARMS = {2: ("pass3/p3a_margin", "pass3/p3b_score", range(4)), 3: ("pass4/p4a_3p_margin", "pass4/p4b_3p_score", range(3)),
        4: ("pass4/p4c_4p_margin", "pass4/p4d_4p_score", range(3)), 5: ("pass5/p5a_5p_margin", "pass5/p5b_5p_score", range(3))}
SELF = ("pass5/p5c_4p_self_margin", "pass5/p5d_4p_self_score", range(3))
METRIC_KEYS = ("margin", "win_share", "score", "opp_score", "route_points", "ticket_points", "tickets_kept",
               "tickets_completed", "tickets_failed", "claims", "mean_claim_length", "longest_bonus",
               "triggered_end", "turns", "trains_left", "draw_blind", "draw_color")


def spec(name: str) -> str:
    return name if ":" in name else f"{PPO}/{name}.json" if name.startswith("pass") else name


def on_paths(game: Game, q: int) -> set:
    """Unclaimed routes on the cheapest paths of q's incomplete tickets, with the
    double-route siblings a claim would close (2-3 players)."""
    board = game.board
    mine = [board.routes[r] for r in game.players[q].routes]
    out = set()
    for tid in game.players[q].tickets:
        t = board.tickets[tid]
        if connected(mine, t.a, t.b):
            continue
        cost, path = cheapest_path(game, q, t.a, t.b)
        if cost == INF:
            continue
        for r in path:
            out.add(r)
            sib = board.routes[r].sibling
            if sib is not None and game.num_players <= 3:
                out.add(sib)
    return out


def play(task) -> list:
    """Games with `labels[k]` (agent specs) in rotating seats; one row per seat per game."""
    labels, specs, games, seed0, tag = task
    board = load_board("usa")
    n = len(specs)
    rows = []
    for g in range(games):
        game = Game(board, n, seed=seed0 + g)
        order = [(g + k) % n for k in range(n)]  # slot sitting in each seat
        agents = [make_agent(specs[slot], seed0 + 97 * g + k) for k, slot in enumerate(order)]
        extra = [defaultdict(float) for _ in range(n)]
        while not game.game_over:
            p = game.current_player
            act = agents[p].act(game, p)
            if isinstance(act, ClaimRoute):
                r = board.routes[act.route_id]
                e = extra[p]
                hits = set().union(*(on_paths(game, q) for q in range(n) if q != p))
                legal = [a.route_id for a in game.legal_actions() if isinstance(a, ClaimRoute)]
                same = [x for x in legal if board.routes[x].length == r.length]
                e["claims"] += 1
                e[f"len{r.length}"] += 1
                e["blocks"] += act.route_id in hits
                e["chance_uniform"] += sum(x in hits for x in legal) / len(legal)
                e["chance_length"] += sum(x in hits for x in same) / len(same)
                if game.players[p].trains - r.length <= 22 and "half_turn" not in e:
                    e["half_turn"] = game.turn
            game.step(act)
        for s in range(n):
            m = game_metrics(game, s)
            others = [q for q in range(n) if q != s]
            row = {"tag": tag, "label": labels[order[s]], "players": n, "game": g,
                   "rank": 1 + sum(game.result.players[q].total > game.result.players[s].total for q in others)}
            row.update({k: m[k] for k in METRIC_KEYS})
            ms = [game_metrics(game, q) for q in others]
            for k in ("tickets_completed", "ticket_points", "route_points", "trains_left"):
                row["opp_" + k] = sum(x[k] for x in ms) / len(ms)
            for k in ("blocks", "chance_uniform", "chance_length", "len1", "len2", "len3", "len4", "len5", "len6"):
                row[k] = extra[s][k]
            row["half_turn"] = extra[s].get("half_turn", game.turn) / n  # in the seat's own turns
            row["own_turns"] = game.turn / n
            rows.append(row)
    return rows


def summarize(rows: list, key=lambda r: (r["tag"], r["label"])) -> list:
    """Means (and standard errors for margin and excess blocks) per key."""
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = []
    for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        claims = sum(r["claims"] for r in rs) or 1
        excess = [r["blocks"] - r["chance_length"] for r in rs]
        margins = [r["margin"] for r in rs]
        row = {"key": list(k) if isinstance(k, tuple) else k, "games": len(rs), "players": rs[0]["players"],
               **{m: round(st.mean(r[m] for r in rs), 3) for m in METRIC_KEYS + ("rank", "half_turn", "own_turns")
                  + ("opp_tickets_completed", "opp_ticket_points", "opp_route_points", "opp_trains_left")
                  + tuple(f"len{i}" for i in range(1, 7))},
               "margin_se": round(st.stdev(margins) / len(margins) ** 0.5, 3) if len(rs) > 1 else 0.0,
               "blocks_per_claim": round(sum(r["blocks"] for r in rs) / claims, 4),
               "chance_length_per_claim": round(sum(r["chance_length"] for r in rs) / claims, 4),
               "chance_uniform_per_claim": round(sum(r["chance_uniform"] for r in rs) / claims, 4),
               "excess_blocks": round(st.mean(excess), 3),
               "excess_blocks_se": round(st.stdev(excess) / len(excess) ** 0.5, 3) if len(rs) > 1 else 0.0}
        out.append(row)
    return out


def run(tasks: list, workers: int) -> list:
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(play, tasks):
            rows += r
    return rows


def anatomy(games: int, workers: int) -> dict:
    tasks = [([label, bot], [ANATOMY[label], bot], games, 424242, f"{label} vs {bot}")
             for label in ANATOMY for bot in ("greedy", "racer")]
    return {"games_per_matchup": games, "rows": summarize(run(tasks, workers))}


def blocking(games: int, workers: int, self_play: bool = False) -> dict:
    """PPO arms with N - 1 copies of greedy or racer, and margin and score arms at one
    table; with `self_play`, pass 5's 4-player self-play arms instead."""
    tasks = []
    arms = {} if self_play else dict(ARMS)
    for n, (m, sc, seeds) in arms.items():
        for arm in (m, sc):
            for s in seeds:
                for bot in ("greedy", "racer"):
                    labels = [arm] + [bot] * (n - 1)
                    tasks.append((labels, [spec(f"{arm}_s{s}")] + [bot] * (n - 1), games, 424242, f"{n}p {arm} vs {bot}"))
        for i in seeds:  # cross-play: margin and score arms at one table
            for j in seeds:
                mi, sj = spec(f"{m}_s{i}"), spec(f"{sc}_s{j}")
                if n == 2:
                    tasks.append(([m, sc], [mi, sj], games // 2, 525252, "2p cross"))
                elif n == 4:
                    tasks.append(([m, m, sc, sc], [mi, mi, sj, sj], games // 2, 525252, "4p cross"))
                else:  # 3 and 5 players: majorities both ways
                    k = n // 2
                    tasks.append(([m] * (k + 1) + [sc] * k, [mi] * (k + 1) + [sj] * k, games // 4, 525252, f"{n}p cross"))
                    tasks.append(([m] * k + [sc] * (k + 1), [mi] * k + [sj] * (k + 1), games // 4, 626262, f"{n}p cross"))
    if self_play:
        m, sc, seeds = SELF
        for arm in (m, sc):
            for s in seeds:
                for bot in ("greedy", "racer"):
                    tasks.append(([arm] + [bot] * 3, [spec(f"{arm}_s{s}")] + [bot] * 3, games, 424242, f"4p {arm} vs {bot}"))
    rows = run(tasks, workers)
    # every seat at learner tables, pooled over seeds: the arms, and the bots beside them as reference
    return {"games_per_seed": games, "rows": summarize(rows)}


def ticket_gap(games: int, workers: int) -> dict:
    matchups = [("PPO margin", "racer"), ("PPO margin", "greedy"), ("PPO score", "racer"), ("PPO score", "greedy"),
                ("racer", "greedy")]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_gap, [(me, opp, games) for me, opp in matchups]))
    return {"games_per_matchup": games, "rows": results}


def _gap(task) -> dict:
    me, opp, games = task
    board = load_board("usa")
    kept_cost, kept_pts, best_pts = [], [], []
    for g in range(games):
        game = Game(board, 2, seed=606060 + g)
        seat = g % 2
        agents = [None, None]
        agents[seat] = make_agent(ANATOMY[me], 606060 + g)
        agents[1 - seat] = make_agent(opp, 606060 + g + 7)
        offer = next(e.private["tickets"] for e in game.log if e.kind == "deal_initial_tickets" and e.player == seat)
        while not game.game_over:
            p = game.current_player
            game.step(agents[p].act(game, p))
        kept = set(game.players[seat].tickets)
        rows = []
        for tid in offer:
            t = board.tickets[tid]
            rows.append((tid in kept, cheapest_path(game, seat, t.a, t.b)[0], t.points))
        kept_cost += [c for k, c, _ in rows if k]
        kept_pts.append(sum(p if c == 0 else -p for k, c, p in rows if k))
        two = sorted(rows, key=lambda r: r[1])[:2]
        best_pts.append(sum(p if c == 0 else -p for _, c, p in two))
    reachable = [c for c in kept_cost if c != INF]
    share = lambda x: round(sum(1 for c in kept_cost if c <= x) / len(kept_cost), 3)
    return {"agent": me, "opponent": opp, "kept_median_trains": st.median(reachable), "kept_done": share(0),
            "kept_within_3": share(3), "kept_within_6": share(6),
            "unreachable": round(1 - len(reachable) / len(kept_cost), 3),
            "ticket_points_kept": round(st.mean(kept_pts), 2), "ticket_points_best_two": round(st.mean(best_pts), 2)}


TABLES = [["greedy", "racer"], ["collector", "racer"], ["wary", "racer"],
          ["greedy", "racer", "racer"], ["racer", "greedy", "greedy"], ["collector", "racer", "racer"],
          ["greedy", "racer", "racer", "racer"], ["racer", "greedy", "greedy", "greedy"],
          ["greedy", "greedy", "racer", "racer"], ["collector", "racer", "racer", "racer"],
          ["greedy", "racer", "racer", "racer", "racer"], ["racer", "greedy", "greedy", "greedy", "greedy"],
          ["greedy", "greedy", "racer", "racer", "racer"], ["wary", "racer", "racer", "racer", "racer"],
          ["collector", "racer", "racer", "racer", "racer"]]


def _table(task) -> dict:
    agents, games = task
    stats = run_matches(agents, games=games, board=load_board("usa"), seed=8080)["stats"]
    by = defaultdict(list)
    for name, s in zip(agents, stats):
        by[name].append(s)
    return {"agents": agents, "players": len(agents), "games": games,
            "by_kind": {name: {"win": round(st.mean(s.wins / s.games for s in ss), 4),
                               "score": round(st.mean(st.mean(s.totals) for s in ss), 2),
                               "tickets_completed": round(st.mean(st.mean(s.tickets_completed) for s in ss), 2)}
                        for name, ss in by.items()}}


def scripted(games: int, workers: int) -> dict:
    with ProcessPoolExecutor(workers) as ex:
        return {"games_per_table": games, "tables": list(ex.map(_table, [(t, games) for t in TABLES]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("anatomy", "blocking", "blocking-self", "ticket-gap", "scripted"))
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.what == "anatomy":
        result = anatomy(args.games or 200, args.workers)
    elif args.what in ("blocking", "blocking-self"):
        result = blocking(args.games or 100, args.workers, self_play=args.what == "blocking-self")
    elif args.what == "ticket-gap":
        result = ticket_gap(args.games or 200, args.workers)
    else:
        result = scripted(args.games or 400, args.workers)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data[args.what] = result
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{args.what} -> {OUT}")


if __name__ == "__main__":
    main()
