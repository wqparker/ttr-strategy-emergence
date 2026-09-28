"""Round robin: every pair of agents plays the same number of 2-player games on
fresh seeds, random seats (PLAN.md "Common evaluation"), then Elo ratings from
all the results.

    python scripts/round_robin.py random greedy wary racer \
        linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best \
        dqn:runs/dqn/pass1/d1a_n1_s2.json@best --out runs/round_robin_2026-09-28

Prints the margin matrix (row agent minus column agent, mean over games, with the
standard error), win shares and the ratings; writes one CSV row per pair (PREFIX.csv)
and the printed tables (PREFIX.txt).

Ratings: a Bradley-Terry fit of the win shares (a shared win counts half to each),
on the Elo scale (400 points = 10:1 odds), greedy at 1000 when it plays (else the
mean). One virtual drawn game per pair keeps an agent that never won finite.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ttr.board import load_board
from ttr.simulate import run_matches


def play(task: Tuple[str, str, int, int]) -> dict:
    a, b, games, seed = task
    stats = run_matches([a, b], games=games, board=load_board("usa"), seed=seed)["stats"]
    sa, sb = stats
    margins = [x - y for x, y in zip(sa.totals, sb.totals)]
    return {"a": a, "b": b, "games": games, "margin": st.mean(margins),
            "se": st.stdev(margins) / len(margins) ** 0.5,
            "a_wins": sa.wins, "b_wins": sb.wins,  # a shared win already counts half to each
            "a_tickets_completed": st.mean(sa.tickets_completed), "b_tickets_completed": st.mean(sb.tickets_completed)}


def ratings(agents: Sequence[str], rows: Sequence[dict], iterations: int = 2000) -> Dict[str, float]:
    """Bradley-Terry by the minorization-maximization update, on the Elo scale."""
    wins = {x: 0.5 * (len(agents) - 1) for x in agents}  # one virtual draw per pair
    games: Dict[Tuple[str, str], float] = {}
    for r in rows:
        wins[r["a"]] += r["a_wins"]
        wins[r["b"]] += r["b_wins"]
        games[r["a"], r["b"]] = games[r["b"], r["a"]] = r["games"] + 1
    strength = {x: 1.0 for x in agents}
    for _ in range(iterations):
        new = {x: wins[x] / sum(games[x, y] / (strength[x] + strength[y]) for y in agents if y != x)
               for x in agents}
        mean = math.exp(st.mean(math.log(v) for v in new.values()))
        strength = {x: v / mean for x, v in new.items()}
    elo = {x: 400 * math.log10(v) for x, v in strength.items()}
    anchor = elo["greedy"] if "greedy" in elo else st.mean(elo.values())
    return {x: 1000 + v - anchor for x, v in elo.items()}


def short(spec: str) -> str:
    """A readable column name: the run file's stem for trained agents."""
    kind, _, arg = spec.partition(":")
    if not arg:
        return spec
    best = arg.endswith("@best")
    stem = Path(arg.removesuffix("@best")).stem
    return f"{kind}:{stem}{'@best' if best else ''}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agents", nargs="+", help="agent specs (ttr.agents.registry)")
    ap.add_argument("--games", type=int, default=400, help="games per pair")
    ap.add_argument("--seed", type=int, default=9101, help="batch seed (fresh: not used in training)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True, help="output prefix: PREFIX.csv and PREFIX.txt")
    args = ap.parse_args()

    agents: List[str] = list(dict.fromkeys(args.agents))
    tasks = [(a, b, args.games, args.seed) for a, b in itertools.combinations(agents, 2)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(play, tasks))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_pair = {(r["a"], r["b"]): r for r in rows}
    names = {a: short(a) for a in agents}
    width = max(len(n) for n in names.values())
    elo = ratings(agents, rows)
    order = sorted(agents, key=lambda x: -elo[x])
    lines = [f"{args.games} games per pair (seed {args.seed}), random seats; row vs column", "",
             "margin (row - column) +- standard error"]
    cols = [f"{i + 1:>11}" for i in range(len(order))]
    lines.append(" " * (width + 5) + "".join(cols))
    for i, a in enumerate(order):
        cells = []
        for b in order:
            if a == b:
                cells.append(f"{'-':>11}")
            else:
                r = by_pair.get((a, b)) or by_pair[(b, a)]
                m = r["margin"] if r["a"] == a else -r["margin"]
                cells.append(f"{m:+6.1f}+-{r['se']:<3.1f}")
        lines.append(f"{i + 1:>2}. {names[a]:<{width}} " + "".join(cells))
    lines += ["", "win share (row vs column)", " " * (width + 5) + "".join(cols)]
    for i, a in enumerate(order):
        cells = []
        for b in order:
            if a == b:
                cells.append(f"{'-':>11}")
            else:
                r = by_pair.get((a, b)) or by_pair[(b, a)]
                wins = r["a_wins"] if r["a"] == a else r["b_wins"]
                cells.append(f"{wins / r['games']:>11.0%}")
        lines.append(f"{i + 1:>2}. {names[a]:<{width}} " + "".join(cells))
    lines += ["", "ratings (Elo scale, greedy = 1000)"]
    lines += [f"{i + 1:>2}. {names[a]:<{width}} {elo[a]:7.0f}" for i, a in enumerate(order)]
    text = "\n".join(lines)
    print(text)
    args.out.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
