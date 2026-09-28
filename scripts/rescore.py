"""Re-score trained agents (linear and DQN runs) against greedy on fresh games.

    python scripts/rescore.py "runs/linear/pass5/*.json" "runs/linear/pass6/*.json" --out runs/linear/rescore.csv

A run's best checkpoint is picked by the same evaluation that scores it, so its recorded
margin is optimistic. This plays the final weights and the best checkpoint of every run
against greedy on the same fresh games (batch seed 9001 by default, never used in
training), so the numbers are unbiased and paired across agents. One row per agent in
the CSV; the printed table averages each setting (NAME_s<seed>) over its seeds.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ttr.board import load_board
from ttr.simulate import run_matches


def score(task):
    spec, games, seed = task
    stats = run_matches([spec, "greedy"], games=games, board=load_board("usa"), seed=seed)["stats"]
    me, opp = stats
    margins = [a - b for a, b in zip(me.totals, opp.totals)]
    return {
        "agent": spec,
        "games": games,
        "margin": st.mean(margins),
        "se": st.stdev(margins) / len(margins) ** 0.5,
        "win_share": me.wins / games,
        "tickets_completed": st.mean(me.tickets_completed),
        "tickets_failed": st.mean(me.tickets_failed),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="+", help="run files (wildcards expanded here)")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True, help="CSV, one row per agent")
    args = ap.parse_args()

    tasks = []
    for pattern in args.patterns:
        for path in sorted(glob.glob(pattern)):
            path = path.replace("\\", "/")
            with open(path) as f:
                run = json.load(f)
            method = run.get("method", "linear")  # DQN runs say so; linear runs predate the field
            tasks.append((f"{method}:{path}", args.games, args.seed))
            if run.get("best"):
                tasks.append((f"{method}:{path}@best", args.games, args.seed))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(score, tasks))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        path = r["agent"].split(":", 1)[1]
        which = "best" if path.endswith("@best") else "final"
        setting = path.removesuffix("@best").rsplit("/", 1)[-1].removesuffix(".json").rsplit("_s", 1)[0]
        groups[setting][which].append(r)
    print(f"vs greedy, {args.games} fresh games each (seed {args.seed}); mean over seeds, per-seed in brackets")
    for setting, by in groups.items():
        for which in ("final", "best"):
            rs = by.get(which)
            if rs:
                seeds = ", ".join(f"{r['margin']:+.0f}" for r in rs)
                print(f"{setting:24} {which:5} {st.mean(r['margin'] for r in rs):+6.1f} ({seeds})"
                      f"  win {st.mean(r['win_share'] for r in rs):4.0%}"
                      f"  tickets {st.mean(r['tickets_completed'] for r in rs):.2f}/{st.mean(r['tickets_failed'] for r in rs):.2f}")


if __name__ == "__main__":
    main()
