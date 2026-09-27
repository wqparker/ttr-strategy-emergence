"""One line per linear training run: progress, latest evaluations and health checks.

    python scripts/run_status.py "runs/linear/pass4/*.json"

Flags a run as BROKEN (non-finite weights) or COLLAPSED (past 3000 games, its last
evaluation against random below -150 margin with fewer than 2 claims a game, the way
pass 3's lambda 0.98 runs stopped claiming). Reads the files `--live` rewrites.
"""

import glob
import json
import math
import sys


def finite(weights):
    return all(math.isfinite(w) for block in weights.values() for w in block)


def status(path):
    try:
        with open(path) as f:
            run = json.load(f)
    except (OSError, ValueError) as e:  # mid-write or unreadable
        return f"{path}: unreadable ({e.__class__.__name__})"
    prog = run.get("progress", {})
    hist = run.get("history", [])
    name = path.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".json")
    line = f"{name:16} {prog.get('games', 0):>5}/{prog.get('of', '?')}"
    line += " done" if prog.get("finished") else "     "
    flags = []
    if not finite(run.get("weights", {})):
        flags.append("BROKEN: non-finite weights")
    if hist:
        ev = hist[-1]["eval"]
        r, g = ev.get("random", {}), ev.get("greedy", {})
        line += (f"  vs random {r.get('margin', float('nan')):+7.1f} ({r.get('win_share', 0):.0%})"
                 f"  vs greedy {g.get('margin', float('nan')):+7.1f} ({g.get('win_share', 0):.0%})"
                 f"  claims {g.get('claims', float('nan')):4.1f}"
                 f"  tickets {g.get('tickets_completed', float('nan')):.2f}/{g.get('tickets_failed', float('nan')):.2f}")
        if hist[-1]["games"] >= 3000 and r.get("margin", 0) < -150 and r.get("claims", 99) < 2:
            flags.append("COLLAPSED")
    best = run.get("best") or {}
    if best:
        line += f"  best {best.get('margin_vs_greedy', float('nan')):+.1f}@{best.get('games')}"
    return line + ("  <-- " + "; ".join(flags) if flags else "")


if __name__ == "__main__":
    for pattern in sys.argv[1:] or ["runs/linear/pass4/*.json"]:
        for path in sorted(glob.glob(pattern)):
            print(status(path))
