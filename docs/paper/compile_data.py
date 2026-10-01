"""Compile the data behind the research paper (docs/paper/) into one small JSON file.

    python docs/paper/compile_data.py                 # -> docs/paper/data.json

Reads the local run files (runs/, gitignored) and the committed summaries:

- every training run (linear, DQN, PPO): its evaluation history, reduced to the mean over
  seeds of each setting at every evaluation, the last five evaluations' means (margin, win
  share and behavior metrics against greedy and racer), and its best checkpoints
- every fresh-game re-score (runs/*/rescore_*.csv)
- every round robin (runs/round_robin_*.csv / .txt): pair results and Elo ratings
- docs/paper/measured.json: the instrumented-game measurements that have no run file
  (behavior and blocking against chance, the ticket gap, scripted tables, engine speed),
  written by docs/paper/measure.py

Runs still training are read from their last live save.
"""

from __future__ import annotations

import csv
import glob
import json
import re
import statistics as st
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("data.json")
MEASURED = Path(__file__).with_name("measured.json")
BEHAVIOR = ("score", "opp_score", "route_points", "ticket_points", "tickets_kept", "tickets_completed",
            "tickets_failed", "ticket_draws", "claims", "mean_claim_length", "long_claims", "longest_bonus",
            "triggered_end", "turns", "draw_blind", "draw_color", "draw_locomotive", "trains_left")
CONFIG_KEYS = ("algo", "opponent", "pool", "games", "alpha", "alpha_end", "lam", "average", "shaping",
               "epsilon_decay", "n_step", "lr", "lr_end", "reward_mode", "ticket_plan", "players", "league")


def setting_of(path: str) -> tuple:
    """(run group such as "ppo/pass3", setting, seed); linear pass 1 files carry no seed (seed 0)."""
    p = Path(path)
    m = re.fullmatch(r"(.+)_s(\d+)", p.stem)
    setting, seed = (m.group(1), int(m.group(2))) if m else (p.stem, 0)
    return p.parent.relative_to(ROOT / "runs").as_posix(), setting, seed


def read_run(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    group, setting, seed = setting_of(path)
    history = []
    for h in run.get("history", []):
        ev = {}
        for opp, m in h["eval"].items():
            row = {"margin": m.get("margin"), "win": m.get("win_share")}
            if opp in ("greedy", "racer"):
                row.update({k: m.get(k) for k in BEHAVIOR})
            ev[opp] = row
        history.append({"games": h["games"], "eval": ev})
    cfg = run.get("config", {})
    best = run.get("best") or {}
    return {
        "group": group, "setting": setting, "seed": seed, "method": run.get("method", "linear"),
        "config": {k: cfg[k] for k in CONFIG_KEYS if k in cfg},
        "progress": run.get("progress", {}),
        "history": history,
        "best": {"games": best.get("games"), "margin_vs_greedy": best.get("margin_vs_greedy")} if best else None,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 3) if xs else None


def summarize(runs: list) -> dict:
    runs = sorted(runs, key=lambda r: r["seed"])
    first = runs[0]
    # evaluation points every seed reached
    common = sorted(set.intersection(*(set(h["games"] for h in r["history"]) for r in runs)))
    by = [{h["games"]: h for h in r["history"]} for r in runs]
    opps = sorted(set.intersection(*(set(r["history"][0]["eval"]) for r in runs if r["history"])))
    curve = {"games": common}
    for opp in opps:
        vals = [[b[g]["eval"][opp]["margin"] for b in by] for g in common]
        curve[opp] = [mean(v) for v in vals]
        if opp in ("greedy", "racer"):
            curve[opp + "_min"] = [round(min(v), 2) for v in vals]
            curve[opp + "_max"] = [round(max(v), 2) for v in vals]
            curve[opp + "_win"] = [mean([b[g]["eval"][opp]["win"] for b in by]) for g in common]
    last = [r["history"][-5:] for r in runs]
    last5 = {}
    for opp in opps:
        row = {"margin": mean([mean([h["eval"][opp]["margin"] for h in hs]) for hs in last]),
               "win": mean([mean([h["eval"][opp]["win"] for h in hs]) for hs in last]),
               "seeds": [mean([h["eval"][opp]["margin"] for h in hs]) for hs in last]}
        if opp in ("greedy", "racer"):
            for k in BEHAVIOR:
                row[k] = mean([mean([h["eval"][opp].get(k) for h in hs]) for hs in last])
        last5[opp] = row
    games_done = [r["progress"].get("games", r["history"][-1]["games"] if r["history"] else 0) for r in runs]
    parity = []
    for r in runs:
        hit = next((h["games"] for h in r["history"] if h["eval"]["greedy"]["margin"] >= 0), None)
        parity.append(hit)
    return {
        "group": first["group"], "setting": first["setting"], "method": first["method"],
        "seeds": [r["seed"] for r in runs], "config": first["config"],
        "games_done": games_done, "finished": all(r["progress"].get("finished", True) for r in runs),
        "curve": curve, "last5": last5, "parity_games": parity,
        "best": [r["best"] for r in runs],
    }


def read_rescores() -> list:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "runs" / "*" / "rescore_*.csv"))):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                agent = r["agent"]
                kind, _, arg = agent.partition(":")
                which = "best" if arg.endswith("@best") else "final"
                group, setting, seed = setting_of(str(ROOT / arg.removesuffix("@best")))
                rows.append({
                    "file": Path(path).name, "method": kind, "group": group, "setting": setting, "seed": seed,
                    "which": which, "opponent": r.get("opponent", "greedy"), "players": int(r.get("players", 2)),
                    "games": int(r["games"]), "margin": round(float(r["margin"]), 3), "se": round(float(r["se"]), 3),
                    "win": round(float(r["win_share"]), 4),
                    "tickets_completed": round(float(r["tickets_completed"]), 3),
                    "tickets_failed": round(float(r["tickets_failed"]), 3),
                })
    return rows


def short(spec: str) -> str:
    kind, _, arg = spec.partition(":")
    if not arg:
        return spec
    return f"{kind}:{Path(arg.removesuffix('@best')).stem}{'@best' if arg.endswith('@best') else ''}"


def read_round_robins() -> list:
    out = []
    for path in sorted(glob.glob(str(ROOT / "runs" / "round_robin_*.csv"))):
        txt = Path(path).with_suffix(".txt").read_text(encoding="utf-8")
        ratings = []
        block = txt.split("ratings (Elo scale, greedy = 1000)", 1)[1]
        for line in block.strip().splitlines():
            m = re.match(r"\s*\d+\.\s+(\S+)\s+(-?\d+)", line)
            if m:
                ratings.append([m.group(1), int(m.group(2))])
        with open(path, encoding="utf-8") as f:
            pairs = [{"a": short(r["a"]), "b": short(r["b"]), "games": int(r["games"]),
                      "margin": round(float(r["margin"]), 2), "se": round(float(r["se"]), 2),
                      "a_win": round(float(r["a_wins"]) / int(r["games"]), 4)} for r in csv.DictReader(f)]
        out.append({"file": Path(path).stem, "ratings": ratings, "pairs": pairs})
    return out


def main() -> None:
    paths = sorted(glob.glob(str(ROOT / "runs" / "*" / "*" / "*.json")))
    with ProcessPoolExecutor(3) as ex:
        runs = list(ex.map(read_run, paths, chunksize=4))
    groups = defaultdict(list)
    for r in runs:
        if r["history"]:
            groups[(r["group"], r["setting"])].append(r)
    settings = {f"{g}/{s}": summarize(rs) for (g, s), rs in sorted(groups.items())}
    data = {
        "generated": datetime.now().isoformat(timespec="minutes"),
        "settings": settings,
        "rescores": read_rescores(),
        "round_robins": read_round_robins(),
        "measured": json.loads(MEASURED.read_text(encoding="utf-8")) if MEASURED.exists() else {},
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"{len(paths)} runs in {len(settings)} settings, {len(data['rescores'])} re-scores, "
          f"{len(data['round_robins'])} round robins -> {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
