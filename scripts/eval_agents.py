"""Evaluate slow agents (MCTS, tier D) on fresh games, split across worker processes.

    python scripts/eval_agents.py "mcts:iterations=400" "mcts:iterations=400,reward=score" \
        --opponents greedy racer --games 200 --out runs/mcts/pass1

A search agent takes minutes a game, so each agent-opponent pair's games are cut into
chunks (`--chunk`) spread over the workers, and every finished chunk is appended to
OUT.csv at once: rerunning with the same OUT skips the games already there (resume after
a stop) or adds games beyond them. Any agent spec works (ttr.agents.registry), so a bot
on the same games gives the reference.

The games are `run_matches`' (ttr.simulate.match_game): batch seed 9001 by default, the
seed `scripts/rescore.py` uses, so game g is the same game in both. At more than 2
players every other seat is a copy of the opponent and the margin is against their mean.
One CSV row per game: the agent's metrics (ttr.learn.metrics), its seat, the game's wall
time and, for MCTS, its searches and their time. OUT.txt gets the summary table; progress
lines go to the terminal and to OUT.log.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ttr.board import load_board
from ttr.learn.metrics import METRICS, game_metrics
from ttr.simulate import match_game, play_game

FIELDS = ["agent", "opponent", "players", "seed", "game", "seat"] + list(METRICS) + [
    "seconds", "searches", "search_seconds"]
SUMMARY = ("win_share", "score", "opp_score", "tickets_kept", "tickets_completed", "tickets_failed", "ticket_draws",
           "claims", "mean_claim_length", "long_claims", "triggered_end", "longest_bonus", "turns")


def keep_awake(on: bool) -> None:
    """Windows: no idle sleep while games are being played (as scripts/run_queue.ps1)."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if on else 0x80000000)


def play_chunk(task) -> list:
    spec, opponent, players, seed, games, max_turns = task
    board = load_board("usa")
    rows = []
    for g in games:
        game, agents, slot_of = match_game([spec] + [opponent] * (players - 1), board, seed, g, max_turns)
        seat = slot_of.index(0)
        start = time.perf_counter()
        play_game(game, agents)
        me = agents[seat]
        rows.append({"agent": spec, "opponent": opponent, "players": players, "seed": seed, "game": g, "seat": seat,
                     **game_metrics(game, seat), "seconds": round(time.perf_counter() - start, 2),
                     "searches": getattr(me, "searches", 0),
                     "search_seconds": round(getattr(me, "search_seconds", 0.0), 2)})
    return rows


def read_rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{k: (v if k in ("agent", "opponent") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def summarize(rows: list) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[r["agent"], r["opponent"], int(r["players"])].append(r)
    lines = []
    for (agent, opponent, players), rs in sorted(groups.items()):
        margins = [r["margin"] for r in rs]
        se = st.stdev(margins) / len(margins) ** 0.5 if len(margins) > 1 else 0.0
        mean = {k: st.mean(r[k] for r in rs) for k in SUMMARY}
        searches = sum(r["searches"] for r in rs)
        per_search = sum(r["search_seconds"] for r in rs) / searches if searches else 0.0
        lines.append(
            f"{agent} vs {opponent} ({players}p, {len(rs)} games): margin {st.mean(margins):+.1f} ± {se:.1f}, "
            f"win {mean['win_share']:.0%}, score {mean['score']:.0f} to {mean['opp_score']:.0f}\n"
            f"    tickets kept {mean['tickets_kept']:.2f}, done {mean['tickets_completed']:.2f}, "
            f"failed {mean['tickets_failed']:.2f}, draws {mean['ticket_draws']:.2f}; "
            f"claims {mean['claims']:.1f} x {mean['mean_claim_length']:.2f} ({mean['long_claims']:.1f} of 5-6); "
            f"ended it {mean['triggered_end']:.0%}, longest bonus {mean['longest_bonus']:.0%}, "
            f"{mean['turns']:.0f} turns\n"
            f"    {st.mean(r['seconds'] for r in rs) / 60:.1f} min a game"
            + (f", {searches / len(rs):.0f} searches of {per_search:.2f} s" if searches else ""))
    return "\n".join(lines)


def report(csv_path: Path, log_path: Path, every: float) -> None:
    while True:
        rows = read_rows(csv_path)
        lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
        progress = next((x for x in reversed(lines) if " games, " in x or "finished" in x), "")
        text = f"{time.strftime('%H:%M:%S')}  {progress.split(':')[0] if ' games, ' in progress else progress}\n\n"
        text += summarize(rows) if rows else f"no games in {csv_path} yet"
        if every:
            print("\033[2J\033[H" + text, flush=True)  # clear the screen, then redraw
            time.sleep(every)
        else:
            print(text)
            return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agents", nargs="*", help="agent specs (ttr.agents.registry)")
    ap.add_argument("--opponents", nargs="+", default=["greedy"], metavar="SPEC")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--games", type=int, default=100, help="games per agent and opponent")
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--chunk", type=int, default=4, help="games per task")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-turns", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True, help="output prefix: OUT.csv, OUT.txt")
    ap.add_argument("--report", nargs="?", type=float, const=0, metavar="SECONDS",
                    help="play nothing: print the summary of the games in OUT.csv so far, and with SECONDS "
                         "keep reprinting it that often (a live view of a running evaluation; Ctrl+C stops)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles and pipes default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")

    csv_path, txt_path, log_path = (args.out.with_suffix(x) for x in (".csv", ".txt", ".log"))
    if args.report is not None:
        report(csv_path, log_path, args.report)
        return
    if not args.agents:
        ap.error("give at least one agent (or --report)")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["agent"], r["opponent"], int(r["players"]), int(r["seed"]), int(r["game"])) for r in read_rows(csv_path)}
    pairs = [(a, o) for o in args.opponents for a in args.agents]
    tasks = []
    for start in range(0, args.games, args.chunk):  # interleaved, so every pair advances together
        for agent, opponent in pairs:
            games = [g for g in range(start, min(start + args.chunk, args.games))
                     if (agent, opponent, args.players, args.seed, g) not in done]
            if games:
                tasks.append((agent, opponent, args.players, args.seed, games, args.max_turns))
    total = sum(len(t[4]) for t in tasks)

    def say(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(line + "\n")

    say(f"{time.strftime('%Y-%m-%d %H:%M')} {total} games to play ({len(done)} already in {csv_path})")

    new = not csv_path.exists()
    played, start = 0, time.time()
    keep_awake(True)
    try:
        with open(csv_path, "a", newline="") as f, ProcessPoolExecutor(args.workers) as pool:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            for future in as_completed([pool.submit(play_chunk, t) for t in tasks]):
                rows = future.result()
                writer.writerows(rows)
                f.flush()
                played += len(rows)
                say(f"{played}/{total} games, {(time.time() - start) / 60:.0f} min: {rows[0]['agent']} vs "
                    f"{rows[0]['opponent']}, margins {', '.join(f'{r['margin']:+.0f}' for r in rows)}")
    finally:
        keep_awake(False)

    summary = summarize(read_rows(csv_path))
    txt_path.write_text(summary + "\n", encoding="utf-8")
    say(f"{time.strftime('%Y-%m-%d %H:%M')} finished\n{summary}")


if __name__ == "__main__":
    main()
