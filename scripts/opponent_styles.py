"""Measure the two play styles the MCTS opponent inference tells apart (ttr.agents.mcts
STYLES): ticket players (greedy, wary, collector) and racers (racer, PPO p2b, linear p7b),
from the public record of their games against greedy and racer.

    python scripts/opponent_styles.py --games 200

Per style: how many opening tickets it keeps (2 or 3), the share of its claims of each
length, and the share of its turns after the opening that draw tickets. Prints the
STYLES table to paste into src/ttr/agents/mcts.py (add-one smoothed counts).
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from ttr.board import load_board
from ttr.simulate import match_game, play_game

# style -> its agents; a style is named for the bot that plays it in the search's sampled worlds
STYLES = {
    "greedy": ["greedy", "wary", "collector"],
    "racer": ["racer", "ppo:runs/ppo/pass2/p2b_pool_s3.json",
              "linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best"],
}
OPPONENTS = ["greedy", "racer"]


def count(task) -> Counter:
    """Public events of slot 0 in `games` games against `opponent`."""
    spec, opponent, games, seed = task
    board = load_board("usa")
    out = Counter()
    for g in range(games):
        game, agents, slot_of = match_game([spec, opponent], board, seed, g)
        play_game(game, agents)
        seat = slot_of.index(0)
        card_turns = set()
        for e in game.log:
            if e.player != seat:
                continue
            if e.kind == "keep_initial_tickets":
                out[f"keep{e.public['count']}"] += 1
            elif e.kind == "claim_route":
                out[f"len{board.routes[e.public['route']].length}"] += 1
                out["turns"] += 1
            elif e.kind == "draw_tickets":
                out["ticket_draws"] += 1
                out["turns"] += 1
            elif e.kind in ("draw_face_up", "draw_blind"):
                card_turns.add(e.turn)
        out["turns"] += len(card_turns)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=200, help="games per agent and opponent")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    tasks = [(spec, opp, args.games, args.seed) for specs in STYLES.values() for spec in specs for opp in OPPONENTS]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(count, tasks))
    totals = {style: Counter() for style in STYLES}
    for (spec, *_), c in zip(tasks, results):
        style = next(s for s, specs in STYLES.items() if spec in specs)
        totals[style] += c
    print("STYLES = {  # scripts/opponent_styles.py, "
          f"{args.games} games per agent and opponent, add-one smoothed")
    for style, c in totals.items():
        claims = sum(c[f"len{n}"] for n in range(1, 7))
        keep = (c["keep3"] + 1) / (c["keep2"] + c["keep3"] + 2)
        lengths = ", ".join(f"{n}: {(c[f'len{n}'] + 1) / (claims + 6):.4f}" for n in range(1, 7))
        draw = (c["ticket_draws"] + 1) / (c["turns"] + 2)
        print(f'    "{style}": {{"keep3": {keep:.3g}, "ticket_draw": {draw:.3g}, "length": {{{lengths}}}}},')
    print("}")
    for style, c in totals.items():
        print(f"# {style}: {sum(c[f'len{n}'] for n in range(1, 7))} claims, {c['turns']} turns, "
              f"{c['ticket_draws']} ticket draws, opening keep 2/3: {c['keep2']}/{c['keep3']}")


if __name__ == "__main__":
    main()
