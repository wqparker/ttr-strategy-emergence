"""Engine speed as the RL environment will drive it: at each state build the
legal mask and the observation, pick a random legal index, decode and step.

    python scripts/bench_env.py [--games 200] [--players 2] [--board usa]

Reports sub-steps per second for the engine alone, then adding the mask, then
the observation, so the cost of each env layer is visible (PLAN.md "Open questions": engine speed).
"""

from __future__ import annotations

import argparse
import random
import time

from ttr.board import load_board
from ttr.env import actions as A
from ttr.env.observation import ObservationEncoder
from ttr.game import Game


def run(games: int, players: int, board_name: str, with_mask: bool, with_obs: bool) -> float:
    board = load_board(board_name)
    rng = random.Random(0)
    steps = 0
    start = time.perf_counter()
    for g in range(games):
        game = Game(board, num_players=players, seed=g, max_turns=1000)
        encoder = ObservationEncoder(players)
        while not game.game_over:
            if with_obs:
                encoder.encode(game, game.current_player)
            if with_mask:
                mask = A.legal_mask(game)
                index = rng.choice([i for i, m in enumerate(mask) if m])
                game.step(A.decode(game, index))
            else:
                game.step(rng.choice(game.legal_actions()))
            steps += 1
    return steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--board", default="usa")
    args = parser.parse_args()
    for label, with_mask, with_obs in (
        ("engine step only", False, False),
        ("mask + decode + step", True, False),
        ("+ observation", True, True),
    ):
        rate = run(args.games, args.players, args.board, with_mask, with_obs)
        print(f"{label:22s} {rate:9,.0f} sub-steps/s")


if __name__ == "__main__":
    main()
