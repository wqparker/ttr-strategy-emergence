"""Random agents through the PettingZoo env, end to end (PLAN.md Phase 4).

    python scripts/smoke_env.py [--games 50] [--players 2] [--board usa] [--reward margin]

Plays whole games with uniformly random legal actions from the action mask and
reports agent steps per second, game length, truncations, win shares per seat,
and the mean reward sum per agent. Checks on the way that every live agent has
a legal action and that the score-mode rewards add up to the final score.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter

import numpy as np

from ttr.env.aec import REWARD_MODES, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--board", default="usa")
    parser.add_argument("--reward", choices=REWARD_MODES, default="margin")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    e = env(num_players=args.players, board=args.board, reward_mode=args.reward)
    rng = np.random.default_rng(args.seed)
    steps, lengths, truncated, wins = 0, [], 0, Counter()
    reward_sums = {a: [] for a in e.possible_agents}
    start = time.perf_counter()
    for g in range(args.games):
        e.reset(seed=args.seed + g)
        total = dict.fromkeys(e.possible_agents, 0.0)
        game_steps = 0
        for agent in e.agent_iter():
            obs, reward, terminated, truncation, info = e.last()
            total[agent] += reward
            if terminated or truncation:
                truncated += truncation
                wins[agent] += info["winner"]
                if args.reward == "score":
                    assert abs(total[agent] - info["score"] * e.unwrapped.reward_scale) < 1e-6
                e.step(None)
                continue
            legal = np.flatnonzero(obs["action_mask"])
            assert len(legal), f"{agent} has no legal action"
            e.step(int(rng.choice(legal)))
            game_steps += 1
        steps += game_steps
        lengths.append(game_steps)
        for a in e.possible_agents:
            reward_sums[a].append(total[a])
    elapsed = time.perf_counter() - start

    print(f"{args.games} games, {args.players} players, {args.board}, reward {args.reward}")
    print(f"  {steps / elapsed:,.0f} agent steps/s   {statistics.mean(lengths):.0f} steps/game"
          f"   {truncated // args.players} truncated")
    for a in e.possible_agents:
        print(f"  {a}: won {wins[a]}   mean reward sum {statistics.mean(reward_sums[a]):+.3f}")


if __name__ == "__main__":
    main()
