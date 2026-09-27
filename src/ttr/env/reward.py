"""Reward modes (PLAN.md "Reward"), shared by the PettingZoo env and the
hand-rolled learners that drive the engine directly.

The reward for a step is the change in `reward_values(game, mode)[seat]`:

    score   my score
    margin  my score - mean of opponents' scores      [default]
    win     0 during the game; at the end +1 / -1, 0 to each winner of a shared win

The score is route points during the game and the final total (tickets and
longest route included) once it is over, so the rewards over a game sum to
exactly the final score or margin. Pure Python: no numpy or PettingZoo.
"""

from __future__ import annotations

from typing import List

from ttr.game import Game

REWARD_MODES = ("score", "margin", "win")


def reward_values(game: Game, mode: str = "margin") -> List[float]:
    """Per seat, the quantity whose change is the reward (unscaled)."""
    if mode not in REWARD_MODES:
        raise ValueError(f"reward mode must be one of {REWARD_MODES}")
    n = game.num_players
    result = game.result
    if mode == "win":
        if result is None:
            return [0.0] * n
        win = 0.0 if len(result.winners) > 1 else 1.0
        return [win if p in result.winners else -1.0 for p in range(n)]
    scores = [r.total for r in result.players] if result else [p.route_points for p in game.players]
    if mode == "score":
        return [float(s) for s in scores]
    total = sum(scores)
    return [s - (total - s) / (n - 1) for s in scores]
