"""The adapter from a policy over (observation, mask) to the `Agent` interface
(PLAN.md Phase 5), so trained networks play through the match runner, the
viewer and the evaluation code like any scripted bot.

    agent = PolicyAgent(policy, num_players=2)
    action = agent.act(game, player)

`policy(observation, mask) -> index` receives the same vector and mask the
PettingZoo env gives that seat (ttr.env.observation, ttr.env.actions) and
returns an index in Discrete(168); the agent decodes it to an engine action.
An index the mask doesn't allow raises, as it would in the env.

Needs the `[env]` extra (numpy); `ttr.agents` itself doesn't import this module.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ttr.actions import Action
from ttr.env import actions as A
from ttr.env.observation import ObservationEncoder
from ttr.game import Game

Policy = Callable[[np.ndarray, np.ndarray], int]


def legal_mask(game: Game) -> np.ndarray:
    """The env's action mask for the player to act, as int8."""
    mask = np.zeros(A.N_ACTIONS, dtype=np.int8)
    mask[A.legal_indices(game)] = 1
    return mask


def random_policy(seed: Optional[int] = None) -> Policy:
    """Uniform over the mask: the smallest policy, for tests and baselines."""
    rng = np.random.default_rng(seed)
    return lambda obs, mask: int(rng.choice(np.flatnonzero(mask)))


class PolicyAgent:
    """An `Agent` that asks `policy` for an action index. It keeps one
    observation encoder, which starts over whenever it sees a different game
    object (the viewer clones the game every step; that stays correct, only
    slower)."""

    def __init__(self, policy: Policy, num_players: int, memory_level: int = 2, name: str = "policy") -> None:
        self.policy = policy
        self.name = name
        self.encoder = ObservationEncoder(num_players, memory_level)

    def act(self, game: Game, player: int) -> Action:
        if player != game.current_player:
            raise ValueError(f"seat {player} asked to act on seat {game.current_player}'s turn")
        mask = legal_mask(game)
        index = int(self.policy(self.encoder.encode(game, player), mask))
        if not 0 <= index < A.N_ACTIONS or not mask[index]:
            raise ValueError(f"policy chose index {index}, which the mask doesn't allow")
        return A.decode(game, index)
