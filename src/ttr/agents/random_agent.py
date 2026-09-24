from __future__ import annotations

import random
from typing import Optional

from ttr.actions import Action
from ttr.game import Game


class RandomAgent:
    """Uniform over legal actions. Sanity baseline; the floor any learner must beat."""

    name = "random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def act(self, game: Game, player: int) -> Action:
        return self.rng.choice(game.legal_actions())
