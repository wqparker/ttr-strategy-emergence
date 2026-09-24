from __future__ import annotations

from typing import Protocol

from ttr.actions import Action
from ttr.game import Game


class Agent(Protocol):
    name: str

    def act(self, game: Game, player: int) -> Action:
        """Choose one of game.legal_actions() for `player` (the current player)."""
        ...
