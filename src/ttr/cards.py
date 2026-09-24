"""Train car card colors and the standard deck (RULES.md §1)."""

from __future__ import annotations

from enum import Enum
from typing import List


class Color(str, Enum):
    PURPLE = "purple"
    BLUE = "blue"
    ORANGE = "orange"
    WHITE = "white"
    GREEN = "green"
    YELLOW = "yellow"
    BLACK = "black"
    RED = "red"
    LOCOMOTIVE = "locomotive"


# The 8 regular colors, in a fixed order (used for indexing/encoding later).
TRAIN_COLORS: List[Color] = [c for c in Color if c is not Color.LOCOMOTIVE]

CARDS_PER_COLOR = 12
LOCOMOTIVE_COUNT = 14


def standard_deck() -> List[Color]:
    """The 110-card train deck, unshuffled."""
    deck: List[Color] = []
    for color in TRAIN_COLORS:
        deck.extend([color] * CARDS_PER_COLOR)
    deck.extend([Color.LOCOMOTIVE] * LOCOMOTIVE_COUNT)
    return deck
