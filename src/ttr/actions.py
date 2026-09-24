"""Action types. A turn may span several sub-step actions (see Phase in game.py).

    CHOOSE_ACTION     -> DrawFaceUp | DrawBlind | ClaimRoute | DrawTickets | Pass
    DRAW_SECOND_CARD  -> DrawFaceUp (non-locomotive) | DrawBlind
    CHOOSE_PAYMENT    -> Pay
    KEEP_TICKETS / CHOOSE_INITIAL_TICKETS -> KeepTickets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Union

from ttr.cards import Color


@dataclass(frozen=True)
class DrawFaceUp:
    color: Color


@dataclass(frozen=True)
class DrawBlind:
    pass


@dataclass(frozen=True)
class ClaimRoute:
    route_id: int


@dataclass(frozen=True)
class Pay:
    """Pay for the pending route: (length - locomotives) cards of `color` plus
    `locomotives` Locomotives. `color` is None only for an all-Locomotive payment."""

    color: Optional[Color]
    locomotives: int


@dataclass(frozen=True)
class DrawTickets:
    pass


@dataclass(frozen=True)
class KeepTickets:
    ticket_ids: FrozenSet[int]


@dataclass(frozen=True)
class Pass:
    """Only legal when nothing else is (RULES.md §9 #11)."""


Action = Union[DrawFaceUp, DrawBlind, ClaimRoute, Pay, DrawTickets, KeepTickets, Pass]
