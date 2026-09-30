"""The flat action space: every engine action as one index in Discrete(168),
plus the legal-action mask for a state (PLAN.md "Action space", Phase 4).

    index = encode(game, action)
    action = decode(game, index)
    mask = legal_mask(game)         # N_ACTIONS 0/1 entries, 1 = legal now

| Block          | Indices  | Action                                                   |
| -------------- | -------- | -------------------------------------------------------- |
| FACE_UP        | 0-8      | DrawFaceUp(color), colors in cards.ALL_COLORS order      |
| BLIND          | 9        | DrawBlind                                                |
| CLAIM          | 10-109   | ClaimRoute(route id)                                     |
| PAY            | 110-157  | Pay(color, k): 8 colors x k = 0-5 Locomotives            |
| PAY_ALL_LOCO   | 158      | Pay(None, route length): all Locomotives                 |
| DRAW_TICKETS   | 159      | DrawTickets                                              |
| KEEP           | 160-166  | KeepTickets, by offer position: bitmask 1-7 of the offer |
| PASS           | 167      | Pass                                                     |

Each sub-step (engine Phase) only ever makes one block family legal, so the
mask carries the sub-step too. Two indices need the state to decode: an
all-Locomotive payment takes the pending route's length, and a ticket choice
names positions in the player's current offer (`pending_tickets` order), which
the observation shows. Boards with fewer than 100 routes leave the rest of the
CLAIM block always masked.
"""

from __future__ import annotations

from typing import List

from ttr.actions import Action, ClaimRoute, DrawBlind, DrawFaceUp, DrawTickets, KeepTickets, Pass, Pay
from ttr.cards import ALL_COLORS, TRAIN_COLORS
from ttr.game import Game

MAX_ROUTES = 100
MAX_LOCOS_WITH_COLOR = 5  # a colored payment for a 6-route uses at most 5 Locomotives
OFFER = 3  # tickets per offer (RULES.md §2, §3C)

FACE_UP = 0
BLIND = FACE_UP + len(ALL_COLORS)  # 9
CLAIM = BLIND + 1  # 10
PAY = CLAIM + MAX_ROUTES  # 110
PAY_ALL_LOCO = PAY + len(TRAIN_COLORS) * (MAX_LOCOS_WITH_COLOR + 1)  # 158
DRAW_TICKETS = PAY_ALL_LOCO + 1  # 159
KEEP = DRAW_TICKETS + 1  # 160
PASS = KEEP + (2 ** OFFER - 1)  # 167
N_ACTIONS = PASS + 1  # 168

_COLOR_INDEX = {c: i for i, c in enumerate(ALL_COLORS)}
_TRAIN_INDEX = {c: i for i, c in enumerate(TRAIN_COLORS)}


class ActionIndexError(ValueError):
    pass


def encode(game: Game, action: Action) -> int:
    """The index of `action` in `game`'s current state."""
    if isinstance(action, DrawFaceUp):
        return FACE_UP + _COLOR_INDEX[action.color]
    if isinstance(action, DrawBlind):
        return BLIND
    if isinstance(action, ClaimRoute):
        if not 0 <= action.route_id < MAX_ROUTES:
            raise ActionIndexError(f"route {action.route_id} is outside the {MAX_ROUTES}-route block")
        return CLAIM + action.route_id
    if isinstance(action, Pay):
        if action.color is None:
            return PAY_ALL_LOCO
        if not 0 <= action.locomotives <= MAX_LOCOS_WITH_COLOR:
            raise ActionIndexError(f"{action!r} uses more Locomotives than a colored payment can")
        return PAY + _TRAIN_INDEX[action.color] * (MAX_LOCOS_WITH_COLOR + 1) + action.locomotives
    if isinstance(action, DrawTickets):
        return DRAW_TICKETS
    if isinstance(action, KeepTickets):
        offer = game.players[game.current_player].pending_tickets
        bits = 0
        for tid in action.ticket_ids:
            if tid not in offer:
                raise ActionIndexError(f"ticket {tid} is not in the current offer {offer}")
            bits |= 1 << offer.index(tid)
        if not bits:
            raise ActionIndexError("keeping no tickets has no index")
        return KEEP + bits - 1
    if isinstance(action, Pass):
        return PASS
    raise TypeError(f"not an action: {action!r}")


def decode(game: Game, index: int) -> Action:
    """The action `index` stands for in `game`'s current state. It may still be
    illegal there; the mask says which are legal."""
    if not 0 <= index < N_ACTIONS:
        raise ActionIndexError(f"index {index} is outside 0-{N_ACTIONS - 1}")
    if index < BLIND:
        return DrawFaceUp(ALL_COLORS[index - FACE_UP])
    if index == BLIND:
        return DrawBlind()
    if index < PAY:
        return ClaimRoute(index - CLAIM)
    if index < PAY_ALL_LOCO:
        color, k = divmod(index - PAY, MAX_LOCOS_WITH_COLOR + 1)
        return Pay(TRAIN_COLORS[color], k)
    if index == PAY_ALL_LOCO:
        if game.pending_route is None:
            raise ActionIndexError("an all-Locomotive payment needs a route being paid for")
        return Pay(None, game.board.routes[game.pending_route].length)
    if index == DRAW_TICKETS:
        return DrawTickets()
    if index < PASS:
        offer = game.players[game.current_player].pending_tickets
        bits = index - KEEP + 1
        if bits >> len(offer):
            raise ActionIndexError(f"index {index} names offer positions beyond the {len(offer)} on offer")
        return KeepTickets(frozenset(t for i, t in enumerate(offer) if bits >> i & 1))
    return Pass()


def legal_indices(game: Game) -> List[int]:
    """Indices of the legal actions, in the engine's order."""
    return [encode(game, a) for a in game.legal_actions()]


def legal_mask(game: Game) -> List[int]:
    """N_ACTIONS entries: 1 where the action is legal now, else 0."""
    mask = [0] * N_ACTIONS
    for i in legal_indices(game):
        mask[i] = 1
    return mask


def check_board(game: Game) -> None:
    """Raise if the board doesn't fit the action space (more than 100 routes)."""
    if len(game.board.routes) > MAX_ROUTES:
        raise ActionIndexError(f"board {game.board.name!r} has {len(game.board.routes)} routes; "
                               f"the action space holds {MAX_ROUTES}")
