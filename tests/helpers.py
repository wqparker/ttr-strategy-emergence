from __future__ import annotations

import random
from collections import Counter
from typing import Optional

from ttr.actions import KeepTickets
from ttr.board import load_board
from ttr.cards import Color
from ttr.game import Game, Phase


def started_game(board: str = "usa", num_players: int = 2, seed: int = 0, **kw) -> Game:
    """A game past the initial ticket choice (everyone keeps all 3), player 0 to act."""
    game = Game(load_board(board), num_players=num_players, seed=seed, first_player=0, **kw)
    while game.phase is Phase.CHOOSE_INITIAL_TICKETS:
        pending = game.players[game.current_player].pending_tickets
        game.step(KeepTickets(frozenset(pending)))
    assert game.phase is Phase.CHOOSE_ACTION and game.current_player == 0
    return game


def set_hand(game: Game, player: int, **cards: int) -> None:
    """Replace a player's hand, e.g. set_hand(g, 0, red=3, locomotive=1).
    Cards are moved to/from the deck so the 110-card total is preserved."""
    old = game.players[player].hand
    game.deck.extend(old.elements())
    new = Counter({Color(k): v for k, v in cards.items() if v})
    for color, n in new.items():
        for _ in range(n):
            game.deck.remove(color)
    game.players[player].hand = new


def set_market(game: Game, *colors: Color) -> None:
    """Replace the face-up cards, moving cards to/from the deck."""
    game.deck.extend(game.market)
    game.market = []
    for c in colors:
        game.deck.remove(c)
        game.market.append(c)


def empty_the_piles(game: Game, into: int = 1) -> None:
    """Move the deck and discard into a player's hand, so nothing can be drawn
    blind and the market cannot refill. Used to test the empty-pile rules."""
    hand = game.players[into].hand
    for card in game.deck + game.discard:
        hand[card] += 1
    game.deck, game.discard = [], []


def total_cards(game: Game) -> int:
    return (
        len(game.deck)
        + len(game.discard)
        + len(game.market)
        + sum(sum(p.hand.values()) for p in game.players)
    )


def route_id(game: Game, a: str, b: str, color: Optional[str] = "any") -> int:
    for r in game.board.routes_between(a, b):
        if color == "any" or (r.color.value if r.color else "gray") == color:
            return r.id
    raise KeyError((a, b, color))


def play_random(game: Game, rng: random.Random, check=None) -> None:
    while not game.game_over:
        game.step(rng.choice(game.legal_actions()))
        if check:
            check(game)
