"""The flat Discrete(168) action space: encode/decode and the legal mask."""

import random

import pytest

from ttr.actions import ClaimRoute, DrawBlind, DrawFaceUp, DrawTickets, KeepTickets, Pass, Pay
from ttr.board import load_board
from ttr.cards import ALL_COLORS, TRAIN_COLORS, Color
from ttr.env import actions as A
from ttr.game import Game, Phase

from helpers import set_hand, started_game


def test_layout_matches_the_plan():
    assert (A.FACE_UP, A.BLIND, A.CLAIM, A.PAY, A.PAY_ALL_LOCO) == (0, 9, 10, 110, 158)
    assert (A.DRAW_TICKETS, A.KEEP, A.PASS, A.N_ACTIONS) == (159, 160, 167, 168)


def test_fixed_indices_decode_without_state():
    game = started_game()
    assert A.decode(game, 0) == DrawFaceUp(ALL_COLORS[0])
    assert A.decode(game, 8) == DrawFaceUp(Color.LOCOMOTIVE)
    assert A.decode(game, A.BLIND) == DrawBlind()
    assert A.decode(game, A.CLAIM + 37) == ClaimRoute(37)
    assert A.decode(game, A.PAY) == Pay(TRAIN_COLORS[0], 0)
    assert A.decode(game, A.PAY_ALL_LOCO - 1) == Pay(TRAIN_COLORS[-1], 5)
    assert A.decode(game, A.DRAW_TICKETS) == DrawTickets()
    assert A.decode(game, A.PASS) == Pass()
    for bad in (-1, A.N_ACTIONS):
        with pytest.raises(A.ActionIndexError):
            A.decode(game, bad)


def test_all_locomotive_payment_takes_the_route_length():
    game = started_game()
    route = next(r for r in game.board.routes if r.length == 4)
    set_hand(game, 0, locomotive=4)
    game.step(ClaimRoute(route.id))
    assert game.phase is Phase.CHOOSE_PAYMENT
    assert A.decode(game, A.PAY_ALL_LOCO) == Pay(None, 4)
    assert A.encode(game, Pay(None, 4)) == A.PAY_ALL_LOCO
    assert A.PAY_ALL_LOCO in A.legal_indices(game)


def test_ticket_choices_are_offer_positions():
    game = Game(load_board("usa"), num_players=2, seed=3, first_player=0)
    offer = game.players[0].pending_tickets
    keep_first_and_last = KeepTickets(frozenset({offer[0], offer[2]}))
    assert A.encode(game, keep_first_and_last) == A.KEEP + 0b101 - 1
    assert A.decode(game, A.KEEP + 0b101 - 1) == keep_first_and_last
    # Initial choice keeps at least 2: the four subsets of size 2 or 3.
    assert sorted(A.legal_indices(game)) == [A.KEEP + b - 1 for b in (0b011, 0b101, 0b110, 0b111)]
    with pytest.raises(A.ActionIndexError):
        A.encode(game, KeepTickets(frozenset({999})))


def test_short_offer_masks_missing_positions():
    """With fewer than 3 tickets left to draw, positions past the offer don't decode."""
    game = started_game()
    game.ticket_deck = game.ticket_deck[:2]
    game.step(DrawTickets())
    assert len(game.players[0].pending_tickets) == 2
    assert sorted(A.legal_indices(game)) == [A.KEEP + b - 1 for b in (0b01, 0b10, 0b11)]
    with pytest.raises(A.ActionIndexError):
        A.decode(game, A.KEEP + 0b100 - 1)


BLOCKS = {
    Phase.CHOOSE_INITIAL_TICKETS: range(A.KEEP, A.PASS),
    Phase.KEEP_TICKETS: range(A.KEEP, A.PASS),
    Phase.CHOOSE_PAYMENT: range(A.PAY, A.DRAW_TICKETS),
    Phase.DRAW_SECOND_CARD: range(A.FACE_UP, A.CLAIM),
    Phase.CHOOSE_ACTION: set(range(A.FACE_UP, A.PAY)) | {A.DRAW_TICKETS, A.PASS},
}


@pytest.mark.parametrize("board,players,seed", [
    ("usa", 2, 1), ("usa", 3, 2), ("usa", 5, 3), ("toy", 2, 4), ("toy", 4, 5),
])
def test_every_legal_action_round_trips_through_random_games(board, players, seed):
    """At every state of random games: each legal action has a distinct index
    that decodes back to it, the mask is exactly those indices, and they sit in
    the current sub-step's block."""
    rng = random.Random(seed)
    game = Game(load_board(board), num_players=players, seed=seed, max_turns=400)
    A.check_board(game)
    steps = 0
    while not game.game_over:
        legal = game.legal_actions()
        indices = [A.encode(game, a) for a in legal]
        assert len(set(indices)) == len(indices)
        for a, i in zip(legal, indices):
            assert A.decode(game, i) == a
            assert i in BLOCKS[game.phase]
        mask = A.legal_mask(game)
        assert len(mask) == A.N_ACTIONS and {i for i, m in enumerate(mask) if m} == set(indices)
        game.step(A.decode(game, rng.choice(indices)))
        steps += 1
    assert steps > 50
