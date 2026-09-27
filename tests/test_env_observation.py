"""The flat observation vector: layout, per-block contents, hidden information."""

import random
from collections import Counter

import pytest

np = pytest.importorskip("numpy")

from ttr.actions import ClaimRoute, DrawBlind, DrawTickets, KeepTickets, Pay
from ttr.board import load_board
from ttr.cards import ALL_COLORS, Color
from ttr.env import observation as O
from ttr.env.actions import MAX_ROUTES
from ttr.game import Game, Phase

from helpers import route_id, set_hand, started_game


def block(enc, obs, name):
    return obs[enc.layout[name]]


def test_size_matches_the_plan():
    assert O.ObservationEncoder(2).size == 765
    for n in range(2, 6):
        enc = O.ObservationEncoder(n)
        assert enc.size == 100 * (n + 1) + 4 * n + 10 * (n - 1) + 447
        slices = list(enc.layout.values())
        assert slices[0].start == 0 and slices[-1].stop == enc.size
        assert all(a.stop == b.start for a, b in zip(slices, slices[1:]))


def test_opening_state():
    game = Game(load_board("usa"), num_players=2, seed=3, first_player=0)
    enc = O.ObservationEncoder(2)
    obs = enc.encode(game, 0)
    assert obs.dtype == np.float32 and obs.shape == (765,)
    assert list(block(enc, obs, "phase")) == [1, 0, 0, 0, 0]
    owner = block(enc, obs, "route_owner").reshape(MAX_ROUTES, 3)
    assert (owner[:, 0] == 1).all() and (owner[:, 1:] == 0).all()
    assert block(enc, obs, "route_open").sum() == 100
    assert block(enc, obs, "hand").sum() > 0
    offer = block(enc, obs, "tickets_offer").reshape(3, 30)
    for i, tid in enumerate(game.players[0].pending_tickets):
        assert offer[i, tid] == 1 and offer[i].sum() == 1
    assert block(enc, obs, "tickets_held").sum() == 0
    # Player 1 has an offer too, but it is private to them.
    assert block(enc, enc.encode(game, 1), "tickets_offer").sum() == 3
    assert block(enc, O.ObservationEncoder(2).encode(game, 0), "tickets_offer").sum() == 3


def test_route_ownership_is_relative_to_the_viewer():
    game = started_game(num_players=3)
    rid = route_id(game, "Vancouver", "Seattle")
    set_hand(game, 0, locomotive=1)
    game.step(ClaimRoute(rid))
    enc = O.ObservationEncoder(3)
    paying = enc.encode(game, 0)
    assert list(block(enc, paying, "phase")) == [0, 0, 0, 1, 0]
    assert np.flatnonzero(block(enc, paying, "route_paying")).tolist() == [rid]
    game.step(Pay(None, 1))
    for viewer, owner_slot in ((0, 1), (1, 3), (2, 2)):  # me, then seats after me
        obs = enc.encode(game, viewer)
        assert block(enc, obs, "route_owner").reshape(MAX_ROUTES, 4)[rid].tolist() == [
            1.0 if k == owner_slot else 0.0 for k in range(4)
        ]
        assert block(enc, obs, "route_paying").sum() == 0
    players = block(enc, enc.encode(game, 2), "players").reshape(3, 4)
    assert players[1, 1] == pytest.approx(1 / O.SCORE_SCALE)  # seat 0 is two seats after 2
    assert players[1, 0] == pytest.approx(44 / 45)


def test_double_route_closes_in_two_player_games():
    game = started_game()
    rid = route_id(game, "Vancouver", "Seattle")
    sibling = game.board.routes[rid].sibling
    set_hand(game, 0, locomotive=1)
    game.step(ClaimRoute(rid))
    game.step(Pay(None, 1))
    enc = O.ObservationEncoder(2)
    for viewer in (0, 1):
        open_ = block(enc, enc.encode(game, viewer), "route_open")
        assert open_[rid] == 0 and open_[sibling] == 0
        assert open_.sum() == 98


def test_route_open_needs_trains_left():
    game = started_game()
    game.players[0].trains = 3
    enc = O.ObservationEncoder(2)
    open_ = block(enc, enc.encode(game, 0), "route_open")
    for r in game.board.routes:
        assert open_[r.id] == (r.length <= 3)


def test_trains_to_finish_on_the_toy_board():
    # Ticket 0 is A-D: A-C-D or A-B-C-D, 5 trains.
    game = started_game("toy")
    game.players[0].tickets = [0]
    enc = O.ObservationEncoder(2)
    assert enc.trains_to_finish(game, 0, 0) == 5
    trains = block(enc, enc.encode(game, 0), "tickets_trains")
    assert trains[0] == pytest.approx(5 / 12) and trains[30] == 0

    # My claim of C-D leaves A-C (3) or A-B-C (3).
    set_hand(game, 0, yellow=2)
    game.step(ClaimRoute(route_id(game, "C", "D")))
    game.step(Pay(Color.YELLOW, 0))
    assert enc.trains_to_finish(game, 0, 0) == 3
    # The opponent sees C-D as closed: their A-D path is now 6 (A-B-D).
    assert enc.trains_to_finish(game, 1, 0) == 6


def test_completed_and_impossible_tickets():
    game = started_game("toy")
    game.players[0].tickets = [7, 0]  # C-D, A-D
    set_hand(game, 0, yellow=2)
    game.step(ClaimRoute(route_id(game, "C", "D")))
    game.step(Pay(Color.YELLOW, 0))
    enc = O.ObservationEncoder(2)
    obs = enc.encode(game, 0)
    assert block(enc, obs, "tickets_done")[7] == 1 and block(enc, obs, "tickets_done")[0] == 0
    assert block(enc, obs, "tickets_trains")[7] == 0

    # Too few trains left for the 3 still needed: impossible, but the count stays.
    game.players[0].trains = 2
    enc.reset()
    trains = block(enc, enc.encode(game, 0), "tickets_trains")
    assert trains[0] == pytest.approx(3 / 12) and trains[30 + 0] == 1

    # Every route out of A claimed by the opponent: no path at all.
    game.players[0].trains = 10
    for r in game.board.routes:
        if "A" in (r.a, r.b):
            game.route_owner[r.id] = 1
    enc.reset()
    trains = block(enc, enc.encode(game, 0), "tickets_trains")
    assert enc.trains_to_finish(game, 0, 0) is None
    assert trains[0] == 1 and trains[30] == 1


def test_endgame_flags():
    game = started_game()
    enc = O.ObservationEncoder(2)
    assert block(enc, enc.encode(game, 0), "endgame").tolist() == [0, 0]
    game.players[0].trains = 2
    game.step(DrawBlind())
    game.step(DrawBlind())  # player 0's turn ends with 2 trains: final round
    assert game.current_player == 1
    assert block(enc, enc.encode(game, 1), "endgame").tolist() == [1, 1]
    assert block(enc, enc.encode(game, 0), "endgame").tolist() == [1, 1]
    game.step(DrawBlind())
    game.step(DrawBlind())
    assert game.current_player == 0 and not game.game_over
    assert block(enc, enc.encode(game, 0), "endgame").tolist() == [1, 1]
    assert block(enc, enc.encode(game, 1), "endgame").tolist() == [1, 0]  # already played


def test_memory_levels():
    game = started_game()
    game.step(DrawTickets())  # player 0 is an opponent here: their face-up takes below
    game.step(KeepTickets(frozenset(game.players[0].pending_tickets)))
    rng = random.Random(0)
    for _ in range(30):
        game.step(rng.choice(game.legal_actions()))
    level0, level1, level2 = (O.ObservationEncoder(2, level) for level in (0, 1, 2))
    assert block(level0, level0.encode(game, 1), "memory").sum() == 0
    m1 = block(level1, level1.encode(game, 1), "memory")
    m2 = block(level2, level2.encode(game, 1), "memory")
    assert (m1[:10] == m2[:10]).all() and m1[10:].sum() == 0
    # Unseen pool = deck + the opponent's unknown cards.
    hidden = Counter(game.deck)
    unseen = m2[10:] * O._COLOR_SCALE
    known = m2[:9] * O._COLOR_SCALE
    hidden.update(game.players[0].hand)
    assert sum(hidden.values()) == pytest.approx(unseen.sum() + known.sum())


def test_hidden_information_does_not_leak():
    """The opponent's hand, tickets, offer, and the deck order don't change my view."""
    game = started_game(seed=5)
    game.step(DrawTickets())  # player 0 now holds a private offer
    base = O.ObservationEncoder(2).encode(game, 1)

    other = game.clone()
    size = other.players[0].hand_size
    set_hand(other, 0, red=size)  # same size, different cards
    held = set(other.players[0].tickets) | set(other.players[0].pending_tickets)
    spare = [t for t in range(30) if t not in held]
    other.players[0].tickets = spare[:len(other.players[0].tickets)]
    other.players[0].pending_tickets = spare[-3:]
    other.deck.reverse()
    assert (O.ObservationEncoder(2).encode(other, 1) == base).all()


@pytest.mark.parametrize("board,players,seed", [
    ("usa", 2, 1), ("usa", 3, 2), ("usa", 5, 3), ("toy", 2, 4), ("toy", 4, 5),
])
def test_random_games_stay_in_range_and_match_a_fresh_encoder(board, players, seed):
    """Over whole random games, for every seat: values are finite and in [0, 1]
    here, and the cached encoder agrees with a fresh one."""
    rng = random.Random(seed)
    game = Game(load_board(board), num_players=players, seed=seed, max_turns=400)
    enc = O.ObservationEncoder(players)
    steps = 0
    while not game.game_over:
        for viewer in range(players):
            obs = enc.encode(game, viewer)
            assert np.isfinite(obs).all() and obs.min() >= 0 and obs.max() <= 1
            if steps % 17 == 0:
                assert (O.ObservationEncoder(players).encode(game, viewer) == obs).all()
        assert block(enc, enc.encode(game, game.current_player), "phase").sum() == 1
        game.step(rng.choice(game.legal_actions()))
        steps += 1
    assert block(enc, enc.encode(game, 0), "phase").sum() == 0


def test_rejects_the_wrong_player_count():
    with pytest.raises(O.ObservationError):
        O.ObservationEncoder(2).encode(started_game(num_players=3), 0)
    with pytest.raises(O.ObservationError):
        O.ObservationEncoder(2, memory_level=3)
