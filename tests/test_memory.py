import random
from collections import Counter

import pytest

from ttr.actions import ClaimRoute, DrawBlind, DrawFaceUp, Pay
from ttr.agents import GreedyAgent
from ttr.board import load_board
from ttr.cards import Color
from ttr.game import Game, Phase
from ttr.memory import CardMemory

from helpers import route_id, set_hand, set_market, started_game


def check_invariants(game: Game, memories) -> None:
    for viewer, memory in memories.items():
        view = memory.view(game)
        unaccounted = Counter()
        for p, known in view.known.items():
            hand = game.players[p].hand
            # Known counts are lower bounds on the true hand.
            for color, n in known.items():
                assert 0 < n <= hand[color], (viewer, p, color)
            assert view.unknown[p] == sum(hand.values()) - sum(known.values()) >= 0
            unaccounted += hand - known
        # Unseen pool = deck + opponents' cards not accounted for, exactly.
        assert view.unseen == Counter(game.deck) + unaccounted


@pytest.mark.parametrize("num_players", [2, 3, 5])
@pytest.mark.parametrize("seed", range(4))
def test_invariants_hold_every_step_random_play(num_players, seed):
    game = Game(load_board("usa"), num_players=num_players, seed=seed, max_turns=400)
    memories = {v: CardMemory(num_players, v) for v in range(num_players)}
    rng = random.Random(seed)
    while not game.game_over:
        game.step(rng.choice(game.legal_actions()))
        check_invariants(game, memories)


def test_invariants_hold_in_greedy_games():
    for seed in range(3):
        game = Game(load_board("usa"), num_players=2, seed=seed)
        memories = {v: CardMemory(2, v) for v in range(2)}
        agents = [GreedyAgent(seed), GreedyAgent(seed + 1)]
        while not game.game_over:
            p = game.current_player
            game.step(agents[p].act(game, p))
            check_invariants(game, memories)


def test_face_up_take_then_paying_more_than_known():
    game = started_game(seed=1)
    set_market(game, Color.RED, Color.RED, Color.BLUE, Color.GREEN, Color.WHITE)
    game.step(DrawFaceUp(Color.RED))
    game.step(DrawFaceUp(Color.RED))
    view = CardMemory.at(game, viewer=1).view(game)
    assert view.known[0][Color.RED] == 2

    # Player 1 passes the turn by drawing blind twice.
    game.step(DrawBlind())
    game.step(DrawBlind())
    assert game.current_player == 0 and game.phase is Phase.CHOOSE_ACTION
    # Player 0 pays 4 red: the 2 known reds, plus 2 that came from blind draws.
    set_hand(game, 0, red=4, blue=1)
    game.step(ClaimRoute(route_id(game, "Duluth", "Omaha")))  # gray, length 2
    game.step(Pay(Color.RED, 0))
    view = CardMemory.at(game, viewer=1).view(game)
    assert view.known[0][Color.RED] == 0
    assert Color.RED not in view.known[0]


def test_levels_hide_what_they_should():
    game = started_game(seed=2)
    game.step(DrawFaceUp(game.market[0]) if game.market[0] is not Color.LOCOMOTIVE else DrawBlind())
    v0 = CardMemory.at(game, viewer=1, level=0).view(game)
    v1 = CardMemory.at(game, viewer=1, level=1).view(game)
    v2 = CardMemory.at(game, viewer=1, level=2).view(game)
    assert not v0.known and not v0.unknown and v0.unseen is None
    assert 0 in v1.known and v1.unseen is None
    assert v2.unseen is not None


def test_viewer_is_not_tracked_as_opponent():
    game = started_game()
    view = CardMemory.at(game, viewer=0).view(game)
    assert set(view.known) == {1}


def test_fresh_tracker_matches_incremental():
    game = Game(load_board("usa"), num_players=3, seed=9)
    incremental = CardMemory(3, viewer=2)
    rng = random.Random(9)
    for _ in range(150):
        if game.game_over:
            break
        game.step(rng.choice(game.legal_actions()))
        incremental.update(game)
    fresh = CardMemory.at(game, viewer=2)
    assert fresh.view(game) == incremental.view(game)


def test_invalid_level():
    with pytest.raises(ValueError):
        CardMemory(2, 0, level=3)
