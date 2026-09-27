"""Batch seating: agents dealt into random seats, a random first seat, and
records that replay it."""

from collections import Counter

from ttr.board import load_board
from ttr.game import Game
from ttr.record import GameRecord
from ttr.simulate import run_matches, seating


def play_order(slots, first):
    """The agent slots in the order they move, starting from the first seat."""
    n = len(slots)
    return tuple(slots[(first + i) % n] for i in range(n))


def test_seating_is_a_seeded_permutation():
    assert sorted(seating(5, 7, 3)) == [0, 1, 2, 3, 4]
    assert seating(5, 7, 3) == seating(5, 7, 3)
    assert len({tuple(seating(4, 7, g)) for g in range(40)}) > 10


def test_every_order_of_play_comes_up():
    """Rotating seats only ever gave the cyclic orders (A->B->C and its shifts).
    With shuffled seats and a random first seat, every order appears."""
    seed = 11
    orders = set()
    for g in range(80):
        slots = seating(3, seed, g)
        first = Game(load_board("usa"), num_players=3, seed=seed * 100000 + g).first_player
        # Cycles are what matter for who follows whom: normalise to start at slot 0.
        order = play_order(slots, first)
        i = order.index(0)
        orders.add(order[i:] + order[:i])
    assert orders == {(0, 1, 2), (0, 2, 1)}


def test_batch_records_hold_the_seating_and_replay(tmp_path):
    run_matches(["greedy", "random", "greedy"], 6, load_board("usa"), seed=4,
                record_dir=tmp_path, board_ref="usa")
    firsts = Counter()
    for path in sorted(tmp_path.glob("*.json")):
        record = GameRecord.load(path)
        assert record.first_player_drawn and sorted(record.slots) == [0, 1, 2]
        assert record.agents == [["greedy", "random", "greedy"][s] for s in record.slots]
        record.replay()  # raises if the replay drifts from the recorded result
        firsts[record.first_player] += 1
    assert len(firsts) > 1  # not always seat 0 any more


def test_old_records_with_a_fixed_first_seat_still_replay(tmp_path):
    game = Game(load_board("usa"), num_players=2, seed=9, first_player=0, max_turns=50)
    record = GameRecord.from_game(game, [], agents=["a", "b"], seed=9)
    d = record.to_dict()
    del d["first_player_drawn"], d["slots"]  # the fields older records lack
    old = GameRecord.from_dict(d)
    assert not old.first_player_drawn and old.slots is None
    assert old.new_game().first_player == 0
