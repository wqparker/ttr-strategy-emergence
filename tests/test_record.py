import json
import random

import pytest

from ttr.actions import ClaimRoute, DrawBlind, DrawFaceUp, DrawTickets, KeepTickets, Pass, Pay
from ttr.agents import GreedyAgent, RandomAgent
from ttr.board import load_board
from ttr.cards import Color
from ttr.game import Game
from ttr.record import GameRecord, RecordError, action_from_dict, action_to_dict
from ttr.simulate import play_game, run_matches

from helpers import started_game


def recorded_game(seed: int = 7, board: str = "usa"):
    game = Game(load_board(board), num_players=2, seed=seed, first_player=0, max_turns=1000)
    actions = []
    play_game(game, [GreedyAgent(seed), RandomAgent(seed + 1)], actions_out=actions)
    record = GameRecord.from_game(game, actions, agents=["greedy", "random"], board=board, seed=seed)
    return game, record


def assert_same_state(a: Game, b: Game) -> None:
    assert a.route_owner == b.route_owner
    assert a.phase == b.phase and a.turn == b.turn and a.current_player == b.current_player
    assert a.market == b.market and a.deck == b.deck and a.discard == b.discard
    assert a.ticket_deck == b.ticket_deck
    for p, q in zip(a.players, b.players):
        assert p.hand == q.hand and p.tickets == q.tickets and p.routes == q.routes
        assert p.trains == q.trains and p.route_points == q.route_points
        assert p.pending_tickets == q.pending_tickets


@pytest.mark.parametrize(
    "action",
    [
        DrawFaceUp(Color.RED),
        DrawFaceUp(Color.LOCOMOTIVE),
        DrawBlind(),
        ClaimRoute(42),
        Pay(Color.BLUE, 2),
        Pay(None, 4),
        DrawTickets(),
        KeepTickets(frozenset({3, 11})),
        Pass(),
    ],
)
def test_action_round_trip(action):
    d = action_to_dict(action)
    assert action_from_dict(json.loads(json.dumps(d))) == action


def test_unknown_action_type_rejected():
    with pytest.raises(RecordError):
        action_from_dict({"type": "teleport"})


@pytest.mark.parametrize("board", ["usa", "toy"])
def test_save_load_replay_reproduces_game(tmp_path, board):
    game, record = recorded_game(board=board)
    path = record.save(tmp_path / "g.json")
    loaded = GameRecord.load(path)
    assert loaded.actions == record.actions
    replayed = loaded.replay()
    assert_same_state(game, replayed)
    assert replayed.result == game.result


def test_replay_states_cover_every_step():
    game, record = recorded_game()
    states = record.replay_states()
    assert len(states) == len(record.actions) + 1
    assert states[0].turn == 0 and not states[0].log  # fresh setup, nothing logged yet
    assert_same_state(states[-1], game)
    # Each snapshot is independent: stepping one leaves the next unchanged.
    before = states[1].clone()
    states[0].step(record.actions[0])
    assert_same_state(states[1], before)


def test_replay_detects_mismatched_result():
    _, record = recorded_game()
    record.result = dict(record.result, totals=[0, 0])
    with pytest.raises(RecordError):
        record.replay()


def test_record_requires_seed():
    game = started_game()
    with pytest.raises(RecordError):
        GameRecord.from_game(game, [])


def test_rejects_other_json(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(RecordError):
        GameRecord.load(path)


def test_clone_is_independent():
    game = started_game(seed=3)
    rng = random.Random(0)
    for _ in range(20):
        game.step(rng.choice(game.legal_actions()))
    copy = game.clone()
    assert copy.board is game.board  # shared, immutable
    snapshot = game.clone()
    while not copy.game_over:
        copy.step(rng.choice(copy.legal_actions()))
    assert_same_state(game, snapshot)
    # Same RNG state: both continue identically from the same action.
    a, b = game.clone(), game.clone()
    action = a.legal_actions()[0]
    a.step(action)
    b.step(action)
    assert_same_state(a, b)


def test_run_matches_records_every_game(tmp_path):
    board = load_board("usa")
    run_matches(["greedy", "random"], 3, board, seed=5, record_dir=tmp_path)
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 3
    for f in files:
        record = GameRecord.load(f)
        assert sorted(record.agents) == ["greedy", "random"]
        record.replay()  # raises if the recorded result isn't reproduced
