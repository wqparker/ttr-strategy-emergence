"""Expert iteration's first step: recording a search's decisions and training a network on them."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from ttr.agents.greedy import GreedyAgent
from ttr.agents.mcts import MCTSAgent
from ttr.board import load_board
from ttr.env import actions as A
from ttr.env.observation import PHASES, ObservationEncoder
from ttr.game import Game, Phase
from ttr.learn import ppo
from ttr.learn.distill import (DistillConfig, SearchRecorder, chunk_path, load_data, policy_targets, save_chunk,
                               train, value_targets)
from ttr.simulate import play_game, run_matches

ITERATIONS = 6


def recorded_game(seed: int, seat: int = 1):
    game = Game(load_board("usa"), num_players=2, seed=seed)
    recorder = SearchRecorder(MCTSAgent(seed, iterations=ITERATIONS, reward="score"), 2)
    agents = [GreedyAgent(seed + 1), GreedyAgent(seed + 1)]
    agents[seat] = recorder
    play_game(game, agents)
    return game, recorder


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """Two chunks of recorded games against greedy, and the first chunk saved again under
    another name (a chunk replayed after a stop with another chunk size)."""
    directory = tmp_path_factory.mktemp("data")
    chunks = [[0, 1], [2]]
    for games in chunks:
        records = []
        for g in games:
            game, recorder = recorded_game(100 + g, seat=g % 2)
            records.append(recorder.take(game, g % 2, g))
        save_chunk(chunk_path(directory, "mcts", "greedy", 2, 7, games), records, "mcts", "greedy", 2, 7)
    replay = directory / "replayed.npz"
    chunk_path(directory, "mcts", "greedy", 2, 7, [0, 1]).replace(replay)
    save_chunk(chunk_path(directory, "mcts", "greedy", 2, 7, [0, 1]), [load_game(replay, 0), load_game(replay, 1)],
               "mcts", "greedy", 2, 7)
    return directory


def load_game(path, g):
    with np.load(path) as f:
        rows = f["game"] == g
        return {k: f[k][rows] for k in f.files if k not in ("agent", "opponent", "players", "seed")}


def test_recorder_keeps_every_choice_with_the_searchs_visits():
    game, recorder = recorded_game(5)
    rows = recorder.take(game, 1, 0)
    n = len(rows["action"])
    assert n > 20 and recorder.rows == []
    assert rows["obs"].shape == (n, ObservationEncoder(2, ticket_plan=True).size)
    assert rows["mask"][np.arange(n), rows["action"]].all()
    assert (rows["mask"].sum(1) > 1).all()  # forced moves are left out
    s = rows["searched"]
    assert s.sum() == recorder.searches
    assert (rows["visits"][s].sum(1) == ITERATIONS).all()
    assert (rows["visits"][~rows["mask"]] == 0).all() and (rows["visits"][~s] == 0).all()
    played = rows["visits"][s][np.arange(s.sum()), rows["action"][s]]
    assert (played == rows["visits"][s].max(1)).all()  # the most visited move (ties: any of them)
    minor = {PHASES.index(Phase.DRAW_SECOND_CARD), PHASES.index(Phase.CHOOSE_PAYMENT)}
    assert set(rows["phase"][~s]) <= minor and not set(rows["phase"][s]) & minor
    total = game.result.players[1].total
    assert (rows["final_score"] == total).all() and rows["score"][0] == 0
    assert (rows["final_margin"] == total - game.result.players[0].total).all()
    assert (rows["won"] == float(game.result.winners == [1])).all()
    assert (np.diff(rows["score"]) >= 0).all() and (rows["seat"] == 1).all() and (rows["game"] == 0).all()


def test_loading_counts_a_replayed_game_once(data_dir):
    data = load_data(data_dir)
    assert data["batches"] == [("mcts", "greedy", 2, 7)]
    assert sorted(np.unique(data["game"]).tolist()) == [0, 1, 2]
    per_game = [int((data["game"] == g).sum()) for g in (0, 1, 2)]
    assert all(k > 20 for k in per_game) and len(data["action"]) == sum(per_game)
    assert len(load_data(data_dir, opponents=["greedy"])["action"]) == sum(per_game)
    assert sorted(np.unique(load_data(data_dir, first_games=2)["game"]).tolist()) == [0, 1]
    with pytest.raises(ValueError):
        load_data(data_dir, opponents=["racer"])
    with pytest.raises(ValueError):
        load_data(data_dir, players=3)


def test_policy_and_value_targets(data_dir):
    data = load_data(data_dir)
    s = data["searched"]
    t = policy_targets(data, "visits")
    assert np.allclose(t.sum(1), 1) and (t[~data["mask"]] == 0).all()
    assert np.allclose(t[s], data["visits"][s] / ITERATIONS)
    assert (t[~s].argmax(1) == data["action"][~s]).all() and (t[~s].max(1) == 1).all()
    sharp = policy_targets(data, "visits", temperature=0.5)
    v = data["visits"][s].astype(float) ** 2
    assert np.allclose(sharp[s], v / v.sum(1, keepdims=True))
    move = policy_targets(data, "move")
    assert (move.argmax(1) == data["action"]).all() and (move.max(1) == 1).all()
    first = np.flatnonzero(data["score"] == 0)[0]
    assert value_targets(data, "score")[first] == pytest.approx(data["final_score"][first] / 100)
    assert np.allclose(value_targets(data, "margin"), (data["final_margin"] - data["margin"]) / 100)
    assert set(value_targets(data, "win")) <= {-1.0, 1.0}


@pytest.fixture(scope="module")
def distilled(data_dir, tmp_path_factory):
    cfg = DistillConfig(data=str(data_dir), epochs=3, batch=32, lr=3e-3, hidden=(32,), val_every=3, device="cpu",
                        seed=1)
    result = train(cfg, eval_every=2, eval_games=2, eval_linear=None, eval_opponents=("greedy",),
                   log=lambda line: None)
    path = tmp_path_factory.mktemp("distill") / "x_s1.json"
    result.save(path)
    return path, result


def test_training_imitates_and_saves_a_ppo_run(distilled):
    path, result = distilled
    epochs = result.epochs
    assert [e["epoch"] for e in epochs] == [1, 2, 3]
    assert epochs[-1]["train_ce"] < epochs[0]["train_ce"]
    assert result.data["validation"] > 0 and result.data["by_opponent"]["greedy"]["games"] == 3
    assert [h["epoch"] for h in result.history] == [2, 3] and result.best["epoch"] in (2, 3)
    run = json.loads(path.read_text(encoding="utf-8"))
    assert run["method"] == "ppo" and run["config"]["distill"] and run["progress"]["finished"]
    net, view = ppo.load_policy(path)
    assert view == {"num_players": 2, "memory_level": 2, "ticket_plan": False}
    out = run_matches([f"ppo:{path}", "greedy"], games=2, board=load_board("usa"), seed=3)
    assert out["stats"][0].games == 2


def test_ppo_starts_from_a_distilled_network(distilled):
    path, _ = distilled
    start, _ = ppo.load_policy(path)
    cfg = ppo.PPOConfig(opponent="greedy", games=2, games_per_update=4, hidden=(32,), init=str(path), device="cpu")
    result = ppo.train(cfg, eval_every=0, eval_games=1, eval_linear=None, eval_opponents=("greedy",),
                       log=lambda line: None, baselines=False)
    assert not result.updates  # no update yet: still the distilled network
    for k, v in start.state_dict().items():
        assert torch_equal(result.net.state_dict()[k], v)
    with pytest.raises(ValueError):  # another observation
        ppo.train(ppo.PPOConfig(opponent="greedy", games=1, hidden=(32,), init=str(path), ticket_plan=True,
                                device="cpu"), eval_every=0, eval_games=1, eval_linear=None,
                  eval_opponents=("greedy",), log=lambda line: None, baselines=False)


def torch_equal(a, b) -> bool:
    import torch

    return torch.equal(a.cpu(), b.cpu())


def test_ticket_plan_arm_reads_the_longer_vector(data_dir):
    cfg = DistillConfig(data=str(data_dir), epochs=1, batch=64, hidden=(16,), ticket_plan=True, target="move",
                        device="cpu")
    result = train(cfg, eval_every=0, eval_games=1, eval_linear=None, eval_opponents=("greedy",),
                   log=lambda line: None)
    assert result.net.actor[0].in_features == ObservationEncoder(2, ticket_plan=True).size
    assert A.N_ACTIONS == result.net.actor[-1].out_features
