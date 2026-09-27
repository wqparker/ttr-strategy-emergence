"""The PettingZoo AEC environment: API conformance, reward modes, masks."""

import random

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pettingzoo")

from pettingzoo.test import api_test, seed_test

from ttr.env import actions as A
from ttr.env.aec import env, raw_env


def play(e, rng, seed=0):
    """Random legal play to the end. Returns per-agent reward sums and final infos."""
    e.reset(seed=seed)
    totals = {a: 0.0 for a in e.possible_agents}
    infos = {}
    for agent in e.agent_iter():
        obs, reward, terminated, truncated, info = e.last()
        totals[agent] += reward
        if terminated or truncated:
            infos[agent] = info
            e.step(None)
            continue
        legal = np.flatnonzero(obs["action_mask"])
        assert len(legal) > 0
        e.step(int(rng.choice(legal)))
    return totals, infos


@pytest.mark.parametrize("kwargs", [
    {"num_players": 2, "board": "usa"},
    {"num_players": 3, "board": "toy", "reward_mode": "win"},
])
def test_pettingzoo_api(kwargs):
    api_test(env(**kwargs), num_cycles=300)


def test_seeded_resets_repeat():
    seed_test(lambda: env(board="toy"), num_cycles=200)


@pytest.mark.parametrize("board,players", [("usa", 2), ("toy", 3), ("usa", 5)])
def test_score_rewards_sum_to_the_final_score(board, players):
    e = env(num_players=players, board=board, reward_mode="score", reward_scale=1.0)
    totals, infos = play(e, random.Random(1), seed=1)
    assert set(infos) == set(e.possible_agents)
    for a in e.possible_agents:
        assert totals[a] == pytest.approx(infos[a]["score"])


@pytest.mark.parametrize("board,players", [("usa", 2), ("toy", 4)])
def test_margin_rewards_sum_to_the_final_margin(board, players):
    e = env(num_players=players, board=board)  # margin, scaled by 1/100
    totals, infos = play(e, random.Random(2), seed=2)
    scores = {a: infos[a]["score"] for a in e.possible_agents}
    for a in e.possible_agents:
        others = [s for b, s in scores.items() if b != a]
        assert totals[a] == pytest.approx((scores[a] - sum(others) / len(others)) / 100)
    assert sum(totals.values()) == pytest.approx(0)


def test_win_rewards():
    e = env(board="toy", reward_mode="win")
    for seed in range(10):
        totals, infos = play(e, random.Random(seed), seed=seed)
        winners = [a for a in infos if infos[a]["winner"]]
        for a in e.possible_agents:
            if a not in winners:
                assert totals[a] == -1
            else:
                assert totals[a] == (1 if len(winners) == 1 else 0)


def test_only_the_acting_seat_has_a_mask():
    e = env()
    e.reset(seed=4)
    for _ in range(40):
        acting = e.agent_selection
        for a in e.agents:
            mask = e.observe(a)["action_mask"]
            assert mask.dtype == np.int8
            assert (mask.sum() > 0) == (a == acting)
        e.step(int(np.flatnonzero(e.observe(acting)["action_mask"])[0]))


def test_illegal_action_raises():
    e = env()
    e.reset(seed=0)
    mask = e.observe(e.agent_selection)["action_mask"]
    with pytest.raises(ValueError):
        e.step(int(np.flatnonzero(mask == 0)[0]))
    with pytest.raises(ValueError):
        e.step(A.N_ACTIONS)


def test_max_turns_truncates():
    e = env(board="toy", max_turns=6)
    _, infos = play(e, random.Random(0))
    assert e.unwrapped.game.result.truncated
    assert len(infos) == 2


def test_ansi_render():
    e = raw_env(board="toy", render_mode="ansi")
    e.reset(seed=0)
    text = e.render()
    assert "Face up" in text


def test_bad_settings():
    with pytest.raises(ValueError):
        raw_env(reward_mode="points")
    with pytest.raises(ValueError):
        raw_env(render_mode="rgb_array")
    with pytest.raises(ValueError, match="too few tickets"):
        raw_env(num_players=5, board="toy")
