"""The policy adapter (observation + mask -> action) and the agent registry."""

import argparse

import pytest

np = pytest.importorskip("numpy")

from ttr.agents.policy import PolicyAgent, legal_mask, random_policy
from ttr.agents.registry import agent_spec, make_agent
from ttr.board import load_board
from ttr.env import actions as A
from ttr.env.aec import raw_env
from ttr.game import Game
from ttr.simulate import play_game, run_matches

from helpers import started_game


@pytest.mark.parametrize("players", [2, 4])
def test_policy_agents_play_full_games(players):
    agents = [PolicyAgent(random_policy(i), players) for i in range(players)]
    result = play_game(Game(load_board("usa"), num_players=players, seed=3, max_turns=1000), agents)
    assert result.winners


def test_policy_sees_what_the_env_shows():
    """Same observation and mask as the PettingZoo env gives the acting seat."""
    env = raw_env()
    env.reset(seed=5)
    seen = []

    def policy(obs, mask):
        seen.append((obs.copy(), mask.copy()))
        return int(np.flatnonzero(mask)[0])

    agent = PolicyAgent(policy, 2)
    for _ in range(30):
        game = env.game
        seat = game.current_player
        expected = env.observe(env.agent_selection)
        action = agent.act(game, seat)
        obs, mask = seen[-1]
        assert (obs == expected["observation"]).all() and (mask == expected["action_mask"]).all()
        env.step(A.encode(game, action))


def test_masked_choice_raises():
    game = started_game()
    bad = int(np.flatnonzero(legal_mask(game) == 0)[0])
    with pytest.raises(ValueError):
        PolicyAgent(lambda obs, mask: bad, 2).act(game, 0)
    with pytest.raises(ValueError):
        PolicyAgent(random_policy(0), 2).act(game, 1)  # not seat 1's turn


def test_registry():
    assert make_agent("random", 0).name == "random"
    assert make_agent("greedy", 0).name == "greedy"
    for bad in ("nobody", "greedy:x", "linear", "linear:missing.json"):
        with pytest.raises(argparse.ArgumentTypeError):
            agent_spec(bad)


def test_linear_spec_plays_in_the_match_runner(tmp_path):
    pytest.importorskip("ttr.learn.linear")
    from ttr.learn.linear import LinearAgent

    path = tmp_path / "w.json"
    LinearAgent().save(path)
    spec = f"linear:{path}"
    assert agent_spec(spec) == spec
    summary = run_matches([spec, "random"], games=2, board=load_board("usa"))
    assert summary["stats"][0].games == 2
