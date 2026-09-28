"""DQN: n-step returns, the replay buffer, the masked network and target,
legal play, and a short training run saved, reloaded and played."""

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from ttr.agents import RandomAgent
from ttr.agents.registry import make_agent
from ttr.board import load_board
from ttr.env import actions as A
from ttr.env.observation import ObservationEncoder
from ttr.game import Game
from ttr.agents import GreedyAgent
from ttr.learn.dqn import (DQNAgent, DQNConfig, QNetwork, Replay, load_network, n_step_returns,
                           play_training_game, td_step, train)
from ttr.learn.linear import evaluate
from ttr.simulate import run_matches


def test_n_step_returns_sum_rewards_up_to_the_end():
    rewards = [1.0, 2.0, 3.0, 4.0]
    ret, done = n_step_returns(rewards, 1)
    assert ret.tolist() == rewards and done.tolist() == [False, False, False, True]
    ret, done = n_step_returns(rewards, 2)
    assert ret.tolist() == [3, 5, 7, 4] and done.tolist() == [False, False, True, True]
    ret, done = n_step_returns(rewards, 10)  # longer than the game: Monte Carlo returns
    assert ret.tolist() == [10, 9, 7, 4] and all(done)


def test_replay_points_each_transition_at_its_bootstrap_decision():
    n = 2
    replay = Replay(capacity=7, obs_size=3, n_step=n, seed=0)
    marker = 0
    # 12 decisions through 7 slots: the second game's third decision (slot 5)
    # bootstraps from its fifth, which wrapped around to slot 0.
    for length in (3, 5, 4):
        obs = [np.full(3, marker + t, dtype=np.float32) for t in range(length)]
        masks = [np.eye(A.N_ACTIONS, dtype=bool)[t] for t in range(length)]
        replay.add_game(obs, masks, list(range(length)), [1.0] * length)
        marker += 100
    assert len(replay) == 7
    for i in range(7):
        t = int(replay.action[i])  # the decision's position in its game
        if replay.done[i]:
            assert replay.boot[i] == i
        else:
            b = replay.boot[i]
            assert replay.obs[b][0] == replay.obs[i][0] + n  # same game, n decisions later
            assert replay.mask[b][t + n]
            assert replay.ret[i] == n


def test_network_masks_illegal_actions():
    net = QNetwork(obs_size=5, hidden=(8,), dueling=True)
    mask = torch.zeros(2, A.N_ACTIONS, dtype=torch.bool)
    mask[0, [3, 50, 167]] = True
    mask[1, 10] = True
    q = net(torch.randn(2, 5), mask)
    assert torch.isfinite(q[mask]).all() and torch.isneginf(q[~mask]).all()
    assert q[0].argmax().item() in (3, 50, 167) and q[1].argmax().item() == 10


def test_td_step_is_double_dqn_and_ends_without_bootstrap():
    torch.manual_seed(0)
    online, target = QNetwork(4, (8,), dueling=False), QNetwork(4, (8,), dueling=False)
    opt = torch.optim.SGD(online.parameters(), lr=0.0)  # measure the error without learning
    obs, next_obs = torch.randn(3, 4), torch.randn(3, 4)
    mask = torch.ones(3, A.N_ACTIONS, dtype=torch.bool)
    next_mask = torch.zeros(3, A.N_ACTIONS, dtype=torch.bool)
    next_mask[:, :20] = True
    action = torch.tensor([0, 5, 7])
    ret = torch.tensor([0.5, -0.25, 1.0])
    done = torch.tensor([False, False, True])
    loss, td = td_step(online, target, opt, (obs, mask, action, ret, next_obs, next_mask, done), grad_clip=0)
    with torch.no_grad():
        q = online(obs, mask)[torch.arange(3), action]
        chosen = online(next_obs, next_mask).argmax(1)  # online picks, target values it
        y = ret + torch.where(done, 0.0, target(next_obs, next_mask)[torch.arange(3), chosen])
    assert td.item() == pytest.approx((q - y).abs().mean().item(), rel=1e-5)
    assert torch.isfinite(loss)


def test_agent_plays_only_legal_moves_and_sees_the_env_view():
    net = QNetwork(ObservationEncoder(2).size, (16,))
    for epsilon in (0.0, 1.0):
        agent = DQNAgent(net, epsilon=epsilon, seed=1)
        game = Game(load_board("usa"), num_players=2, seed=5)
        reference = ObservationEncoder(2)
        other = RandomAgent(2)
        while not game.game_over:
            p = game.current_player
            if p == 0:
                obs, mask, _ = agent.observe(game, p)
                assert np.array_equal(obs, reference.encode(game, p))
                assert mask.tolist() == [bool(x) for x in A.legal_mask(game)]
            game.step(agent.act(game, p) if p == 0 else other.act(game, p))  # step raises on an illegal move


@pytest.fixture(scope="module")
def dqn_run(tmp_path_factory):
    cfg = DQNConfig(games=6, hidden=(32,), batch=16, buffer=5000, learning_starts=100, target_every=5,
                    average=3, epsilon_decay=0.5, device="cpu", seed=2)
    result = train(cfg, eval_every=3, eval_games=2, eval_linear=None, log=lambda line: None, baselines=False)
    path = tmp_path_factory.mktemp("dqn") / "dqn_test_s2.json"
    result.save(path)
    return path, result


def test_training_records_the_linear_run_layout(dqn_run):
    path, result = dqn_run
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["method"] == "dqn" and data["progress"] == {"games": 6, "of": 6, "finished": True}
    assert [e["games"] for e in data["history"]] == [3, 6]
    assert set(data["history"][0]["eval"]) == {"random", "greedy", "wary", "racer"}
    best = data["best"]  # picked on the mean margin over the scripted opponents but random
    assert best["selected_on"] == ["greedy", "wary", "racer"]
    means = [sum(e["eval"][o]["margin"] for o in best["selected_on"]) / 3 for e in data["history"]]
    assert best["selection"] == pytest.approx(max(means))
    assert best["games"] == data["history"][means.index(max(means))]["games"]
    assert len(data["games"]) == 6 and data["games"][-1]["grad_steps"] > 0
    assert data["best"]["games"] in (3, 6)
    nets = torch.load(path.with_suffix(".pt"), weights_only=True)
    assert set(nets) == {"final", "best", "raw"}  # averaging on: the raw network is kept too


def test_saved_networks_reload_and_play(dqn_run):
    path, result = dqn_run
    net, level = load_network(path)
    assert level == 2
    for k, v in net.state_dict().items():
        assert torch.equal(v, result.policy.state_dict()[k].cpu())
    board = load_board("usa")
    for spec in (f"dqn:{path}", f"dqn:{path}@best"):
        out = run_matches([spec, "greedy"], games=2, board=board, seed=3)
        assert len(out["stats"]) == 2
    assert evaluate(RandomAgent(0), f"dqn:{path}", 2, board)["margin"] is not None  # a spec as the opponent
    agent = make_agent(f"dqn:{path}@best", 0)
    assert isinstance(agent, DQNAgent) and agent.epsilon == 0


def test_self_play_uses_the_best_checkpoint(tmp_path):
    cfg = DQNConfig(opponent="self", games=6, hidden=(16,), batch=8, learning_starts=50, device="cpu", seed=4)
    result = train(cfg, eval_every=1, eval_games=1, eval_linear=None, log=lambda line: None, baselines=False)
    assert sum(r["vs_self"] for r in result.games) > 0
    assert sum(r["vs_self"] for r in result.games[:1]) == 0  # greedy until the first checkpoint


def test_dashboard_shows_a_dqn_run(dqn_run, tmp_path):
    pytest.importorskip("matplotlib")
    from ttr.learn.dashboard import main

    path, _ = dqn_run
    main([str(path), "--save", str(tmp_path)])
    assert len(list(tmp_path.glob("*.png"))) == 28  # 7 pages x 4 evaluation opponents


def test_pool_draws_every_member_and_itself(tmp_path):
    cfg = DQNConfig(opponent="pool", games=24, hidden=(16,), batch=8, learning_starts=50, device="cpu", seed=5)
    result = train(cfg, eval_every=4, eval_games=1, eval_linear=None, log=lambda line: None, baselines=False)
    for name in ("greedy", "wary", "racer", "self"):
        assert sum(r[f"vs_{name}"] for r in result.games) > 0, name
    assert sum(r["vs_self"] for r in result.games[:4]) == 0  # no frozen copy before the first evaluation
    with pytest.raises(ValueError):
        train(DQNConfig(opponent="pool", pool=("greedy", "nobody"), games=1, device="cpu"), eval_linear=None)


def test_shaping_pays_during_the_game_and_sums_to_zero():
    net = QNetwork(ObservationEncoder(2).size, (16,))
    episodes = {}
    for shaping in (0.0, 1.0):
        game = Game(load_board("usa"), num_players=2, seed=8)
        agent = DQNAgent(net, epsilon=0.0, seed=3)
        ep = play_training_game(agent, GreedyAgent(4), game, 0, DQNConfig(shaping=shaping))
        margin = game.result.players[0].total - game.result.players[1].total
        assert sum(ep.rewards) == pytest.approx(margin / 100)  # shaping telescopes away
        assert ep.shaping == pytest.approx(0.0, abs=1e-9)
        episodes[shaping] = ep
    plain, shaped = episodes[0.0], episodes[1.0]
    assert plain.actions == shaped.actions  # shaping changes rewards, not play
    assert any(abs(a - b) > 1e-9 for a, b in zip(plain.rewards, shaped.rewards))
