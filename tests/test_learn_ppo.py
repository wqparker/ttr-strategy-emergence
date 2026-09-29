"""PPO: masking, GAE, the clipped update, legal play, and a short training run
saved, reloaded and played."""

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from ttr.agents import GreedyAgent, RandomAgent
from ttr.agents.registry import make_agent
from ttr.board import load_board
from ttr.env import actions as A
from ttr.env.observation import ObservationEncoder
from ttr.game import Game
from ttr.learn.ppo import (ActorCritic, Episode, PPOAgent, PPOConfig, gae, load_policy, play_rollout_game,
                           ppo_update, train)
from ttr.simulate import run_matches


def test_masked_policy_gives_illegal_actions_no_probability():
    net = ActorCritic(obs_size=5, hidden=(8,))
    mask = torch.zeros(1, A.N_ACTIONS, dtype=torch.bool)
    mask[0, [4, 40, 160]] = True
    with torch.no_grad():
        probs = torch.softmax(net.logits(torch.randn(1, 5), mask), -1)[0]
    assert float(probs[~mask[0]].sum()) == 0 and float(probs[mask[0]].sum()) == pytest.approx(1)
    agent = PPOAgent(net, sample=True, seed=0)
    agent.encoder = None  # step() needs only the observation and mask
    for _ in range(50):
        i, logp, value, entropy = agent.step(np.random.rand(5).astype(np.float32), mask[0].numpy())
        assert i in (4, 40, 160) and logp <= 0 and np.isfinite(value) and 0 <= entropy <= np.log(3) + 1e-9


def test_gae_matches_the_definition():
    rewards, values = [0.0, 0.0, 1.0], [0.5, 0.2, 0.4]
    adv, ret = gae(rewards, values, gamma=1.0, lam=1.0)  # lambda 1: Monte Carlo returns minus V
    assert ret.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert adv.tolist() == pytest.approx([0.5, 0.8, 0.6])
    adv0, _ = gae(rewards, values, gamma=1.0, lam=0.0)  # lambda 0: one-step TD errors, the last bootstraps 0
    assert adv0.tolist() == pytest.approx([0.2 - 0.5, 0.4 - 0.2, 1.0 - 0.4])


def test_an_update_makes_a_rewarded_action_more_likely():
    torch.manual_seed(0)
    net = ActorCritic(obs_size=4, hidden=(16,))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    obs = np.random.rand(4).astype(np.float32)
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[[1, 2, 3]] = True
    with torch.no_grad():
        logp = torch.log_softmax(net.logits(torch.as_tensor(obs)[None], torch.as_tensor(mask)[None]), -1)[0]
    good, bad = Episode(), Episode()
    for ep, action, reward in ((good, 1, 1.0), (bad, 2, -1.0)):
        ep.obs, ep.masks, ep.actions = [obs], [mask], [action]
        ep.logprobs, ep.values, ep.rewards = [float(logp[action])], [0.0], [reward]
    cfg = PPOConfig(update_epochs=4, minibatches=1, ent_coef=0.0)
    stats = ppo_update(net, opt, [good, bad], cfg, torch.device("cpu"), np.random.default_rng(0))
    with torch.no_grad():
        after = torch.log_softmax(net.logits(torch.as_tensor(obs)[None], torch.as_tensor(mask)[None]), -1)[0]
    assert after[1] > logp[1] and after[2] < logp[2]
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"}


def test_rollouts_play_legal_moves_and_rewards_sum_to_the_margin():
    net = ActorCritic(ObservationEncoder(2).size, (16,))
    game = Game(load_board("usa"), num_players=2, seed=11)
    ep = play_rollout_game(PPOAgent(net, sample=True, seed=2), GreedyAgent(3), game, 1, PPOConfig(shaping=1.0))
    margin = game.result.players[1].total - game.result.players[0].total
    assert sum(ep.rewards) == pytest.approx(margin / 100)  # shaping telescopes away
    assert len(ep.rewards) == len(ep.actions) == len(ep.logprobs) == len(ep.values)
    assert all(m[a] for m, a in zip(ep.masks, ep.actions))
    greedy_play = PPOAgent(net, seed=0)  # argmax: plays a whole game legally too
    game = Game(load_board("usa"), num_players=2, seed=12)
    while not game.game_over:
        p = game.current_player
        game.step(greedy_play.act(game, p) if p == 0 else RandomAgent(5).act(game, p))


@pytest.fixture(scope="module")
def ppo_run(tmp_path_factory):
    cfg = PPOConfig(opponent="pool", games=8, games_per_update=2, hidden=(32,), shaping=1.0, ticket_plan=True,
                    device="cpu", seed=3)
    result = train(cfg, eval_every=4, eval_games=2, eval_linear=None, log=lambda line: None, baselines=False)
    path = tmp_path_factory.mktemp("ppo") / "ppo_test_s3.json"
    result.save(path)
    return path, result


def test_training_records_the_run_layout(ppo_run):
    path, result = ppo_run
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["method"] == "ppo" and data["progress"] == {"games": 8, "of": 8, "finished": True}
    assert [e["games"] for e in data["history"]] == [4, 8]
    assert set(data["history"][0]["eval"]) == {"random", "greedy", "wary", "racer"}
    assert len(data["updates"]) == 4 and data["updates"][-1]["lr"] < data["updates"][0]["lr"]  # annealed
    assert data["best"]["selected_on"] == ["greedy", "wary", "racer"]
    row = data["games"][0]
    assert row["epsilon"] == 0 and row["entropy"] > 0 and row["decisions"] > 0
    assert abs(sum(r["shaping"] for r in data["games"])) < 1e-6


def test_saved_policy_reloads_and_plays(ppo_run):
    path, result = ppo_run
    net, view = load_policy(path)
    assert view == {"memory_level": 2, "ticket_plan": True}
    for k, v in net.state_dict().items():
        assert torch.equal(v, result.net.state_dict()[k].cpu())
    for spec in (f"ppo:{path}", f"ppo:{path}@best"):
        out = run_matches([spec, "racer"], games=2, board=load_board("usa"), seed=4)
        assert len(out["stats"]) == 2
    agent = make_agent(f"ppo:{path}@best", 0)
    assert isinstance(agent, PPOAgent) and not agent.sample


def test_dashboard_shows_a_ppo_run(ppo_run, tmp_path):
    pytest.importorskip("matplotlib")
    from ttr.learn.dashboard import main

    path, _ = ppo_run
    main([str(path), "--save", str(tmp_path)])
    assert len(list(tmp_path.glob("*.png"))) == 28  # 7 pages x 4 evaluation opponents
