"""Linear Q-learning / SARSA: features, updates, saving, and that it learns."""

import random

import pytest

np = pytest.importorskip("numpy")

from ttr.actions import ClaimRoute, DrawFaceUp, DrawTickets, KeepTickets, Pay
from ttr.board import load_board
from ttr.cards import Color
from ttr.game import Game, Phase
from ttr.learn import features as F
from ttr.learn.linear import LinearAgent, TrainConfig, _update, evaluate, train, train_game
from ttr.agents import RandomAgent

from helpers import route_id, set_hand, set_market, started_game

DENVER_EL_PASO = 24  # 4 points; Denver-Santa Fe-El Paso, 2 + 2 trains


def test_feature_names_are_unique_and_match_the_vectors():
    game = started_game()
    for kind in F.ACTION_TYPES:
        names = F.feature_names(kind)
        assert len(set(names)) == len(names)
    for kind, phi in F.all_features(game, 0):
        assert len(phi) == len(F.feature_names(kind))
        assert np.isfinite(phi).all() and phi.min() >= -1 and phi.max() <= 1


def named(game, p, action):
    kind, phi = F.action_features(game, p, F.context(game, p), action)
    return dict(zip(F.feature_names(kind), phi))


def test_claim_features_follow_my_tickets():
    game = started_game()
    game.players[0].tickets = [DENVER_EL_PASO]
    set_hand(game, 0, red=2)
    to_santa_fe = named(game, 0, ClaimRoute(route_id(game, "Denver", "Santa Fe")))
    assert to_santa_fe["on_ticket_path"] == 1 and to_santa_fe["completes_ticket"] == 0
    assert to_santa_fe["ticket_points"] == pytest.approx(4 / 20)
    elsewhere = named(game, 0, ClaimRoute(route_id(game, "Boston", "New York", "yellow")))
    assert elsewhere["on_ticket_path"] == 0

    game.step(ClaimRoute(route_id(game, "Denver", "Santa Fe")))
    game.step(Pay(Color.RED, 0))
    game.current_player = 0
    game.phase = Phase.CHOOSE_ACTION
    game.invalidate()
    last = named(game, 0, ClaimRoute(route_id(game, "Santa Fe", "El Paso")))
    assert last["completes_ticket"] == 1


def test_draw_features_know_needed_colors():
    game = started_game()
    helena_la = next(t for t in game.board.tickets if {t.a, t.b} == {"Helena", "Los Angeles"})
    game.players[0].tickets = [helena_la.id]
    set_hand(game, 0)
    market = (Color.GREEN, Color.RED, Color.BLUE, Color.WHITE, Color.BLACK)
    set_market(game, *market)
    needed = {c for c, n in F.context(game, 0).needs.items() if n > 0}
    assert needed  # an empty hand needs every colored route on the path
    for color in market:
        assert named(game, 0, DrawFaceUp(color))["needed"] == float(color in needed)


def test_keep_features():
    game = Game(load_board("usa"), num_players=2, seed=3, first_player=0)
    offer = game.players[0].pending_tickets
    feats = {a: named(game, 0, a) for a in game.legal_actions()}
    all_three = feats[KeepTickets(frozenset(offer))]
    assert all_three["count"] == 1 and all_three["initial"] == 1
    points = sum(game.board.tickets[t].points for t in offer)
    assert all_three["points"] == pytest.approx(min(points / 30, 1))


def test_features_ignore_the_opponents_hidden_cards_and_tickets():
    game = started_game(seed=4)
    base = [(k, phi.tolist()) for k, phi in F.all_features(game, 0)]
    set_hand(game, 1, red=game.players[1].hand_size)
    game.players[1].tickets = [t for t in range(30) if t not in game.players[0].tickets][:3]
    assert [(k, phi.tolist()) for k, phi in F.all_features(game, 0)] == base


def test_update_moves_q_toward_the_target():
    agent = LinearAgent()
    phi = np.array([1.0, 0.5] + [0.0] * (len(F.feature_names("draw_blind")) - 2))
    for _ in range(3):
        before = float(agent.weights["draw_blind"] @ phi)
        _update(agent, "draw_blind", phi, 1.0, alpha=0.5)
        after = float(agent.weights["draw_blind"] @ phi)
        assert abs(1.0 - after) == pytest.approx(0.5 * abs(1.0 - before))


@pytest.mark.parametrize("algo", ["q", "sarsa"])
def test_training_game_runs(algo):
    agent = LinearAgent(epsilon=0.1, seed=0)
    game = Game(load_board("usa"), num_players=2, seed=1, max_turns=1000)
    row = train_game(agent, RandomAgent(1), game, 0, TrainConfig(algo=algo))
    assert game.game_over and row["decisions"] > 20 and row["mean_abs_td"] > 0
    assert any(w.any() for w in agent.weights.values())


def test_save_and_load(tmp_path):
    agent = LinearAgent(seed=0)
    agent.weights["claim"][:] = np.arange(len(agent.weights["claim"])) / 10
    path = tmp_path / "w.json"
    agent.save(path, note="x")
    loaded = LinearAgent.load(path)
    for k in F.ACTION_TYPES:
        assert np.allclose(loaded.weights[k], agent.weights[k])


def test_learns_to_beat_random():
    """Untrained (all Q equal, so a random policy) wins about half; a short
    training run against random should win nearly every game."""
    board = load_board("usa")
    untrained = evaluate(LinearAgent(seed=0), "random", 30, board, seed=1)
    result = train(TrainConfig(games=100, seed=2), eval_games=5, log=lambda line: None, baselines=False)
    trained = evaluate(result.agent, "random", 30, board, seed=1)
    assert untrained["win_share"] < 0.75
    assert trained["win_share"] >= 0.9 and trained["margin"] > untrained["margin"] + 50
    assert len(result.history) == 1 and result.history[0]["games"] == 100
    assert len(result.games) == 100


def test_payment_carries_the_route_features():
    """Route points are scored at the payment step, so its features name the route."""
    game = started_game()
    game.players[0].tickets = [DENVER_EL_PASO]
    set_hand(game, 0, red=2)
    game.step(ClaimRoute(route_id(game, "Denver", "Santa Fe")))
    pay = named(game, 0, Pay(Color.RED, 0))
    assert pay["route_points"] == pytest.approx(2 / 15)
    assert pay["on_ticket_path"] == 1 and pay["ticket_points"] == pytest.approx(4 / 20)


def test_tickets_kept_on_the_last_turn_are_doomed():
    game = started_game()
    game.final_turns_remaining = 2  # final round: this turn is player 0's last
    game.invalidate()
    game.step(DrawTickets())
    offer = game.players[0].pending_tickets
    keep_all = named(game, 0, KeepTickets(frozenset(offer)))
    points = sum(game.board.tickets[t].points for t in offer)
    assert keep_all["doomed_points"] == pytest.approx(min(points / 30, 1))

    early = started_game()
    early.step(DrawTickets())
    offer = early.players[0].pending_tickets
    assert named(early, 0, KeepTickets(frozenset(offer)))["doomed_points"] == 0


def test_a_run_records_everything_the_dashboard_reads(tmp_path):
    from ttr.learn.metrics import METRICS

    cfg = TrainConfig(games=4, seed=3)
    result = train(cfg, eval_every=2, eval_games=2, log=lambda line: None)
    assert [e["games"] for e in result.history] == [2, 4]
    assert [s["games"] for s in result.snapshots] == [2, 4]
    assert set(result.history[0]["eval"]) == {"random", "greedy"}
    assert set(result.baselines) == {"random", "greedy"}
    assert set(result.baselines["greedy"]["random"]) == set(METRICS)
    for row in result.games:
        assert set(METRICS) <= set(row) and {"game", "epsilon", "seat", "decisions", "mean_abs_td"} <= set(row)
    path = tmp_path / "run.json"
    result.save(path, cfg)
    loaded = LinearAgent.load(path)
    assert np.allclose(loaded.weights["claim"], result.agent.weights["claim"], atol=1e-6)
