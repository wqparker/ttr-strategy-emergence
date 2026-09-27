"""Game metrics and the agent-analysis dashboard (rendered headless)."""

import json
import random
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from ttr.agents import GreedyAgent, RandomAgent
from ttr.board import load_board
from ttr.game import Game
from ttr.learn.metrics import METRICS, game_metrics, mean_metrics
from ttr.simulate import play_game


def test_game_metrics_agree_with_the_result():
    game = Game(load_board("usa"), num_players=2, seed=4)
    play_game(game, [GreedyAgent(1), RandomAgent(2)])
    for seat in (0, 1):
        m = game_metrics(game, seat)
        r = game.result.players[seat]
        assert set(m) == set(METRICS)
        assert m["score"] == r.total and m["route_points"] == r.route_points
        assert m["tickets_completed"] == r.tickets_completed and m["tickets_failed"] == r.tickets_failed
        assert m["claims"] == len(game.players[seat].routes)
        assert m["tickets_kept"] == len(game.players[seat].tickets)
        assert m["trains_left"] == game.players[seat].trains
        assert m["margin"] == r.total - game.result.players[1 - seat].total
    assert game_metrics(game, 0)["ticket_draws"] + game_metrics(game, 1)["ticket_draws"] == sum(
        e.kind == "draw_tickets" for e in game.log)
    avg = mean_metrics([game_metrics(game, 0), game_metrics(game, 1)])
    assert avg["margin"] == pytest.approx(0)


@pytest.fixture(scope="module")
def run_file(tmp_path_factory):
    from ttr.learn.linear import TrainConfig, train

    cfg = TrainConfig(games=12, seed=1)
    result = train(cfg, eval_every=4, eval_games=3, log=lambda line: None)
    path = tmp_path_factory.mktemp("runs") / "q_test.json"
    result.save(path, cfg)
    return path


def test_every_page_renders(run_file, tmp_path):
    pytest.importorskip("matplotlib")
    from ttr.learn.dashboard import main

    main([str(run_file), str(run_file), "--save", str(tmp_path)])
    pngs = sorted(tmp_path.glob("*.png"))
    assert len(pngs) == 14  # 7 pages x 2 opponents
    assert all(p.stat().st_size > 20_000 for p in pngs)


def test_keys_move_between_pages(run_file):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ttr.learn.dashboard import SERIES, Dashboard, load_run

    dash = Dashboard([load_run(run_file, SERIES[0])])
    dash.draw()
    for key, page in (("right", 1), ("left", 0), ("7", 6), ("right", 0)):
        dash.on_key(SimpleNamespace(key=key))
        assert dash.page == page
    dash.on_key(SimpleNamespace(key="o"))
    assert dash.opponent == "random"
    dash.page = 6
    dash.on_key(SimpleNamespace(key="pagedown"))
    dash.on_key(SimpleNamespace(key="end"))
    assert dash.games_page == 0  # 12 games fit on one page
    plt.close(dash.fig)


def test_old_run_files_are_refused(tmp_path):
    pytest.importorskip("matplotlib")
    from ttr.learn.dashboard import RunFormatError, load_run

    path = tmp_path / "old.json"
    path.write_text(json.dumps({"history": [{"games": 1, "vs_random_win_rate": 1}]}), encoding="utf-8")
    with pytest.raises(RunFormatError):
        load_run(path, "#000000")


def test_group_averages_seeds(run_file, tmp_path):
    pytest.importorskip("matplotlib")
    from ttr.learn.dashboard import SERIES, group_runs, load_run

    data = json.loads(run_file.read_text(encoding="utf-8"))
    paths = []
    for seed, shift in ((0, 0.0), (1, 10.0)):
        d = json.loads(json.dumps(data))
        for e in d["history"]:
            e["eval"]["greedy"]["margin"] += shift
        d["config"]["seed"] = seed
        path = tmp_path / f"q_test_s{seed}.json"
        path.write_text(json.dumps(d), encoding="utf-8")
        paths.append(path)
    other = tmp_path / "sarsa_x.json"
    other.write_text(json.dumps(data), encoding="utf-8")
    runs = [load_run(p, SERIES[i]) for i, p in enumerate(paths)] + [load_run(other, SERIES[2])]
    grouped = group_runs(runs)
    assert [r.name for r in grouped] == ["q_test x2", "sarsa_x"]
    mean = grouped[0].history[-1]["eval"]["greedy"]["margin"]
    assert mean == pytest.approx(data["history"][-1]["eval"]["greedy"]["margin"] + 5)
    assert grouped[0].config["seed"] == "0, 1"
