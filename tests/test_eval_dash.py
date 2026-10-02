"""The evaluation dashboard over eval_agents.py CSVs (rendered headless)."""

import csv
import random

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")

from ttr.learn.eval_dash import EvalData, read_csvs, short
from ttr.learn.metrics import METRICS

FIELDS = ["agent", "opponent", "players", "seed", "game", "seat"] + list(METRICS) + [
    "seconds", "searches", "search_seconds"]


def write_csv(path, agents, opponents, games, offset=0.0):
    rng = random.Random(len(agents) + games)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for a, agent in enumerate(agents):
            for opp in opponents:
                for g in range(games):
                    row = {k: rng.random() for k in METRICS}
                    row.update(agent=agent, opponent=opp, players=2, seed=9001, game=g, seat=g % 2,
                               margin=10 * a + g + offset, win_share=float(g % 2), seconds=120.0,
                               searches=40 if agent.startswith("mcts") else 0, search_seconds=140.0)
                    w.writerow(row)


def test_short_names():
    assert short("mcts") == "mcts"
    assert short("mcts:reward=score,opponent=greedy+racer") == "mcts score · opp greedy+racer"
    assert short("mcts:iterations=1600") == "mcts 1600 it"
    assert short("ppo:runs/ppo/pass2/p2b_pool_s3.json") == "ppo p2b_pool_s3"
    assert short("linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best") == "linear p7b…s3@best"
    assert short("racer") == "racer"


def test_games_count_once_and_pair_on_the_same_games(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    write_csv(first, ["mcts", "greedy"], ["ppo:x.json", "greedy"], 5)
    write_csv(second, ["mcts"], ["greedy"], 3, offset=100.0)  # games 0-2 again: these win
    data = EvalData(read_csvs([first, second]))
    assert data.agents == ["mcts", "greedy"]
    assert data.opponents == ["greedy", "ppo:x.json"]  # bots first
    margins = sorted(r["margin"] for r in data.games("mcts", "greedy"))
    assert margins == [3.0, 4.0, 100.0, 101.0, 102.0]
    # mcts - greedy on the same game: (g + 100) - (10 + g) for games 0-2, g - (10 + g) for 3-4
    assert sorted(data.paired("mcts", "greedy", "greedy")) == [-10.0, -10.0, 90.0, 90.0, 90.0]


def test_every_page_renders(tmp_path):
    from ttr.learn.dashboard import main

    path = tmp_path / "pass.csv"
    write_csv(path, ["mcts:reward=score", "mcts", "racer"], ["greedy", "racer", "wary"], 6)
    (tmp_path / "pass.log").write_text("2026-10-01 19:17 18 games to play\n12/18 games, 30 min: x\n",
                                       encoding="utf-8")
    main([str(path), "--save", str(tmp_path / "png"), "--control", "racer"])  # CSVs: the evaluation view
    assert sorted(p.name for p in (tmp_path / "png").iterdir()) == [
        "1_strength.png", "2_paired.png", "3_behavior.png", "4_progress.png"]
