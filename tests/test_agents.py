import io

import pytest
from rich.console import Console

from helpers import route_id, started_game
from ttr.agents import GreedyAgent, RandomAgent
from ttr.agents.greedy import cheapest_path
from ttr.board import load_board
from ttr.game import Game
from ttr.ascii_map import render_map
from ttr.render import describe_event, render_board_view, render_game, render_result, render_routes
from ttr.simulate import play_game, run_matches


def test_cheapest_path_uses_own_routes_free_and_avoids_blocked():
    game = started_game()
    cost, path = cheapest_path(game, 0, "Nashville", "Atlanta")
    assert cost == 1 and len(path) == 1
    game.route_owner[path[0]] = 1  # opponent takes it
    cost2, path2 = cheapest_path(game, 0, "Nashville", "Atlanta")
    assert cost2 > 1 and path[0] not in path2
    own = route_id(game, "Raleigh", "Nashville")
    game.route_owner[own] = 0
    cost3, path3 = cheapest_path(game, 0, "Raleigh", "Nashville")
    assert cost3 == 0 and path3 == []


@pytest.mark.parametrize("board", ["toy", "usa"])
@pytest.mark.parametrize("num_players", [2, 4])
def test_agents_play_full_games(board, num_players):
    for seed in range(3):
        agents = [GreedyAgent(seed), RandomAgent(seed)] * 2
        game = Game(load_board(board), num_players=num_players, seed=seed, max_turns=2000)
        result = play_game(game, agents[:num_players])  # step() rejects illegal moves
        assert result.winners


def test_greedy_beats_random():
    summary = run_matches(["greedy", "random"], games=20, board=load_board("usa"))
    greedy, rnd = summary["stats"]
    assert greedy.wins >= 18


def test_greedy_avoids_final_round_ticket_draws():
    for seed in range(10):
        game = Game(load_board("usa"), seed=seed)
        play_game(game, [GreedyAgent(seed), GreedyAgent(seed + 1)])
        final = False
        for e in game.log:
            if e.kind == "final_round_triggered":
                final = True
            assert not (final and e.kind == "draw_tickets")


def test_render_does_not_crash():
    game = Game(load_board("usa"), seed=0)
    result = play_game(game, [GreedyAgent(0), GreedyAgent(1)])
    console = Console(file=io.StringIO(), width=120)
    console.print(render_game(game))
    console.print(render_game(game, reveal=False))
    console.print(render_routes(game))
    console.print(render_result(result, names=["a", "b"]))
    assert "Final scores" in console.file.getvalue()


@pytest.mark.parametrize("board,num_players", [("usa", 2), ("usa", 5), ("toy", 3)])
@pytest.mark.parametrize("width", [100, 160, 240])
def test_board_view_renders(board, num_players, width):
    game = Game(load_board(board), num_players=num_players, seed=1)
    play_game(game, [GreedyAgent(i) for i in range(num_players)])
    lines = [t for t in (describe_event(game, e) for e in game.log) if t]
    assert any("claimed" in t for t in lines)
    console = Console(file=io.StringIO(), width=width)
    console.print(render_board_view(game, names=["g"] * num_players, recent=lines[-3:], width=width))
    out = console.file.getvalue()
    assert "Face up" in out and "P0" in out


def test_ascii_map_marks_claimed_routes_with_seat_number():
    board = load_board("usa")
    rid = next(r.id for r in board.routes if {r.a, r.b} == {"Seattle", "Helena"})
    plain = render_map(board, {rid: 1}).plain
    assert "SEA" in plain and "HEL" in plain and "1" in plain
    assert "0" not in plain
