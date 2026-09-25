"""Headless checks of the composed viewer: board plus panels (SDL dummy driver)."""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from ttr.viz import theme  # noqa: E402
from ttr.viz.perspective import Perspective  # noqa: E402
from ttr.viz.screen import BOTTOM_H, SIDE_W, TICKER_H, TOP_H, Screen, seats_in_play  # noqa: E402

from helpers import started_game  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def pg():
    pygame.init()
    yield
    pygame.quit()


def test_size_is_board_plus_panels():
    game = started_game()
    screen = Screen(game.board)
    bw, bh = screen.board_view.size
    assert screen.size == (bw + 2 * SIDE_W, bh + TOP_H + TICKER_H + BOTTOM_H)
    assert screen.board_origin == (SIDE_W, TOP_H)


@pytest.mark.parametrize("scale", [0.6, 1.0, 2.0])
def test_scales(scale):
    game = started_game()
    screen = Screen(game.board, scale=scale)
    surface, _ = screen.render(game)
    assert surface.get_size() == screen.size
    bw, bh = screen.board_view.size
    panels_h = round((TOP_H + TICKER_H + BOTTOM_H) * scale)
    assert surface.get_size() == (bw + round(2 * SIDE_W * scale), bh + panels_h)


@pytest.mark.parametrize("num_players", [2, 3, 4, 5])
def test_seat_slots_per_player_count(num_players):
    game = started_game(num_players=num_players)
    screen = Screen(game.board)
    rects = screen.seat_rects(num_players)
    # Every seat gets a slot; seats_in_play is the same set in layout order.
    assert sorted(rects) == sorted(seats_in_play(num_players)) == list(range(num_players))
    board = pygame.Rect(screen.board_origin, screen.board_view.size)
    for seat, rect in rects.items():
        assert screen.size[0] >= rect.right and screen.size[1] >= rect.bottom
        if seat:  # side panels never cover the board
            assert not rect.colliderect(board)
    assert rects[0].width > rects[1].width  # seat 0 is the full-width panel


@pytest.mark.parametrize("viewer", [None, 0, 1])
def test_renders_in_every_perspective(viewer):
    game = started_game(num_players=3, seed=2)
    screen = Screen(game.board, scale=0.8, perspective=Perspective(viewer))
    surface, vm = screen.render(game)
    assert vm.viewer == viewer
    assert surface.get_size() == screen.size
    # The panel background is actually painted where a seat slot sits.
    slot = screen.seat_rects(3)[1]
    assert surface.get_at((slot.centerx, slot.centery))[:3] != theme.PANEL_BG


def test_cycle_perspective():
    game = started_game(num_players=3)
    screen = Screen(game.board, scale=0.5)
    screen.render(game)  # teaches the screen how many seats are in play
    assert [screen.cycle_perspective() for _ in range(4)] == [0, 1, 2, None]


def test_board_point_maps_window_pixels_to_canvas():
    game = started_game()
    screen = Screen(game.board, scale=1.5)
    ox, oy = screen.board_origin
    city = "Denver"
    x, y = screen.board_view.to_screen(screen.board_view.layout.cities[city])
    assert screen.board_point((ox + x, oy + y)) == pytest.approx(
        screen.board_view.layout.cities[city], abs=0.01
    )
    assert screen.board_view.layout.city_at(screen.board_point((ox + x, oy + y))) == city
    assert screen.board_point((0, 0)) is None  # in the side panel, not the board
    assert screen.board_point((ox + screen.board_view.size[0] + 1, oy)) is None


def test_game_over_state_renders():
    game = started_game(seed=11, max_turns=6)
    while not game.game_over:
        game.step(game.legal_actions()[0])
    screen = Screen(game.board, scale=0.5)
    _, vm = screen.render(game)
    assert vm.table.game_over and vm.table.winners


def test_screenshot_cli_full_screen(tmp_path):
    from ttr.viz.screenshot import main

    out = tmp_path / "screen.png"
    main(["--out", str(out), "--scale", "0.5", "--perspective", "0", "--turns", "4"])
    assert out.exists() and out.stat().st_size > 10_000


def test_screenshot_cli_board_only(tmp_path):
    from ttr.viz.screenshot import main

    out = tmp_path / "board.png"
    main(["--out", str(out), "--scale", "0.5", "--board-only", "--turns", "0"])
    assert out.exists() and out.stat().st_size > 10_000
