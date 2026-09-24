"""Headless rendering checks (SDL dummy driver). Skipped without pygame."""

import math
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from ttr.actions import ClaimRoute, Pay  # noqa: E402
from ttr.cards import Color  # noqa: E402
from ttr.viz import theme  # noqa: E402
from ttr.viz.board_view import BoardView  # noqa: E402

from helpers import route_id, set_hand, started_game  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def pg():
    pygame.init()
    yield
    pygame.quit()


def pixel_near(surface, p, color, tol=40):
    got = surface.get_at((round(p[0]), round(p[1])))[:3]
    return all(abs(a - b) <= tol for a, b in zip(got, color)), got


def body_point(view, car):
    """A point on the car body: off a claimed train's center stripe (±1 px) and
    inside an unclaimed car's inner highlight line (3.5 px out)."""
    dx, dy = -math.sin(car.angle) * 2.2, math.cos(car.angle) * 2.2
    return view.to_screen((car.center[0] + dx, car.center[1] + dy))


def test_unclaimed_and_claimed_routes_drawn():
    game = started_game(seed=4)
    rid = route_id(game, "Seattle", "Helena")  # yellow, 6
    view = BoardView(game.board)
    surface = pygame.Surface(view.size)
    view.draw(surface, game)
    car = view.layout.routes[rid].cars[2]
    ok, got = pixel_near(surface, body_point(view, car), theme.CARD[Color.YELLOW])
    assert ok, got

    set_hand(game, 0, yellow=6)
    game.step(ClaimRoute(rid))
    game.step(Pay(Color.YELLOW, 0))
    view.draw(surface, game)
    ok, got = pixel_near(surface, body_point(view, car), theme.PLAYER[0])
    assert ok, got


@pytest.mark.parametrize("scale", [0.6, 1.0, 1.5])
def test_scales(scale):
    game = started_game()
    view = BoardView(game.board, scale=scale)
    surface = pygame.Surface(view.size)
    view.draw(surface, game, highlight_routes=[0], highlight_cities=["Denver"])
    w, h = view.layout.canvas
    assert surface.get_size() == (round(w * scale), round(h * scale))


def test_toy_board_renders():
    game = started_game(board="toy")
    view = BoardView(game.board)
    surface = pygame.Surface(view.size)
    view.draw(surface, game)


def test_screenshot_cli(tmp_path):
    from ttr.viz.screenshot import main

    out = tmp_path / "board.png"
    main(["--out", str(out), "--scale", "0.5"])
    assert out.exists() and out.stat().st_size > 10_000
