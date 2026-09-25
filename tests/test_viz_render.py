"""Headless rendering checks (SDL dummy driver). Skipped without pygame."""

import math
from collections import Counter

import pytest

pygame = pytest.importorskip("pygame")

from ttr.actions import ClaimRoute, Pay  # noqa: E402
from ttr.cards import Color  # noqa: E402
from ttr.viz import theme  # noqa: E402
from ttr.viz.board_view import BoardView  # noqa: E402

from helpers import route_id, set_hand, started_game  # noqa: E402


def close(a, b, tol=40):
    """Colors within tolerance: everything here is drawn through a downscale."""
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def pixel_near(surface, p, color, tol=40):
    got = surface.get_at((round(p[0]), round(p[1])))[:3]
    return close(got, color, tol), got


def body_point(view, car):
    """A point on the car body, off the center line (an unclaimed car's inner
    highlight sits 3.5 px out, so stay inside that)."""
    dx, dy = -math.sin(car.angle) * 2.2, math.cos(car.angle) * 2.2
    return view.to_screen((car.center[0] + dx, car.center[1] + dy))


def car_colors(surface, view, car, inset=3.0):
    """Colors sampled on a grid inside the car. A claimed car is patterned, so
    single points are unreliable; counts are not."""
    ux, uy = math.cos(car.angle), math.sin(car.angle)
    hl, hw = car.length / 2 - inset, car.width / 2 - inset
    seen = Counter()
    for i in range(21):
        for j in range(7):
            a = -hl + 2 * hl * i / 20
            c = -hw + 2 * hw * j / 6
            x, y = view.to_screen((car.center[0] + a * ux - c * uy, car.center[1] + a * uy + c * ux))
            seen[surface.get_at((round(x), round(y)))[:3]] += 1
    return seen


def test_unclaimed_and_claimed_routes_drawn():
    game = started_game(seed=4)
    rid = route_id(game, "Seattle", "Helena")  # yellow, 6
    view = BoardView(game.board)
    surface = view.render(game)
    car = view.layout.routes[rid].cars[2]
    ok, got = pixel_near(surface, body_point(view, car), theme.CARD[Color.YELLOW])
    assert ok, got

    set_hand(game, 0, yellow=6)
    game.step(ClaimRoute(rid))
    game.step(Pay(Color.YELLOW, 0))
    view.draw(surface, game)
    seen = car_colors(surface, view, car)
    assert close(seen.most_common(1)[0][0], theme.PLAYER[0]), seen.most_common(3)


@pytest.mark.parametrize("pattern", ["plain", "track", "bars", "diagonal", "cross"])
def test_claimed_cars_are_patterned(pattern, monkeypatch):
    """A claimed space carries a contrasting pattern, so it never reads as an
    unclaimed tile of the same color (theme.TRAIN_PATTERN picks the shape)."""
    monkeypatch.setattr(theme, "TRAIN_PATTERN", pattern)
    game = started_game(seed=4)
    rid = route_id(game, "Seattle", "Helena")
    set_hand(game, 0, yellow=6)
    game.step(ClaimRoute(rid))
    game.step(Pay(Color.YELLOW, 0))
    view = BoardView(game.board, scale=2.0)
    surface = view.render(game)

    seen = car_colors(surface, view, view.layout.routes[rid].cars[2])
    total = sum(seen.values())
    seat = theme.PLAYER[0]
    ink = theme.chip_text(seat) if pattern != "plain" else theme.lighter(seat, 0.55)
    # Both must show: the seat color says whose it is, the ink says it is claimed.
    # Shares, not the top color: "cross" covers about half the car.
    assert sum(n for c, n in seen.items() if close(c, seat)) / total > 0.3, seen.most_common(3)
    assert sum(n for c, n in seen.items() if close(c, ink, tol=30)) / total > 0.1, seen.most_common(3)


def test_unknown_pattern_is_rejected(monkeypatch):
    monkeypatch.setattr(theme, "TRAIN_PATTERN", "spots")
    game = started_game(seed=4)
    set_hand(game, 0, yellow=6)
    game.step(ClaimRoute(route_id(game, "Seattle", "Helena")))
    game.step(Pay(Color.YELLOW, 0))
    view = BoardView(game.board)
    with pytest.raises(ValueError, match="spots"):
        view.render(game)


@pytest.mark.parametrize("scale", [0.6, 1.0, 1.5])
def test_scales(scale):
    game = started_game()
    view = BoardView(game.board, scale=scale)
    surface = view.render(game, highlight_routes=[0], highlight_cities=["Denver"])
    w, h = view.layout.canvas
    assert surface.get_size() == (round(w * scale), round(h * scale))


def test_toy_board_renders():
    game = started_game(board="toy")
    view = BoardView(game.board)
    surface = view.render(game)


def test_screenshot_cli(tmp_path):
    from ttr.viz.screenshot import main

    out = tmp_path / "board.png"
    main(["--out", str(out), "--scale", "0.5"])
    assert out.exists() and out.stat().st_size > 10_000


def test_fonts_survive_a_pygame_restart():
    """Fonts cached before a pygame.quit() are dead; theme.font must notice."""
    before = theme.font(12, bold=True)
    assert before.render("x", True, theme.LABEL)
    pygame.quit()
    pygame.init()
    after = theme.font(12, bold=True)
    assert after is not before
    assert after.render("x", True, theme.LABEL)
