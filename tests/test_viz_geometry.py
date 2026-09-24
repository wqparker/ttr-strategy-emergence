import json
import math
from importlib import resources

import pytest

from ttr.board import load_board
from ttr.cards import Color
from ttr.viz.geometry import TRACK_CELLS, load_layout, track_cells

from helpers import route_id, started_game


@pytest.fixture(scope="module")
def usa():
    board = load_board("usa")
    return board, load_layout(board)


def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def test_usa_uses_display_file(usa):
    _, layout = usa
    assert layout.source == "display file"
    assert layout.canvas == (1151, 764)


def test_every_route_has_one_car_per_space(usa):
    board, layout = usa
    assert set(layout.routes) == {r.id for r in board.routes}
    for r in board.routes:
        assert len(layout.routes[r.id].cars) == r.length


def test_cars_stay_clear_of_their_cities(usa):
    board, layout = usa
    for r in board.routes:
        for car in layout.routes[r.id].cars:
            for city in (r.a, r.b):
                assert dist(car.center, layout.cities[city]) > layout.city_radius + car.length / 2 - 1


def test_cars_are_reasonably_sized(usa):
    board, layout = usa
    for r in board.routes:
        for car in layout.routes[r.id].cars:
            assert 15 < car.length < 60, (r, car.length)


def test_cities_inside_map_area(usa):
    board, layout = usa
    x, y, w, h = layout.inner
    for city in board.cities:
        cx, cy = layout.cities[city]
        assert x < cx < x + w and y < cy < y + h


def test_double_routes_sit_side_by_side(usa):
    board, layout = usa
    for r in board.routes:
        if r.sibling is None or r.id > r.sibling:
            continue
        mine, theirs = layout.routes[r.id].cars, layout.routes[r.sibling].cars
        for a, b in zip(mine, theirs):
            assert 10 < dist(a.center, b.center) < 20  # two lanes, 2 x 7.5 apart


def test_double_route_colors_on_the_board_sides(usa):
    board, layout = usa
    game = started_game()
    # West coast: green outside (west) of purple; Boston-New York: yellow west of red.
    green = layout.routes[route_id(game, "Portland", "San Francisco", "green")].midpoint
    purple = layout.routes[route_id(game, "Portland", "San Francisco", "purple")].midpoint
    assert green[0] < purple[0]
    yellow = layout.routes[route_id(game, "Boston", "New York", "yellow")].midpoint
    red = layout.routes[route_id(game, "Boston", "New York", "red")].midpoint
    assert yellow[0] < red[0]
    # Salt Lake City-Denver: red on top (north) of yellow.
    red = layout.routes[route_id(game, "Salt Lake City", "Denver", "red")].midpoint
    yellow = layout.routes[route_id(game, "Salt Lake City", "Denver", "yellow")].midpoint
    assert red[1] < yellow[1]


def test_display_data_matches_board():
    board = load_board("usa")
    d = json.loads(resources.files("ttr").joinpath("data", "usa_display.json").read_text(encoding="utf-8"))
    assert set(d["cities"]) == set(board.cities)
    pairs = {}
    for r in board.routes:
        pairs.setdefault(frozenset((r.a, r.b)), []).append(r)
    seen = set()
    for entry in d["routes"]:
        key = frozenset((entry["a"], entry["b"]))
        assert key in pairs, f"no route {entry['a']}-{entry['b']}"
        assert key not in seen, f"duplicate entry {entry['a']}-{entry['b']}"
        seen.add(key)
        colors = {r.color.value if r.color else "gray" for r in pairs[key]}
        for color, side in entry.get("sides", {}).items():
            assert color in colors and side in (1, -1)


def test_score_track_goes_around_the_border():
    canvas = (1151, 764)
    cells = track_cells(canvas, 28)
    assert len(cells) == TRACK_CELLS
    x, y, w, h = cells[0]
    assert x == 0 and y + h == canvas[1]  # bottom-left
    assert cells[20][:2] == (0, 0)  # top-left
    assert cells[50][0] + cells[50][2] == canvas[0] and cells[50][1] == 0  # top-right
    assert cells[70][0] + cells[70][2] == canvas[0]  # bottom-right
    assert cells[1][1] < cells[0][1] and cells[19][1] > cells[20][1]  # up the left side
    assert cells[21][0] < cells[49][0]  # left to right along the top
    assert cells[71][0] > cells[99][0]  # right to left along the bottom
    for cx, cy, cw, ch in cells:
        assert cx >= -1e-6 and cy >= -1e-6 and cx + cw <= canvas[0] + 1e-6 and cy + ch <= canvas[1] + 1e-6


def test_hit_testing(usa):
    board, layout = usa
    game = started_game()
    rid = route_id(game, "Seattle", "Helena")
    assert layout.route_at(layout.routes[rid].cars[2].center) == rid
    assert layout.city_at(layout.cities["Denver"]) == "Denver"
    assert layout.route_at(layout.cities["Denver"]) is None


def test_toy_board_gets_automatic_layout():
    board = load_board("toy")
    layout = load_layout(board)
    assert layout.source == "auto"
    assert set(layout.cities) == set(board.cities)
    for r in board.routes:
        assert len(layout.routes[r.id].cars) == r.length
