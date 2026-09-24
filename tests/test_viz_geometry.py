import json
import math
from importlib import resources

import pytest

from ttr.board import load_board
from ttr.cards import Color
from ttr.viz.geometry import load_layout

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


def test_no_cars_of_different_routes_overlap(usa):
    board, layout = usa
    ids = sorted(layout.routes)
    bad = []
    for i, r in enumerate(ids):
        for q in ids[i + 1:]:
            if any(a.overlaps(b) for a in layout.routes[r].cars for b in layout.routes[q].cars):
                bad.append((board.routes[r].a, board.routes[r].b, board.routes[q].a, board.routes[q].b))
    assert not bad


def test_cars_within_a_route_do_not_overlap(usa):
    _, layout = usa
    for shape in layout.routes.values():
        for a, b in zip(shape.cars, shape.cars[1:]):
            assert not a.overlaps(b, clearance=-1.0)  # neighbors may touch, not overlap


def test_measured_car_counts_match_route_lengths():
    board = load_board("usa")
    d = json.loads(resources.files("ttr").joinpath("data", "usa_display.json").read_text(encoding="utf-8"))
    lengths = {frozenset((r.a, r.b)): r.length for r in board.routes}
    for entry in d["routes"]:
        if "cars" in entry:
            assert len(entry["cars"]) == lengths[frozenset((entry["a"], entry["b"]))], entry


def test_backdrop_loaded_and_inside_canvas(usa):
    _, layout = usa
    b = layout.backdrop
    assert b is not None and b.land and b.states and b.lakes
    w, h = layout.canvas
    # The warp pins cities, so each city is on (or, for coastal cities like
    # Miami, within a few pixels of) drawn land.
    for city, (x, y) in layout.cities.items():
        near = [(x + dx, y + dy) for dx in (-6, 0, 6) for dy in (-6, 0, 6)]
        assert any(_inside(p, ring) for p in near for ring in b.land), city
    # The five Great Lakes: large lakes between Duluth and Montreal.
    great = [
        ring for ring in b.lakes
        if _area(ring) > 1000 and all(640 <= x <= 980 and 110 <= y <= 320 for x, y in ring)
    ]
    assert len(great) == 5


def _area(ring):
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]))) / 2


def _inside(p, ring):
    x, y = p
    inside = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


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
