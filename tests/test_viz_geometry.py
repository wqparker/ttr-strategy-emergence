import json
import math
from importlib import resources

import pytest

from ttr.board import load_board
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


def test_no_car_covers_a_city(usa):
    board, layout = usa
    for shape in layout.routes.values():
        for car in shape.cars:
            for city, (cx, cy) in layout.cities.items():
                assert not car.contains((cx, cy)), (board.routes[shape.route_id], city)
                assert all(dist(c, (cx, cy)) > layout.city_radius - 1 for c in car.corners())


def test_every_car_is_the_same_size(usa):
    _, layout = usa
    sizes = {(car.length, car.width) for shape in layout.routes.values() for car in shape.cars}
    assert len(sizes) == 1


def test_cars_run_from_city_a_to_city_b(usa):
    board, layout = usa
    for r in board.routes:
        cars = layout.routes[r.id].cars
        a, b = layout.cities[r.a], layout.cities[r.b]
        assert dist(cars[0].center, a) <= dist(cars[-1].center, a)
        assert dist(cars[-1].center, b) <= dist(cars[0].center, b)
        # Consecutive cars are about one pitch (~40 px) apart. On the board the outer
        # lane of a curved double route is a little wider, and Raleigh-Charleston
        # turns a sharp corner between its two cars (centers ~30 px apart).
        for p, q in zip(cars, cars[1:]):
            assert 28 < dist(p.center, q.center) < 49, r


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
            assert 10 < dist(a.center, b.center) < 20, r  # two lanes about 14 px apart


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
    assert len(d["tiles"]) == len(board.routes)
    for r, entry in zip(board.routes, d["tiles"]):
        assert (entry["a"], entry["b"]) == (r.a, r.b)
        assert entry["color"] == (r.color.value if r.color else "gray")
        assert len(entry["tiles"]) == r.length


def test_mismatched_display_data_rejected():
    from ttr.viz.geometry import layout_from_display

    board = load_board("usa")
    d = json.loads(resources.files("ttr").joinpath("data", "usa_display.json").read_text(encoding="utf-8"))
    d["tiles"][0]["tiles"].pop()
    with pytest.raises(ValueError):
        layout_from_display(board, d)


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
