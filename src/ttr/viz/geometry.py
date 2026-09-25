"""Board geometry for drawing: city positions, train-car rectangles along every
route, the map backdrop. Pure math, no Pygame.

Coordinates are "canvas" pixels (x right, y down). The USA board uses
`data/usa_display.json`, measured from a photo of the physical board, and
`data/usa_backdrop.json`, state and country borders warped onto the board (built
by `scripts/build_backdrop.py`). A board without a display file (e.g. the toy
map) gets an automatic layout from the `layout` block in its board data.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import resources
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from ttr.board import Board

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]  # x, y, w, h

DEFAULT_CANVAS = (1151, 764)
CAR_CLEARANCE = 1.0  # minimum gap between cars of different routes
SLIDE_STEP = 0.5  # how far a route's cars slide per step to clear a crowded city
MAX_AUTO_CAR_LENGTH = 40.0


@dataclass(frozen=True)
class Car:
    """One train-car space: a rectangle centered on `center`, rotated to `angle`
    (radians, along the route)."""

    center: Point
    angle: float
    length: float
    width: float

    def corners(self, inset: float = 0.0) -> List[Point]:
        hl, hw = self.length / 2 - inset, self.width / 2 - inset
        ux, uy = math.cos(self.angle), math.sin(self.angle)
        vx, vy = -uy, ux
        cx, cy = self.center
        return [
            (cx + sx * hl * ux + sy * hw * vx, cy + sx * hl * uy + sy * hw * vy)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]

    def contains(self, p: Point) -> bool:
        ux, uy = math.cos(self.angle), math.sin(self.angle)
        dx, dy = p[0] - self.center[0], p[1] - self.center[1]
        along = dx * ux + dy * uy
        across = -dx * uy + dy * ux
        return abs(along) <= self.length / 2 and abs(across) <= self.width / 2

    def overlaps(self, other: "Car", clearance: float = 0.0) -> bool:
        """True if the two rectangles overlap or come within `clearance`
        (separating-axis test)."""
        if math.dist(self.center, other.center) > (self.length + other.length) / 2 + self.width + clearance:
            return False
        p, q = self.corners(), other.corners()
        for angle in (self.angle, self.angle + math.pi / 2, other.angle, other.angle + math.pi / 2):
            ax, ay = math.cos(angle), math.sin(angle)
            pp = [x * ax + y * ay for x, y in p]
            qq = [x * ax + y * ay for x, y in q]
            if max(pp) + clearance <= min(qq) or max(qq) + clearance <= min(pp):
                return False
        return True


@dataclass(frozen=True)
class RouteShape:
    route_id: int
    cars: Tuple[Car, ...]
    path: Tuple[Point, ...]  # the lane's centerline, city to city

    def contains(self, p: Point) -> bool:
        return any(car.contains(p) for car in self.cars)

    @property
    def midpoint(self) -> Point:
        return self.path[len(self.path) // 2]


@dataclass
class Backdrop:
    """Map backdrop in canvas coordinates: land to fill, lakes to fill with
    water, and border lines (state/province, and country)."""

    land: List[List[Point]] = field(default_factory=list)
    lakes: List[List[Point]] = field(default_factory=list)
    states: List[List[Point]] = field(default_factory=list)
    countries: List[List[Point]] = field(default_factory=list)


@dataclass
class BoardLayout:
    canvas: Tuple[int, int]
    cities: Dict[str, Point]
    labels: Dict[str, Point]  # label center, absolute
    city_radius: float
    routes: Dict[int, RouteShape]
    border: float  # thickness of the frame around the map
    table: Optional[Rect] = None  # route-points table
    source: str = "auto"  # "display file" or "auto"
    label_text: Dict[str, str] = field(default_factory=dict)  # "\n" splits lines
    backdrop: Optional[Backdrop] = None

    def label(self, city: str) -> str:
        return self.label_text.get(city, city)

    @property
    def inner(self) -> Rect:
        w, h = self.canvas
        return (self.border, self.border, w - 2 * self.border, h - 2 * self.border)

    def route_at(self, p: Point) -> Optional[int]:
        for rid, shape in self.routes.items():
            if shape.contains(p):
                return rid
        return None

    def city_at(self, p: Point, slack: float = 4.0) -> Optional[str]:
        for city, (x, y) in self.cities.items():
            if math.hypot(p[0] - x, p[1] - y) <= self.city_radius + slack:
                return city
        return None


# ------------------------------------------------------------------ loading


def _data(name: str) -> Optional[str]:
    try:
        return resources.files("ttr").joinpath("data", name).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def load_layout(board: Board) -> BoardLayout:
    """Layout from `data/<board>_display.json` if it exists, else automatic."""
    text = _data(f"{board.name}_display.json")
    layout = auto_layout(board) if text is None else layout_from_display(board, json.loads(text))
    backdrop = _data(f"{board.name}_backdrop.json")
    if backdrop is not None:
        layout.backdrop = backdrop_from_dict(json.loads(backdrop))
    return layout


def backdrop_from_dict(d: dict) -> Backdrop:
    def rings(key: str) -> List[List[Point]]:
        return [[(p[0], p[1]) for p in ring] for ring in d.get(key, [])]

    return Backdrop(land=rings("land"), lakes=rings("lakes"), states=rings("states"), countries=rings("countries"))


def layout_from_display(board: Board, d: dict) -> BoardLayout:
    canvas = tuple(d.get("canvas", DEFAULT_CANVAS))
    missing = set(board.cities) - set(d["cities"])
    if missing:
        raise ValueError(f"display data has no position for {sorted(missing)}")
    cities = {c: tuple(v["pos"]) for c, v in d["cities"].items() if c in board.cities}
    labels = {
        c: (cities[c][0] + v.get("label", (0, -16))[0], cities[c][1] + v.get("label", (0, -16))[1])
        for c, v in d["cities"].items()
        if c in board.cities
    }
    radius = float(d.get("city_radius", 8))
    routes = _traced_routes(board, cities, d["tiles"], float(d["car_length"]), float(d["car_width"]))

    t = d.get("table")
    table = (t["x"], t["y"], t["w"], t["h"]) if t else None
    label_text = {c: v["text"] for c, v in d["cities"].items() if "text" in v and c in board.cities}
    return BoardLayout(
        canvas=canvas, cities=cities, labels=labels, city_radius=radius, routes=routes,
        border=float(d.get("border", 22)), table=table, source="display file", label_text=label_text,
    )


def auto_layout(board: Board) -> BoardLayout:
    """Fit the board's `layout` positions (y grows northward) into the canvas."""
    if not board.layout:
        raise ValueError(f"board {board.name!r} has neither display data nor a layout")
    canvas = DEFAULT_CANVAS
    border = 22.0
    margin = border + 70
    xs = [p.x for p in board.layout.values()]
    ys = [p.y for p in board.layout.values()]
    span_x = (max(xs) - min(xs)) or 1
    span_y = (max(ys) - min(ys)) or 1
    s = min((canvas[0] - 2 * margin) / span_x, (canvas[1] - 2 * margin) / span_y)
    ox = (canvas[0] - span_x * s) / 2
    oy = (canvas[1] - span_y * s) / 2
    cities = {
        c: (ox + (p.x - min(xs)) * s, oy + (max(ys) - p.y) * s) for c, p in board.layout.items()
    }
    labels = {c: (x, y - 18) for c, (x, y) in cities.items()}
    routes = _auto_routes(board, cities, radius=10, car_width=14, gap=3, lane=8)
    return BoardLayout(
        canvas=canvas, cities=cities, labels=labels, city_radius=10, routes=routes, border=border,
    )


# ------------------------------------------------------------ route shapes


def _traced_routes(
    board: Board, cities: Dict[str, Point], tiles: Sequence[dict], length: float, width: float
) -> Dict[int, RouteShape]:
    """Routes from tiles traced off the board photo: one entry per board route, in
    board order, each tile [x, y, angle in degrees], ordered from city a to b.
    Every tile has the same size, as on the printed board."""
    if len(tiles) != len(board.routes):
        raise ValueError(f"display data has {len(tiles)} traced routes, board has {len(board.routes)}")
    shapes = {}
    for route, entry in zip(board.routes, tiles):
        color = route.color.value if route.color else "gray"
        if (entry["a"], entry["b"], entry["color"]) != (route.a, route.b, color):
            raise ValueError(f"traced route {entry['a']}-{entry['b']} doesn't match board route {route}")
        if len(entry["tiles"]) != route.length:
            raise ValueError(f"{route.a}-{route.b}: {len(entry['tiles'])} tiles for length {route.length}")
        cars = tuple(Car((x, y), math.radians(a), length, width) for x, y, a in entry["tiles"])
        path = (cities[route.a], *(c.center for c in cars), cities[route.b])
        shapes[route.id] = RouteShape(route_id=route.id, cars=cars, path=path)
    return shapes


def _auto_routes(
    board: Board, cities: Dict[str, Point], radius: float, car_width: float, gap: float, lane: float
) -> Dict[int, RouteShape]:
    """Straight routes for boards without traced display data (the toy map): every
    car the same size (the longest that fits every route), each route's block of
    cars centered between its cities, double routes side by side. Where end cars
    collide at a city, the blocks slide away from it within their spare length."""
    paths: Dict[int, List[Point]] = {}
    for route in board.routes:
        side = 0 if route.sibling is None else (-1 if route.id < route.sibling else 1)
        paths[route.id] = _line(cities[route.a], cities[route.b], side * lane)
    cums = {rid: _cumulative(path) for rid, path in paths.items()}
    margin = radius + 2  # path kept clear at each city
    car_length = min(
        MAX_AUTO_CAR_LENGTH,
        min((cums[r.id][-1] - 2 * margin - (r.length - 1) * gap) / r.length for r in board.routes),
    )
    spare = {r.id: cums[r.id][-1] - 2 * margin - (r.length * car_length + (r.length - 1) * gap) for r in board.routes}
    offset = {rid: spare[rid] / 2 for rid in paths}

    def build(rid: int) -> List[Car]:
        start = margin + offset[rid]
        return [
            _car_at(paths[rid], cums[rid], start + i * (car_length + gap), car_length, car_width)
            for i in range(board.routes[rid].length)
        ]

    by_city: Dict[str, List[int]] = defaultdict(list)
    for r in board.routes:
        by_city[r.a].append(r.id)
        by_city[r.b].append(r.id)

    cars = {rid: build(rid) for rid in paths}
    for _ in range(100):
        moves: Dict[int, float] = {}
        for city, rids in by_city.items():
            for r, q in combinations(rids, 2):
                near_r = cars[r][0] if board.routes[r].a == city else cars[r][-1]
                near_q = cars[q][0] if board.routes[q].a == city else cars[q][-1]
                if near_r.overlaps(near_q, CAR_CLEARANCE):
                    for rid in (r, q):  # slide away from this city
                        step = SLIDE_STEP if board.routes[rid].a == city else -SLIDE_STEP
                        moves[rid] = moves.get(rid, 0.0) + step
        moved = False
        for rid, step in moves.items():
            new = min(max(offset[rid] + step, 0.0), max(spare[rid], 0.0))
            if abs(new - offset[rid]) > 1e-9:
                offset[rid] = new
                cars[rid] = build(rid)
                moved = True
        if not moved:
            break

    return {rid: RouteShape(route_id=rid, cars=tuple(cars[rid]), path=tuple(paths[rid])) for rid in paths}


def _line(a: Point, b: Point, offset: float, samples: int = 16) -> List[Point]:
    """Straight path from a to b, shifted sideways by `offset` (y down)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    ox, oy = -dy / n * offset, dx / n * offset
    return [(a[0] + dx * t / samples + ox, a[1] + dy * t / samples + oy) for t in range(samples + 1)]

def _cumulative(path: List[Point]) -> List[float]:
    cum = [0.0]
    for p, q in zip(path, path[1:]):
        cum.append(cum[-1] + math.hypot(q[0] - p[0], q[1] - p[1]))
    return cum


def _car_at(path: List[Point], cum: List[float], s: float, length: float, width: float) -> Car:
    """The car whose back end is at arc length s: its ends sit on the path, so it
    follows curves like a chord."""
    back, _ = _at(path, cum, s)
    front, _ = _at(path, cum, s + length)
    center = ((back[0] + front[0]) / 2, (back[1] + front[1]) / 2)
    angle = math.atan2(front[1] - back[1], front[0] - back[0])
    return Car(center, angle, length, width)


def _at(path: List[Point], cum: List[float], s: float) -> Tuple[Point, float]:
    s = min(max(s, 0.0), cum[-1])
    lo, hi = 0, len(cum) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cum[mid] <= s:
            lo = mid
        else:
            hi = mid
    p, q = path[lo], path[hi]
    seg = (cum[hi] - cum[lo]) or 1.0
    f = (s - cum[lo]) / seg
    point = (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)
    return point, math.atan2(q[1] - p[1], q[0] - p[0])
