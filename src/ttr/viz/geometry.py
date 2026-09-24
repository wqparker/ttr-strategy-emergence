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

from ttr.board import Board, Route

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]  # x, y, w, h

DEFAULT_CANVAS = (1151, 764)
SAMPLES = 64  # points per route centerline
CAR_CLEARANCE = 1.0  # minimum gap between cars of different routes
MIN_CAR_LENGTH = 16.0  # end-trimming never shrinks cars below this
TRIM_STEP = 1.0
MAX_EXTRA_TRIM = 12.0  # trimming only resolves crowding at cities; overlaps
# further out mean a route's shape needs fixing in the display data


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
    routes = _route_shapes(
        board,
        cities,
        entries=d.get("routes", []),
        radius=radius,
        car_width=float(d.get("car_width", 13)),
        gap=float(d.get("car_gap", 3)),
        lane=float(d.get("lane_offset", 7.5)),
    )
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
    routes = _route_shapes(board, cities, entries=[], radius=10, car_width=14, gap=3, lane=8)
    return BoardLayout(
        canvas=canvas, cities=cities, labels=labels, city_radius=10, routes=routes, border=border,
    )


# ------------------------------------------------------------ route shapes


def _route_shapes(
    board: Board,
    cities: Dict[str, Point],
    entries: Sequence[dict],
    radius: float,
    car_width: float,
    gap: float,
    lane: float,
) -> Dict[int, RouteShape]:
    """Build each route's lane path, then fill it with cars. Where routes meet at a
    city, their end cars can collide; those ends are pulled back from the city a
    step at a time until every pair clears (as the printed board does)."""
    by_pair = {frozenset((e["a"], e["b"])): e for e in entries}
    paths: Dict[int, List[Point]] = {}
    start_city: Dict[int, str] = {}  # which city the path starts at
    fixed: Dict[int, List[Car]] = {}  # routes with measured car positions
    for route in board.routes:
        entry = by_pair.get(frozenset((route.a, route.b)), {})
        a, b = (entry["a"], entry["b"]) if entry else (route.a, route.b)
        offset = _lane_side(route, entry) * lane
        via = entry.get("via")
        start_city[route.id] = a
        if "cars" in entry:
            if len(entry["cars"]) != route.length:
                raise ValueError(f"{a}-{b}: {len(entry['cars'])} car positions for length {route.length}")
            fixed[route.id] = _measured_cars([tuple(c) for c in entry["cars"]], offset, car_width, gap)
            paths[route.id] = [cities[a], *(c.center for c in fixed[route.id]), cities[b]]
        elif via and isinstance(via[0], (list, tuple)):
            paths[route.id] = _spline([cities[a], *map(tuple, via), cities[b]], offset)
        else:
            paths[route.id] = _curve(cities[a], cities[b], tuple(via) if via else None, offset)

    base = radius + 3
    trims = {r.id: [base, base] for r in board.routes}  # [start, end]
    for rid, measured in fixed.items():
        # A measured end car never reaches into its city's circle.
        r = board.routes[rid]
        a, b = start_city[rid], (r.b if start_city[rid] == r.a else r.a)
        for e, car, city in ((0, measured[0], a), (1, measured[-1], b)):
            intrude = car.length / 2 - (math.dist(car.center, cities[city]) - base)
            trims[rid][e] = base + max(0.0, math.ceil(intrude))
    by_city: Dict[str, List[int]] = defaultdict(list)
    for r in board.routes:
        by_city[r.a].append(r.id)
        by_city[r.b].append(r.id)

    def end(rid: int, city: str) -> int:
        return 0 if start_city[rid] == city else 1

    def build(rid: int) -> List[Car]:
        if rid in fixed:
            # Measured cars stay put; only the car at a crowded end shrinks back
            # from the city by the extra trim.
            out = list(fixed[rid])
            out[0] = _shrink(out[0], trims[rid][0] - base, forward=True)
            out[-1] = _shrink(out[-1], trims[rid][1] - base, forward=False)
            return out
        return _cars(paths[rid], board.routes[rid].length, trims[rid][0], trims[rid][1], car_width, gap)

    def room(rid: int, e: int) -> bool:
        """Can this end be trimmed one more step?"""
        if trims[rid][e] + TRIM_STEP > base + MAX_EXTRA_TRIM:
            return False
        if rid in fixed:
            car = fixed[rid][0 if e == 0 else -1]
            return car.length - (trims[rid][e] - base) - TRIM_STEP >= MIN_CAR_LENGTH
        return cars[rid][0].length - TRIM_STEP / board.routes[rid].length >= MIN_CAR_LENGTH

    cars = {r.id: build(r.id) for r in board.routes}
    for _ in range(200):
        grow = set()
        for city, rids in by_city.items():
            for r, q in combinations(rids, 2):
                er, eq = end(r, city), end(q, city)
                near_r = cars[r][:2] if er == 0 else cars[r][-2:]
                near_q = cars[q][:2] if eq == 0 else cars[q][-2:]
                if any(a.overlaps(b, CAR_CLEARANCE) for a in near_r for b in near_q):
                    grow.update({(r, er), (q, eq)})
        grown = False
        for rid, e in grow:
            if room(rid, e):
                trims[rid][e] += TRIM_STEP
                grown = True
        if not grown:
            break
        for rid in {rid for rid, _ in grow}:
            cars[rid] = build(rid)

    return {
        rid: RouteShape(route_id=rid, cars=tuple(cars[rid]), path=tuple(paths[rid]))
        for rid in paths
    }


def _lane_side(route: Route, entry: dict) -> int:
    if route.sibling is None:
        return 0
    sides = entry.get("sides", {})
    color = route.color.value if route.color else "gray"
    if color in sides and not route.is_gray:
        return 1 if sides[color] > 0 else -1
    return -1 if route.id < route.sibling else 1


def _curve(a: Point, b: Point, via: Optional[Point], offset: float) -> List[Point]:
    """Quadratic Bezier from a to b passing through `via` at t=0.5, shifted
    sideways by `offset` (positive = right-hand side of a->b, y down)."""
    mx, my = via if via is not None else ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    cx, cy = 2 * mx - (a[0] + b[0]) / 2, 2 * my - (a[1] + b[1]) / 2
    pts = []
    for i in range(SAMPLES + 1):
        t = i / SAMPLES
        u = 1 - t
        x = u * u * a[0] + 2 * u * t * cx + t * t * b[0]
        y = u * u * a[1] + 2 * u * t * cy + t * t * b[1]
        dx = 2 * u * (cx - a[0]) + 2 * t * (b[0] - cx)
        dy = 2 * u * (cy - a[1]) + 2 * t * (b[1] - cy)
        n = math.hypot(dx, dy) or 1.0
        pts.append((x - dy / n * offset, y + dx / n * offset))
    return pts


def _spline(points: List[Point], offset: float) -> List[Point]:
    """Catmull-Rom spline through every point (city, via points..., city),
    shifted sideways by `offset` like `_curve`."""
    ext = [points[0], *points, points[-1]]  # repeat the ends as phantom neighbors
    per = max(4, SAMPLES // (len(points) - 1))
    pts: List[Point] = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        last = i == len(ext) - 3
        for k in range(per + (1 if last else 0)):
            t = k / per
            (x, y), (dx, dy) = _catmull_rom(p0, p1, p2, p3, t)
            n = math.hypot(dx, dy) or 1.0
            pts.append((x - dy / n * offset, y + dx / n * offset))
    return pts


def _catmull_rom(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Tuple[Point, Point]:
    """Point and tangent of a uniform Catmull-Rom segment from p1 to p2."""
    t2, t3 = t * t, t * t * t
    out = []
    for i in (0, 1):
        a = 2 * p1[i]
        b = -p0[i] + p2[i]
        c = 2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]
        d = -p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]
        out.append((0.5 * (a + b * t + c * t2 + d * t3), 0.5 * (b + 2 * c * t + 3 * d * t2)))
    (x, dx), (y, dy) = out
    return (x, y), (dx, dy)


MAX_CAR_LENGTH = 38.0


def _measured_cars(centers: List[Point], offset: float, width: float, gap: float) -> List[Car]:
    """Cars at measured centerline positions (from the board photo). Each car
    points along its neighbors; a double route's lane is offset sideways."""
    cars = []
    for i, (x, y) in enumerate(centers):
        prev = centers[i - 1] if i > 0 else None
        nxt = centers[i + 1] if i + 1 < len(centers) else None
        a, b = prev or (x, y), nxt or (x, y)
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        spacing = min(math.dist((x, y), p) for p in (prev, nxt) if p is not None) if len(centers) > 1 else MAX_CAR_LENGTH + gap
        dx, dy = -math.sin(angle) * offset, math.cos(angle) * offset
        cars.append(Car((x + dx, y + dy), angle, min(MAX_CAR_LENGTH, spacing - gap), width))
    return cars


def _shrink(car: Car, by: float, forward: bool) -> Car:
    """Shorten a car by `by` from its city end. The first car of a route loses its
    back end (it moves forward, along its angle); the last car loses its front."""
    if by <= 0:
        return car
    step = by / 2 if forward else -by / 2
    x = car.center[0] + math.cos(car.angle) * step
    y = car.center[1] + math.sin(car.angle) * step
    return Car((x, y), car.angle, car.length - by, car.width)


def _cars(path: List[Point], n: int, trim_start: float, trim_end: float, width: float, gap: float) -> List[Car]:
    """Fill the path with n equal cars, leaving `trim_start` / `trim_end` of path
    free at each city."""
    cum = [0.0]
    for p, q in zip(path, path[1:]):
        cum.append(cum[-1] + math.hypot(q[0] - p[0], q[1] - p[1]))
    total = cum[-1]
    usable = max(total - trim_start - trim_end, n * 4.0)
    length = (usable - (n - 1) * gap) / n
    cars = []
    for i in range(n):
        s = trim_start + i * (length + gap) + length / 2
        point, angle = _at(path, cum, s)
        cars.append(Car(point, angle, length, width))
    return cars


def _at(path: List[Point], cum: List[float], s: float) -> Tuple[Point, float]:
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
