"""Board geometry for drawing: city positions, train-car rectangles along every
route, the score track around the border. Pure math, no Pygame.

Coordinates are "canvas" pixels (x right, y down). The USA board uses
`data/usa_display.json`, measured from a photo of the physical board. A board
without a display file (e.g. the toy map) gets an automatic layout from the
`layout` block in its board data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import resources
from typing import Dict, List, Optional, Sequence, Tuple

from ttr.board import Board, Route

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]  # x, y, w, h

TRACK_CELLS = 100
DEFAULT_CANVAS = (1151, 764)
SAMPLES = 64  # points per route centerline


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
class BoardLayout:
    canvas: Tuple[int, int]
    cities: Dict[str, Point]
    labels: Dict[str, Point]  # label center, absolute
    city_radius: float
    routes: Dict[int, RouteShape]
    track: List[Rect]  # index = score mod 100
    band: float  # score-track thickness
    table: Optional[Rect] = None  # route-points table
    source: str = "auto"  # "display file" or "auto"
    label_text: Dict[str, str] = field(default_factory=dict)  # "\n" splits lines

    def label(self, city: str) -> str:
        return self.label_text.get(city, city)

    @property
    def inner(self) -> Rect:
        w, h = self.canvas
        return (self.band, self.band, w - 2 * self.band, h - 2 * self.band)

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


def load_layout(board: Board) -> BoardLayout:
    """Layout from `data/<board>_display.json` if it exists, else automatic."""
    try:
        text = resources.files("ttr").joinpath("data", f"{board.name}_display.json").read_text(
            encoding="utf-8"
        )
    except FileNotFoundError:
        return auto_layout(board)
    return layout_from_display(board, json.loads(text))


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
    band = float(d.get("track", {}).get("band", 28))
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
        track=track_cells(canvas, band), band=band, table=table, source="display file",
        label_text=label_text,
    )


def auto_layout(board: Board) -> BoardLayout:
    """Fit the board's `layout` positions (y grows northward) into the canvas."""
    if not board.layout:
        raise ValueError(f"board {board.name!r} has neither display data nor a layout")
    canvas = DEFAULT_CANVAS
    band = 28.0
    margin = band + 70
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
        canvas=canvas, cities=cities, labels=labels, city_radius=10, routes=routes,
        track=track_cells(canvas, band), band=band,
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
    by_pair = {frozenset((e["a"], e["b"])): e for e in entries}
    shapes = {}
    for route in board.routes:
        entry = by_pair.get(frozenset((route.a, route.b)), {})
        a, b = (entry["a"], entry["b"]) if entry else (route.a, route.b)
        offset = _lane_side(board, route, entry) * lane
        via = entry.get("via")
        if via and isinstance(via[0], (list, tuple)):
            path = _spline([cities[a], *map(tuple, via), cities[b]], offset)
        else:
            path = _curve(cities[a], cities[b], tuple(via) if via else None, offset)
        shapes[route.id] = RouteShape(
            route_id=route.id,
            cars=tuple(_cars(path, route.length, radius, car_width, gap)),
            path=tuple(path),
        )
    return shapes


def _lane_side(board: Board, route: Route, entry: dict) -> int:
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
    raw: List[Tuple[Point, Point]] = []  # (point, tangent)
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        last = i == len(ext) - 3
        for k in range(per + (1 if last else 0)):
            t = k / per
            t2, t3 = t * t, t * t * t
            x = 0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            dx = 0.5 * ((-p0[0] + p2[0]) + 2 * (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t
                        + 3 * (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t2)
            dy = 0.5 * ((-p0[1] + p2[1]) + 2 * (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t
                        + 3 * (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t2)
            raw.append(((x, y), (dx, dy)))
    pts = []
    for (x, y), (dx, dy) in raw:
        n = math.hypot(dx, dy) or 1.0
        pts.append((x - dy / n * offset, y + dx / n * offset))
    return pts


def _cars(path: List[Point], n: int, radius: float, width: float, gap: float) -> List[Car]:
    """Fill the path between the two city circles with n equal cars."""
    cum = [0.0]
    for p, q in zip(path, path[1:]):
        cum.append(cum[-1] + math.hypot(q[0] - p[0], q[1] - p[1]))
    total = cum[-1]
    margin = radius + 3
    usable = max(total - 2 * margin, n * 4.0)
    length = (usable - (n - 1) * gap) / n
    start = (total - usable) / 2
    cars = []
    for i in range(n):
        s = start + i * (length + gap) + length / 2
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


# ------------------------------------------------------------ score track


def track_cells(canvas: Tuple[int, int], band: float) -> List[Rect]:
    """The 100 score cells around the border, as on the physical board: 0 in the
    bottom-left corner, 1-19 up the left side, 20 top-left, 21-49 along the top,
    50 top-right, 51-69 down the right side, 70 bottom-right, 71-99 along the
    bottom back toward 0."""
    w, h = canvas
    cw = (w - 2 * band) / 29
    ch = (h - 2 * band) / 19
    cells: List[Rect] = [None] * TRACK_CELLS  # type: ignore[list-item]
    cells[0] = (0, h - band, band, band)
    for k in range(1, 20):
        cells[k] = (0, h - band - k * ch, band, ch)
    cells[20] = (0, 0, band, band)
    for k in range(21, 50):
        cells[k] = (band + (k - 21) * cw, 0, cw, band)
    cells[50] = (w - band, 0, band, band)
    for k in range(51, 70):
        cells[k] = (w - band, band + (k - 51) * ch, band, ch)
    cells[70] = (w - band, h - band, band, band)
    for k in range(71, 100):
        cells[k] = (w - band - (k - 70) * cw, h - band, cw, band)
    return cells
