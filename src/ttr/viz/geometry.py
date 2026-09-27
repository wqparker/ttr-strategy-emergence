"""Board geometry for drawing: city positions, train-car rectangles along every
route, the map backdrop. Pure math, no Pygame.

Coordinates are "canvas" pixels (x right, y down). The USA board uses
`data/usa_display.json`, measured from a photo of the physical board, and
`data/usa_backdrop.json`, state and country borders warped onto the board (built
by `scripts/build_backdrop.py`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import resources
from typing import Dict, List, Optional, Sequence, Tuple

from ttr.board import Board

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]  # x, y, w, h

DEFAULT_CANVAS = (1151, 764)


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
    """Layout from `data/<board>_display.json`."""
    text = _data(f"{board.name}_display.json")
    if text is None:
        raise ValueError(f"board {board.name!r} has no display data (data/{board.name}_display.json)")
    layout = layout_from_display(board, json.loads(text))
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
        border=float(d.get("border", 22)), table=table, label_text=label_text,
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


