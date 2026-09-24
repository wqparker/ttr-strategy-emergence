"""Build the viewer's map backdrop (state, province and country borders) for the
USA board from Natural Earth data, warped so each city's real location lands on
its position on the board.

    python scripts/build_backdrop.py NE_DIR [--out src/ttr/data/usa_backdrop.json]

NE_DIR must hold these Natural Earth 1:50m GeoJSON files (public domain, from
github.com/nvkelso/natural-earth-vector, folder geojson/):
    ne_50m_admin_0_countries.geojson
    ne_50m_admin_1_states_provinces_lakes.geojson
    ne_50m_lakes.geojson

The board is a stylized map, not a projection. Real coordinates are projected
(Lambert conformal conic), then a thin-plate spline pinned at the 36 cities maps
them onto the board canvas. Output is display-only data; stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]

# Real (latitude, longitude) of each board city.
CITY_LATLON: Dict[str, Tuple[float, float]] = {
    "Vancouver": (49.28, -123.12), "Seattle": (47.61, -122.33), "Portland": (45.52, -122.68),
    "San Francisco": (37.77, -122.42), "Los Angeles": (34.05, -118.24), "Las Vegas": (36.17, -115.14),
    "Salt Lake City": (40.76, -111.89), "Phoenix": (33.45, -112.07), "El Paso": (31.76, -106.49),
    "Santa Fe": (35.69, -105.94), "Denver": (39.74, -104.99), "Helena": (46.59, -112.04),
    "Calgary": (51.05, -114.07), "Winnipeg": (49.90, -97.14), "Duluth": (46.79, -92.10),
    "Omaha": (41.26, -95.93), "Kansas City": (39.10, -94.58), "Oklahoma City": (35.47, -97.52),
    "Dallas": (32.78, -96.80), "Houston": (29.76, -95.37), "Little Rock": (34.75, -92.29),
    "New Orleans": (29.95, -90.07), "Saint Louis": (38.63, -90.20), "Chicago": (41.88, -87.63),
    "Nashville": (36.16, -86.78), "Atlanta": (33.75, -84.39), "Miami": (25.76, -80.19),
    "Charleston": (32.78, -79.93), "Raleigh": (35.78, -78.64), "Washington": (38.91, -77.04),
    "Pittsburgh": (40.44, -79.99), "New York": (40.71, -74.01), "Boston": (42.36, -71.06),
    "Montreal": (45.50, -73.57), "Toronto": (43.65, -79.38), "Sault St. Marie": (46.52, -84.35),
}

COUNTRIES = {"USA", "CAN", "MEX", "CUB", "BHS"}
STATE_COUNTRIES = {"USA", "CAN"}
BBOX = (-140.0, 18.0, -55.0, 60.0)  # lon/lat clip box, well outside the canvas
SIMPLIFY = 0.45  # canvas px
MIN_LAKE_AREA = 60.0  # canvas px^2


# ------------------------------------------------------------- projection


def lcc(lat: float, lon: float) -> Point:
    """Spherical Lambert conformal conic, standard parallels 33N/45N, centered 96W 39N."""
    p1, p2, p0, l0 = map(math.radians, (33.0, 45.0, 39.0, -96.0))
    n = math.log(math.cos(p1) / math.cos(p2)) / math.log(
        math.tan(math.pi / 4 + p2 / 2) / math.tan(math.pi / 4 + p1 / 2)
    )
    f = math.cos(p1) * math.tan(math.pi / 4 + p1 / 2) ** n / n
    rho = f / math.tan(math.pi / 4 + math.radians(lat) / 2) ** n
    rho0 = f / math.tan(math.pi / 4 + p0 / 2) ** n
    theta = n * (math.radians(lon) - l0)
    return rho * math.sin(theta), rho0 - rho * math.cos(theta)


# ------------------------------------------------------ thin-plate spline


def _solve(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def _u(r2: float) -> float:
    return r2 * math.log(r2) / 2 if r2 > 0 else 0.0  # r^2 log r


class ThinPlate:
    def __init__(self, src: Sequence[Point], dst: Sequence[Point], smoothing: float = 0.0) -> None:
        self.src = list(src)
        n = len(src)
        a = [[0.0] * (n + 3) for _ in range(n + 3)]
        for i, (xi, yi) in enumerate(src):
            for j, (xj, yj) in enumerate(src):
                a[i][j] = _u((xi - xj) ** 2 + (yi - yj) ** 2) + (smoothing if i == j else 0.0)
            a[i][n], a[i][n + 1], a[i][n + 2] = 1.0, xi, yi
            a[n][i], a[n + 1][i], a[n + 2][i] = 1.0, xi, yi
        self.wx = _solve(a, [p[0] for p in dst] + [0.0] * 3)
        self.wy = _solve(a, [p[1] for p in dst] + [0.0] * 3)

    def __call__(self, p: Point) -> Point:
        n = len(self.src)
        x, y = p
        ks = [_u((x - sx) ** 2 + (y - sy) ** 2) for sx, sy in self.src]
        out = []
        for w in (self.wx, self.wy):
            out.append(w[n] + w[n + 1] * x + w[n + 2] * y + sum(wi * ki for wi, ki in zip(w, ks)))
        return out[0], out[1]


# -------------------------------------------------------------- geometry


def rings_of(geometry: dict) -> List[List[Point]]:
    """Exterior rings (lon, lat) of a Polygon or MultiPolygon."""
    if geometry["type"] == "Polygon":
        return [[tuple(p) for p in geometry["coordinates"][0]]]
    if geometry["type"] == "MultiPolygon":
        return [[tuple(p) for p in poly[0]] for poly in geometry["coordinates"]]
    return []


def clip_ring(ring: List[Point], box: Tuple[float, float, float, float]) -> List[Point]:
    """Sutherland-Hodgman clip of a polygon ring to an axis-aligned box."""
    x0, y0, x1, y1 = box
    edges = [
        (lambda p: p[0] >= x0, lambda p, q: _cross_x(p, q, x0)),
        (lambda p: p[0] <= x1, lambda p, q: _cross_x(p, q, x1)),
        (lambda p: p[1] >= y0, lambda p, q: _cross_y(p, q, y0)),
        (lambda p: p[1] <= y1, lambda p, q: _cross_y(p, q, y1)),
    ]
    out = ring
    for inside, cross in edges:
        if not out:
            break
        src, out = out, []
        prev = src[-1]
        for cur in src:
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
            prev = cur
    return out


def _cross_x(p: Point, q: Point, x: float) -> Point:
    t = (x - p[0]) / (q[0] - p[0])
    return x, p[1] + t * (q[1] - p[1])


def _cross_y(p: Point, q: Point, y: float) -> Point:
    t = (y - p[1]) / (q[1] - p[1])
    return p[0] + t * (q[0] - p[0]), y


def simplify(points: List[Point], tol: float) -> List[Point]:
    """Douglas-Peucker (iterative)."""
    if len(points) < 3:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        (ax, ay), (bx, by) = points[i], points[j]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        best, idx = -1.0, -1
        for k in range(i + 1, j):
            px, py = points[k]
            # A closed ring starts and ends on the same point: use plain distance.
            d = abs(dy * (px - ax) - dx * (py - ay)) / norm if norm > 1e-9 else math.hypot(px - ax, py - ay)
            if d > best:
                best, idx = d, k
        if best > tol:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return [p for p, k in zip(points, keep) if k]


def area(ring: List[Point]) -> float:
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]))) / 2


def in_view(ring: List[Point], canvas: Tuple[int, int], pad: float = 40) -> bool:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return max(xs) >= -pad and min(xs) <= canvas[0] + pad and max(ys) >= -pad and min(ys) <= canvas[1] + pad


# ------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ne_dir", type=Path)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--out", type=Path, default=root / "src" / "ttr" / "data" / "usa_backdrop.json")
    parser.add_argument("--smoothing", type=float, default=0.0)
    args = parser.parse_args()

    display = json.loads((root / "src" / "ttr" / "data" / "usa_display.json").read_text(encoding="utf-8"))
    canvas = tuple(display["canvas"])
    cities = sorted(CITY_LATLON)
    src = [lcc(*CITY_LATLON[c]) for c in cities]
    dst = [tuple(display["cities"][c]["pos"]) for c in cities]
    # Normalize projected coordinates so the spline is well conditioned.
    mx = sum(p[0] for p in src) / len(src)
    my = sum(p[1] for p in src) / len(src)
    sd = math.sqrt(sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in src) / len(src))
    norm = lambda p: ((p[0] - mx) / sd, (p[1] - my) / sd)  # noqa: E731
    tps = ThinPlate([norm(p) for p in src], dst, args.smoothing)

    def to_canvas(ring: List[Point]) -> List[Point]:
        return [tps(norm(lcc(lat, lon))) for lon, lat in ring]

    def load(name: str) -> List[dict]:
        return json.loads((args.ne_dir / f"{name}.geojson").read_text(encoding="utf-8"))["features"]

    def build(features, keep, closed_min_area=0.0) -> List[List[Point]]:
        out = []
        for ft in features:
            if not keep(ft["properties"]):
                continue
            for ring in rings_of(ft["geometry"]):
                ring = clip_ring(ring, BBOX)
                if len(ring) < 3:
                    continue
                pts = simplify(to_canvas(ring), SIMPLIFY)
                if len(pts) < 3 or not in_view(pts, canvas) or area(pts) < closed_min_area:
                    continue
                out.append([(round(x, 1), round(y, 1)) for x, y in pts])
        return out

    countries = build(load("ne_50m_admin_0_countries"), lambda p: p["ADM0_A3"] in COUNTRIES)
    states = build(load("ne_50m_admin_1_states_provinces_lakes"), lambda p: p["adm0_a3"] in STATE_COUNTRIES)
    lakes = build(load("ne_50m_lakes"), lambda p: True, closed_min_area=MIN_LAKE_AREA)

    for name, (lat, lon) in CITY_LATLON.items():  # sanity: the warp pins every city
        x, y = tps(norm(lcc(lat, lon)))
        tx, ty = display["cities"][name]["pos"]
        assert math.hypot(x - tx, y - ty) < 1.0 or args.smoothing, name

    out = {
        "board": "usa",
        "note": (
            "Display-only map backdrop in canvas pixels. Built by scripts/build_backdrop.py from "
            "Natural Earth 1:50m data (public domain), warped so real city locations land on the "
            "board's city positions."
        ),
        "land": countries,
        "lakes": lakes,
        "states": states,
        "countries": countries,
    }
    args.out.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    n = sum(len(r) for r in countries + states + lakes)
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes): {len(countries)} country rings, "
          f"{len(states)} state rings, {len(lakes)} lakes, {n:,} points")


if __name__ == "__main__":
    main()
