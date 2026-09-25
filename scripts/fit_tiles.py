"""Fit the USA board's train tiles to the board photo.

    python scripts/fit_tiles.py fit [--out PATH]      # re-fit every tile, write the display file
    python scripts/fit_tiles.py size                  # measure the tile size from the photo
    python scripts/fit_tiles.py check                 # list the weakest fits
    python scripts/fit_tiles.py overlay OUT.png [--zoom 4]   # draw tile outlines on the photo

Needs the photo at docs/ticket-to-ride_usa_map.jpg (kept locally, not in the repo)
and the `[photo]` extra (numpy, OpenCV): pip install -e ".[photo]".

Each tile is a rotated rectangle (center, angle) of one fixed size. Starting from
the tiles already in `src/ttr/data/usa_display.json`, a grid search around each
one scores candidate rectangles on the photo:
  - color: how much the pixels inside look like the route's color, minus how much a
    thin ring just outside does (colors are the median of each color's tiles);
  - outline: the printed tiles have a dark outline, so the ring right at the edge
    is darker than the fill;
  - evenness: the fill is one flat color, so its brightness varies little.
The search fits the colored fill (FILL_LENGTH x FILL_WIDTH). The stored car is
CAR_LENGTH x CAR_WIDTH, the fill plus its outline, because the renderer strokes the
outline on the car's edge. `size` re-measures the fill size; brightness profiles
across fitted tiles put the outline about 1 px outside it.

Starting positions must be within about 5 px and 12 degrees of the real tile, which
the committed data is. City positions are not touched.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "docs" / "ticket-to-ride_usa_map.jpg"
DISPLAY = ROOT / "src" / "ttr" / "data" / "usa_display.json"

FILL_LENGTH, FILL_WIDTH = 35.5, 10.5  # colored fill, measured by `size`
CAR_LENGTH, CAR_WIDTH = 38, 13  # fill plus outline, as stored and drawn
COLOR_SIGMA = 12.0  # Lab distance scale for "looks like the route color"

LAB: np.ndarray  # the photo in Lab (L 0..100), set by load_photo()


def load_photo() -> None:
    global LAB
    if not PHOTO.exists():
        raise SystemExit(f"needs the board photo at {PHOTO}")
    img = cv2.imread(str(PHOTO))
    LAB = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def sample(pts: np.ndarray) -> np.ndarray:
    """Bilinear Lab values at points (..., 2) -> (..., 3)."""
    shape = pts.shape[:-1]
    p = pts.reshape(-1, 2).astype(np.float32)
    n, cols = p.shape[0], 1024
    rows = -(-n // cols)
    pad = rows * cols - n
    mx = np.pad(p[:, 0], (0, pad)).reshape(rows, cols)
    my = np.pad(p[:, 1], (0, pad)).reshape(rows, cols)
    out = cv2.remap(LAB, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1, 3)[:n].reshape(*shape, 3)


def grid(length: float, width: float, inset: float, n_u: int = 0, n_v: int = 0) -> np.ndarray:
    """Points filling a rectangle centered at 0 (u along, v across), inset from its edge."""
    n_u = n_u or max(4, int(length - 2 * inset))
    n_v = n_v or max(2, int(width - 2 * inset))
    u, v = np.meshgrid(
        np.linspace(-length / 2 + inset, length / 2 - inset, n_u),
        np.linspace(-width / 2 + inset, width / 2 - inset, n_v),
    )
    return np.stack([u.ravel(), v.ravel()], 1)


def ring(length: float, width: float, d: float) -> np.ndarray:
    """Points on a rectangle `d` px outside the edge (corners left out)."""
    pts = []
    for s in (-1, 1):
        u = np.linspace(-length / 2 + 1, length / 2 - 1, int(length / 1.5))
        pts.append(np.stack([u, np.full_like(u, s * (width / 2 + d))], 1))
        v = np.linspace(-width / 2 + 1, width / 2 - 1, max(3, int(width / 2)))
        pts.append(np.stack([np.full_like(v, s * (length / 2 + d)), v], 1))
    return np.concatenate(pts)


def place(local: np.ndarray, cx, cy, th) -> np.ndarray:
    """Local points placed at each candidate (cx, cy, th): (K, N, 2)."""
    cx, cy, th = np.atleast_1d(cx), np.atleast_1d(cy), np.atleast_1d(th)
    ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
    x = cx[:, None] + local[None, :, 0] * ct - local[None, :, 1] * st
    y = cy[:, None] + local[None, :, 0] * st + local[None, :, 1] * ct
    return np.stack([x, y], -1)


class Scorer:
    def __init__(self, length: float, width: float, color_lab: np.ndarray, color: str) -> None:
        self.inside = grid(length, width, 1.5)
        self.core = grid(length, width, 2.5)
        self.edge = ring(length, width, 0.8)  # the dark outline
        self.outer = ring(length, width, 2.2)
        self.col = np.asarray(color_lab, np.float32)
        self.outline_weight = 0.3 if color == "black" else 1.0  # black fill is darker than its outline

    def like(self, lab: np.ndarray) -> np.ndarray:
        return np.exp(-((lab - self.col) ** 2).sum(-1) / (2 * COLOR_SIGMA**2))

    def score(self, cx, cy, th) -> np.ndarray:
        pin = sample(place(self.inside, cx, cy, th))
        edge = sample(place(self.edge, cx, cy, th))
        outer = sample(place(self.outer, cx, cy, th))
        core = sample(place(self.core, cx, cy, th))
        s = 2.0 * (self.like(pin).mean(1) - np.concatenate([self.like(edge), self.like(outer)], 1).mean(1))
        s -= 0.02 * core[..., 0].std(1)
        s += 0.02 * self.outline_weight * (pin[..., 0].mean(1) - edge[..., 0].mean(1))
        return s


def search(sc: Scorer, x: float, y: float, th: float, r: float, step: float, dth: float, thstep: float):
    ds = np.arange(-r, r + 1e-9, step)
    ts = np.radians(np.arange(-dth, dth + 1e-9, thstep))
    dx, dy, dt = np.meshgrid(ds, ds, ts, indexing="ij")
    cx, cy, tt = x + dx.ravel(), y + dy.ravel(), th + dt.ravel()
    s = sc.score(cx, cy, tt)
    i = int(np.argmax(s))
    return float(cx[i]), float(cy[i]), float(tt[i]), float(s[i])


def fit_tile(sc: Scorer, x: float, y: float, th: float) -> Tuple[float, float, float]:
    x, y, th, _ = search(sc, x, y, th, 5, 1.0, 12, 2)  # coarse: +-5 px, +-12 deg
    x, y, th, _ = search(sc, x, y, th, 1.0, 0.25, 2, 0.5)  # fine
    return x, y, th


def color_models(display: dict) -> Dict[str, np.ndarray]:
    """Median Lab of the middle of every tile, per route color."""
    pools: Dict[str, List[np.ndarray]] = {}
    for e in display["tiles"]:
        for x, y, a in e["tiles"]:
            pts = place(grid(8, 3, 0, 5, 3), x, y, math.radians(a))
            pools.setdefault(e["color"], []).append(sample(pts)[0])
    return {c: np.median(np.concatenate(v), 0) for c, v in pools.items()}


# ------------------------------------------------------------------ commands


def cmd_fit(display: dict, out: Path) -> None:
    cols = color_models(display)
    moved = 0.0
    for e in display["tiles"]:
        sc = Scorer(FILL_LENGTH, FILL_WIDTH, cols[e["color"]], e["color"])
        new = []
        for x, y, a in e["tiles"]:
            fx, fy, th = fit_tile(sc, x, y, math.radians(a))
            moved = max(moved, math.hypot(fx - x, fy - y))
            new.append([round(fx, 1), round(fy, 1), round(math.degrees(th), 1)])
        e["tiles"] = new
    display["car_length"], display["car_width"] = CAR_LENGTH, CAR_WIDTH
    write_display(display, out)
    print(f"wrote {out}; largest move {moved:.1f} px")


def cmd_size(display: dict) -> None:
    """Best fill size over the colored tiles of single routes (gray and white have
    too little contrast to measure size well). Sibling routes share a+b."""
    cols = color_models(display)
    pairs = {}
    for e in display["tiles"]:
        pairs[(e["a"], e["b"])] = pairs.get((e["a"], e["b"]), 0) + 1
    tiles = [
        (e["color"], t)
        for e in display["tiles"]
        if pairs[(e["a"], e["b"])] == 1 and e["color"] not in ("gray", "white")
        for t in e["tiles"]
    ]
    located = []
    for color, (x, y, a) in tiles:
        sc = Scorer(FILL_LENGTH, FILL_WIDTH, cols[color], color)
        located.append((color,) + search(sc, x, y, math.radians(a), 3, 1.0, 6, 2)[:3])
    results = []
    for length in np.arange(33, 40.01, 0.5):
        for width in np.arange(9, 13.01, 0.5):
            total = sum(
                search(Scorer(length, width, cols[c], c), x, y, th, 1.0, 0.5, 2, 1)[3] for c, x, y, th in located
            )
            results.append((total, length, width))
    results.sort(reverse=True)
    print(f"fill size, best first ({len(tiles)} tiles):")
    for total, length, width in results[:6]:
        print(f"  {length:4.1f} x {width:4.1f}  score {total:.2f}")


def cmd_check(display: dict, n: int = 20) -> None:
    cols = color_models(display)
    rows = []
    for rid, e in enumerate(display["tiles"]):
        sc = Scorer(FILL_LENGTH, FILL_WIDTH, cols[e["color"]], e["color"])
        for i, (x, y, a) in enumerate(e["tiles"]):
            rows.append((float(sc.score(x, y, math.radians(a))[0]), rid, e["a"], e["b"], e["color"], i))
    rows.sort()
    print(f"weakest {n} fits (gray and white score lowest; view them with `overlay`):")
    for s, rid, a, b, color, i in rows[:n]:
        print(f"  {s:5.2f}  route {rid:3d} {a}-{b} {color} tile {i}")


def cmd_overlay(display: dict, out: Path, zoom: int) -> None:
    img = cv2.imread(str(PHOTO))
    img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LANCZOS4)
    hl, hw = display["car_length"] / 2, display["car_width"] / 2
    for e in display["tiles"]:
        for x, y, a in e["tiles"]:
            t = math.radians(a)
            ux, uy, vx, vy = math.cos(t), math.sin(t), -math.sin(t), math.cos(t)
            pts = np.array(
                [[(x + su * hl * ux + sv * hw * vx) * zoom, (y + su * hl * uy + sv * hw * vy) * zoom]
                 for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
                np.int32,
            )
            cv2.polylines(img, [pts], True, (255, 255, 0), 1, cv2.LINE_AA)
    r = display["city_radius"] * zoom
    for v in display["cities"].values():
        x, y = v["pos"]
        cv2.circle(img, (round(x * zoom), round(y * zoom)), round(r), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), img)
    print(f"saved {out} {img.shape[1]}x{img.shape[0]}")


def write_display(d: dict, path: Path) -> None:
    """Same layout as the committed file: one city and one route per line."""
    head = {k: v for k, v in d.items() if k not in ("cities", "tiles")}
    lines = ["{"] + [f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    lines.append('  "cities": {')
    cities = list(d["cities"].items())
    for i, (c, v) in enumerate(cities):
        lines.append(f"    {json.dumps(c)}: {json.dumps(v)}" + ("," if i < len(cities) - 1 else ""))
    lines += ["  },", '  "tiles": [']
    for i, e in enumerate(d["tiles"]):
        lines.append("    " + json.dumps(e) + ("," if i < len(d["tiles"]) - 1 else ""))
    lines += ["  ]", "}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--out", type=Path, default=DISPLAY)
    sub.add_parser("size")
    sub.add_parser("check")
    p = sub.add_parser("overlay")
    p.add_argument("out", type=Path)
    p.add_argument("--zoom", type=int, default=4)
    args = parser.parse_args()

    load_photo()
    display = json.loads(DISPLAY.read_text(encoding="utf-8"))
    if args.cmd == "fit":
        cmd_fit(display, args.out)
    elif args.cmd == "size":
        cmd_size(display)
    elif args.cmd == "check":
        cmd_check(display)
    else:
        cmd_overlay(display, args.out, args.zoom)


if __name__ == "__main__":
    main()
