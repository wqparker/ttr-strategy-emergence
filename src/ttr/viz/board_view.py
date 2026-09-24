"""Pygame drawing of the board: map backdrop, routes as train cars, cities,
labels, the frame and the route-points table, plus claimed trains on top.

Shapes are drawn at SUPERSAMPLE x the display scale and smoothly scaled down,
which anti-aliases every edge. The static board is rendered once per scale; the
claims layer is re-rendered only when claims or highlights change. Labels are
drawn at the display scale (text is already anti-aliased).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pygame

from ttr.board import ROUTE_POINTS, Board
from ttr.game import Game
from ttr.viz import theme
from ttr.viz.geometry import BoardLayout, Car, Point, load_layout

SUPERSAMPLE = 2
SERIF = "georgia,palatinolinotype,timesnewroman,serif"
SANS = "segoeui,arial,helvetica,sans"


def font(size: int, bold: bool = False, serif: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont(SERIF if serif else SANS, max(6, size), bold=bold)


class BoardView:
    def __init__(self, board: Board, scale: float = 1.0, layout: Optional[BoardLayout] = None) -> None:
        self.board = board
        self.layout = layout or load_layout(board)
        self.scale = scale
        self._static: Optional[pygame.Surface] = None
        self._labels: Optional[pygame.Surface] = None
        self._overlay: Optional[pygame.Surface] = None
        self._overlay_key: object = None

    # ------------------------------------------------------------ transforms

    @property
    def size(self) -> Tuple[int, int]:
        w, h = self.layout.canvas
        return (round(w * self.scale), round(h * self.scale))

    def to_screen(self, p: Point) -> Tuple[float, float]:
        return (p[0] * self.scale, p[1] * self.scale)

    def to_canvas(self, p: Point) -> Point:
        return (p[0] / self.scale, p[1] / self.scale)

    def set_scale(self, scale: float) -> None:
        if abs(scale - self.scale) > 1e-6:
            self.scale = scale
            self._static = self._labels = self._overlay = None
            self._overlay_key = None

    # --------------------------------------------------------------- drawing

    def draw(
        self,
        target: pygame.Surface,
        game: Optional[Game] = None,
        origin: Tuple[int, int] = (0, 0),
        route_tint: Optional[Dict[int, theme.RGB]] = None,
        highlight_routes: Iterable[int] = (),
        highlight_cities: Iterable[str] = (),
    ) -> None:
        """Draw the board onto `target` at `origin`. `route_tint` recolors routes
        (analysis overlays); highlights outline routes and ring cities."""
        if self._static is None:
            self._static = self._supersampled(self._paint_static, alpha=False)
            self._labels = self._render_labels()
        owners = tuple(sorted(game.route_owner.items())) if game is not None else ()
        key = (
            owners,
            tuple(sorted(route_tint.items())) if route_tint else (),
            tuple(highlight_routes),
            tuple(highlight_cities),
        )
        if key != self._overlay_key:
            self._overlay_key = key
            self._overlay = (
                self._supersampled(lambda s, k: self._paint_overlay(s, k, *key), alpha=True)
                if any(key) else None
            )
        target.blit(self._static, origin)
        if self._overlay is not None:
            target.blit(self._overlay, origin)
        target.blit(self._labels, origin)

    def _supersampled(self, paint, alpha: bool) -> pygame.Surface:
        k = self.scale * SUPERSAMPLE
        w, h = self.layout.canvas
        big = pygame.Surface((round(w * k), round(h * k)), pygame.SRCALPHA if alpha else 0)
        if alpha:
            big.fill((0, 0, 0, 0))
        paint(big, k)
        return pygame.transform.smoothscale(big, self.size)

    # ----------------------------------------------------------- static layer

    def _paint_static(self, s: pygame.Surface, k: float) -> None:
        w, h = self.layout.canvas
        s.fill(theme.FRAME)
        inner = _rect(self.layout.inner, k)
        s.fill(theme.WATER if self.layout.backdrop else theme.MAP_BG, inner)
        if self.layout.backdrop:
            s.set_clip(inner)
            self._backdrop(s, k)
            s.set_clip(None)
        pygame.draw.rect(s, theme.FRAME_LINE, inner.inflate(round(2 * k), round(2 * k)), max(1, round(2 * k)))
        if self.layout.table:
            self._table(s, k)
        for shape in self.layout.routes.values():
            route = self.board.routes[shape.route_id]
            color = theme.CARD[route.color]
            for car in shape.cars:
                _car(s, car, k, color, theme.darker(color, 0.45))
        self._cities(s, k)

    def _backdrop(self, s: pygame.Surface, k: float) -> None:
        b = self.layout.backdrop
        for ring in b.land:
            if len(ring) >= 3:
                pygame.draw.polygon(s, theme.LAND, _pts(ring, k))
        for ring in b.lakes:
            if len(ring) >= 3:
                pygame.draw.polygon(s, theme.WATER, _pts(ring, k))
        for ring in b.states:
            if len(ring) >= 2:
                pygame.draw.lines(s, theme.STATE_LINE, True, _pts(ring, k), max(1, round(0.8 * k)))
        for ring in b.lakes:
            if len(ring) >= 3:
                pygame.draw.lines(s, theme.COAST_LINE, True, _pts(ring, k), max(1, round(0.9 * k)))
        for ring in b.countries:  # country borders and coastlines
            if len(ring) >= 2:
                pygame.draw.lines(s, theme.COUNTRY_LINE, True, _pts(ring, k), max(1, round(1.4 * k)))

    def _table(self, s: pygame.Surface, k: float) -> None:
        x, y, w, h = self.layout.table
        rect = _rect((x, y, w, h), k)
        pygame.draw.rect(s, theme.TABLE_BG, rect, border_radius=round(4 * k))
        pygame.draw.rect(s, theme.OUTLINE, rect, max(1, round(k)), border_radius=round(4 * k))
        f = font(round(9 * k), bold=True)
        row_h = h / 6
        for i, length in enumerate(range(1, 7)):
            cy = y + (i + 0.5) * row_h
            s.blit(f.render(str(length), True, theme.LABEL), ((x + 4) * k, cy * k - f.get_height() / 2))
            for n in range(length):
                cx = x + 16 + n * 13
                car = _rect((cx, cy - 3, 11, 6), k)
                pygame.draw.rect(s, theme.CARD[None], car)
                pygame.draw.rect(s, theme.OUTLINE, car, max(1, round(k * 0.8)))
            pts = f.render(str(ROUTE_POINTS[length]), True, theme.LABEL)
            s.blit(pts, ((x + w - 5) * k - pts.get_width(), cy * k - pts.get_height() / 2))

    def _cities(self, s: pygame.Surface, k: float) -> None:
        r = self.layout.city_radius * k
        for pos in self.layout.cities.values():
            x, y = pos[0] * k, pos[1] * k
            pygame.draw.circle(s, theme.CITY_RING, (x, y), r + 1.5 * k)
            pygame.draw.circle(s, theme.CITY_FILL, (x, y), r)
            pygame.draw.circle(s, theme.CITY_SHINE, (x - r * 0.3, y - r * 0.3), r * 0.3)

    # ---------------------------------------------------------- claims layer

    def _paint_overlay(self, s, k, owners, tint, highlight_routes, highlight_cities) -> None:
        for rid, color in tint:
            for car in self.layout.routes[rid].cars:
                _car(s, car, k, color, theme.darker(color, 0.5))
        for rid, owner in owners:
            color = theme.PLAYER[owner % len(theme.PLAYER)]
            for car in self.layout.routes[rid].cars:
                _train(s, car, k, color)
        for rid in highlight_routes:
            for car in self.layout.routes[rid].cars:
                pygame.draw.polygon(s, theme.HIGHLIGHT, _pts(car.corners(inset=-1.5), k), max(2, round(2 * k)))
        for city in highlight_cities:
            x, y = self.layout.cities[city]
            r = (self.layout.city_radius + 5) * k
            pygame.draw.circle(s, theme.HIGHLIGHT, (x * k, y * k), r, max(2, round(3 * k)))

    # ---------------------------------------------------------------- labels

    def _render_labels(self) -> pygame.Surface:
        s = pygame.Surface(self.size, pygame.SRCALPHA)
        f = font(round(11 * self.scale), bold=True, serif=True)
        for city, pos in self.layout.labels.items():
            lines = self.layout.label(city).upper().split("\n")
            x, y = self.to_screen(pos)
            line_h = f.get_linesize() * 0.85
            top = y - line_h * (len(lines) - 1) / 2
            for i, line in enumerate(lines):
                _label(s, line, (x, top + i * line_h), f)
        return s


# ------------------------------------------------------------------ helpers


def _pts(points, k: float) -> List[Tuple[float, float]]:
    return [(x * k, y * k) for x, y in points]


def _rect(r, k: float) -> pygame.Rect:
    x, y, w, h = r
    return pygame.Rect(round(x * k), round(y * k), round(w * k), round(h * k))


def _car(s: pygame.Surface, car: Car, k: float, fill: theme.RGB, edge: theme.RGB) -> None:
    pts = _pts(car.corners(), k)
    pygame.draw.polygon(s, fill, pts)
    pygame.draw.polygon(s, edge, pts, max(1, round(1.2 * k)))
    pygame.draw.polygon(s, theme.lighter(fill, 0.25), _pts(car.corners(inset=3), k), max(1, round(0.8 * k)))


def _train(s: pygame.Surface, car: Car, k: float, color: theme.RGB) -> None:
    """A plastic train piece: covers the space, with a dark rim and a light
    center stripe so it stands out even on a route of the same color."""
    pygame.draw.polygon(s, theme.TRAIN_RIM, _pts(car.corners(inset=-1), k))
    pygame.draw.polygon(s, color, _pts(car.corners(inset=1), k))
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = car.corners(inset=3)
    a = ((x0 + x3) / 2 * k, (y0 + y3) / 2 * k)
    b = ((x1 + x2) / 2 * k, (y1 + y2) / 2 * k)
    pygame.draw.line(s, theme.lighter(color, 0.55), a, b, max(1, round(2 * k)))


def _label(s: pygame.Surface, text: str, center: Point, f: pygame.font.Font) -> None:
    halo = f.render(text, True, theme.LABEL_HALO)
    body = f.render(text, True, theme.LABEL)
    rect = body.get_rect(center=(round(center[0]), round(center[1])))
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        s.blit(halo, rect.move(dx, dy))
    s.blit(body, rect)
