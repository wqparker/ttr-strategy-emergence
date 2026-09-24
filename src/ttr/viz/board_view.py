"""Pygame drawing of the board: map area, routes as train cars, cities, labels,
the score track with player markers, and the route-points table.

The static board (everything but claims and scores) is drawn once per scale and
cached; `draw()` then adds claimed trains, score markers and any highlights.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import pygame

from ttr.board import ROUTE_POINTS, Board
from ttr.game import Game
from ttr.viz import theme
from ttr.viz.geometry import BoardLayout, Car, Point, load_layout

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
        self._labels: Optional[pygame.Surface] = None  # drawn above claimed trains

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
            self._static = self._labels = None

    # --------------------------------------------------------------- drawing

    def draw(
        self,
        target: pygame.Surface,
        game: Optional[Game] = None,
        origin: Tuple[int, int] = (0, 0),
        route_tint: Optional[Dict[int, theme.RGB]] = None,
        highlight_routes: Iterable[int] = (),
        highlight_cities: Iterable[str] = (),
        show_scores: bool = True,
    ) -> None:
        """Draw the board onto `target` at `origin`. `route_tint` recolors routes
        (analysis overlays); highlights outline routes and ring cities."""
        if self._static is None:
            self._static = self._render_static()
            self._labels = self._render_labels()
        target.blit(self._static, origin)
        layer = pygame.Surface(self.size, pygame.SRCALPHA)
        if route_tint:
            for rid, color in route_tint.items():
                for car in self.layout.routes[rid].cars:
                    self._car(layer, car, color, theme.darker(color, 0.5))
        if game is not None:
            for rid, owner in game.route_owner.items():
                color = theme.PLAYER[owner % len(theme.PLAYER)]
                for car in self.layout.routes[rid].cars:
                    self._train(layer, car, color)
        for rid in highlight_routes:
            for car in self.layout.routes[rid].cars:
                pts = [self.to_screen(p) for p in car.corners(inset=-1.5)]
                pygame.draw.polygon(layer, theme.HIGHLIGHT, pts, max(2, round(2 * self.scale)))
        for city in highlight_cities:
            x, y = self.to_screen(self.layout.cities[city])
            r = (self.layout.city_radius + 5) * self.scale
            pygame.draw.circle(layer, theme.HIGHLIGHT, (x, y), r, max(2, round(3 * self.scale)))
        if game is not None and show_scores:
            self._score_markers(layer, [p.route_points for p in game.players])
        target.blit(layer, origin)
        target.blit(self._labels, origin)

    def _render_static(self) -> pygame.Surface:
        s = pygame.Surface(self.size)
        s.fill(theme.MAP_BG)
        self._terrain(s)
        self._track(s)
        if self.layout.table:
            self._table(s)
        for shape in self.layout.routes.values():
            route = self.board.routes[shape.route_id]
            color = theme.CARD[route.color]
            for car in shape.cars:
                self._car(s, car, color, theme.darker(color, 0.45))
        self._cities(s)
        return s

    def _terrain(self, s: pygame.Surface) -> None:
        x, y, w, h = (v * self.scale for v in self.layout.inner)
        step = 40 * self.scale
        gx = x
        while gx < x + w:
            pygame.draw.line(s, theme.MAP_GRID, (gx, y), (gx, y + h))
            gx += step
        gy = y
        while gy < y + h:
            pygame.draw.line(s, theme.MAP_GRID, (x, gy), (x + w, gy))
            gy += step

    def _track(self, s: pygame.Surface) -> None:
        f = font(round(13 * self.scale), bold=True, serif=True)
        for i, (x, y, w, h) in enumerate(self.layout.track):
            rect = pygame.Rect(*(round(v * self.scale) for v in (x, y, w, h)))
            pygame.draw.rect(s, theme.TRACK_BG if i % 2 == 0 else theme.TRACK_ALT, rect)
            pygame.draw.rect(s, theme.darker(theme.TRACK_BG, 0.5), rect, 1)
            color = theme.TRACK_TEXT_5 if i % 5 == 0 else theme.TRACK_TEXT
            label = f.render(str(i if i else 100), True, color)
            s.blit(label, label.get_rect(center=rect.center))

    def _table(self, s: pygame.Surface) -> None:
        x, y, w, h = (v * self.scale for v in self.layout.table)
        rect = pygame.Rect(round(x), round(y), round(w), round(h))
        pygame.draw.rect(s, theme.TABLE_BG, rect, border_radius=round(4 * self.scale))
        pygame.draw.rect(s, theme.OUTLINE, rect, 1, border_radius=round(4 * self.scale))
        f = font(round(9 * self.scale), bold=True)
        row_h = h / 6
        for i, length in enumerate(range(1, 7)):
            cy = y + (i + 0.5) * row_h
            s.blit(f.render(str(length), True, theme.LABEL), (x + 4 * self.scale, cy - f.get_height() / 2))
            cw, gap = 11 * self.scale, 2 * self.scale
            for k in range(length):
                cx = x + 16 * self.scale + k * (cw + gap)
                car = pygame.Rect(round(cx), round(cy - 3 * self.scale), round(cw), round(6 * self.scale))
                pygame.draw.rect(s, theme.CARD[None], car)
                pygame.draw.rect(s, theme.OUTLINE, car, 1)
            pts = f.render(str(ROUTE_POINTS[length]), True, theme.LABEL)
            s.blit(pts, (x + w - pts.get_width() - 5 * self.scale, cy - pts.get_height() / 2))

    def _cities(self, s: pygame.Surface) -> None:
        r = self.layout.city_radius * self.scale
        for city, pos in self.layout.cities.items():
            x, y = self.to_screen(pos)
            pygame.draw.circle(s, theme.CITY_RING, (x, y), r + 1.5 * self.scale)
            pygame.draw.circle(s, theme.CITY_FILL, (x, y), r)
            pygame.draw.circle(s, theme.CITY_SHINE, (x - r * 0.3, y - r * 0.3), r * 0.3)

    def _render_labels(self) -> pygame.Surface:
        s = pygame.Surface(self.size, pygame.SRCALPHA)
        f = font(round(11 * self.scale), bold=True, serif=True)
        for city, pos in self.layout.labels.items():
            lines = self.layout.label(city).upper().split("\n")
            x, y = self.to_screen(pos)
            line_h = f.get_linesize() * 0.85
            top = y - line_h * (len(lines) - 1) / 2
            for i, line in enumerate(lines):
                self._label(s, line, (x, top + i * line_h), f)
        return s

    def _label(self, s: pygame.Surface, text: str, center: Point, f: pygame.font.Font) -> None:
        halo = f.render(text, True, theme.LABEL_HALO)
        body = f.render(text, True, theme.LABEL)
        rect = body.get_rect(center=(round(center[0]), round(center[1])))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
            s.blit(halo, rect.move(dx, dy))
        s.blit(body, rect)

    def _car(self, s: pygame.Surface, car: Car, fill: theme.RGB, edge: theme.RGB) -> None:
        pts = [self.to_screen(p) for p in car.corners()]
        pygame.draw.polygon(s, fill, pts)
        pygame.draw.polygon(s, edge, pts, max(1, round(1.2 * self.scale)))
        inner = [self.to_screen(p) for p in car.corners(inset=3)]
        pygame.draw.polygon(s, theme.lighter(fill, 0.25), inner, 1)

    def _train(self, s: pygame.Surface, car: Car, color: theme.RGB) -> None:
        """A plastic train piece: covers the space, with a dark rim and a light
        center stripe so it stands out even on a route of the same color."""
        outer = [self.to_screen(p) for p in car.corners(inset=-1)]
        pygame.draw.polygon(s, theme.TRAIN_RIM, outer)
        body = [self.to_screen(p) for p in car.corners(inset=1)]
        pygame.draw.polygon(s, color, body)
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = car.corners(inset=3)
        a = self.to_screen(((x0 + x3) / 2, (y0 + y3) / 2))
        b = self.to_screen(((x1 + x2) / 2, (y1 + y2) / 2))
        pygame.draw.line(s, theme.lighter(color, 0.55), a, b, max(1, round(2 * self.scale)))

    def _score_markers(self, s: pygame.Surface, scores: Sequence[int]) -> None:
        """One disc per player on cell (score mod 100); players sharing a cell
        are spread over the cell's quadrants."""
        cells = [score % len(self.layout.track) for score in scores]
        r = 6 * self.scale
        spread = ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0))
        placed: Dict[int, int] = {}
        for seat, cell in enumerate(cells):
            k = placed.get(cell, 0)
            placed[cell] = k + 1
            dx, dy = spread[k] if cells.count(cell) > 1 else (0, 0)
            x, y, w, h = self.layout.track[cell]
            cx = (x + w / 2) * self.scale + dx * r * 0.8
            cy = (y + h / 2) * self.scale + dy * r * 0.8
            color = theme.PLAYER[seat % len(theme.PLAYER)]
            pygame.draw.circle(s, theme.darker(color, 0.4), (cx, cy), r + 1.5)
            pygame.draw.circle(s, color, (cx, cy), r)
