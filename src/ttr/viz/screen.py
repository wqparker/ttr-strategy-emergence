"""The whole viewer surface: the board with panels around it.

Layout (PLAN.md Phase 3, milestone 4), in canvas units scaled by `scale`:

    +---------------------------------------------------------------+
    | table strip: 5 face-up cards, pile counts, sub-step            |
    +---------+-----------------------------------------+-----------+
    | seat 1  |                                         | seat 2    |
    | seat 3  |                board                    | seat 4    |
    +---------+-----------------------------------------+-----------+
    | event ticker                                                  |
    | seat 0, full width: the human seat                            |
    +---------------------------------------------------------------+

Seat 0 is the bottom panel whatever the perspective is; the perspective only
decides what each panel may show. Side slots exist for seats in play only.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pygame

from ttr.board import Board
from ttr.game import Game
from ttr.viz import panels, theme
from ttr.viz.board_view import BoardView
from ttr.viz.geometry import BoardLayout, Point
from ttr.viz.perspective import Perspective, ViewModel, code_map

SIDE_W = 200.0  # each side panel column
TOP_H = 96.0  # table strip
TICKER_H = 22.0  # event ticker above the bottom panel
BOTTOM_H = 132.0  # seat 0's panel
GAP = 5.0  # between panel boxes

LEFT_SEATS = (1, 3)
RIGHT_SEATS = (2, 4)

# Key legend for the table strip's top-left corner. The bindings themselves live
# in viz/app.py; this is only what the viewer tells the player about them.
CONTROLS = (
    ("space", "play / pause"),
    (". ,", "step fwd / back"),
    ("] [", "faster / slower"),
    ("v", "perspective"),
    ("end home", "jump to end / start"),
    ("esc", "quit"),
)
BUTTON_W, BUTTON_H, BUTTON_GAP = 46.0, 22.0, 5.0


class Screen:
    """Composes a `BoardView` with the panels. Holds the perspective; the live
    and replay viewers (milestone 5) drive it."""

    def __init__(
        self,
        board: Board,
        scale: float = 1.0,
        names: Optional[Sequence[str]] = None,
        perspective: Optional[Perspective] = None,
        layout: Optional[BoardLayout] = None,
    ) -> None:
        self.board = board
        self.names = list(names) if names else None
        self.perspective = perspective or Perspective()
        self.board_view = BoardView(board, scale=scale, layout=layout)
        self.codes = code_map(board)
        self._num_players = 2

    # -------------------------------------------------------------- geometry

    @property
    def scale(self) -> float:
        return self.board_view.scale

    def set_scale(self, scale: float) -> None:
        self.board_view.set_scale(scale)

    @property
    def size(self) -> Tuple[int, int]:
        bw, bh = self.board_view.size
        k = self.scale
        return (round(2 * SIDE_W * k) + bw, round((TOP_H + TICKER_H + BOTTOM_H) * k) + bh)

    @property
    def board_origin(self) -> Tuple[int, int]:
        k = self.scale
        return (round(SIDE_W * k), round(TOP_H * k))

    def _rect(self, x: float, y: float, w: float, h: float) -> pygame.Rect:
        k = self.scale
        return pygame.Rect(round(x * k), round(y * k), round(w * k), round(h * k))

    @property
    def table_rect(self) -> pygame.Rect:
        w = self.size[0] / self.scale
        return self._rect(GAP, GAP, w - 2 * GAP, TOP_H - 2 * GAP)

    @property
    def ticker_rect(self) -> pygame.Rect:
        w, h = (v / self.scale for v in self.size)
        return self._rect(0, h - BOTTOM_H - TICKER_H, w, TICKER_H)

    @property
    def bottom_rect(self) -> pygame.Rect:
        w, h = (v / self.scale for v in self.size)
        return self._rect(GAP, h - BOTTOM_H, w - 2 * GAP, BOTTOM_H - GAP)

    def seat_rects(self, num_players: int) -> Dict[int, pygame.Rect]:
        """Seat 0's bottom panel plus a slot for every other seat in play."""
        rects = {0: self.bottom_rect}
        bh = self.board_view.size[1] / self.scale
        slot_h = (bh - GAP) / 2 - GAP
        w, _ = (v / self.scale for v in self.size)
        for seats, x in ((LEFT_SEATS, GAP), (RIGHT_SEATS, w - SIDE_W + GAP)):
            for row, seat in enumerate(seats):
                if seat < num_players:
                    y = TOP_H + GAP + row * (slot_h + GAP)
                    rects[seat] = self._rect(x, y, SIDE_W - 2 * GAP, slot_h)
        return rects

    def button_rects(self, names: Sequence[str]) -> List[pygame.Rect]:
        """Slots for the on-screen controls, in the bottom panel's left margin
        (seat 0's fields are centered, so nothing else sits there)."""
        r = self.bottom_rect
        rows = 1 if len(names) <= 4 else 2
        per_row = -(-len(names) // rows)
        top = (r.centery / self.scale) - rows * (BUTTON_H + BUTTON_GAP) / 2
        left = r.left / self.scale + 9
        out = []
        for i in range(len(names)):
            col, row = i % per_row, i // per_row
            out.append(self._rect(left + col * (BUTTON_W + BUTTON_GAP),
                                  top + row * (BUTTON_H + BUTTON_GAP), BUTTON_W, BUTTON_H))
        return out

    def button_at(self, pos: Tuple[float, float], buttons) -> Optional[str]:
        """Which on-screen control a click landed on, if any."""
        for name, _, rect in buttons:
            if rect.collidepoint(pos):
                return name
        return None

    def board_point(self, pos: Tuple[float, float]) -> Optional[Point]:
        """Window pixel -> board canvas coordinate, or None if off the board.
        Human play (milestone 6) turns this into a route or city."""
        ox, oy = self.board_origin
        bw, bh = self.board_view.size
        x, y = pos[0] - ox, pos[1] - oy
        if not (0 <= x < bw and 0 <= y < bh):
            return None
        return self.board_view.to_canvas((x, y))

    # ----------------------------------------------------------- perspective

    def set_perspective(self, viewer: Optional[int], memory_level: Optional[int] = None) -> None:
        level = self.perspective.memory_level if memory_level is None else memory_level
        self.perspective = Perspective(viewer, level)

    def cycle_perspective(self) -> Optional[int]:
        """All-seeing -> seat 0 -> ... -> last seat -> all-seeing."""
        current = self.perspective.viewer
        nxt = 0 if current is None else (None if current + 1 >= self._num_players else current + 1)
        self.set_perspective(nxt)
        return nxt

    # -------------------------------------------------------------- drawing

    def draw(
        self,
        target: pygame.Surface,
        game: Game,
        route_tint: Optional[Dict[int, theme.RGB]] = None,
        highlight_routes: Iterable[int] = (),
        highlight_cities: Iterable[str] = (),
        events: int = 3,
        controls: Optional[Sequence[Tuple[str, str]]] = None,
        buttons: Sequence[Tuple[str, str, pygame.Rect]] = (),
        active: Sequence[str] = (),
    ) -> ViewModel:
        """Draw everything and return the view model that was drawn."""
        self._num_players = game.num_players
        vm = self.perspective.view(game, names=self.names, events=events)
        k = self.scale
        target.fill(theme.PANEL_BG)
        self.board_view.draw(
            target,
            game,
            origin=self.board_origin,
            route_tint=route_tint,
            highlight_routes=highlight_routes,
            highlight_cities=highlight_cities,
        )
        r = self.table_rect
        legend = panels.controls_layout(controls or (), r, k=k)[3]
        panels.draw_table(target, r, vm, k=k,
                          left_bound=r.left + 8 * k + legend + (24 * k if legend else 2 * k))
        if controls:
            panels.draw_controls(target, r, controls, k=k)
        panels.draw_ticker(target, self.ticker_rect, vm.events, k=k)
        rects = self.seat_rects(game.num_players)
        for seat, rect in rects.items():
            panels.draw_seat(target, rect, vm.seats[seat], k=k, wide=(seat == 0), codes=self.codes)
        if buttons:
            panels.draw_buttons(target, buttons, k=k, active=active)
        return vm

    def render(self, game: Game, **kw) -> Tuple[pygame.Surface, ViewModel]:
        """A fresh surface with everything drawn on it (screenshots, tests)."""
        surface = pygame.Surface(self.size)
        vm = self.draw(surface, game, **kw)
        return surface, vm


def seats_in_play(num_players: int) -> List[int]:
    """Seat order as the panels lay them out: bottom, then left, then right."""
    return [0] + [s for s in LEFT_SEATS + RIGHT_SEATS if s < num_players]
