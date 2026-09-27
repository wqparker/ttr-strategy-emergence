"""Analysis overlays (PLAN.md Phase 3, milestone 7): every route colored by a
statistic from `ttr.analysis`, with a legend in the board's lower-left corner.

    summary = analyze("runs/records")
    overlay = make_overlay(summary, "claim_rate", agent="greedy")  # or seat=0
    board_view.draw(surface, None, route_tint=overlay.tint)
    draw_legend(surface, legend_rect(board_view), overlay, k=board_view.scale)

The scale runs dark blue (least significant) through light yellow to dark red
(most). Rates start at 0, so a 0% route is blue: measured, never claimed. The
average turn spans its observed range and runs the other way: an early claim
is red, a late one blue. A statistic shares one scale across the overall view,
every agent and every seat, so switching filters compares like with like. Routes
with nothing to measure (never claimed for the average turn, or not a double for
`contested`) are drawn in a flat dark gray.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pygame

from ttr.analysis import STATS, PlayerSummary, Summary
from ttr.board import Board
from ttr.viz import theme
from ttr.viz.panels import _text_w, fit_label, fitted_text, text

# Least to most significant: dark blue, light blue, light yellow, light red, dark
# red (ColorBrewer RdYlBu / RdBu stops).
RAMP: Tuple[theme.RGB, ...] = (
    (44, 123, 182),
    (171, 217, 233),
    (255, 255, 191),
    (244, 165, 130),
    (178, 24, 43),
)
NO_DATA: theme.RGB = (110, 114, 120)
RATES = ("claim_rate", "contested")
# Statistics where a low value is the significant end: an early claim marks a
# route in demand, so the average turn runs red (early) to blue (late).
REVERSED = ("avg_turn",)

# Legend box in board canvas units, inset from the map's lower-left corner
# (open water on the USA map).
LEGEND_W, LEGEND_H, LEGEND_HOVER_H, LEGEND_INSET = 290.0, 62.0, 16.0, 10.0


def ramp(t: float) -> theme.RGB:
    """The color at `t` in [0, 1] along RAMP (clamped)."""
    t = min(1.0, max(0.0, t))
    pos = t * (len(RAMP) - 1)
    i = min(int(pos), len(RAMP) - 2)
    f = pos - i
    a, b = RAMP[i], RAMP[i + 1]
    return tuple(round(a[j] + (b[j] - a[j]) * f) for j in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class PlayerPanel:
    """What a seat's panel shows while an overlay is on: that seat's average
    results over the records, and its top routes for the overlay's statistic."""

    seat: int
    label: str  # the agent(s) that sat there
    player: Optional[PlayerSummary]  # None: no finished games for this filter
    top_title: str
    top: Tuple[Tuple[str, str], ...]  # (route, value) pairs, most significant first
    selected: bool  # the overlay is filtered to this seat


# Panel caption for each statistic's top-routes list.
TOP_TITLES = {
    "claim_rate": "MOST CLAIMED",
    "avg_turn": "EARLIEST CLAIMS",
    "contested": "DOUBLES IT CLOSED",
}
TOP_N = 8
TOP_GAP = 12.0  # canvas units after a top-routes column


@dataclass(frozen=True)
class Overlay:
    stat: str
    agent: Optional[str]
    seat: Optional[int]
    values: Dict[int, Optional[float]]
    lo: float
    hi: float
    games: int  # games behind the values (the filtered player's games)
    players: Dict[int, PlayerPanel] = field(default_factory=dict, compare=False)
    tint: Dict[int, theme.RGB] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tint", {rid: self.color(v) for rid, v in self.values.items()})

    @property
    def title(self) -> str:
        return f"{STATS[self.stat]} · {filter_label(self.agent, self.seat)}"

    @property
    def reversed(self) -> bool:
        return self.stat in REVERSED

    def color(self, v: Optional[float]) -> theme.RGB:
        if v is None:
            return NO_DATA
        t = (v - self.lo) / (self.hi - self.lo)
        return ramp(1.0 - t if self.reversed else t)

    def fmt(self, v: Optional[float]) -> str:
        if v is None:
            return "no data"
        if self.stat in RATES:
            return f"{100 * v:.0f}%"
        return f"turn {v:.1f}"


Filter = Tuple[Optional[str], Optional[int]]  # (agent, seat); (None, None) = everyone


def filter_label(agent: Optional[str], seat: Optional[int]) -> str:
    """"all players", "greedy", "P1", "greedy as P1"."""
    if agent is None and seat is None:
        return "all players"
    if seat is None:
        return agent  # type: ignore[return-value]
    return f"P{seat}" if agent is None else f"{agent} as P{seat}"


def filters(summary: Summary) -> List[Filter]:
    """What the viewer cycles through: everyone, each agent, then each seat."""
    return [(None, None), *((a, None) for a in summary.agents), *((None, s) for s in summary.seats)]


def make_overlay(summary: Summary, stat: str, agent: Optional[str] = None,
                 seat: Optional[int] = None) -> Overlay:
    values = summary.values(stat, agent, seat)
    family = [summary.values(stat, a, s) for a, s in filters(summary)]
    seen = [v for vs in family for v in vs.values() if v is not None]
    if stat in RATES:
        lo, hi = 0.0, max(seen, default=0.0)
    else:
        lo, hi = min(seen, default=0.0), max(seen, default=0.0)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    overlay = Overlay(stat, agent, seat, values, lo, hi, summary.games_for(agent, seat))
    for s in summary.seats:
        overlay.players[s] = player_panel(summary, overlay, s)
    return overlay


def player_panel(summary: Summary, overlay: Overlay, seat: int) -> PlayerPanel:
    """Seat `seat` under the overlay's agent filter (a seat filter only marks
    the panel as selected; every seat keeps its own numbers)."""
    agent = overlay.agent
    player = summary.player(agent, seat)
    names = sorted(player.agents) if player else ([agent] if agent else [])
    values = summary.values(overlay.stat, agent, seat)
    ranked = sorted(
        ((v, rid) for rid, v in values.items()
         if v is not None and (overlay.stat not in RATES or v > 0)),
        key=lambda vr: (vr[0] if overlay.reversed else -vr[0], vr[1]),
    )[:TOP_N]
    top = tuple((route_name(summary.board, rid), overlay.fmt(v).replace("turn ", ""))
                for v, rid in ranked)
    return PlayerPanel(seat, " / ".join(names), player, TOP_TITLES[overlay.stat], top,
                       selected=overlay.seat == seat)


def route_name(board: Board, route_id: int) -> str:
    """"Omaha – Kansas City", plus the color for one side of a double route."""
    r = board.routes[route_id]
    name = f"{r.a} – {r.b}"
    if r.sibling is not None:
        name += f" ({r.color.value if r.color else 'gray'})"
    return name


# ------------------------------------------------------------------- legend


def legend_rect(board_view, origin: Tuple[int, int] = (0, 0), hover: bool = False) -> pygame.Rect:
    """Where the legend goes, in window pixels, for a board drawn at `origin`."""
    k = board_view.scale
    x, y, _, h = board_view.layout.inner
    height = LEGEND_H + (LEGEND_HOVER_H if hover else 0)
    top = y + h - LEGEND_INSET - height
    return pygame.Rect(round(origin[0] + (x + LEGEND_INSET) * k), round(origin[1] + top * k),
                       round(LEGEND_W * k), round(height * k))


def hover_label(board: Board, overlay: Overlay, route_id: int) -> str:
    """"Denver – Omaha, 4 gray: 62%" for the route under the cursor."""
    r = board.routes[route_id]
    color = r.color.value if r.color else "gray"
    return f"{r.a} – {r.b}, {r.length} {color}: {overlay.fmt(overlay.values[route_id])}"


def draw_legend(s: pygame.Surface, rect: pygame.Rect, overlay: Overlay, k: float = 1.0,
                hover: Optional[str] = None) -> None:
    """Title, the color bar with its end values, the no-data swatch, the game
    count, and the hovered route's value if there is one."""
    pygame.draw.rect(s, theme.PANEL_BG, rect, border_radius=round(5 * k))
    pygame.draw.rect(s, theme.PANEL_EDGE, rect, max(1, round(k)), border_radius=round(5 * k))
    pad = 8 * k
    left, top = rect.left + pad, rect.top + 6 * k
    text(s, overlay.title, (left, top), 11 * k, theme.PANEL_TEXT, bold=True)

    bar = pygame.Rect(round(left), round(top + 18 * k), round(rect.width * 0.56), round(10 * k))
    for i in range(bar.width):
        pygame.draw.line(s, ramp(i / max(1, bar.width - 1)),
                         (bar.left + i, bar.top), (bar.left + i, bar.bottom - 1))
    pygame.draw.rect(s, theme.PANEL_EDGE, bar, max(1, round(k)))
    below = bar.bottom + 2 * k
    # The bar always runs blue -> red; a reversed statistic puts its high end on the left.
    left_v, right_v = (overlay.hi, overlay.lo) if overlay.reversed else (overlay.lo, overlay.hi)
    text(s, overlay.fmt(left_v), (bar.left, below), 9 * k, theme.PANEL_DIM)
    text(s, overlay.fmt(right_v), (bar.right, below), 9 * k, theme.PANEL_DIM, right=True)

    swatch = pygame.Rect(round(bar.right + 12 * k), bar.top, round(14 * k), bar.height)
    pygame.draw.rect(s, NO_DATA, swatch)
    pygame.draw.rect(s, theme.PANEL_EDGE, swatch, max(1, round(k)))
    text(s, "no data", (swatch.right + 5 * k, swatch.top - 2 * k), 9 * k, theme.PANEL_DIM)
    text(s, f"{overlay.games} games", (swatch.left, below), 9 * k, theme.PANEL_DIM)

    if hover:
        label, size = fit_label(hover, rect.width - 2 * pad, 11 * k, k, bold=True)
        text(s, label, (left, rect.bottom - 18 * k), size, theme.HIGHLIGHT, bold=True)


def draw_board_overlay(s: pygame.Surface, board_view, overlay: Overlay,
                       origin: Tuple[int, int] = (0, 0), hover_route: Optional[int] = None) -> None:
    """The board with the overlay's colors and no claimed trains, plus the legend."""
    board_view.draw(s, None, origin=origin, route_tint=overlay.tint,
                    highlight_routes=[hover_route] if hover_route is not None else [])
    hover = hover_label(board_view.board, overlay, hover_route) if hover_route is not None else None
    draw_legend(s, legend_rect(board_view, origin, hover=hover is not None), overlay,
                k=board_view.scale, hover=hover)


# ------------------------------------------------------------- seat panels


def player_items(p: Optional[PlayerSummary]) -> Tuple[Tuple[str, str], ...]:
    """(caption, value) pairs for a seat's analysis panel, averages per game."""
    if p is None:
        return ()
    return (
        ("GAMES", str(p.games)),
        ("WIN %", f"{100 * p.win_rate:.0f}"),
        ("SCORE", f"{p.total:.1f}"),
        ("ROUTE PTS", f"{p.route_points:.1f}"),
        ("TICKET PTS", f"{p.ticket_points:+.1f}"),
        ("COMPLETED %", f"{100 * p.completion_rate:.0f}"),  # of the tickets it held
        ("LONGEST %", f"{100 * p.longest_rate:.0f}"),
        ("CLAIMS", f"{p.claims:.1f}"),
        ("TRAINS", f"{p.trains:.1f}"),
    )


def draw_player_panel(s: pygame.Surface, rect: pygame.Rect, panel: Optional[PlayerPanel],
                      seat: int, k: float = 1.0, wide: bool = False) -> None:
    """A seat's box while an overlay is on: its average results over the records
    in place of the replayed game's cards and tickets, then its top routes for
    the overlay's statistic. `wide` is the full-width bottom panel."""
    selected = panel is not None and panel.selected
    radius = max(2, round(min(rect.width, rect.height) * 0.05))
    pygame.draw.rect(s, theme.PANEL_SLOT, rect, border_radius=radius)
    pygame.draw.rect(s, theme.HIGHLIGHT if selected else theme.PANEL_EDGE, rect, 2, border_radius=radius)
    pad = 9 * k
    x, y = rect.left + pad, rect.top + pad

    swatch = pygame.Rect(round(x), round(y + 2 * k), round(11 * k), round(11 * k))
    pygame.draw.rect(s, theme.seat_color(seat), swatch, border_radius=max(1, round(2 * k)))
    pygame.draw.rect(s, theme.PANEL_DIM, swatch, max(1, round(k)), border_radius=max(1, round(2 * k)))
    label = f"P{seat}" + (f" {panel.label}" if panel and panel.label else "")
    text(s, label, (x + 16 * k, y), 13 * k, theme.PANEL_TEXT, bold=True)
    if selected:
        text(s, "SELECTED", (rect.right - pad, y + 1 * k), 10 * k, theme.HIGHLIGHT, bold=True, right=True)
    y += 20 * k

    items = player_items(panel.player if panel else None)
    if not items:
        text(s, "no games in these records", (x, y + 4 * k), 11 * k, theme.PANEL_DIM)
        return
    if wide:
        col_w, gap = 74 * k, 34 * k
        stats_w = len(items) * col_w
        # Two columns of top routes beside the stats, as wide as their longest
        # line unless that would pass the panel's right edge.
        room = rect.right - pad - (x + stats_w + gap)
        top_w = min(2 * top_col_w(panel, k), room)
        left = max(x, rect.centerx - (stats_w + gap + top_w) / 2)
        left = min(left, rect.right - pad - (stats_w + gap + top_w))
        for i, (name, value) in enumerate(items):
            cx = left + i * col_w
            text(s, name, (cx, y), 9 * k, theme.PANEL_LABEL, bold=True)
            text(s, value, (cx, y + 11 * k), 17 * k, theme.PANEL_TEXT, bold=True)
        _top_routes(s, panel, (left + stats_w + gap, y), k, per_col=4, col_w=top_w / 2)
        return
    # One column in a default side panel; two once full screen has widened it.
    inner = rect.width - 2 * pad
    cols = 2 if inner >= 280 * k else 1
    col_w = inner / cols
    rows = -(-len(items) // cols)
    row_h = 17 * k
    for i, (name, value) in enumerate(items):
        col, row = i // rows, i % rows
        cx, cy = x + col * col_w, y + row * row_h
        text(s, name, (cx, cy + 2 * k), 9 * k, theme.PANEL_LABEL, bold=True)
        right = cx + col_w - (14 * k if col < cols - 1 else 0)
        text(s, value, (right, cy), 12 * k, theme.PANEL_TEXT, bold=True, right=True)
    y += rows * row_h + 8 * k
    room = int((rect.bottom - pad - y - 14 * k) // (15 * k))
    _top_routes(s, panel, (x, y), k, per_col=max(0, min(TOP_N, room)),
                col_w=rect.width - 2 * pad + TOP_GAP * k, cols=1)


def _top_routes(s, panel: PlayerPanel, pos, k: float, per_col: int, col_w: float,
                cols: int = 2) -> None:
    x, y = pos
    text(s, panel.top_title, (x, y), 9 * k, theme.PANEL_LABEL, bold=True)
    if not panel.top:
        text(s, "none", (x, y + 13 * k), 11 * k, theme.PANEL_DIM)
        return
    shown = panel.top[:per_col * cols] if per_col else ()
    size = 11 * k
    for i, (route, value) in enumerate(shown):
        col, row = i // per_col, i % per_col
        lx, ly = x + col * col_w, y + 14 * k + row * 15 * k
        room = col_w - TOP_GAP * k - 8 * k - _text_w(value, size, bold=True)
        fitted_text(s, route, (lx, ly), room, size, theme.PANEL_TEXT, k)
        text(s, value, (lx + col_w - TOP_GAP * k, ly), size, theme.PANEL_TEXT, bold=True, right=True)


def top_col_w(panel: PlayerPanel, k: float) -> float:
    """Width one top-routes column needs for its longest line at full size."""
    size = 11 * k
    return max((_text_w(route, size) + 8 * k + _text_w(value, size, bold=True)
                for route, value in panel.top), default=0.0) + TOP_GAP * k
