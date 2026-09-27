"""Ticket markers: one seat's open destination tickets shown on the map.

Each open ticket gets a marker, a (shape, color) pair drawn beside both of its
cities and leading the ticket in that seat's panel, so a glance at the board
shows which cities each ticket joins. Markers go to one seat only: the seat
being viewed, else the human seat, else P0 (Screen picks it).

Assignment rules:

- Markers come in a fixed order: the five base pairs first (pink moon, yellow
  star, light green square, light blue circle, orange triangle), then the same
  shapes with the colors rotated one step (yellow moon, light green star, ...),
  and so on: 25 distinct pairs.
- A ticket takes the first pair no open ticket is using, and keeps it until it
  is completed. Then its marker disappears and the pair is free again.
- A ticket offer reserves pairs for every offered ticket as it is dealt, so a
  ticket shows the same pair while it is being chosen and after it is kept.
  Offered tickets are drawn hollow, and only those currently selected.

Everything is rebuilt from the game log (`assign_markers`), so stepping back
and forth through a replay gives the same markers every time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Sequence, Tuple

import pygame

from ttr.game import Game
from ttr.scoring import connected
from ttr.viz import theme

SHAPES: Tuple[str, ...] = ("moon", "star", "square", "circle", "triangle")
COLORS: Tuple[theme.RGB, ...] = (
    (240, 110, 190),  # pink
    (255, 214, 10),  # yellow
    (150, 225, 110),  # light green
    (120, 200, 245),  # light blue
    (255, 150, 40),  # orange
)
# Base pairs first, then each shape with the colors rotated one step further.
MARKERS: Tuple[Tuple[str, theme.RGB], ...] = tuple(
    (SHAPES[i], COLORS[(i + turn) % len(COLORS)])
    for turn in range(len(COLORS))
    for i in range(len(SHAPES))
)
OUTLINE: theme.RGB = (25, 25, 30)
SUPERSAMPLE = 4

# Canvas units: marker size, gap from the city dot, and the gap between two
# markers sharing a city.
MARKER_SIZE = 22.0
MARKER_GAP = 2.0
MARKER_SPACING = 1.0

_ICONS: Dict[Tuple[str, theme.RGB, int, bool], pygame.Surface] = {}


def marker(index: int) -> Tuple[str, theme.RGB]:
    return MARKERS[index % len(MARKERS)]


# --------------------------------------------------------------- assignment


@dataclass(frozen=True)
class Assignment:
    held: Dict[int, int]  # open (uncompleted) held ticket id -> marker index
    offered: Dict[int, int]  # ticket on offer now -> the marker it would keep


def assign_markers(game: Game, seat: int) -> Assignment:
    """Replay `seat`'s ticket history from the log: offers reserve the first
    free markers, kept tickets take their reservation, and a ticket frees its
    marker when a claim completes it (or at once, if it is kept complete)."""
    board = game.board
    held: Dict[int, int] = {}
    offered: Dict[int, int] = {}
    routes = []

    def free() -> List[int]:
        used = set(held.values())
        out = [i for i in range(len(MARKERS)) if i not in used]
        return out or list(range(len(MARKERS)))  # more open tickets than pairs: reuse

    def drop_completed() -> None:
        for tid in [t for t in held if connected(routes, board.tickets[t].a, board.tickets[t].b)]:
            del held[tid]

    for event in game.log:
        if event.player != seat:
            continue
        if event.kind in ("deal_initial_tickets", "draw_tickets"):
            slots = free()
            offered = {t: slots[i % len(slots)] for i, t in enumerate(event.private["tickets"])}
        elif event.kind in ("keep_initial_tickets", "keep_tickets"):
            for t in event.private["tickets"]:
                held[t] = offered[t]
            offered = {}
            drop_completed()
        elif event.kind == "claim_route":
            routes.append(board.routes[event.public["route"]])
            drop_completed()
    pending = game.players[seat].pending_tickets
    return Assignment(held, {t: offered[t] for t in pending if t in offered})


@dataclass(frozen=True)
class Mark:
    ticket: int
    a: str
    b: str
    index: int  # into MARKERS
    hollow: bool  # on offer, not yet kept


def visible_marks(game: Game, seat: int, selected: Optional[Collection[int]] = None) -> Dict[int, Mark]:
    """Ticket id -> the marker to draw, for open held tickets and for offered
    tickets that are selected (`selected` None means every offered ticket, as
    for a bot, whose choice isn't shown until made)."""
    a = assign_markers(game, seat)
    out: Dict[int, Mark] = {}
    for tid, index in a.held.items():
        t = game.board.tickets[tid]
        out[tid] = Mark(tid, t.a, t.b, index, hollow=False)
    for tid, index in a.offered.items():
        if selected is None or tid in selected:
            t = game.board.tickets[tid]
            out[tid] = Mark(tid, t.a, t.b, index, hollow=True)
    return out


# ------------------------------------------------------------------ drawing


def icon(shape: str, fill: theme.RGB, size: int, hollow: bool = False) -> pygame.Surface:
    """A size x size marker with a dark outline, anti-aliased by drawing it large
    and scaling down. `hollow` draws the outline only, in the fill color."""
    key = (shape, fill, size, hollow)
    cached = _ICONS.get(key)
    if cached is not None:
        return cached
    big = max(4, size) * SUPERSAMPLE
    s = pygame.Surface((big, big), pygame.SRCALPHA)
    _paint(s, shape, fill, big)
    edge = pygame.mask.from_surface(s).outline()
    if hollow:
        s.fill((0, 0, 0, 0))
    if len(edge) > 2:
        width = max(2, round(big * (0.16 if hollow else 0.09)))
        pygame.draw.lines(s, fill if hollow else OUTLINE, True, edge, width)
        if hollow:  # a thin dark rim keeps the colored ring readable on any background
            pygame.draw.lines(s, OUTLINE, True, edge, max(1, width // 3))
    out = pygame.transform.smoothscale(s, (max(4, size), max(4, size)))
    _ICONS[key] = out
    return out


def mark_icon(mark: Mark, size: int) -> pygame.Surface:
    shape, fill = marker(mark.index)
    return icon(shape, fill, size, hollow=mark.hollow)


def _paint(s: pygame.Surface, shape: str, fill: theme.RGB, big: int) -> None:
    c = big / 2
    r = big * 0.44
    if shape == "circle":
        pygame.draw.circle(s, fill, (c, c), r * 0.92)
    elif shape == "square":
        side = r * 1.6
        pygame.draw.rect(s, fill, pygame.Rect(round(c - side / 2), round(c - side / 2), round(side), round(side)))
    elif shape == "triangle":
        pts = [(c + r * math.cos(a), c + r * 1.05 * math.sin(a) + r * 0.12)
               for a in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)]
        pygame.draw.polygon(s, fill, pts)
    elif shape == "star":
        pts = []
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append((c + rr * math.cos(a), c + rr * math.sin(a) + r * 0.06))
        pygame.draw.polygon(s, fill, pts)
    elif shape == "moon":
        pygame.draw.circle(s, fill, (c, c), r)
        # Cut a second disc out of the first: drawing with a transparent color
        # replaces the pixels rather than blending over them.
        pygame.draw.circle(s, (0, 0, 0, 0), (c + r * 0.55, c - r * 0.3), r * 0.72)
    else:
        raise ValueError(f"unknown marker shape {shape!r}")


def placements(board_view, marks: Sequence[Mark]) -> List[Tuple[int, str, Tuple[float, float]]]:
    """(position in `marks`, city, marker center in canvas units) for every
    marker. Markers sit just outside the city dot on the side away from its
    label, fanned out so markers sharing a city don't overlap."""
    layout = board_view.layout
    at_city: Dict[str, List[int]] = {}
    for i, m in enumerate(marks):
        for city in (m.a, m.b):
            at_city.setdefault(city, []).append(i)
    out = []
    dist = layout.city_radius + MARKER_GAP + MARKER_SIZE / 2
    # The angle at which neighbours on that circle clear each other.
    spread = 2 * math.asin(min(1.0, (MARKER_SIZE + MARKER_SPACING) / (2 * dist)))
    for city, indices in at_city.items():
        x, y = layout.cities[city]
        lx, ly = layout.labels.get(city, (x, y - 1))
        base = math.atan2(y - ly, x - lx)  # pointing away from the label
        n = len(indices)
        for j, i in enumerate(indices):
            a = base + (j - (n - 1) / 2) * spread
            out.append((i, city, (x + dist * math.cos(a), y + dist * math.sin(a))))
    return out


def draw_ticket_markers(target: pygame.Surface, board_view, marks: Sequence[Mark],
                        origin: Tuple[int, int] = (0, 0)) -> None:
    """Draw each mark beside both its cities on a board drawn at `origin`."""
    size = round(MARKER_SIZE * board_view.scale)
    for i, _city, (cx, cy) in placements(board_view, marks):
        img = mark_icon(marks[i], size)
        sx, sy = board_view.to_screen((cx, cy))
        target.blit(img, img.get_rect(center=(round(origin[0] + sx), round(origin[1] + sy))))
