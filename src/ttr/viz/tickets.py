"""Ticket markers: one seat's destination tickets shown on the map.

Each ticket gets a colored shape, drawn beside both of its cities and beside
the ticket in that seat's panel, so a glance at the board shows which cities
each ticket joins. Markers go to one seat only: the seat being viewed, or P0 in
the all-seeing view (Screen picks it). Shapes follow the ticket's place in the
seat's list (held, then on offer) and repeat after the fifth.

    draw_ticket_markers(surface, board_view, facts.tickets, origin)
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import pygame

from ttr.viz import theme
from ttr.viz.perspective import TicketFact

# (shape, fill), in the order tickets take them.
MARKERS: Tuple[Tuple[str, theme.RGB], ...] = (
    ("moon", (240, 110, 190)),  # pink
    ("star", (255, 214, 10)),  # yellow
    ("square", (150, 225, 110)),  # light green
    ("circle", (120, 200, 245)),  # light blue
    ("triangle", (255, 150, 40)),  # orange
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


def icon(shape: str, fill: theme.RGB, size: int, hollow: bool = False) -> pygame.Surface:
    """A size x size marker with a dark outline, anti-aliased by drawing it large
    and scaling down. `hollow` draws the outline only, in the fill color (a
    ticket still on offer)."""
    key = (shape, fill, size, hollow)
    cached = _ICONS.get(key)
    if cached is not None:
        return cached
    big = max(4, size) * SUPERSAMPLE
    s = pygame.Surface((big, big), pygame.SRCALPHA)
    _paint(s, shape, fill, big)
    mask = pygame.mask.from_surface(s)
    edge = mask.outline()
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


def placements(board_view, tickets: Sequence[TicketFact]) -> List[Tuple[int, str, Tuple[float, float]]]:
    """(ticket index, city, marker center in canvas units) for every marker.
    Markers sit just outside the city dot on the side away from its label,
    fanned out when several tickets share a city."""
    layout = board_view.layout
    at_city: Dict[str, List[int]] = {}
    for i, t in enumerate(tickets):
        for city in (t.a, t.b):
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


def draw_ticket_markers(target: pygame.Surface, board_view, tickets: Sequence[TicketFact],
                        origin: Tuple[int, int] = (0, 0)) -> None:
    """Draw every ticket's marker beside both its cities on a board drawn at
    `origin`. Tickets still on offer are drawn hollow."""
    size = round(MARKER_SIZE * board_view.scale)
    for i, _city, (cx, cy) in placements(board_view, tickets):
        shape, fill = marker(i)
        img = icon(shape, fill, size, hollow=tickets[i].pending)
        sx, sy = board_view.to_screen((cx, cy))
        target.blit(img, img.get_rect(center=(round(origin[0] + sx), round(origin[1] + sy))))
