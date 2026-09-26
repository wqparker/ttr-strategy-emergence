"""Ticket markers: one seat's tickets drawn as shapes beside their cities."""

import math

import pytest

pygame = pytest.importorskip("pygame")

from ttr.viz.board_view import BoardView  # noqa: E402
from ttr.viz.perspective import Perspective, TicketFact  # noqa: E402
from ttr.viz.screen import Screen  # noqa: E402
from ttr.viz.tickets import MARKER_SIZE, MARKERS, icon, marker, placements  # noqa: E402

from helpers import started_game  # noqa: E402


def fact(i, a, b, pending=False):
    return TicketFact(id=i, a=a, b=b, points=10, done=False, pending=pending)


def test_markers_cycle_in_the_asked_order():
    assert [shape for shape, _ in MARKERS] == ["moon", "star", "square", "circle", "triangle"]
    assert marker(0) == marker(len(MARKERS)) == MARKERS[0]


@pytest.mark.parametrize("shape,fill", MARKERS)
def test_every_icon_draws_filled_and_hollow(shape, fill):
    solid = icon(shape, fill, 24)
    ring = icon(shape, fill, 24, hollow=True)
    assert solid.get_size() == ring.get_size() == (24, 24)
    count = lambda s: sum(s.get_at((x, y)).a > 128 for x in range(24) for y in range(24))  # noqa: E731
    assert count(solid) > count(ring) > 20


def test_each_ticket_marks_both_cities_and_shared_cities_fan_out():
    game = started_game()
    view = BoardView(game.board, scale=1.0)
    tickets = [fact(0, "Los Angeles", "Miami"), fact(1, "Los Angeles", "New York")]
    spots = placements(view, tickets)
    assert sorted((i, c) for i, c, _ in spots) == [
        (0, "Los Angeles"), (0, "Miami"), (1, "Los Angeles"), (1, "New York"),
    ]
    la = [p for _, c, p in spots if c == "Los Angeles"]
    assert math.dist(*la) >= MARKER_SIZE  # side by side, not overlapping
    city = view.layout.cities["Los Angeles"]
    for p in la:
        assert math.dist(p, city) == pytest.approx(view.layout.city_radius + 2 + MARKER_SIZE / 2)


def star_pixel(screen, game, seat):
    """The window pixel at the center of seat's second ticket's first marker (a star)."""
    tickets = screen.perspective.view(game).seats[seat].tickets
    _, _, center = next(s for s in placements(screen.board_view, tickets) if s[0] == 1)
    x, y = screen.board_view.to_screen(center)
    ox, oy = screen.board_origin
    return round(ox + x), round(oy + y)


@pytest.mark.parametrize("viewer,preferred,expected", [
    (None, None, 0),  # all-seeing: P0 by default
    (None, 1, 1),  # all-seeing with a human seat
    (1, None, 1),  # a seat's view marks that seat
    (1, 0, 1),
])
def test_marked_seat(viewer, preferred, expected):
    game = started_game()
    screen = Screen(game.board, perspective=Perspective(viewer))
    assert screen.marked_seat(game, preferred) == expected


def test_markers_are_drawn_for_the_marked_seat_only():
    game = started_game(seed=4)
    screen = Screen(game.board, scale=1.0)
    star = MARKERS[1][1]
    surface, _ = screen.render(game)
    assert surface.get_at(star_pixel(screen, game, 0))[:3] == star
    plain, _ = screen.render(game, ticket_seat=1)  # P1's tickets now, not P0's
    assert plain.get_at(star_pixel(screen, game, 0))[:3] != star or \
        star_pixel(screen, game, 0) == star_pixel(screen, game, 1)
