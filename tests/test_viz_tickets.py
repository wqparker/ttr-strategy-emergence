"""Ticket markers: one seat's open tickets drawn as shapes beside their cities."""

import math

import pytest

pygame = pytest.importorskip("pygame")

from ttr.actions import DrawTickets, KeepTickets  # noqa: E402
from ttr.viz.board_view import BoardView  # noqa: E402
from ttr.viz.perspective import Perspective  # noqa: E402
from ttr.viz.screen import Screen  # noqa: E402
from ttr.viz.tickets import (  # noqa: E402
    COLORS, MARKER_SIZE, MARKERS, SHAPES, Mark, assign_markers, icon, placements, visible_marks,
)

from helpers import started_game  # noqa: E402
from test_analysis import claim  # noqa: E402


def dealt_game(seed=0):
    """A fresh game, still in the initial ticket choice, P0 to choose."""
    from ttr.board import load_board
    from ttr.game import Game

    return Game(load_board("usa"), num_players=2, seed=seed, first_player=0)


def keep(game, tickets):
    game.step(KeepTickets(frozenset(tickets)))


def pass_turn(game):
    """The current player draws two blind cards, ending the turn."""
    from ttr.actions import DrawBlind

    game.step(DrawBlind())
    game.step(DrawBlind())


# ------------------------------------------------------------------- order


def test_base_pairs_first_then_rotated_colors():
    assert MARKERS[:5] == tuple(zip(SHAPES, COLORS))
    assert MARKERS[5] == ("moon", COLORS[1]) and MARKERS[6] == ("star", COLORS[2])  # yellow moon, green star
    assert len(set(MARKERS)) == len(MARKERS) == 25


@pytest.mark.parametrize("shape,fill", MARKERS[:5])
def test_every_icon_draws_filled_and_hollow(shape, fill):
    solid = icon(shape, fill, 24)
    ring = icon(shape, fill, 24, hollow=True)
    assert solid.get_size() == ring.get_size() == (24, 24)
    count = lambda s: sum(s.get_at((x, y)).a > 128 for x in range(24) for y in range(24))  # noqa: E731
    assert count(solid) > count(ring) > 20


# -------------------------------------------------------------- assignment


def test_offer_reserves_markers_and_kept_tickets_keep_them():
    game = dealt_game()
    offer = list(game.players[0].pending_tickets)
    a = assign_markers(game, 0)
    assert a.held == {} and a.offered == {t: i for i, t in enumerate(offer)}
    keep(game, [offer[0], offer[2]])  # return the middle one
    a = assign_markers(game, 0)
    assert a.held == {offer[0]: 0, offer[2]: 2} and a.offered == {}


def test_new_tickets_fill_the_first_free_markers_without_repeats():
    game = dealt_game()
    offer = list(game.players[0].pending_tickets)
    keep(game, [offer[0], offer[2]])  # markers 0 and 2; 1 is free
    keep(game, game.players[1].pending_tickets)
    game.step(DrawTickets())
    new = list(game.players[0].pending_tickets)
    assert assign_markers(game, 0).offered == dict(zip(new, [1, 3, 4]))
    keep(game, new)
    held = assign_markers(game, 0).held
    assert sorted(held.values()) == [0, 1, 2, 3, 4]
    pass_turn(game)
    game.step(DrawTickets())  # all five base pairs taken: rotated colors next
    assert sorted(assign_markers(game, 0).offered.values()) == [5, 6, 7]


def path_routes(board, a, b):
    """Route ids along a fewest-hops path from a to b (breadth-first)."""
    from collections import deque

    prev = {a: None}
    queue = deque([a])
    while queue:
        city = queue.popleft()
        for r in board.routes:
            if city in (r.a, r.b):
                nxt = r.b if city == r.a else r.a
                if nxt not in prev:
                    prev[nxt] = (city, r.id)
                    queue.append(nxt)
    out, city = [], b
    while prev[city] is not None:
        city, rid = prev[city]
        out.append(rid)
    return out


def test_completing_a_ticket_frees_its_marker_for_the_next_offer():
    game = started_game(seed=1)
    tid, index = min(assign_markers(game, 0).held.items(), key=lambda kv: kv[1])
    t = game.board.tickets[tid]
    for rid in path_routes(game.board, t.a, t.b):
        claim(game, rid)  # P0 claims
        pass_turn(game)  # P1 draws
    a = assign_markers(game, 0)
    assert tid not in a.held and index not in a.held.values()
    assert tid not in visible_marks(game, 0)  # no marker once completed
    game.step(DrawTickets())
    assert min(assign_markers(game, 0).offered.values()) == min(
        i for i in range(len(MARKERS)) if i not in a.held.values()
    )


def test_selection_hides_unticked_offers():
    game = dealt_game()
    offer = list(game.players[0].pending_tickets)
    assert set(visible_marks(game, 0)) == set(offer)  # no selection given: all shown
    marks = visible_marks(game, 0, selected={offer[1]})
    assert set(marks) == {offer[1]} and marks[offer[1]].hollow
    assert marks[offer[1]].index == 1  # the same pair it keeps once confirmed
    keep(game, [offer[1], offer[2]])
    marks = visible_marks(game, 0)
    assert marks[offer[1]].index == 1 and not marks[offer[1]].hollow


def test_markers_survive_stepping_through_a_replay():
    game = started_game(seed=3)
    before = assign_markers(game, 0).held
    copy = game.clone()
    pass_turn(copy)
    assert assign_markers(copy, 0).held == before
    assert assign_markers(game, 0).held == before  # the earlier state is unchanged


# -------------------------------------------------------------- placement


def mark(i, a, b, index=0):
    return Mark(i, a, b, index, hollow=False)


def test_each_ticket_marks_both_cities_and_shared_cities_fan_out():
    game = started_game()
    view = BoardView(game.board, scale=1.0)
    marks = [mark(0, "Los Angeles", "Miami"), mark(1, "Los Angeles", "New York", 1)]
    spots = placements(view, marks)
    assert sorted((i, c) for i, c, _ in spots) == [
        (0, "Los Angeles"), (0, "Miami"), (1, "Los Angeles"), (1, "New York"),
    ]
    la = [p for _, c, p in spots if c == "Los Angeles"]
    assert math.dist(*la) >= MARKER_SIZE  # side by side, not overlapping
    city = view.layout.cities["Los Angeles"]
    for p in la:
        assert math.dist(p, city) == pytest.approx(view.layout.city_radius + 2 + MARKER_SIZE / 2)


# ------------------------------------------------------------------ screen


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


def marker_pixel(screen, game, seat, tid):
    marks = list(visible_marks(game, seat).values())
    i = next(i for i, m in enumerate(marks) if m.ticket == tid)
    _, _, center = next(s for s in placements(screen.board_view, marks) if s[0] == i)
    x, y = screen.board_view.to_screen(center)
    ox, oy = screen.board_origin
    return round(ox + x), round(oy + y)


def test_markers_are_drawn_for_the_marked_seat():
    game = started_game(seed=4)
    screen = Screen(game.board, scale=1.0)
    star = next(t for t, i in assign_markers(game, 0).held.items() if i == 1)  # yellow star
    surface, _ = screen.render(game)
    assert surface.get_at(marker_pixel(screen, game, 0, star))[:3] == COLORS[1]


def city_pixel(screen, city):
    """A point on the city dot's lower right, clear of the highlight."""
    x, y = screen.board_view.layout.cities[city]
    r = screen.board_view.layout.city_radius * 0.5
    sx, sy = screen.board_view.to_screen((x + r, y + r))
    ox, oy = screen.board_origin
    return round(ox + sx), round(oy + sy)


def test_ticket_cities_turn_yellow():
    from ttr.viz import theme

    game = started_game(seed=4)
    screen = Screen(game.board, scale=2.0)
    surface, _ = screen.render(game)
    marked = {c for m in visible_marks(game, 0).values() for c in (m.a, m.b)}
    other = next(c for c in game.board.cities if c not in marked)
    near = lambda got, want: all(abs(g - w) <= 6 for g, w in zip(got, want))  # noqa: E731
    for city in marked:  # the overlay layer is alpha-blended, so allow rounding
        assert near(surface.get_at(city_pixel(screen, city))[:3], theme.CITY_TICKET)
    assert near(surface.get_at(city_pixel(screen, other))[:3], theme.CITY_FILL)
