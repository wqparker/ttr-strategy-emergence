"""Seat panels: ticket lists with full city names stay inside their boxes."""

import pytest

pygame = pytest.importorskip("pygame")

from ttr.game import GameResult, PlayerResult  # noqa: E402
from ttr.viz import panels  # noqa: E402
from ttr.viz.perspective import Perspective  # noqa: E402

from helpers import started_game  # noqa: E402

SENTINEL = (255, 0, 255)
LONG_NAMES = ["ppo:runs/ppo/pass2/p2b_pool_s3.json", "dqn:runs/dqn/" + "d" * 90 + ".json@best"]


def seat_with_long_tickets(count):
    """P0's facts holding the `count` tickets with the longest city names."""
    game = started_game()
    board = game.board
    longest = sorted(range(len(board.tickets)), key=lambda t: -len(board.tickets[t].a + board.tickets[t].b))
    game.players[0].tickets = longest[:count]
    return Perspective().view(game).seats[0]


def drawn_outside(surface, rect):
    """Whether anything was painted outside `rect` on a sentinel-filled surface."""
    w, h = surface.get_size()
    for x in range(0, w, 2):
        for y in range(0, h, 2):
            if not rect.collidepoint(x, y) and surface.get_at((x, y))[:3] != SENTINEL:
                return True
    return False


@pytest.mark.parametrize("k", [0.6, 1.0])
def test_side_panel_fits_full_names(k):
    facts = seat_with_long_tickets(6)
    surface = pygame.Surface((round(260 * k), round(420 * k)))
    surface.fill(SENTINEL)
    rect = pygame.Rect(round(10 * k), round(10 * k), round(190 * k), round(380 * k))  # SIDE_W - gaps
    panels.draw_seat(surface, rect, facts, k=k)
    assert not drawn_outside(surface, rect)


@pytest.mark.parametrize("k", [0.6, 1.0])
def test_side_panel_fits_a_long_agent_name(k):
    facts = Perspective().view(started_game(), names=LONG_NAMES).seats[1]
    surface = pygame.Surface((round(260 * k), round(420 * k)))
    surface.fill(SENTINEL)
    rect = pygame.Rect(round(10 * k), round(10 * k), round(190 * k), round(380 * k))
    panels.draw_seat(surface, rect, facts, k=k)
    assert not drawn_outside(surface, rect)


@pytest.mark.parametrize("spec,shown", [
    ("ppo:runs/ppo/pass2/p2b_pool_s3.json", "ppo:p2b_pool_s3"),
    ("linear:runs\\linear\\pass7\\p7b_sarsa_lam98_random_s3.json@best", "linear:p7b_sarsa_lam98_random_s3@best"),
    ("greedy", "greedy"),
    ("human", "human"),
])
def test_short_name_keeps_the_run_name(spec, shown):
    assert panels.short_name(spec) == shown


@pytest.mark.parametrize("k", [0.6, 1.0])
def test_scoreboard_names_stay_clear_of_the_score_columns(monkeypatch, k):
    drawn = []
    real = panels.text

    def spy(s, body, pos, size, color=panels.theme.PANEL_TEXT, bold=False, right=False):
        drawn.append((body, pos[0], size, bold))
        return real(s, body, pos, size, color, bold, right)

    monkeypatch.setattr(panels, "text", spy)
    players = [PlayerResult(54, 52, 5, 0, 27, True, 116), PlayerResult(98, -21, 0, 2, 18, False, 77),
               PlayerResult(60, -4, 1, 1, 9, False, 56)]
    surface = pygame.Surface((round(1100 * k), round(700 * k)))
    panels.draw_result(surface, surface.get_rect(), GameResult(players, [0], False), ["human"] + LONG_NAMES, k=k)

    first = next(x for body, x, _, _ in drawn if body == "ROUTES")
    rows = drawn[drawn.index(next(d for d in drawn if d[0] == "TOTAL")) + 1:]  # after the title and headers
    labels = [(body, x, size, bold) for body, x, size, bold in rows if body[:1] == "P" and body[1:2].isdigit()]
    assert [body for body, *_ in labels[:2]] == ["P0 human", "P1 ppo:p2b_pool_s3"]
    assert labels[2][0].startswith("P2 dqn:ddd") and labels[2][0].endswith("…")
    for body, x, size, bold in labels:
        assert x + panels._text_w(body, size, bold) < first


@pytest.mark.parametrize("count", [3, 9, 16])
def test_wide_panel_uses_columns_of_three_and_stays_inside(count):
    facts = seat_with_long_tickets(count)
    surface = pygame.Surface((1600, 160))
    surface.fill(SENTINEL)
    rect = pygame.Rect(20, 10, 1540, 127)  # the bottom panel at scale 1
    panels.draw_seat(surface, rect, facts, wide=True)
    assert not drawn_outside(surface, rect)


@pytest.mark.parametrize("completed,held,shown", [(5, 6, "83%"), (3, 3, "100%"), (0, 4, "0%"), (0, 0, "–")])
def test_completion_percentage(completed, held, shown):
    assert panels.completion_pct(completed, held) == shown


def test_wide_panel_respects_a_right_bound():
    facts = seat_with_long_tickets(9)
    surface = pygame.Surface((1600, 160))
    surface.fill(SENTINEL)
    rect = pygame.Rect(20, 10, 1540, 127)
    right = 20 + round(1540 * 0.62)  # where a human's move chips begin
    panels.draw_seat(surface, rect, facts, wide=True, right=right)
    for x in range(right + 2, rect.right - 8, 2):  # short of the rounded border
        for y in range(rect.top + 34, rect.bottom - 4, 2):  # below the header's "TO ACT"
            assert surface.get_at((x, y))[:3] == pygame.Color(*panels.theme.PANEL_SLOT)[:3]
