"""Analysis overlays: statistic -> colors, the legend, and the viewer's keys."""

from argparse import Namespace

import pytest

pygame = pytest.importorskip("pygame")

from ttr.analysis import STATS, Summary, analyze  # noqa: E402
from ttr.board import load_board  # noqa: E402
from ttr.simulate import run_matches  # noqa: E402
from ttr.viz.app import Viewer, build  # noqa: E402
from ttr.viz import theme  # noqa: E402
from ttr.viz.board_view import BoardView  # noqa: E402
from ttr.viz.overlay import (  # noqa: E402
    NO_DATA, RAMP, draw_board_overlay, filters, legend_rect, make_overlay, ramp,
)
from ttr.viz.screen import OVERLAY_CONTROLS  # noqa: E402


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    folder = tmp_path_factory.mktemp("records")
    run_matches(["greedy", "random"], 6, load_board("usa"), seed=5, record_dir=folder, board_ref="usa")
    return folder


@pytest.fixture(scope="module")
def summary(records):
    return analyze(records)


def test_ramp_ends_and_clamps():
    assert ramp(0.0) == RAMP[0] and ramp(1.0) == RAMP[-1]
    for i, stop in enumerate(RAMP):  # evenly spaced stops
        assert ramp(i / (len(RAMP) - 1)) == stop
    assert ramp(-3) == RAMP[0] and ramp(7) == RAMP[-1]


def test_zero_rate_is_data_not_no_data(summary):
    overlay = make_overlay(summary, "claim_rate")
    assert overlay.color(0.0) == RAMP[0] != NO_DATA


@pytest.mark.parametrize("stat", list(STATS))
def test_every_route_gets_a_color(summary, stat):
    overlay = make_overlay(summary, stat)
    assert set(overlay.tint) == set(summary.routes)
    for rid, v in overlay.values.items():
        assert (overlay.tint[rid] == NO_DATA) == (v is None)
    assert overlay.lo < overlay.hi


def test_rates_start_at_zero_and_share_a_scale_across_filters(summary):
    overall = make_overlay(summary, "claim_rate")
    others = [make_overlay(summary, "claim_rate", a, s) for a, s in filters(summary)]
    assert overall.lo == 0.0
    assert all((o.lo, o.hi) == (overall.lo, overall.hi) for o in others)
    assert make_overlay(summary, "claim_rate", summary.agents[0]).games == \
        summary.games_by_agent[summary.agents[0]]
    assert make_overlay(summary, "claim_rate", seat=1).title.endswith("· P1")


def test_filters_list_everyone_then_agents_then_seats(summary):
    assert filters(summary) == [(None, None), ("greedy", None), ("random", None), (None, 0), (None, 1)]


def test_highest_value_gets_the_darkest_color(summary):
    overlay = make_overlay(summary, "claim_rate")
    top = max((v, rid) for rid, v in overlay.values.items() if v is not None)[1]
    assert overlay.tint[top] == RAMP[-1]


def test_average_turn_runs_early_red_to_late_blue(summary):
    overlay = make_overlay(summary, "avg_turn")
    assert overlay.color(overlay.lo) == RAMP[-1]
    assert overlay.color(overlay.hi) == RAMP[0]


def test_player_panels_follow_the_filter(summary):
    overlay = make_overlay(summary, "claim_rate")
    assert sorted(overlay.players) == summary.seats
    p0 = overlay.players[0]
    assert p0.player == summary.player(seat=0) and p0.label == "greedy / random"
    assert not p0.selected and p0.top_title == "MOST CLAIMED"
    # Top routes: this seat's own claim rates, highest first, none at 0%.
    values = summary.values("claim_rate", seat=0)
    best = max(v for v in values.values() if v is not None)
    assert p0.top and p0.top[0][1] == f"{100 * best:.0f}%"

    by_agent = make_overlay(summary, "claim_rate", agent="random")
    assert by_agent.players[1].player == summary.player("random", 1)
    assert by_agent.players[1].label == "random"
    assert make_overlay(summary, "claim_rate", seat=1).players[1].selected


def test_average_turn_panels_list_the_earliest_claims(summary):
    overlay = make_overlay(summary, "avg_turn")
    turns = [float(v) for _, v in overlay.players[0].top]
    assert turns == sorted(turns) and overlay.players[0].top_title == "EARLIEST CLAIMS"


def test_screen_draws_player_panels(summary, records):
    from ttr.viz.screen import Screen
    from helpers import started_game

    game = started_game()
    overlay = make_overlay(summary, "contested", seat=0)
    for scale in (0.6, 1.2):
        screen = Screen(game.board, scale=scale)
        surface, _ = screen.render(game, overlay=overlay)
        rect = screen.seat_rects(2)[0]
        # The selected seat's box is outlined in the highlight color.
        assert surface.get_at((rect.centerx, rect.top))[:3] == theme.HIGHLIGHT
    empty = make_overlay(Summary(load_board("usa")), "claim_rate")
    Screen(game.board, scale=0.5).render(game, overlay=empty)  # no seats in the records


def test_empty_summary_still_draws():
    summary = Summary(load_board("usa"))  # no games: every value is None
    overlay = make_overlay(summary, "avg_turn")
    assert set(overlay.tint.values()) == {NO_DATA}


def test_board_overlay_renders_legend(summary):
    view = BoardView(summary.board, scale=0.6)
    surface = pygame.Surface(view.size)
    overlay = make_overlay(summary, "contested")
    draw_board_overlay(surface, view, overlay, hover_route=0)
    rect = legend_rect(view, hover=True)
    assert surface.get_rect().contains(rect)
    assert surface.get_at((rect.left + 3, rect.centery))[:3] != NO_DATA


def test_screenshot_cli_overlay(records, tmp_path):
    from ttr.viz.screenshot import main

    out = tmp_path / "overlay.png"
    main(["--out", str(out), "--scale", "0.5", "--overlay", str(records),
          "--stat", "avg_turn", "--agent", "greedy", "--seat", "1"])
    assert out.exists() and out.stat().st_size > 10_000
    with pytest.raises(SystemExit):
        main(["--out", str(out), "--overlay", str(records), "--agent", "nobody"])
    with pytest.raises(SystemExit):
        main(["--out", str(out), "--overlay", str(records), "--seat", "4"])


# -------------------------------------------------------------------- viewer


def overlay_viewer(records):
    args = Namespace(record=None, overlay=records, agents=["greedy", "random"], board="usa",
                     seed=0, max_turns=1000, perspective="all", memory_level=2, human=None)
    timeline, screen, human, summary = build(args)
    screen.set_scale(0.5)
    return Viewer(timeline, screen, playing=False, human=human, summary=summary, overlay_on=True)


def test_viewer_replays_the_first_record_with_the_overlay_on(records):
    v = overlay_viewer(records)
    assert v.summary.games == 6
    assert len(v.timeline.states) > 1  # a replay, not a fresh live game
    assert v.overlay is not None and v.overlay.stat == next(iter(STATS))
    assert all(entry in v.status() for entry in OVERLAY_CONTROLS)


def test_overlay_keys_cycle(records):
    v = overlay_viewer(records)
    seen = [v.stat]
    for _ in range(len(STATS) + 1):
        v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_o))
        seen.append(v.stat)
        if v.stat is None:
            assert v.overlay is None
    assert seen == list(STATS) + [None, next(iter(STATS))]  # each statistic, off, around

    options = filters(v.summary)
    seen = []
    for _ in range(len(options)):
        seen.append(v.filter)
        v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
    assert seen == options
    assert v.filter == (None, None)


def test_overlay_draws_with_hover(records):
    v = overlay_viewer(records)
    v.hover = 0
    v.timeline.to_end()
    surface = pygame.Surface(v.screen.size)
    v.draw(surface)  # overlay on at game over: legend drawn, no scoreboard
    v.cycle_filter()
    assert v.overlay.agent == v.summary.agents[0]
    v.draw(surface)
    v.filter = (None, 1)
    assert v.overlay.seat == 1
    v.draw(surface)


def test_no_summary_means_no_overlay_keys(records):
    v = overlay_viewer(records)
    v.summary = None
    v.act("overlay")
    v.act("filter")
    assert v.overlay is None
    assert not any(entry in v.status() for entry in OVERLAY_CONTROLS)
