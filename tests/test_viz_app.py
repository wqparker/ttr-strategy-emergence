"""The live and replay viewers (ttr.viz.app), driven without a window."""

from argparse import Namespace

import pytest

pygame = pytest.importorskip("pygame")

from ttr.agents import GreedyAgent, RandomAgent  # noqa: E402
from ttr.board import load_board  # noqa: E402
from ttr.game import Game  # noqa: E402
from ttr.record import GameRecord  # noqa: E402
from ttr.simulate import play_game  # noqa: E402
from ttr.viz.app import (  # noqa: E402
    BUTTONS, SPEEDS, Timeline, Viewer, build, fit_scale, layout_for, live_timeline, replay_timeline,
)
from ttr.viz.perspective import Perspective  # noqa: E402
from ttr.viz.screen import SIDE_W, Screen  # noqa: E402


@pytest.fixture(scope="module")
def record():
    board, seed = "usa", 7
    game = Game(load_board(board), num_players=2, seed=seed, first_player=0, max_turns=1000)
    actions = []
    play_game(game, [GreedyAgent(seed), RandomAgent(seed + 1)], actions_out=actions)
    return GameRecord.from_game(game, actions, agents=["greedy", "random"], board=board, seed=seed)


def live(num_players=2, seed=3):
    agents = [GreedyAgent(seed + i) for i in range(num_players)]
    game = Game(load_board("usa"), num_players=num_players, seed=seed, max_turns=1000)
    return live_timeline(game, agents)


def viewer(timeline, **kw):
    screen = Screen(timeline.current.board, scale=0.5, **kw)
    return Viewer(timeline, screen)


# ------------------------------------------------------------------ timeline


def test_replay_walks_every_substep(record):
    timeline = replay_timeline(record)
    assert len(timeline.states) == len(record.actions) + 1
    assert timeline.index == 0 and not timeline.at_end

    assert timeline.forward() and timeline.index == 1
    assert timeline.back() and timeline.index == 0
    assert not timeline.back()  # already at the first state

    timeline.to_end()
    assert timeline.at_end and timeline.current.game_over
    assert not timeline.forward()
    timeline.to_start()
    assert timeline.index == 0 and not timeline.current.game_over


def test_replay_end_matches_the_record(record):
    timeline = replay_timeline(record)
    timeline.to_end()
    final = record.replay()
    assert timeline.current.route_owner == final.route_owner
    assert timeline.current.result.winners == final.result.winners


def test_live_extends_on_demand_and_keeps_history():
    timeline = live()
    assert len(timeline.states) == 1
    for _ in range(12):
        assert timeline.forward()
    assert len(timeline.states) == 13

    kept = timeline.states[8]
    for _ in range(4):
        timeline.back()
    assert timeline.index == 8 and timeline.current is kept
    # Going forward again reuses the stored state; it is not re-simulated.
    timeline.forward()
    assert timeline.current is timeline.states[9]
    assert len(timeline.states) == 13


def test_live_stops_at_game_over():
    timeline = live(seed=5)
    timeline.to_end()
    assert timeline.current.game_over and timeline.at_end
    assert not timeline.forward()


def test_empty_timeline_is_rejected():
    with pytest.raises(ValueError):
        Timeline([])


# -------------------------------------------------------------------- viewer


def test_tick_advances_on_the_clock():
    v = viewer(live())
    v.playing = True
    v.tick(0.0)
    first = v.timeline.index
    v.tick(v.interval / 2)  # too soon
    assert v.timeline.index == first
    v.tick(v.interval * 1.5)
    assert v.timeline.index == first + 1


def test_pause_stops_the_clock():
    v = viewer(live())
    v.playing = False
    for i in range(20):
        v.tick(i * 1.0)
    assert v.timeline.index == 0


def test_stepping_by_hand_pauses():
    v = viewer(live())
    v.playing = True
    v.act("forward")
    assert not v.playing and v.timeline.index == 1
    v.act("back")
    assert v.timeline.index == 0


def test_play_stops_at_the_end(record):
    v = viewer(replay_timeline(record))
    v.timeline.index = len(v.timeline.states) - 2
    v.playing = True
    v.tick(0.0)
    v.tick(100.0)
    v.tick(200.0)
    assert v.timeline.at_end and not v.playing


def test_speed_is_bounded():
    v = viewer(live())
    for _ in range(20):
        v.act("faster")
    assert v.interval == min(SPEEDS)
    for _ in range(20):
        v.act("slower")
    assert v.interval == max(SPEEDS)


def test_keys_drive_the_viewer():
    v = viewer(live(num_players=3))
    v.playing = False
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    assert v.playing
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PERIOD))
    assert v.timeline.index == 1 and not v.playing
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_COMMA))
    assert v.timeline.index == 0

    v.draw(pygame.Surface(v.screen.size))  # the screen learns the seat count
    assert v.screen.perspective.viewer is None
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v))
    assert v.screen.perspective.viewer == 0

    assert v.running
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert not v.running
    v.handle(pygame.event.Event(pygame.QUIT))
    assert not v.running


def test_buttons_are_clickable():
    v = viewer(live())
    v.playing = False
    v.draw(pygame.Surface(v.screen.size))  # lays the buttons out
    assert [name for name, _, _ in v._buttons] == [name for name, _ in BUTTONS]

    rects = {name: rect for name, _, rect in v._buttons}
    v.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rects["forward"].center))
    assert v.timeline.index == 1
    v.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rects["play"].center))
    assert v.playing
    # A click on the board is not a button.
    v.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=v.screen.board_origin))
    assert v.playing


def test_play_button_reads_pause_while_playing():
    v = viewer(live())
    v.playing = True
    v.draw(pygame.Surface(v.screen.size))
    assert dict((name, label) for name, label, _ in v._buttons)["play"] == "pause"
    v.act("play")
    v.draw(pygame.Surface(v.screen.size))
    assert dict((name, label) for name, label, _ in v._buttons)["play"] == "play"


def test_draw_renders_the_whole_screen(record):
    v = viewer(replay_timeline(record), perspective=Perspective(0))
    v.timeline.forward()
    surface = pygame.Surface(v.screen.size)
    v.draw(surface)
    assert surface.get_size() == v.screen.size
    keys = [k for k, _ in v.status()]
    assert "space" in keys and "step" in keys  # legend plus position


def test_fullscreen_layout_and_back():
    v = viewer(live())
    size, offset = layout_for(v.screen, True, display=(1920, 1080))
    assert size == (1920, 1080) and v.screen.size[0] == 1920
    assert offset == (0, (1080 - v.screen.size[1]) // 2)
    size, offset = layout_for(v.screen, False, want=0.5)
    assert offset == (0, 0) and size == v.screen.size
    assert v.screen.side_w == SIDE_W and v.screen.scale == 0.5


def test_f11_toggles_fullscreen():
    v = viewer(live())
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F11))
    assert v.fullscreen
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F11))
    assert not v.fullscreen


def test_mouse_positions_account_for_the_offset():
    v = viewer(live())
    v.playing = False
    v.draw(pygame.Surface(v.screen.size))
    rect = {name: r for name, _, r in v._buttons}["forward"]
    v.offset = (100, 40)
    v.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center))
    assert v.timeline.index == 0  # the unshifted spot is no longer the button
    v.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                pos=(rect.centerx + 100, rect.centery + 40)))
    assert v.timeline.index == 1


# -------------------------------------------------------------------- series


def series_viewer(games=3, agents=("greedy", "random"), advance=2.0, **kw):
    args = Namespace(record=None, overlay=None, agents=list(agents), board="usa", seed=5,
                     max_turns=1000, perspective="all", memory_level=2, human=None, games=games)
    for key, value in kw.items():
        setattr(args, key, value)
    timeline, screen, human, summary, series = build(args)
    screen.set_scale(0.5)
    return Viewer(timeline, screen, playing=True, human=human, series=series, advance=advance)


def test_one_game_is_not_a_series():
    v = series_viewer(games=1)
    assert v.series is None
    assert not any(k == "game" for k, _ in v.status())


def test_single_live_game_keeps_its_old_seeds():
    """--games 1 plays exactly what ttr-view always played for a seed."""
    v = series_viewer(games=1)
    board = v.timeline.current.board
    old = Game(board, num_players=2, seed=5, max_turns=1000)
    assert v.timeline.current.players[0].pending_tickets == old.players[0].pending_tickets


def test_agents_rotate_seats_between_games():
    v = series_viewer(games=3)
    assert v.screen.names == ["greedy", "random"]
    v.act("next_game")
    assert v.game_no == 1 and v.screen.names == ["random", "greedy"]
    v.act("next_game")
    assert v.screen.names == ["greedy", "random"]
    assert not v.next_game()  # no fourth game
    v.act("prev_game")
    assert v.game_no == 1


def test_finished_game_hands_over_after_the_delay():
    v = series_viewer(games=2, advance=2.0)
    v.timeline.to_end()
    first = v.timeline
    assert first.current.game_over
    v.tick(0.0)  # the step that finds the end schedules the hand-over for t=2
    assert v.game_no == 0 and v.playing
    assert dict(v.status())["speed"].endswith("next game soon")
    v.tick(1.9)
    assert v.game_no == 0  # the scoreboard is still up
    v.tick(2.1)
    assert v.game_no == 1 and v.timeline is not first and v.playing
    # Going back finds the first game as it was left.
    v.act("prev_game")
    assert v.timeline is first and first.current.game_over


def test_last_game_just_stops():
    v = series_viewer(games=2)
    v.show_game(1)
    v.timeline.to_end()
    v.tick(0.0)
    v.tick(1.0)
    assert not v.playing and v.game_no == 1


def test_pausing_cancels_the_hand_over():
    v = series_viewer(games=2, advance=2.0)
    v.timeline.to_end()
    v.tick(0.0)
    v.tick(1.0)
    v.act("play")  # pause on the scoreboard
    v.tick(10.0)
    assert v.game_no == 0 and not v.playing
    v.act("play")  # play again at the end of a finished game: moves on
    assert v.playing
    v.tick(11.0)
    v.tick(13.5)
    assert v.game_no == 1


def test_series_keys_and_legend():
    v = series_viewer(games=4)
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
    assert v.game_no == 1
    status = dict(v.status())
    assert status["game"] == "2/4" and "n p" in status
    v.draw(pygame.Surface(v.screen.size))


def test_legend_shows_the_seed_of_the_game_on_screen():
    v = series_viewer(games=3)
    v.seed = 5
    assert dict(v.status())["seed"] == "5"
    v.next_game()
    assert dict(v.status())["seed"] == "6"
    assert "seed" not in dict(viewer(live()).status())  # not given: not shown


def test_caption_names_the_seed():
    from ttr.viz.app import caption

    v = series_viewer(games=3)
    args = Namespace(overlay=None, record=None, seed=5)
    v.next_game()
    assert caption(args, v) == "Ticket to Ride — live · game 2/3 · seed 6"
    assert caption(args, series_viewer(games=1)) == "Ticket to Ride — live · seed 5"


def test_seed_defaults_to_random(monkeypatch, capsys):
    """Without --seed, main draws one and prints how to replay it."""
    import ttr.viz.app as app

    seen = {}

    def fake_build(args):
        seen["seed"] = args.seed
        raise SystemExit(0)  # stop before a window opens

    monkeypatch.setattr(app, "build", fake_build)
    monkeypatch.setattr(app.random, "randrange", lambda n: 424242)
    with pytest.raises(SystemExit):
        app.main(["--games", "3"])
    assert seen["seed"] == 424242
    assert "rerun with --seed 424242 --games 3" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        app.main(["--seed", "7"])
    assert seen["seed"] == 7


def test_human_seat_stays_put():
    v = series_viewer(games=2, human=0)
    assert v.screen.names == ["human", "random"]
    v.next_game()
    assert v.screen.names == ["human", "random"]


def test_games_cannot_replay_records(record, tmp_path):
    path = record.save(tmp_path / "g.json")
    with pytest.raises(SystemExit):
        series_viewer(games=2, record=path)


def test_fit_scale_respects_an_explicit_scale():
    assert fit_scale((1551, 1014), 1.25) == 1.25
    assert 0.5 <= fit_scale((1551, 1014)) <= 1.6
