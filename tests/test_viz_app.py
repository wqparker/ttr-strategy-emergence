"""The live and replay viewers (ttr.viz.app), driven without a window."""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from ttr.agents import GreedyAgent, RandomAgent  # noqa: E402
from ttr.board import load_board  # noqa: E402
from ttr.game import Game  # noqa: E402
from ttr.record import GameRecord  # noqa: E402
from ttr.simulate import play_game  # noqa: E402
from ttr.viz.app import (  # noqa: E402
    BUTTONS, SPEEDS, Timeline, Viewer, fit_scale, live_timeline, replay_timeline,
)
from ttr.viz.perspective import Perspective  # noqa: E402
from ttr.viz.screen import Screen  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def pg():
    pygame.init()
    yield
    pygame.quit()


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


def test_fit_scale_respects_an_explicit_scale():
    assert fit_scale((1551, 1014), 1.25) == 1.25
    assert 0.5 <= fit_scale((1551, 1014)) <= 1.6
