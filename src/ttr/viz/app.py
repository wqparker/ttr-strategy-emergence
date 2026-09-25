"""The live and replay viewers (PLAN.md Phase 3, milestone 5).

    ttr-view                                   # live: greedy vs random
    ttr-view --agents greedy greedy greedy     # three seats
    ttr-view --record runs/records/g.json      # replay a saved game
    ttr-view --record g.json --perspective 0   # from one seat's view

(or `python -m ttr.viz.app ...`, which is the same entry point)

Both modes walk a `Timeline` of game states, one per sub-step, so stepping back
is an index move rather than a re-simulation. A replay loads every state from the
record up front; a live game produces the next state on demand by cloning the
last one and letting the seat's agent act, which keeps the same history.

Controls: space play/pause, `.`/`,` step, `]`/`[` speed, `v` perspective,
home/end jump, esc quit. The same actions sit as buttons in the bottom panel.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import pygame

from ttr.agents import Agent
from ttr.board import Board, load_board
from ttr.game import Game
from ttr.record import GameRecord
from ttr.viz.perspective import Perspective, parse_viewer
from ttr.viz.screen import CONTROLS, Screen

# Seconds between sub-steps, slowest first. Index 2 is the default pace.
SPEEDS: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.12, 0.06, 0.02)
# Button labels name the action. They stay ASCII on purpose: the system font
# pygame resolves here has no geometric shapes, and a missing glyph draws a box.
BUTTONS: Tuple[Tuple[str, str], ...] = (
    ("play", "play"),  # reads "pause" while playing
    ("back", "<<"),
    ("forward", ">>"),
    ("slower", "−"),
    ("faster", "+"),
    ("view", "view"),
)


class Timeline:
    """Game states in order, with an optional producer for the next one."""

    def __init__(self, states: Sequence[Game], produce: Optional[Callable[[Game], Optional[Game]]] = None):
        if not states:
            raise ValueError("a timeline needs at least one state")
        self.states: List[Game] = list(states)
        self.produce = produce
        self.index = 0

    @property
    def current(self) -> Game:
        return self.states[self.index]

    @property
    def at_end(self) -> bool:
        return self.index >= len(self.states) - 1 and not self._can_extend()

    def _can_extend(self) -> bool:
        return self.produce is not None and not self.states[-1].game_over

    def forward(self) -> bool:
        if self.index < len(self.states) - 1:
            self.index += 1
            return True
        if not self._can_extend():
            return False
        nxt = self.produce(self.states[-1])  # type: ignore[misc]
        if nxt is None:
            return False
        self.states.append(nxt)
        self.index = len(self.states) - 1
        return True

    def back(self) -> bool:
        if self.index == 0:
            return False
        self.index -= 1
        return True

    def to_start(self) -> None:
        self.index = 0

    def to_end(self) -> None:
        """Jump to the last state, playing a live game out if needed."""
        while self.forward():
            pass


def live_timeline(game: Game, agents: Sequence[Agent]) -> Timeline:
    def produce(last: Game) -> Optional[Game]:
        if last.game_over:
            return None
        nxt = last.clone()
        p = nxt.current_player
        nxt.step(agents[p].act(nxt, p))
        return nxt

    return Timeline([game], produce)


def replay_timeline(record: GameRecord, board: Optional[Board] = None) -> Timeline:
    return Timeline(record.replay_states(board))


class Viewer:
    """Drives a timeline and draws it. Free of the window loop, so the tests can
    feed it events and frames without a display."""

    def __init__(self, timeline: Timeline, screen: Screen, playing: bool = True, speed: int = 2):
        self.timeline = timeline
        self.screen = screen
        self.playing = playing
        self.speed = speed
        self.running = True
        self._next_step = 0.0
        self._buttons: List[Tuple[str, str, pygame.Rect]] = []

    # ---------------------------------------------------------------- actions

    @property
    def interval(self) -> float:
        return SPEEDS[self.speed]

    def toggle_play(self) -> None:
        self.playing = not self.playing and not self.timeline.at_end

    def act(self, name: str) -> None:
        """One control, by name. Stepping by hand pauses, as on any player."""
        if name == "play":
            self.toggle_play()
        elif name == "forward":
            self.playing = False
            self.timeline.forward()
        elif name == "back":
            self.playing = False
            self.timeline.back()
        elif name == "faster":
            self.speed = min(len(SPEEDS) - 1, self.speed + 1)
        elif name == "slower":
            self.speed = max(0, self.speed - 1)
        elif name == "view":
            self.screen.cycle_perspective()
        elif name == "start":
            self.playing = False
            self.timeline.to_start()
        elif name == "end":
            self.playing = False
            self.timeline.to_end()
        elif name == "quit":
            self.running = False

    KEYS = {
        pygame.K_SPACE: "play",
        pygame.K_PERIOD: "forward",
        pygame.K_COMMA: "back",
        pygame.K_RIGHTBRACKET: "faster",
        pygame.K_LEFTBRACKET: "slower",
        pygame.K_v: "view",
        pygame.K_HOME: "start",
        pygame.K_END: "end",
        pygame.K_ESCAPE: "quit",
    }

    def handle(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            name = self.KEYS.get(event.key)
            if name:
                self.act(name)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            name = self.screen.button_at(event.pos, self._buttons)
            if name:
                self.act(name)

    def tick(self, now: float) -> None:
        """Advance the timeline if enough time has passed. `now` is seconds."""
        if not self.playing:
            self._next_step = now + self.interval
            return
        if now < self._next_step:
            return
        self._next_step = now + self.interval
        if not self.timeline.forward():
            self.playing = False

    # ---------------------------------------------------------------- drawing

    def status(self) -> Tuple[Tuple[str, str], ...]:
        """The key legend, plus where the timeline stands, in the same columns."""
        where = f"{self.timeline.index}/{len(self.timeline.states) - 1}"
        state = "playing" if self.playing else ("end" if self.timeline.at_end else "paused")
        return CONTROLS + (("step", where), ("speed", f"x{self.speed + 1}  ·  {state}"))

    def draw(self, target: pygame.Surface) -> None:
        self._buttons = [
            (name, "pause" if name == "play" and self.playing else label, rect)
            for (name, label), rect in zip(BUTTONS, self.screen.button_rects([b[0] for b in BUTTONS]))
        ]
        self.screen.draw(
            target,
            self.timeline.current,
            controls=self.status(),
            buttons=self._buttons,
            active=("play",) if self.playing else (),
        )


# -------------------------------------------------------------------- window


def fit_scale(size: Tuple[int, int], want: Optional[float] = None) -> float:
    """Largest scale whose window fits the display, unless one is asked for."""
    if want:
        return want
    try:
        info = pygame.display.Info()
        room = (info.current_w * 0.92, info.current_h * 0.88)
    except pygame.error:
        return 1.0
    return max(0.5, min(1.6, room[0] / size[0], room[1] / size[1]))


def build(args: argparse.Namespace) -> Tuple[Timeline, Screen]:
    from ttr.simulate import AGENTS  # local: keeps the viewer out of the engine's import path

    if args.record:
        record = GameRecord.load(args.record)
        timeline = replay_timeline(record)
        names = record.agents
        board = timeline.current.board
    else:
        board = load_board(args.board)
        agents = [AGENTS[name](args.seed + i) for i, name in enumerate(args.agents)]
        game = Game(board, num_players=len(args.agents), seed=args.seed, max_turns=args.max_turns)
        timeline = live_timeline(game, agents)
        names = list(args.agents)
    screen = Screen(board, names=names,
                    perspective=Perspective(parse_viewer(args.perspective), args.memory_level))
    return timeline, screen


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, help="replay a saved game instead of one played live")
    parser.add_argument("--agents", nargs="+", default=["greedy", "random"],
                        help="live mode: one agent per seat")
    parser.add_argument("--board", default="usa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--perspective", default="all", help="'all' or a seat number")
    parser.add_argument("--memory-level", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--scale", type=float, default=None, help="default: fit the display")
    parser.add_argument("--paused", action="store_true", help="start paused")
    args = parser.parse_args(argv)

    pygame.init()
    pygame.display.set_caption("Ticket to Ride" + (" — replay" if args.record else " — live"))
    timeline, screen = build(args)
    screen.set_scale(fit_scale(screen.size, args.scale))
    surface = pygame.display.set_mode(screen.size)
    viewer = Viewer(timeline, screen, playing=not args.paused)

    clock = pygame.time.Clock()
    while viewer.running:
        for event in pygame.event.get():
            viewer.handle(event)
        viewer.tick(pygame.time.get_ticks() / 1000)
        viewer.draw(surface)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":  # pragma: no cover
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        raise SystemExit("the viewer needs a real display; unset SDL_VIDEODRIVER")
    main()
