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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pygame

from ttr.actions import Action, ClaimRoute, DrawBlind, DrawFaceUp, DrawTickets, KeepTickets, Pass, Pay
from ttr.agents import Agent
from ttr.board import Board, load_board
from ttr.game import Game, Phase
from ttr.record import GameRecord
from ttr.viz.perspective import Perspective, code_map, parse_viewer
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

    def apply(self, action: Action) -> None:
        """Play `action` from the state on screen. Stepping back and then acting
        forks: the states after this one are dropped, as an undo would."""
        del self.states[self.index + 1:]
        nxt = self.current.clone()
        nxt.step(action)
        self.states.append(nxt)
        self.index = len(self.states) - 1

    def to_start(self) -> None:
        self.index = 0

    def to_end(self) -> None:
        """Jump to the last state, playing a live game out if needed."""
        while self.forward():
            pass


def live_timeline(game: Game, agents: Sequence[Agent], human: Sequence[int] = ()) -> Timeline:
    """A game the bots play themselves. A human seat produces nothing, so the
    timeline stalls there until a click supplies the move."""
    seats = set(human)

    def produce(last: Game) -> Optional[Game]:
        if last.game_over or last.current_player in seats:
            return None
        nxt = last.clone()
        p = nxt.current_player
        nxt.step(agents[p].act(nxt, p))
        return nxt

    return Timeline([game], produce)


def replay_timeline(record: GameRecord, board: Optional[Board] = None) -> Timeline:
    return Timeline(record.replay_states(board))


def pay_label(game: Game, action: Pay) -> str:
    """"4 red", "3 red + 1 loco", "4 loco"."""
    route = game.board.routes[game.pending_route]
    plain = route.length - action.locomotives
    parts = []
    if plain:
        parts.append(f"{plain} {action.color.value}")
    if action.locomotives:
        parts.append(f"{action.locomotives} loco")
    return " + ".join(parts)


class HumanControl:
    """Turns clicks into actions for the seats a person plays.

    Every move it offers comes from `game.legal_actions()`, so the UI can never
    propose something the rules forbid. Keeping tickets is the one sub-step with
    no single click: the chips toggle a selection, and the last chip confirms it.
    """

    def __init__(self, seats: Sequence[int] = ()) -> None:
        self.seats = set(seats)
        self._codes: Dict[str, str] = {}
        self.keep: set = set()  # ticket ids ticked in the current keep sub-step
        self._keep_for: Optional[int] = None  # log length the selection belongs to

    def acts_now(self, game: Game) -> bool:
        return not game.game_over and game.current_player in self.seats

    # ------------------------------------------------------------- the chips

    def _pending(self, game: Game) -> List[int]:
        return list(game.players[game.current_player].pending_tickets)

    def _sync_keep(self, game: Game) -> None:
        """Start a fresh selection (all kept) for each new batch of tickets."""
        if self._keep_for != len(game.log):
            self._keep_for = len(game.log)
            self.keep = set(self._pending(game))

    def code(self, game: Game, city: str) -> str:
        if not self._codes:
            self._codes = code_map(game.board)
        return self._codes[city]

    def title(self, game: Game) -> str:
        if game.phase is Phase.CHOOSE_PAYMENT:
            route = game.board.routes[game.pending_route]
            return f"PAY FOR {self.code(game, route.a)}–{self.code(game, route.b)}"
        if game.phase in (Phase.KEEP_TICKETS, Phase.CHOOSE_INITIAL_TICKETS):
            return "KEEP WHICH TICKETS?"
        return "YOUR MOVE"

    def choices(self, game: Game) -> List[Tuple[str, bool, bool]]:
        """(label, selected, enabled) per chip, for the bottom panel."""
        if not self.acts_now(game):
            return []
        legal = game.legal_actions()
        if game.phase is Phase.CHOOSE_PAYMENT:
            return [(pay_label(game, a), False, True) for a in legal if isinstance(a, Pay)]
        if game.phase in (Phase.KEEP_TICKETS, Phase.CHOOSE_INITIAL_TICKETS):
            self._sync_keep(game)
            out = []
            for tid in self._pending(game):
                t = game.board.tickets[tid]
                a, b = self.code(game, t.a), self.code(game, t.b)
                out.append((f"{a}–{b}  {t.points}", tid in self.keep, True))
            out.append((f"keep {len(self.keep)}", False, self._keep_action(game) in legal))
            return out
        out = []
        if any(isinstance(a, DrawTickets) for a in legal):
            out.append(("draw tickets", False, True))
        if any(isinstance(a, Pass) for a in legal):
            out.append(("pass", False, True))
        return out

    def _keep_action(self, game: Game) -> KeepTickets:
        return KeepTickets(frozenset(self.keep))

    # ------------------------------------------------------------- the clicks

    def click_choice(self, game: Game, index: int) -> Optional[Action]:
        if not self.acts_now(game):
            return None
        legal = game.legal_actions()
        if game.phase is Phase.CHOOSE_PAYMENT:
            pays = [a for a in legal if isinstance(a, Pay)]
            return pays[index] if index < len(pays) else None
        if game.phase in (Phase.KEEP_TICKETS, Phase.CHOOSE_INITIAL_TICKETS):
            self._sync_keep(game)
            pending = self._pending(game)
            if index < len(pending):  # a ticket chip toggles
                tid = pending[index]
                self.keep.symmetric_difference_update({tid})
                return None
            action = self._keep_action(game)
            return action if action in legal else None
        labels = [label for label, _, _ in self.choices(game)]
        if index < len(labels):
            wanted = DrawTickets() if labels[index] == "draw tickets" else Pass()
            return wanted if wanted in legal else None
        return None

    def click_market(self, game: Game, slot: int) -> Optional[Action]:
        if not self.acts_now(game) or slot >= len(game.market):
            return None
        action = DrawFaceUp(game.market[slot])
        return action if action in game.legal_actions() else None

    def click_deck(self, game: Game) -> Optional[Action]:
        if not self.acts_now(game):
            return None
        return DrawBlind() if DrawBlind() in game.legal_actions() else None

    def click_ticket_deck(self, game: Game) -> Optional[Action]:
        if not self.acts_now(game):
            return None
        return DrawTickets() if DrawTickets() in game.legal_actions() else None

    def click_route(self, game: Game, route_id: Optional[int]) -> Optional[Action]:
        if route_id is None or not self.acts_now(game):
            return None
        action = ClaimRoute(route_id)
        return action if action in game.legal_actions() else None

    def claimable(self, game: Game) -> List[int]:
        """Routes this seat could claim right now, for the board highlight."""
        if not self.acts_now(game) or game.phase is not Phase.CHOOSE_ACTION:
            return []
        return [a.route_id for a in game.legal_actions() if isinstance(a, ClaimRoute)]


class Viewer:
    """Drives a timeline and draws it. Free of the window loop, so the tests can
    feed it events and frames without a display."""

    def __init__(self, timeline: Timeline, screen: Screen, playing: bool = True, speed: int = 2,
                 human: Optional[HumanControl] = None):
        self.timeline = timeline
        self.screen = screen
        self.playing = playing
        self.speed = speed
        self.human = human or HumanControl()
        self.running = True
        self.hover: Optional[int] = None  # route under the cursor
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
        elif event.type == pygame.MOUSEMOTION:
            self.hover = self._route_at(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            name = self.screen.button_at(event.pos, self._buttons)
            if name:
                self.act(name)
            else:
                self.click(event.pos)

    def _route_at(self, pos) -> Optional[int]:
        point = self.screen.board_point(pos)
        return self.screen.board_view.layout.route_at(point) if point else None

    def click(self, pos) -> Optional[Action]:
        """A click outside the transport buttons: a human move, or nothing."""
        game = self.timeline.current
        if not self.human.acts_now(game):
            return None
        hit = self.screen.hit(pos)
        if hit is None:
            action = self.human.click_route(game, self._route_at(pos))
        else:
            kind, index = hit
            action = {
                "market": lambda: self.human.click_market(game, index),
                "deck": lambda: self.human.click_deck(game),
                "tickets": lambda: self.human.click_ticket_deck(game),
                "choice": lambda: self.human.click_choice(game, index),
            }[kind]()
        if action is not None:
            self.playing = False
            self.timeline.apply(action)
        return action

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
        game = self.timeline.current
        self._buttons = [
            (name, "pause" if name == "play" and self.playing else label, rect)
            for (name, label), rect in zip(BUTTONS, self.screen.button_rects([b[0] for b in BUTTONS]))
        ]
        claimable = self.human.claimable(game)
        highlight = [self.hover] if self.hover in claimable else []
        self.screen.draw(
            target,
            game,
            controls=self.status(),
            buttons=self._buttons,
            active=("play",) if self.playing else (),
            highlight_routes=highlight,
            choices=self.human.choices(game),
            choices_title=self.human.title(game),
            result=game.result,
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


def build(args: argparse.Namespace) -> Tuple[Timeline, Screen, HumanControl]:
    from ttr.simulate import AGENTS  # local: keeps the viewer out of the engine's import path

    human = [] if args.human is None else [args.human]
    if args.record:
        record = GameRecord.load(args.record)
        timeline = replay_timeline(record)
        names = record.agents
        board = timeline.current.board
        human = []  # a replay is fixed; nothing to play
    else:
        board = load_board(args.board)
        agents = [AGENTS[name](args.seed + i) for i, name in enumerate(args.agents)]
        game = Game(board, num_players=len(args.agents), seed=args.seed, max_turns=args.max_turns)
        timeline = live_timeline(game, agents, human)
        names = [("human" if i in human else name) for i, name in enumerate(args.agents)]
    screen = Screen(board, names=names,
                    perspective=Perspective(parse_viewer(args.perspective), args.memory_level))
    return timeline, screen, HumanControl(human)


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
    parser.add_argument("--human", type=int, default=None, metavar="SEAT",
                        help="play a seat yourself (live games only), e.g. --human 0")
    parser.add_argument("--paused", action="store_true", help="start paused")
    args = parser.parse_args(argv)

    pygame.init()
    pygame.display.set_caption("Ticket to Ride" + (" — replay" if args.record else " — live"))
    timeline, screen, human = build(args)
    screen.set_scale(fit_scale(screen.size, args.scale))
    surface = pygame.display.set_mode(screen.size)
    viewer = Viewer(timeline, screen, playing=not args.paused, human=human)

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
