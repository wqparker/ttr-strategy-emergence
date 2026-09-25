"""Render the viewer to a PNG without opening a window.

    ttr-shot --out shot.png                   # board + panels
    ttr-shot --out shot.png --perspective 0   # seat 0's view
    ttr-shot --record runs/records/g.json --step 120 --out mid.png
    ttr-shot --board-only --out board.png
    ttr-shot --compare --out check.png        # blend with the board photo

(or `python -m ttr.viz.screenshot ...`, which is the same entry point)

With no --record, greedy agents play `--turns` turns of a seeded game so the
panels and the claimed routes have something to show. --compare overlays the
render on docs/ticket-to-ride_usa_map.jpg (kept locally, not in the repo) at 50%
so positions can be checked against the real board; it implies --board-only.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from ttr.agents import GreedyAgent  # noqa: E402
from ttr.board import load_board  # noqa: E402
from ttr.game import Game  # noqa: E402
from ttr.record import GameRecord  # noqa: E402
from ttr.viz.board_view import BoardView  # noqa: E402
from ttr.viz.perspective import Perspective, parse_viewer  # noqa: E402
from ttr.viz.screen import Screen  # noqa: E402

PHOTO = Path(__file__).resolve().parents[3] / "docs" / "ticket-to-ride_usa_map.jpg"


def demo_game(board_name: str, num_players: int, seed: int, turns: int,
              first_player: Optional[int] = None) -> Game:
    """A seeded game with greedy agents played for `turns` turns. The starting
    seat is drawn from the seed unless `first_player` fixes it (§9 #8), so seat 0
    is not always the one to move."""
    game = Game(load_board(board_name), num_players=num_players, seed=seed, first_player=first_player)
    agents = [GreedyAgent(seed + i) for i in range(num_players)]
    while not game.game_over and game.turn < turns:
        p = game.current_player
        game.step(agents[p].act(game, p))
    return game


def render_board(game: Game, scale: float) -> pygame.Surface:
    return BoardView(game.board, scale=scale).render(game)


def render_screen(game: Game, scale: float, viewer: Optional[int], level: int) -> pygame.Surface:
    screen = Screen(game.board, scale=scale, perspective=Perspective(viewer, level))
    return screen.render(game)[0]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--board", default="usa")
    parser.add_argument("--record", type=Path, help="draw the state from a game record")
    parser.add_argument("--step", type=int, default=None, help="with --record: action index (default: end)")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--turns", type=int, default=12, help="turns to play when there is no --record")
    parser.add_argument("--first-player", type=int, default=None,
                        help="fix the starting seat (default: drawn from the seed)")
    parser.add_argument("--perspective", default="all", help="'all' or a seat number")
    parser.add_argument("--memory-level", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--board-only", action="store_true", help="no panels")
    parser.add_argument("--compare", action="store_true", help="blend with the board photo at 50%%")
    args = parser.parse_args(argv)

    pygame.init()
    if args.record:
        record = GameRecord.load(args.record)
        states = record.replay_states()
        game = states[-1 if args.step is None else args.step]
    else:
        game = demo_game(args.board, args.players, args.seed, args.turns, args.first_player)

    if args.compare or args.board_only:
        surface = render_board(game, 1.0 if args.compare else args.scale)
    else:
        surface = render_screen(game, args.scale, parse_viewer(args.perspective), args.memory_level)

    if args.compare:
        if not PHOTO.exists():
            raise SystemExit(f"--compare needs the board photo at {PHOTO}")
        photo = pygame.transform.smoothscale(pygame.image.load(str(PHOTO)), surface.get_size())
        surface.set_alpha(128)
        photo.blit(surface, (0, 0))
        surface = photo
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(args.out))
    print(f"saved {args.out} {surface.get_size()}")


if __name__ == "__main__":
    main()
