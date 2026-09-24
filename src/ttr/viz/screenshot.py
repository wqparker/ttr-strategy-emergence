"""Render the board to a PNG without opening a window.

    python -m ttr.viz.screenshot --out board.png
    python -m ttr.viz.screenshot --record runs/records/g.json --step 120 --out mid.png
    python -m ttr.viz.screenshot --compare --out check.png   # blend with the board photo

--compare overlays the render on docs/ticket-to-ride_usa_map.jpg (kept locally,
not in the repo) at 50% so positions can be checked against the real board.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from ttr.board import load_board  # noqa: E402
from ttr.game import Game  # noqa: E402
from ttr.record import GameRecord  # noqa: E402
from ttr.viz.board_view import BoardView  # noqa: E402

PHOTO = Path(__file__).resolve().parents[3] / "docs" / "ticket-to-ride_usa_map.jpg"


def render(board_name: str, game: Optional[Game], scale: float) -> pygame.Surface:
    board = game.board if game is not None else load_board(board_name)
    view = BoardView(board, scale=scale)
    surface = pygame.Surface(view.size)
    view.draw(surface, game)
    return surface


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--board", default="usa")
    parser.add_argument("--record", type=Path, help="draw the state from a game record")
    parser.add_argument("--step", type=int, default=None, help="with --record: action index (default: end)")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--compare", action="store_true", help="blend with the board photo at 50%%")
    args = parser.parse_args(argv)

    pygame.init()
    game = None
    if args.record:
        record = GameRecord.load(args.record)
        states = record.replay_states()
        game = states[-1 if args.step is None else args.step]
    surface = render(args.board, game, 1.0 if args.compare else args.scale)
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
