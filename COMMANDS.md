# Commands

Installing the package puts three commands in the venv: `ttr-view` (the viewer),
`ttr-shot` (a PNG of it) and `ttr-sim` (bot games in the terminal). They are shown
below with the venv path, `.venv/Scripts/` on Windows and `.venv/bin/` elsewhere;
activate the venv and the prefix goes away. Each one is also reachable as
`python -m ttr.viz.app`, `python -m ttr.viz.screenshot` and `python -m ttr.simulate`.

Roadmap and reasoning live in [PLAN.md](PLAN.md).

## Watch a game (viewer window)

```
.venv/Scripts/ttr-view                              # live: greedy vs random
.venv/Scripts/ttr-view --agents greedy greedy greedy random
.venv/Scripts/ttr-view --record runs/records/g.json # replay a saved game
.venv/Scripts/ttr-view --record g.json --perspective 0 --paused
```

| Key | Does |
| --- | --- |
| `space` | play / pause |
| `.` `,` | step forward / back one sub-step |
| `]` `[` | faster / slower |
| `v` | cycle perspective: all-seeing, then each seat |
| `end` `home` | jump to the end / back to the start |
| `esc` | quit |

The same actions are buttons in the bottom-left of the panel, and the legend is
in the top-left of the window. Other flags: `--seed`, `--board toy`,
`--memory-level 0|1|2`, `--scale` (default: fit the display), `--max-turns`.

## See the board (PNG, no window)

```
.venv/Scripts/ttr-shot --out shot.png --scale 1.0 --turns 30
```

Greedy agents play `--turns` turns of a seeded game first, so the panels and the
claimed routes have something to show. Add `--players 4` for four seats.

| Want | Command |
| --- | --- |
| All-seeing view (every hand and ticket) | `--out shot.png` (the default) |
| One seat's view, with card-memory estimates | `--out p0.png --perspective 0` |
| Weaken that seat's memory | `--perspective 0 --memory-level 1` (0, 1 or 2) |
| The board alone, no panels | `--board-only --out board.png` |
| A bigger image | `--scale 2.0` (canvas is 1151x764 at scale 1) |
| A state from a saved game | `--record runs/records/g.json --step 120` |
| Fix who starts (default: drawn from the seed) | `--first-player 0` |
| Check tiles against the board photo | `--compare --out check.png` |

`--compare` blends the render with `docs/ticket-to-ride_usa_map.jpg`, which is
gitignored as publisher artwork and has to be put back locally first.

## Play games in the terminal

```
.venv/Scripts/ttr-sim --games 100              # batch, summary table
.venv/Scripts/ttr-sim --show                   # one game, rich text log
.venv/Scripts/ttr-sim --show board             # one game, ASCII map view
.venv/Scripts/ttr-sim --show board --step      # Enter between turns
.venv/Scripts/ttr-sim --show board --delay 0.3 # or auto-advance
```

Useful flags: `--agents greedy random` (two to five of `greedy`, `random`; the count
sets the number of players), `--board toy`, `--seed N` (the seed is printed at the
start of every run, so any game can be replayed exactly), `--max-turns N`.

## Save and replay games

```
.venv/Scripts/ttr-sim --games 20 --record runs/records
```

A record is the seed plus the list of actions, so it replays the game exactly
(`ttr.record.GameRecord`). Feed one to `ttr-view --record` to watch it, or to
`ttr-shot --record ... --step N` for one moment of it.

## Tests

```
.venv/Scripts/python -m pytest                     # the whole suite
.venv/Scripts/python -m pytest tests/test_game.py  # one file
.venv/Scripts/python scripts/stress.py             # long random-play invariant run (~1 min)
```

The viz tests render headlessly (SDL dummy driver) and skip themselves if pygame is
not installed. To test an old commit, see the note at the end of PLAN.md: the
editable install points at this working tree, so a worktree needs `PYTHONPATH`.

## Setup

```
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # engine + pytest
.venv/Scripts/python -m pip install -e ".[viz]"     # pygame-ce, for the viewer
.venv/Scripts/python -m pip install -e ".[photo]"   # numpy + OpenCV, board-data tools only
```

Any of these installs the three commands. Re-run one after changing
`[project.scripts]` in `pyproject.toml`; editing the code itself needs no reinstall.

## Board data tools

Only needed when the map data changes. The first two need the board photo at
`docs/ticket-to-ride_usa_map.jpg` and the `[photo]` extra. These stay plain scripts
rather than installed commands: they are development tools, not part of the package.

```
.venv/Scripts/python scripts/fit_tiles.py fit             # re-fit every tile to the photo
.venv/Scripts/python scripts/fit_tiles.py size            # measure the tile size
.venv/Scripts/python scripts/fit_tiles.py check           # list the weakest fits
.venv/Scripts/python scripts/fit_tiles.py overlay out.png --zoom 4
.venv/Scripts/python scripts/build_backdrop.py NE_DIR     # rebuild the map backdrop
.venv/Scripts/python scripts/board_checklist.py usa > docs/usa_map_checklist.md
```

`usa.json` is hand-verified against the physical board (`"verified": true`). Check the
board before changing it; re-running `fit` on the committed data moves no tile more
than about a pixel.
