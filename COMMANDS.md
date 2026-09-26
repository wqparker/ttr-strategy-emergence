# Commands

Activate once per terminal — `.venv\Scripts\Activate.ps1` (PowerShell), `.venv\Scripts\activate`
(cmd), `source .venv/Scripts/activate` (bash, `.venv/bin/` on macOS/Linux). Without it, prefix
each command with `.venv/Scripts/`. Roadmap and reasoning: [PLAN.md](PLAN.md).

| Command | Does | Also |
| --- | --- | --- |
| `ttr-view` | the viewer: watch, replay or play | `python -m ttr.viz.app` |
| `ttr-shot` | one frame of it as a PNG | `python -m ttr.viz.screenshot` |
| `ttr-sim` | bot games in the terminal | `python -m ttr.simulate` |

## ttr-view

```
ttr-view [--record FILE] [--agents NAME ...] [--human SEAT] [--board usa|toy]
         [--perspective all|SEAT] [--memory-level 0|1|2] [--seed N]
         [--max-turns N] [--scale F] [--paused]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--record FILE` | — | replay a saved game instead of playing one live |
| `--agents NAME ...` | `greedy random` | one per seat, 2–5 of `greedy`, `random` |
| `--human SEAT` | — | play that seat yourself (live games only) |
| `--perspective` | `all` | `all` sees every hand; a seat sees only its own |
| `--memory-level` | `2` | what a seat's view knows: 0 none, 1 seen cards, 2 + unseen pool |
| `--scale F` | fit display | 1.0 is a 1551x1014 window |
| `--paused` | off | start stopped |

**Keys** (the first six are also buttons, bottom-left; the legend is top-left):

| `space` | `.` `,` | `]` `[` | `v` | `end` `home` | `esc` |
| --- | --- | --- | --- | --- | --- |
| play / pause | step fwd / back | faster / slower | perspective | jump to end / start | quit |

**Playing a seat** (`--human SEAT`): bots play until your turn, then the viewer waits.

| Click | Move |
| --- | --- |
| a route | claim it — payment chips then appear |
| a payment chip | pay for it (`2 red`, `1 red + 1 loco`, …) |
| a face-up card, or the deck | draw it |
| the ticket count | draw tickets — chips tick each one, last chip confirms |

Only legal moves are offered; anything else does nothing. Claimable routes light up under the
cursor. No cancel button: step back with `,` (the rules have no such move). Acting after stepping
back forks, dropping the states that followed. `--human` is ignored with `--record`. `end` plays a
live game out, stopping at your turn. Stepping back never re-simulates. The final scoreboard
draws over the board when the game ends.

## ttr-shot

```
ttr-shot --out FILE [--record FILE [--step N]] [--players N] [--turns N] [--seed N]
         [--first-player SEAT] [--perspective all|SEAT] [--memory-level 0|1|2]
         [--scale F] [--board-only] [--compare]
```

The viewer's screen without its controls. With no `--record`, greedy agents play `--turns` turns
(default 12) of a seeded game first, so the panels have something to show.

| Want | Add |
| --- | --- |
| a seat's view, with card-memory estimates | `--perspective 0` |
| four seats | `--players 4` |
| a moment from a saved game | `--record g.json --step 120` |
| the board alone | `--board-only` |
| a bigger image | `--scale 2.0` (canvas is 1151x764 at 1.0) |
| a fixed starting seat (default: from the seed) | `--first-player 0` |
| tiles checked against the board photo | `--compare` |

`--compare` blends the render with `docs/ticket-to-ride_usa_map.jpg`, gitignored as publisher
artwork and needed locally.

## ttr-sim

```
ttr-sim [--agents {greedy,random} ...] [--games N] [--board usa|toy] [--seed N]
        [--max-turns N] [--show [log|board]] [--step] [--delay S] [--record DIR]
```

| Command | Does |
| --- | --- |
| `ttr-sim --games 100` | batch, summary table |
| `ttr-sim --show` | one game, `rich` text log |
| `ttr-sim --show board --step` | one game, ASCII map, Enter between turns |
| `ttr-sim --games 20 --record runs/records` | save replayable games |

Seats rotate across games so first-player advantage averages out. The seed is printed at the
start of every run; `--seed N` replays that batch exactly. A record is a seed plus a list of
actions, so `ttr-view --record` and `ttr-shot --record` reproduce the game exactly.

## Tests and setup

```
python -m pytest                     # the whole suite
python -m pytest tests/test_game.py  # one file
python scripts/stress.py             # long random-play invariant run (~1 min)
```

```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"    # engine + pytest
python -m pip install -e ".[viz]"    # pygame-ce, for the viewer
python -m pip install -e ".[photo]"  # numpy + OpenCV, board-data tools only
```

Any install creates the three commands; re-run one only after changing `[project.scripts]`. The
viz tests render headlessly and skip without pygame. Testing an old commit needs `PYTHONPATH` —
see the note at the end of PLAN.md.

## Board data tools

Only when the map data changes; the first two need the board photo and the `[photo]` extra.

```
python scripts/fit_tiles.py fit | size | check | overlay out.png --zoom 4
python scripts/build_backdrop.py NE_DIR                 # rebuild the map backdrop
python scripts/board_checklist.py usa > docs/usa_map_checklist.md
```

`usa.json` is hand-verified against the physical board (`"verified": true`) — check the board
before changing it. Re-running `fit` on the committed data moves no tile more than about a pixel.
