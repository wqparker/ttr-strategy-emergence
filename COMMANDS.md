# Commands

Activate the venv once per terminal, then every command below works as written:

```
.venv\Scripts\Activate.ps1     # PowerShell
.venv\Scripts\activate         # cmd
source .venv/Scripts/activate   # Git Bash (.venv/bin/activate on macOS/Linux)
```

Without activating, put `.venv/Scripts/` in front of each command
(`.venv/bin/` on macOS/Linux).

Installing the package puts three commands in the venv: `ttr-view` (the viewer),
`ttr-shot` (a PNG of it) and `ttr-sim` (bot games in the terminal). Each is also
reachable as `python -m ttr.viz.app`, `python -m ttr.viz.screenshot` and
`python -m ttr.simulate`.

Roadmap and reasoning live in [PLAN.md](PLAN.md).

## Watch a game (viewer window)

```
ttr-view                              # live: greedy vs random
ttr-view --agents greedy greedy greedy random
ttr-view --human 0                    # play seat 0 yourself
ttr-view --record runs/records/g.json # replay a saved game
ttr-view --record g.json --perspective 0 --paused
```

| Key | Does |
| --- | --- |
| `space` | play / pause |
| `.` `,` | step forward / back one sub-step |
| `]` `[` | faster / slower (six speeds, 1s to 0.02s per sub-step) |
| `v` | cycle perspective: all-seeing, then each seat |
| `end` `home` | jump to the end / back to the start |
| `esc` | quit |

The same six are buttons in the bottom-left of the panel, and the key legend is
down the left of the top strip, with the step count and speed beside it. In a live
game `end` plays the rest of it out; with `--human` it stops when it is your turn.
Stepping back never re-simulates — every state is kept — so scrubbing is instant
either way.

Flags: `--agents` (one per seat, `greedy` or `random`, two to five), `--seed`,
`--board toy`, `--perspective all|SEAT`, `--memory-level 0|1|2`, `--scale`
(default: fit the display), `--max-turns`, `--paused` to start stopped.

### What is on screen

- **Top strip:** the key legend and step count on the left, the five face-up cards and
  the deck / discard / ticket counts in the middle, and on the right whose view this is,
  the turn, and the current sub-step. In a seat's view the unseen pool sits beside the
  pile counts.
- **Board:** claimed spaces carry the owner's color under a rail-and-crosstie pattern,
  so a claim never reads as an unclaimed tile of the same color.
- **Sides:** the other seats, two per side, with trains, cards, tickets and score. In a
  seat's view opponents show card-memory bounds (`≥2 red`, `+3 unknown`) instead of hands.
- **Bottom:** seat 0 in full, with the transport buttons on the left and, when you are
  playing, your move chips on the right.
- **Between them:** a ticker of the last three public events.

### Playing a seat yourself

`--human SEAT` in a live game. The bots play until it is your turn, then the viewer
waits for a click:

| Click | Move |
| --- | --- |
| A route on the board | Claims it; the payment options then appear as chips |
| A payment chip | Pays for the pending route (`2 red`, `1 red + 1 loco`, …) |
| A face-up card, or the deck count | Draws that card; a second draw follows if the rules allow one |
| The ticket count | Draws tickets; chips then tick each one, and the last chip confirms |

Only legal moves are offered: a route you cannot pay for, a click during a bot's
turn, or confirming fewer tickets than the minimum all do nothing. Claimable routes
light up under the cursor. To back out of a claim, step back with `,` — there is no
cancel, because the rules have no such move once a route is chosen. Acting after
stepping back forks: the states that followed are dropped.

`--human` is ignored with `--record`; a replay is a fixed list of moves.

When the game ends, the final scoreboard is drawn over the board: the winner, then
every seat's route points, ticket points with completed/failed counts, longest path,
bonus and total.

## See the board (PNG, no window)

```
ttr-shot --out shot.png --scale 1.0 --turns 30
```

Greedy agents play `--turns` turns of a seeded game first, so the panels and the
claimed routes have something to show. Add `--players 4` for four seats. This is the
same screen as the viewer without its controls: no key legend, no buttons.

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
ttr-sim --games 100              # batch, summary table
ttr-sim --show                   # one game, rich text log
ttr-sim --show board             # one game, ASCII map view
ttr-sim --show board --step      # Enter between turns
ttr-sim --show board --delay 0.3 # or auto-advance
```

Useful flags: `--agents greedy random` (two to five of `greedy`, `random`; the count
sets the number of players), `--board toy`, `--seed N` (the seed is printed at the
start of every run, so any game can be replayed exactly), `--max-turns N`.

## Save and replay games

```
ttr-sim --games 20 --record runs/records
```

A record is the seed plus the list of actions, so it replays the game exactly
(`ttr.record.GameRecord`). Feed one to `ttr-view --record` to watch it, or to
`ttr-shot --record ... --step N` for one moment of it.

## Tests

```
python -m pytest                     # the whole suite
python -m pytest tests/test_game.py  # one file
python scripts/stress.py             # long random-play invariant run (~1 min)
```

The viz tests render headlessly (SDL dummy driver) and skip themselves if pygame is
not installed. To test an old commit, see the note at the end of PLAN.md: the
editable install points at this working tree, so a worktree needs `PYTHONPATH`.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1            # activate, then the rest is unprefixed
python -m pip install -e ".[dev]"     # engine + pytest
python -m pip install -e ".[viz]"     # pygame-ce, for the viewer
python -m pip install -e ".[photo]"   # numpy + OpenCV, board-data tools only
```

Any of these installs the three commands. Re-run one after changing
`[project.scripts]` in `pyproject.toml`; editing the code itself needs no reinstall.

## Board data tools

Only needed when the map data changes. The first two need the board photo at
`docs/ticket-to-ride_usa_map.jpg` and the `[photo]` extra. These stay plain scripts
rather than installed commands: they are development tools, not part of the package.

```
python scripts/fit_tiles.py fit             # re-fit every tile to the photo
python scripts/fit_tiles.py size            # measure the tile size
python scripts/fit_tiles.py check           # list the weakest fits
python scripts/fit_tiles.py overlay out.png --zoom 4
python scripts/build_backdrop.py NE_DIR     # rebuild the map backdrop
python scripts/board_checklist.py usa > docs/usa_map_checklist.md
```

`usa.json` is hand-verified against the physical board (`"verified": true`). Check the
board before changing it; re-running `fit` on the committed data moves no tile more
than about a pixel.
