# Commands

Activate once per terminal — `.venv\Scripts\Activate.ps1` (PowerShell), `.venv\Scripts\activate`
(cmd), `source .venv/Scripts/activate` (bash, `.venv/bin/` on macOS/Linux). Without it, prefix
each command with `.venv/Scripts/`. Roadmap and reasoning: [PLAN.md](PLAN.md).

| Command | Does | Also |
| --- | --- | --- |
| `ttr-view` | the viewer: watch, replay, play or analyze | `python -m ttr.viz.app` |
| `ttr-shot` | one frame of it as a PNG | `python -m ttr.viz.screenshot` |
| `ttr-sim` | bot games in the terminal | `python -m ttr.simulate` |

## ttr-view

```
ttr-view [--record FILE] [--agents NAME ...] [--human SEAT] [--board usa|toy]
         [--perspective all|SEAT] [--memory-level 0|1|2] [--seed N]
         [--max-turns N] [--scale F] [--paused] [--overlay DIR] [--fullscreen]
         [--games N [--advance S]]
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
| `--overlay DIR` | — | color routes by statistics over the records in `DIR` (see below) |
| `--fullscreen` | off | start full screen at the display's resolution (`F11` toggles) |
| `--seed N` | random | live: fix the game; the seed used is printed at start and shown in the legend and title |
| `--games N` | `1` | live: N games in a row; seeds `--seed`, `+1`, …; agents take random seats each game |
| `--advance S` | `5` | with `--games`: seconds the final scoreboard stays up before the next game |

**Keys** (the first six are also buttons, bottom-left; the legend is top-left):

| `space` | `.` `,` | `]` `[` | `v` | `end` `home` | `esc` | `F11` |
| --- | --- | --- | --- | --- | --- | --- |
| play / pause | step fwd / back | faster / slower | perspective | jump to end / start | quit | full screen |

**Seats and colors:** live games deal the agents into random seats and start at a random seat
(a `--human` game keeps `--agents` order). Colors follow the agent, not the seat: the first
agent in `--agents` is red wherever it sits, then blue, green, yellow, black — on its trains,
panel swatch and the scoreboard. Replays of `ttr-sim` records use the same colors.

**Ticket markers:** one seat's open tickets are drawn on the map as a shape beside both of each
ticket's cities, with the same shape leading the ticket in that seat's panel. The marked seat is
the `--perspective` seat, else the `--human` seat, else P0. Hidden while an overlay is on.

- Pairs go out in order: pink moon, yellow star, light green square, light blue circle, orange
  triangle; then the same shapes with colors rotated (yellow moon, light green star, …). No two
  open tickets share a pair.
- A ticket keeps its pair until it is completed; then its markers disappear and the pair is
  free for the next ticket.
- While choosing tickets, ticked ones show hollow markers in the pair they will keep; unticked
  ones show none.
- Every city on a marked ticket (open or ticked) has its dot turned from red to yellow.

**Several games** (`--games N`, e.g. `ttr-view --fullscreen --agents greedy greedy --games 5`):
when a game ends its scoreboard stays up for `--advance` seconds, then the next game starts.
`n` / `p` jump to the next / previous game (a revisited game is as you left it); pausing on the
scoreboard holds it, and `space` there moves on. The legend shows `game 2/5`. A `--human` seat
keeps its seat and the agents keep theirs. Live games only, not `--record` or `--overlay`.

Full screen scales the viewer to the display's height and widens the side panels to fill the
width (1920x1080: scale 1.07, no bars).

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

**Analysis overlays** (`--overlay DIR`): every route colored by a statistic over the saved games
in `DIR`, dark blue (least significant) → light blue → light yellow → light red → dark red (most),
dark gray where there is nothing to measure. Rates run blue = low to red = high; the average turn
runs the other way, red = claimed early. A 0% rate is blue, not gray: it was measured (never
claimed in N games). A turn is one player's whole turn (draw two cards, claim a route, or draw
tickets), counted across all players; the initial ticket choice is not a turn.
Starts paused, replaying the folder's first game (or `--record FILE`) in the panels.

| Key | Does |
| --- | --- |
| `o` | next statistic: claim rate → average turn claimed → contested → off |
| `a` | whose claims: all players → each agent in the records → each seat (P0, P1, …) |
| hover a route | its value, in the legend |

While an overlay is on, each seat panel shows that seat's averages over the records instead of
the replayed game's cards and tickets: games, win %, score, route and ticket points, tickets
completed % (of all tickets it held), longest-route bonus %, routes claimed, trains placed, and its top routes for the
current statistic (most claimed, earliest claims, or doubles it closed). With an agent filter
the panels count only that agent's games in each seat; a seat filter outlines that seat's panel.
`o` to off brings the game's own panels back.

Contested = a double route closed because its other side was claimed (2–3 players). Each
statistic keeps one scale across all players, agents and seats, so switching compares like with
like — a single agent or seat therefore reads paler than everyone together. `ttr-sim` batches
always start with seat 0, so a seat is also a place in turn order.

## ttr-shot

```
ttr-shot --out FILE [--record FILE [--step N]] [--players N] [--turns N] [--seed N]
         [--first-player SEAT] [--perspective all|SEAT] [--memory-level 0|1|2]
         [--scale F] [--fit WxH] [--board-only] [--compare]
ttr-shot --out FILE --overlay DIR [--stat claim_rate|avg_turn|contested] [--agent NAME]
         [--seat N] [--scale F]
```

The viewer's screen without its controls. With no `--record`, greedy agents play `--turns` turns
(default 12) of a seeded game first, so the panels have something to show. Its `--seed` defaults
to 0, not random, so the same command always gives the same image.

| Want | Add |
| --- | --- |
| a seat's view, with card-memory estimates | `--perspective 0` |
| four seats | `--players 4` |
| a moment from a saved game | `--record g.json --step 120` |
| the board alone | `--board-only` |
| a bigger image | `--scale 2.0` (canvas is 1151x764 at 1.0) |
| the full-screen layout | `--fit 1920x1080` |
| a fixed starting seat (default: from the seed) | `--first-player 0` |
| tiles checked against the board photo | `--compare` |
| route statistics over saved games (board only) | `--overlay runs/records --stat avg_turn` |
| the same, one agent's claims | `--agent greedy` |
| the same, one seat's claims (both: that agent in that seat) | `--seat 0` |

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

Each game deals the agents into random seats and starts at a random seat, so every order of play comes up and first-player advantage averages out. The seed is printed at the
start of every run; `--seed N` replays that batch exactly. A record is a seed plus a list of
actions, so `ttr-view --record` and `ttr-shot --record` reproduce the game exactly. A folder of
records feeds `--overlay`; from Python, `ttr.analysis.analyze(DIR).rows()` gives the same
statistics as one row per route, with per-agent (`claim_rate[greedy]`) and per-seat
(`claim_rate[P0]`) columns, and `player_rows()` gives one row per player per game (agent, seat,
win, score breakdown, claims, trains), both ready for pandas:

```python
import pandas as pd
from ttr.analysis import analyze
df = pd.DataFrame(analyze("runs/records").rows())
df.sort_values("claim_rate", ascending=False).head(10)
```

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
python -m pip install -e ".[analysis]"  # pandas, for ttr.analysis rows
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
