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
ttr-view [--record FILE] [--agents NAME ...] [--human SEAT]
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
ttr-sim [--agents AGENT ...] [--games N] [--seed N]
        [--max-turns N] [--show [log|board]] [--step] [--delay S] [--record DIR]
```

| Command | Does |
| --- | --- |
| `ttr-sim --games 100` | batch, summary table |
| `ttr-sim --show` | one game, `rich` text log |
| `ttr-sim --show board --step` | one game, ASCII map, Enter between turns |
| `ttr-sim --games 20 --record runs/records` | save replayable games |
| `ttr-sim --agents linear:runs/linear/q_greedy.json greedy --games 200` | a trained agent vs greedy |

An agent is `random`, `greedy`, `linear:PATH` (trained linear Q / SARSA weights, needs the
`[env]` extra) or `linear:PATH@best` (that run's best checkpoint). `ttr-view --agents` takes the same names.

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

## Training

Linear Q-learning / SARSA on hand-made features (`ttr.learn`, `[env]` extra). One learner seat
against a scripted bot, random seat each game; evaluates greedy play against random and
greedy every `--eval-every` games and prints a line per evaluation.

```
ttr-train-linear --algo q --opponent random --games 2000 --out runs/linear/q_random.json
ttr-train-linear --algo sarsa --opponent greedy --games 2000 --out runs/linear/sarsa_greedy.json
ttr-train-linear --algo q --opponent mixed --seed 1 --games 2000 --out runs/linear/q_mixed_s1.json
ttr-train-linear ... --init runs/linear/q_random.json    # continue from saved weights
ttr-sim --agents linear:runs/linear/q_mixed_s1.json@best greedy   # the run's best checkpoint
```

| Option | Default | |
| --- | --- | --- |
| `--algo` | `q` | `q` (Q-learning) or `sarsa` |
| `--opponent` | `random` | `random`, `greedy`, `mixed` (either, a coin flip per game), or `self` (greedy or the run's own best checkpoint so far, a coin flip per game) |
| `--alpha` | `0.05` | step size (normalized by the feature vector's squared length) |
| `--alpha-end` | off | let alpha fall linearly to this by the last game |
| `--lambda` | `0` | eligibility traces, TD(λ). A game is ~90 decisions, so use 0.97–0.99 to reach the opening; the step is scaled by (1 − λ) so `--alpha` stays comparable |
| `--average GAMES` | off | evaluate, checkpoint and save an exponential average of the weights over about GAMES games (the raw weights are saved too) |
| `--shaping POINTS` | off | potential-based shaping: POINTS per train still needed for my incomplete tickets. Doesn't change the best policy; evaluations report the true score |
| `--epsilon-start` / `--epsilon-end` / `--epsilon-decay` | `0.2` / `0.02` / `0.5` | exploration, falling linearly over that share of the games |
| `--reward` | `margin` | `score`, `margin` or `win` |
| `--eval-every` / `--eval-games` | `200` / `100` | evaluation schedule |
| `--live [N]` | off | also rewrite the run file every N games (25 if no N) and after every evaluation, for `ttr-dash --live`; off means one write at the end |

The output JSON holds the weights by action type with feature names, the config, one row of
game metrics per training game, every evaluation (mean metrics against random and greedy),
the weights at each evaluation, and the random and greedy bots' metrics on the final
evaluation's games, and the best checkpoint (the weights at the evaluation with the best margin
against greedy; play it as `linear:PATH@best`). Name runs `NAME_s<seed>.json` to average seeds
in `ttr-dash --group`. Feature definitions: `src/ttr/learn/features.py`; metrics:
`src/ttr/learn/metrics.py`.

### DQN

Double DQN with a dueling head on the env's 765-number observation and the 168-action mask
(`ttr.learn.dqn`, `[env]` + `[deep]` extras). Same setting as the linear learner: one seat
against an opponent, reward = margin / 100 between its decisions, γ = 1. Gradient steps run on
the GPU (`--device auto`); games are played by CPU copies of the network.

```
ttr-train-dqn --opponent greedy --n-step 8 --average 100 --games 30000 --live 500 --out runs/dqn/n8_s0.json
ttr-sim --agents dqn:runs/dqn/n8_s0.json@best greedy    # the run's best checkpoint
```

| Option | Default | |
| --- | --- | --- |
| `--opponent` | `greedy` | `random`, `greedy`, or `self` (greedy or the run's own best checkpoint so far) |
| `--n-step` | `1` | n-step returns; the counterpart of the linear `--lambda` |
| `--hidden UNITS ...` | `512 256` | hidden layer widths |
| `--no-dueling` | dueling on | a plain Q head instead of value + advantage |
| `--lr` / `--lr-end` | `1e-4` / off | Adam step size, optionally falling linearly |
| `--batch` / `--train-every` | `256` / `8` | one gradient step per 8 learner decisions |
| `--buffer` / `--learning-starts` | `200000` / `10000` | replay size and first step, in decisions |
| `--target-every` | `1000` | gradient steps between target-network copies |
| `--average GAMES` | off | evaluate, checkpoint and save an exponential average of the weights |
| `--epsilon-start` / `--epsilon-end` / `--epsilon-decay` | `1.0` / `0.02` / `0.05` | exploration |
| `--eval-linear SPEC` | pass 7's best linear agent | a third evaluation opponent; `none` to skip |
| `--eval-every` / `--eval-games` | `500` / `100` | evaluation schedule |
| `--memory-level`, `--reward`, `--device`, `--threads`, `--live [N]` | | as named |

`RUN.json` has the same layout as a linear run (`ttr-dash` reads it; no weight pages), with
`"method": "dqn"`; the networks (final, best, raw when averaging) are in `RUN.pt` beside it.
`scripts/rescore.py` and `scripts/run_status.py` take both kinds of run.

## ttr-dash

Agent analysis: a full-screen matplotlib window over one or more training runs (`[analysis]`
extra).

```
ttr-dash runs/linear/q_greedy.json                              # one run
ttr-dash runs/linear/q_greedy.json runs/linear/sarsa_greedy.json   # compare (up to 8)
ttr-dash runs/linear/*.json --save runs/linear/dash               # all pages as PNG, no window
ttr-dash runs/linear/pass2/*.json --group                        # average NAME_s0, NAME_s1, ... into NAME xN
```

**Live view while training.** Start the run with `--live`, then point `ttr-dash --live` at the
same file in a second terminal. The window rereads the file every 3 seconds (`--live 5` for 5)
and redraws the page on screen when it changed; the header shows each run's progress
(`1225/2000 training`) and the time of the last update. It waits for files that don't exist yet,
so it can be started first. Training-game pages move every N games; evaluation curves, the final
table and weight snapshots move at each evaluation (`--eval-every 50` for a finer curve, at the
cost of 200 evaluation games each time).

```
ttr-train-linear --algo q --opponent random --seed 0 --live --out runs/linear/pass3/q_random_s0.json
ttr-dash --live runs/linear/pass3/q_random_s0.json
```

Several seeds at once (PowerShell). Terminal 1 starts three seeds in the background, each logging
to its own file, and waits for them; terminal 2 follows them as one averaged line (drop `--group`
to see each seed):

```powershell
# terminal 1
New-Item -ItemType Directory -Force runs\linear\pass3 | Out-Null
$p = 0..2 | ForEach-Object { Start-Process -NoNewWindow -PassThru -FilePath .venv\Scripts\ttr-train-linear.exe `
    -ArgumentList "--algo q --opponent random --seed $_ --games 2000 --live --out runs\linear\pass3\q_random_s$_.json" `
    -RedirectStandardOutput "runs\linear\pass3\q_random_s$_.log" }
$p | Wait-Process

# terminal 2
.venv\Scripts\ttr-dash.exe --live --group "runs/linear/pass3/q_random_s*.json"
```

`ttr-dash` expands wildcards itself (PowerShell passes `*.json` through unexpanded), and in live
mode re-expands them on every poll, so seed files that appear later are picked up.

A ladder of settings × seeds (the third pass: each step adds one change to the one before), with
the second pass's `q_random` as the reference line:

```powershell
# terminal 1: 4 settings x 3 seeds = 12 runs in parallel (~6-8 min on 12 cores)
New-Item -ItemType Directory -Force runs\linear\pass3 | Out-Null
$runs = [ordered]@{
  "p3a_avg"   = "--alpha 0.02 --average 100"
  "p3b_lam"   = "--alpha 0.02 --average 100 --lambda 0.98"
  "p3c_shape" = "--alpha 0.02 --average 100 --lambda 0.98 --shaping 1"
  "p3d_self"  = "--alpha 0.02 --average 100 --lambda 0.98 --shaping 1 --opponent self"
}
$p = foreach ($name in $runs.Keys) { foreach ($s in 0..2) {
  Start-Process -NoNewWindow -PassThru -FilePath .venv\Scripts\ttr-train-linear.exe `
    -ArgumentList "--algo q --opponent random --seed $s --games 2000 --live $($runs[$name]) --out runs\linear\pass3\${name}_s$s.json" `
    -RedirectStandardOutput "runs\linear\pass3\${name}_s$s.log" -RedirectStandardError "runs\linear\pass3\${name}_s$s.err" } }
$p | Wait-Process

# terminal 2
.venv\Scripts\ttr-dash.exe --live --group "runs/linear/pass3/*.json" "runs/linear/pass2/q_random_s*.json"
```

A later `--opponent` in the arguments overrides the earlier one (`p3d_self`).

A live save of a full 2000-game run takes about 20 ms, so `--live 25` adds about 1–2 s to a
5-minute run.

| Page | Shows |
| --- | --- |
| 1 Overview | evaluation win share and margin over training, training margin, TD error, epsilon, decisions; final evaluation table next to the random and greedy bots on the same games; run settings |
| 2 Behavior | 16 evaluation metrics over training (score, tickets, claims, claim length, ticket draws, final-round ticket draws, ...), bots as reference lines |
| 3 Training games | the same over the training games themselves (exploring), rolling means |
| 4 Action mix | share of turn actions over training; actions per game against the bots |
| 5 Weights | final weights as heatmaps; the weights that moved most, over training |
| 6 Evaluations | raw table of every evaluation |
| 7 Games | raw table of every training game, 32 per page |

Keys: `1`-`7` or `←`/`→` pages, `o` evaluation opponent (greedy / random), `r` next run
(pages 4-7), `PgUp`/`PgDn`/`Home`/`End` raw games, `s` save the page as PNG, `f` full screen,
`q` quit. `--opponent`, `--page` and `--windowed` set the start.

## Tests and setup

```
python -m pytest                     # the whole suite
python -m pytest tests/test_game.py  # one file
python scripts/stress.py             # long random-play invariant run (~1 min)
python scripts/bench_env.py          # env speed: engine, + mask, + observation
python scripts/smoke_env.py          # random agents through the PettingZoo env (--players, --board, --reward)
```

```
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"    # engine + pytest
python -m pip install -e ".[viz]"    # pygame-ce, for the viewer
python -m pip install -e ".[analysis]"  # pandas + matplotlib: ttr.analysis rows, ttr-dash
python -m pip install -e ".[env]"    # numpy, PettingZoo, Gymnasium: the RL env (ttr.env)
python -m pip install -e ".[photo]"  # numpy + OpenCV, board-data tools only
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # GPU PyTorch
python -m pip install -e ".[deep]"   # torch: ttr-train-dqn (install the GPU build first)
```

Any install creates the three commands; re-run one only after changing `[project.scripts]`. The
viz tests render headlessly and skip without pygame; the env tests skip without the `[env]` extra. Testing an old commit needs `PYTHONPATH` —
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
