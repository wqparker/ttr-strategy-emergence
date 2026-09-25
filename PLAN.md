# Project Plan & Roadmap

This is the living plan: the goals, the decisions made so far with their reasoning, the
roadmap, and the questions still open. Session-by-session notes go in `DEVLOG.md`.

## Goal

Recreate Ticket to Ride as a multi-agent RL environment, train agents on it, and study
the strategies that emerge. The analysis of what the agents converge on matters as much
as the engineering.

Research questions to explore:

- **Route-hoarding vs. blocking**: do agents learn to take routes their opponents need,
  or do they focus only on their own tickets?
- **Risk tolerance on routes**: do they play long, high-value routes that tie up many
  cards, or short, safe ones?
- **Ticket risk**: do they draw extra destination tickets late in the game? How many do
  they keep, and how often does it backfire?
- **Tempo**: how do they balance drawing cards against claiming routes before
  someone else takes them?
- **Value of memory**: how much does card counting (memory Level 0 vs. Level 2) change
  play and win rate?

## Scope decisions

### Start with 2 players

The first experiments use two-player games. Credit assignment is simpler and interactions
are easier to interpret. Scaling to 3–5 players is a later phase. The engine should
support N players from the start so that step needs no rewrite.

### Full map and full ruleset from the start

Bulk development targets the **full base game on the USA map**. The full rules, with
rulings for edge cases, are in [docs/RULES.md](docs/RULES.md). In summary:

- 45 trains per player
- 110 train cards: 12 of each of the 8 colors, plus 14 locomotives
- Face-up market of 5 cards. If 3 locomotives are ever face up, all 5 go to the
  **discard pile** and 5 new cards are dealt from the draw deck. The discard pile is
  shuffled into a new draw deck only when the deck runs out.
  - Keep resetting until fewer than 3 locomotives are face up. The only exception is when
    the remaining cards mathematically can't form a legal market; see RULES.md §9 #2.
    Possible future safeguard: guarantee a clean deal after 1–2 resets in a row.
- Drawing a face-up locomotive uses the whole draw turn, and a face-up locomotive can't
  be taken as the second draw.
- Destination tickets: deal 3 at the start and keep at least 2; later draws take 3 and
  keep at least 1
- Double routes: the second track of a double route is **only in play with 4–5
  players**. In 2–3 player games, once one side is claimed the other is closed to
  everyone. With 4–5 players, both sides are open, but one player can't claim both.
  - Implementation: store the map's double routes as pairs, and apply the rule by player
    count so the same board data works for every game size.
- Longest continuous path bonus (10 points)
- End trigger: when a player has 2 or fewer trains left, every player gets one final turn
- Scoring: route points, plus or minus ticket values, plus the longest-route bonus

**Why not a simplified ruleset:** agents trained on a simplified game can learn strategies
that work there but fail in the real game. Simplifying the board and mechanics while
keeping the core gameplay intact is also a hard design problem of its own. The time is
better spent on the real thing.

**Toy map only as a smoke test:** a tiny map exists only to check that the engine, the
environment and the training loop run end to end. The board is loaded from data files,
so the toy map is just another data file with the same rules. It gets no separate
development or tuning.

## Current progress

*Updated at the end of each session. Last updated: 2026-09-25.*

- **Just completed: engine ready, and every RL design question settled.**
  - Phases 0–2 (engine, random/greedy bots, `rich` log, temporary ASCII board view,
    seat-rotated match runner, stress test) merged into `main`.
  - USA map data hand-verified against the physical board (`"verified": true`).
  - Moved to Python 3.11.
  - Settled: card memory (Level 2), the methods to compare (CleanRL-style), the action
    space (flat 168 with sub-step masks), no final-turn guard for trained agents, the
    observation (flat vector, about 760 numbers), and reward (a setting, default dense
    score margin).
- **Current: Phase 3, visualization.** A proper Pygame viewer comes before any training
  (see "Visualization").
  - Done: milestone 1, game records and replay (`record.py`, `Game.clone()`,
    `simulate.py --record DIR`).
  - Done: milestone 2, the card-memory tracker (`memory.py`, Levels 0–2).
  - Done: milestone 3, the board renderer (`viz/geometry.py`, `viz/board_view.py`,
    `viz/screenshot.py`, display data in `data/usa_display.json`, measured from the
    board photo). Every tile is fitted to the photo; outlines drawn over the photo match
    the printed tiles to about a pixel, double-route pairs sit flush as on the board.
  - Done: milestone 4, side panels and the perspective toggle (`viz/perspective.py`,
    `viz/panels.py`, `viz/screen.py`). `Perspective` turns a state into a view model
    of what one viewer may see; the panels draw only that, so the hiding is tested
    without a display. `ttr.viz.screenshot` renders the whole screen
    (`--perspective all|<seat>`, `--memory-level`, `--board-only`).
  - Done: milestone 5, the live and replay viewers (`viz/app.py`). Both modes walk a
    `Timeline` of states, one per sub-step, so stepping back is an index move rather
    than a re-simulation; a live game produces the next state on demand by cloning
    the last one. Keys and on-screen buttons for play/pause, step, speed, perspective
    and jumping to either end, with the key legend in the table strip's top-left.
  - Next: milestone 6, human play (click a route, then choose the payment).
- **Next: Phase 4, the PettingZoo environment.**

## Roadmap

0. **Scaffolding**: package layout, `pyproject.toml`, test runner, `.gitignore`.
1. **Game engine**: pure Python with no RL dependencies. It covers board data (USA map plus
   the toy map), game state, legal-move generation, rule enforcement, and scoring,
   including the longest route. Unit tests cover the rule edge cases.
2. **Debugging tools and baselines**: `rich` terminal rendering of the game state.
   Random and simple greedy/heuristic agents serve as sanity checks and as evaluation
   opponents.
3. **Visualization**: a Pygame viewer with live, replay and human-play modes, a
   perspective toggle, and analysis overlays (see "Visualization"). Built before any
   training so every later phase can be watched and checked.
4. **PettingZoo environment**: an AEC wrapper around the engine, with action masking and
   the observation design (see "Agent design decisions").
5. **Training and search agents**: the method roster (see "Methods to compare"), each
   plugged into the same agent interface and evaluated the same way. Smoke-test each on
   the toy map first, then train on the full map.
6. **Strategy analysis**: metrics that capture play style (blocking rate, route-length
   distribution, ticket draw/keep behavior, tempo), compared across methods, plus
   pandas/matplotlib notebooks and the viewer's overlays.
7. **Later / optional**: 3–5 players, the AlphaZero-style stretch agent, and Unity +
   ML-Agents for presentation.

## Visualization (Phase 3)

A robust viewer before any training, replacing the temporary ASCII board view. The
engine stays free of Pygame: the viewer lives in `src/ttr/viz/` and Pygame is an
optional `[viz]` dependency.

**Decided scope:** live viewer, replay viewer, human play, analysis overlays, and a
perspective toggle between an all-seeing view and one player's view.

**Key idea: a game is a seed plus a list of actions.** The engine is deterministic given
its seed, so a record of `{board, players, seed, first player, max turns, agent names,
actions}` replays the game exactly. Records are small JSON files. They drive the replay
viewer and the overlays, and later let us review any trained agent's game move by move.

**Milestones, in build order:**

1. **Game records and replay** (`src/ttr/record.py`): save and load records, serialize
   actions, rebuild the game state at any step. `simulate.py --record DIR` saves games.
   Tests: replaying a record reproduces the original final state and result.
2. **Card-memory tracker** (`src/ttr/memory.py`): Levels 0–2 from "Card memory", built
   from the event log. It moves up from Phase 4 because the player view shows it too;
   the env reuses it. Tests: known counts never exceed the true hand, and the unseen
   pool always equals deck plus opponents' unknown cards.
3. **Board renderer:** replicate the physical board as closely as practical, using a
   photo of it (`docs/ticket-to-ride_usa_map.jpg`, gitignored as publisher artwork) as
   the reference. Match city positions, route shapes (many routes are curved), train-car
   placement for each space, double routes side by side, and the route-points table.
   Coordinates are measured from the photo and stored as display-only data. Everything
   is drawn by us; the photo is never used as a background. Claimed routes are filled in
   the owner's color. A `--screenshot` option renders a PNG without a window, for
   checking against the photo and for tests.
   - **Revisions after review:** a plain red frame instead of the numbered score track
     (scores go in the side panels); a map backdrop of state, province and country
     borders (Natural Earth, public domain) warped so real city locations land on the
     board's cities (`scripts/build_backdrop.py`); supersampled drawing for smooth
     edges; claimed spaces carry a rail-and-crosstie pattern in black or white
     (whichever the seat color carries), so a train never reads as an unclaimed
     tile of the same color. The shape is `theme.TRAIN_PATTERN`; `track` (chosen),
     `bars`, `diagonal`, `cross` and `plain` (the old center stripe) all draw.
   - **Measured from the photo (final):** every city center and every one of the 309
     tiles comes from the photo, not from estimates. City dots were detected by color.
     Each tile was then fitted as a rotated rectangle (position and angle) by searching
     around a rough first trace for the best score: route color inside the rectangle vs.
     a thin ring outside, the tile's dark outline, and an even fill. Brightness profiles
     across and along the fitted tiles gave one tile size for the whole board: a 35.5 x
     10.5 px colored fill with a ~1.5 px dark outline, stored as a 38 x 13 px car (the
     renderer strokes the outline on the car's edge) on the 1151 x 764 canvas. Double-
     route pairs came out ~14 px apart center to center, so they touch without overlapping,
     as on the board. The first trace (36 x 11, angles set by eye) was off by a few
     pixels and degrees per tile, which showed as gaps and odd angles. The fit is
     `scripts/fit_tiles.py` (`fit`, `size`, `check`, `overlay`); it needs the local photo
     and the optional `[photo]` extra (numpy, OpenCV). Re-running `fit` on the committed
     data moves no tile more than about 1 px. The photo itself stays out of the
     repo (gitignored as publisher artwork), so only the resulting coordinates
     are committed.
4. **Side panels and perspective toggle** (layout decided):
   - **Bottom, full width: player 0**, the seat a human plays against bots. Their hand
     of train cards by color, destination tickets, trains left, and score.
   - **Left and right sides: the other players**, two per side, shown only for seats
     in play. Each shows trains left, ticket count, train-card count, card counts by
     color where known (exact in the all-seeing view; card-memory estimates in a
     player's view), and score from routes claimed. The board has no score track, so
     scores live here.
   - **Top: the table.** The 5 face-up cards and the draw-deck, discard-pile and
     ticket-deck counts.
   - **Decided while building:** the current sub-step goes at the right end of the top
     strip (with the turn, whose view this is, and the final-round warning), and recent
     public events run as a one-line ticker along the top edge of seat 0's panel. A
     dedicated log column was rejected: it costs ~200px of width and crowds the seats.
   - The unseen pool (memory Level 2) sits in the top strip next to the pile counts,
     since it is a fact about the table rather than about one seat.
   - All-seeing view shows every hand and ticket; player view shows only that
     player's information plus their card-memory estimates.
5. **Live and replay viewers:** play, pause, step, speed; replay scrubs forward and back
   through every sub-step. Keyboard shortcuts plus on-screen buttons.
   - **Built as:** `viz/app.py`. A `Timeline` holds the states in order plus an optional
     producer for the next one, which is what makes live and replay the same code path:
     a replay preloads every state from the record, a live game clones the last state and
     lets the seat's agent act. Either way stepping back is an index move.
   - `Viewer` owns the timeline, the clock and the bindings but not the window loop, so
     the tests feed it events and frames with no display.
   - The key legend sits in the table strip's top-left, with the timeline position and
     speed in the same columns. Button labels stay ASCII: the system font pygame resolves
     here has no geometric shapes, and a missing glyph draws a box.
6. **Human play:** click a route, then choose a payment; click market cards or the deck
   to draw; tick tickets to keep. The UI only offers `game.legal_actions()`.
7. **Analysis overlays:** from a folder of records, color routes by a statistic (claim
   rate, claim rate per agent, average turn claimed, how often contested).
8. **Retire the ASCII board view** once the viewer covers it. The `rich` text log stays
   for terminal debugging.

### Viewer backlog (tweaks and fixes, in no fixed order)

Noticed while using the viewer; none of them blocks Phase 4.

- **Exact deck composition in the all-seeing view.** The player view already shows the
  unseen pool by color (memory Level 2). The all-seeing view should show the real thing:
  how many of each color are left in the draw deck. Same widget, different source — the
  deck itself rather than `MemoryView.unseen` — and it stays out of the player view,
  where it would leak hidden information.
- **Uniform card borders.** `panels.card_chip` draws its edge as `darker(fill, 0.45)`, so
  the edge's contrast depends on the card color: the locomotive and the white cards read
  as fully outlined, while blue, green and black nearly lose their edge against the dark
  panel. Give every chip the same edge treatment (one fixed outline color, or a light rim
  like the seat swatches) so the five face-up cards look like one set.
- **The bottom panel's empty space.** Centering seat 0's fields left a wide gap on the
  right (the left side now holds the viewer's buttons). Candidates: the current legal
  moves for a human player (Phase 3 milestone 6), a longest-path / ticket-progress
  summary, the last few actions in full rather than the one-line ticker, or simply a
  larger hand and ticket display.
- **Better ticket display.** Tickets are city codes plus points (`MTL–ATL 9`). Either
  write the full city names, or draw them as small ticket cards with the route drawn on
  them — closer to the physical card, and easier to read at a glance. Points are the
  weakest part of the current row: they look like a card count.
- **End-of-game results popup.** When the game ends the viewer just stops. It should
  overlay the scoreboard: the winner, and for every seat the route points, ticket points
  with completed/failed counts, the longest-path bonus and the total. `GameResult` already
  carries all of it, and `render.render_result` is the same table in text.
- **Charts for the strategy analysis (Phase 6).** Beyond the per-route overlays in
  milestone 7: claim rate over time, which tickets get completed and when, route length
  distribution, blocking frequency, how the methods differ on each. These belong with the
  analysis work (matplotlib/pandas out of a folder of records), not in the Pygame viewer,
  but the records already hold everything they need.
- **In-place Replacement of Face-up pile.** When a face-up card is taken, any remaining face-up 
  cards to the right of it are 'slid' down to the left, and new card from deck is always on the 
  right most spot. Behavior should be taking a card from face up and that position is
  then replaced by newly drawn card.

**Library:** `pygame-ce` (decided), the actively maintained drop-in fork of Pygame
(same `import pygame`), with Python 3.11 wheels.

## Methods to compare

The point is to exercise several RL and search methods and compare what each learns,
not just to find the strongest one. Each tier teaches something different:

| Tier | Method | What it lets us study | Implementation |
| --- | --- | --- | --- |
| A. Classic TD | Linear approximate **Q-learning vs. SARSA**, full map, hand-made features | Off- vs. on-policy, ε-greedy schedules; readable weights show what each agent values | Hand-rolled |
| B. Deep value-based | **DQN** with action masking (+ Double DQN) | Replay and target networks; how value methods cope with a large masked action space | CleanRL-style script |
| C. Policy gradient | **PPO** with self-play and action masking | Main agent for studying emergent strategy | CleanRL-style script |
| D. Search | **MCTS** with determinization (sample hidden hands and tickets, then search) | Planning with no training, as a contrast to the learned agents | Hand-rolled |
| Stretch | AlphaZero-style (MCTS guided by learned policy/value networks) | Combines C and D | Later |

- **Why linear rather than tabular for tier A:** tabular methods are only feasible on the
  toy map, which is a smoke test only. Linear features keep the same update rules and
  the same Q-learning vs. SARSA comparison on the full map.
- **Why no expectiminimax:** with 100+ legal moves per turn, chance nodes on every
  draw and hidden hands, it could only search 1–2 plies deep. MCTS with
  determinization handles all three better.
- **Why CleanRL-style scripts rather than SB3 or RLlib:** single-file implementations
  are readable and easy to change for masking and self-play, which suits learning the
  methods. SB3's DQN has no masking support, and RLlib is heavy to configure and debug.
- **Common interface:** every method implements `Agent.act(game, player)`
  (`src/ttr/agents/base.py`), so the existing match runner plays any method against any
  other.
- **Common evaluation:** win rate against the random and greedy bots; a head-to-head
  round robin turned into Elo ratings; the Phase 6 strategy metrics; and curves against
  training budget (games played and wall time), so methods are compared fairly.

## Agent design decisions

- **Card memory for trained agents: Level 2, computed by the env.**
  - **Memory vs. inference.** Memory means exact bookkeeping of public facts
    (RULES.md §9 #17), which the env computes into the observation. Inference, such as
    guessing an opponent's tickets or intent, is left to the agent. That is where
    strategies like blocking come from, so handing it over would build in the behavior
    being studied.
  - **Levels** (each adds to the one before):
    - **0, snapshot:** what the table shows now: market, claimed routes, hand sizes,
      ticket counts, trains, scores, discard pile.
    - **1, known opponent cards:** per opponent and color, a lower bound on cards held.
      A face-up take adds 1. Paying subtracts the amount paid, floored at 0 (paying 4 red
      with 2 known red means 2 came from blind draws). Unknown = hand size − known total.
    - **2, unseen pool:** the 110-card makeup minus your own hand, the market, the
      discard pile and opponents' known cards. With no other information, each unseen
      card is equally likely to be in the deck or in an opponent's unknown slots, so this
      one count vector estimates both blind draws and opponents' hidden cards.
    - **3 (not planned):** reshuffle-aware deck contents (after a reshuffle the new deck
      is exactly the old discard pile, order unknown) and knowing your returned tickets
      sit at the bottom of the ticket deck.
  - **The level is an env observation setting**, not hard-wired. Trained agents default
    to Level 2, and training at Level 0 vs. Level 2 is an experiment (see research
    questions). Scripted bots use the same levels as difficulty settings.
  - Since memory is precomputed, policies don't need to be recurrent.
- **Scripted/baseline bots can be weakened on purpose.** Each option is a per-bot setting
  that the engine doesn't enforce:
  - **Memory:** a memory level from 0–2 (see above).
  - **Final-turn ticket guard:** whether the bot avoids drawing tickets on its last turn.
  These give a range of difficulty for evaluation opponents.
  - **Trained agents don't get the guard.** Drawing tickets on the final turn stays
    unmasked, so the agent has to learn to avoid it. How often it still does is an
    endgame metric (see "Ticket risk").
- **Choosing the payment is a separate sub-step.** Claiming a route is two steps: choose
  the route, then choose the payment (the color for a gray route, how many Locomotives).
  This keeps each action mask small and the logic organized.
- **Action space: one flat `Discrete(168)` with a mask per sub-step.** Each engine action
  (`src/ttr/actions.py`) maps to a fixed index; the observation includes the current
  phase, and the mask allows only that phase's legal actions.

  | Actions | Count | Legal in phase |
  | --- | --- | --- |
  | `DrawFaceUp(color)`: 8 colors + Locomotive | 9 | choose action; second draw (no Locomotive) |
  | `DrawBlind` | 1 | choose action; second draw |
  | `ClaimRoute(id)` | 100 | choose action |
  | `Pay(color, k)`: 8 colors × k = 0–5 Locomotives, plus all-Locomotive | 49 | choose payment |
  | `DrawTickets` | 1 | choose action |
  | `KeepTickets`: non-empty subsets of the offer, by offer position | 7 | keep tickets (initial: subsets of 2+) |
  | `Pass` | 1 | choose action |

  - **Sub-steps rather than whole-turn actions:** a blind draw reveals a card, and a
    face-up take refills the market, before the second draw. A combined "pair of draws"
    action would force the agent to commit before seeing either. Merging claim and payment
    would give about 4,900 actions.
  - **Ticket choices are indexed by offer position**, not ticket ID. The observation shows
    which tickets are on offer, which keeps this to 7 actions.
  - Every method uses the same 168 outputs. MCTS can work on engine actions directly.
  - Discounting per sub-step rather than per turn is uneven, so keep γ ≈ 1 (see "Reward").
- **Observation: one flat vector from the acting player's view**, for a plain multilayer
  network. "Me" comes first and opponents follow in seat order, so one network can play
  any seat (needed for self-play with shared weights). Counts are scaled to about [0, 1].
  About 760 numbers for 2 players:

  | Block | Encoding | Size (2p) |
  | --- | --- | --- |
  | Current sub-step | one-hot over the 5 phases | 5 |
  | Route ownership | per route: unclaimed / mine / each opponent | 300 |
  | Route open to me | per route: claimable by me (double-route rule, trains left) | 100 |
  | Route being paid for | one-hot, payment step only | 100 |
  | Market | counts per color | 9 |
  | Pile sizes | train deck, discard, ticket deck | 3 |
  | Per player | trains left, public score, hand size, ticket count | 8 |
  | Endgame | final round started; this is my last turn | 2 |
  | Card memory (Level 2) | per opponent: 9 known + 1 unknown; unseen pool: 9 | 19 |
  | My hand | counts per color | 9 |
  | My tickets | held, multi-hot by ticket ID | 30 |
  | Ticket completed | per held ticket: already connected by my routes | 30 |
  | Trains to finish | per held ticket: fewest trains still needed + an "impossible" flag | 60 |
  | Tickets on offer | 3 offer slots × one-hot ticket ID (matches `KeepTickets` action order) | 90 |

  - **Tickets are identified by ID.** The board never changes, so the network can learn
    what each ticket means.
  - **Computed values** ("open to me", "completed", "trains to finish", endgame flags) are
    exact computations on information the agent already has. The endgame flags are
    required because the final-turn ticket draw is unmasked.
  - **Trains to finish** is the fewest trains needed to connect a held ticket's cities,
    counting the player's own routes as free and using only routes still open to them.
    It changes only when a route is claimed, so compute it then and cache it. It leans
    furthest toward strategy of the computed values; switching it off is a possible
    experiment.
  - **Tier A doesn't use this vector.** Linear Q-learning/SARSA needs hand-made features
    of state *and* action, designed with that tier. The vector is for DQN, PPO and a
    later AlphaZero-style agent.
  - **A graph network** (cities as nodes, routes as edges) is a possible later experiment.
- **Reward: an env setting, defaulting to dense score margin.** The reward decides which
  strategies are worth learning at all, so it is an experiment variable, like memory level.

  | Mode | Signal | Rewards blocking? |
  | --- | --- | --- |
  | Own score | + route points on each claim; tickets and longest route at game end | No |
  | **Score margin (default)** | my points − opponents' points, same schedule | Yes |
  | Win/loss | +1 / −1 at game end, 0 for a shared win | Yes |

  - **Dense isn't shaping:** in the score modes, rewards summed over a game equal exactly
    the final score or margin. Earlier feedback, same objective.
  - **Default is margin** because it rewards blocking, learns faster than sparse win/loss,
    and suits every tier (DQN and linear TD struggle with win/loss alone). Scale by about
    1/100.
  - **Win/loss** cares only about winning, so once safely ahead it has no reason to push
    the margin. Comparing it with margin speaks to the risk-tolerance question.
  - **Experiment:** train in all three modes. "Does blocking only emerge when the reward
    includes the opponent?" tests the route-hoarding vs. blocking question directly.
  - **γ = 1**, since every game ends. Win rate is the evaluation metric whatever the
    reward mode.
  - **Shaping is off.** If added later, use potential-based shaping,
    `F(s,a,s') = γΦ(s') − Φ(s)` (Ng et al., 1999), which doesn't change which policy is
    optimal. For example, Φ = −(trains to finish held tickets).

## Open questions

- **Engine speed**: about 10k steps/s now, and `step()` recomputes the legal-move list to
  validate each action. Revisit when legal moves become action masks in Phase 4,
  probably with a cached mask per state.

## Notes

- **Testing an old commit needs `PYTHONPATH`.** The venv installs `ttr` editable, so its
  `.pth` entry points at this working tree's `src/` whatever is checked out elsewhere. A
  `git worktree` or a bisected checkout therefore runs its own tests against the *current*
  renderer and engine, which fails or passes for the wrong reason. Run those with
  `PYTHONPATH=<worktree>/src` so the checkout's own source wins:

      git worktree add /tmp/wt <commit>
      cd /tmp/wt && PYTHONPATH=/tmp/wt/src <repo>/.venv/Scripts/python -m pytest -q
