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

**No small test map.** A tiny 6-city map used to exist for smoke tests. It was removed
on 2026-09-26: it only caught problems of its own (e.g. too few tickets to deal 5
players) and needed its own viewer layout code. Smoke tests, stress tests and first
training runs all use the USA map; USA games are short enough (about 200 sub-steps
for 2 players).

## Current progress

*Updated at the end of each session. Last updated: 2026-09-29.*

- **Earlier: engine ready, and every RL design question settled.**
  - Phases 0–2 (engine, random/greedy bots, `rich` log, ASCII board view,
    seat-rotated match runner, stress test) merged into `main`.
  - USA map data hand-verified against the physical board (`"verified": true`).
  - Moved to Python 3.11.
  - Settled: card memory (Level 2), the methods to compare (CleanRL-style), the action
    space (flat 168 with sub-step masks), no final-turn guard for trained agents, the
    observation (flat vector, about 760 numbers), and reward (a setting, default dense
    score margin).
- **Earlier: Phase 3, visualization.** A proper Pygame viewer before any training
  (see "Visualization"). All seven milestones done; the viewer backlog stays open.
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
  - Done: milestone 6, human play (`HumanControl` in `viz/app.py`, `--human SEAT`).
    Click a route then a payment chip, click a face-up card or either deck, tick
    tickets and confirm. Every chip comes from `game.legal_actions()`, so the UI
    cannot offer an illegal move; a click on anything else does nothing. Acting
    after stepping back forks the timeline, dropping the states that followed.
    The end-of-game scoreboard (viewer backlog) draws over the board.
  - Done: milestone 7, analysis overlays (`analysis.py`, `viz/overlay.py`). Per-route
    claim rate, average turn claimed and contested rate over a folder of records,
    overall, per agent or per seat; `ttr-view --overlay DIR` (`o`/`a` keys, hover for
    a value) and `ttr-shot --overlay DIR --stat S [--agent A] [--seat N]`. pandas is
    in a new `[analysis]` extra. The ASCII board view is kept.
  - Since then (viewer backlog): ticket markers and yellow ticket cities, full city
    names in panels, full screen, `--games N`, random default seed in `ttr-view`,
    random seating with colors that follow the agent.
- **Just completed: Phase 4, the PettingZoo environment.**
  - Done: step 1, the action space (`src/ttr/env/actions.py`): `encode` / `decode`
    between engine actions and Discrete(168) indices, and `legal_mask`. A test walks
    random games on both boards at 2–5 players and checks at every state that each
    legal action round-trips to a distinct index inside its sub-step's block.
  - Done: step 2, engine speed. `Game.legal_actions()` is cached per state (cleared by
    `step()`; hand edits call `invalidate()`), so choosing and validating no longer
    compute legal moves twice: 7.1k → 12.6k sub-steps/s, 11.2k with mask and decode
    (`scripts/bench_env.py`, USA, 2 players, random play).
  - Done: step 3, the observation encoder (`src/ttr/env/observation.py`): the
    765-number vector in "Agent design decisions", numpy float32, memory level 0–2.
    Card memory reuses `ttr.memory`; trains-to-finish is a Dijkstra per ticket source,
    cached until the next claim. Tests cover each block, the relative seat order, that
    an opponent's hand, tickets and offer and the deck order don't change the vector,
    and that the cached encoder matches a fresh one across random games at 2–5 players.
    Speed: about 4.4k sub-steps/s with mask and observation (random play, which hoards
    tickets, is the worst case for the Dijkstra cache). New `[env]` extra (numpy).
  - Done: step 4, the AEC env (`src/ttr/env/aec.py`, `env()` / `raw_env`). Agents
    `player_0..N-1` by seat; one agent step per engine sub-step. Observations are
    `{"observation", "action_mask"}` (mask all 0 for a waiting seat); an illegal index
    raises. Reward modes `score` / `margin` (default) / `win`, `reward_scale` 1/100 for
    the score modes; `max_turns` (default 1000) ends a game as a truncation. Final
    scores, winner and ticket counts go in `infos` at the end. `human` / `ansi` render
    via the `rich` board view. Passes PettingZoo's `api_test` and `seed_test`; tests
    check that score and margin rewards sum to the final score and margin at 2–5
    players. `scripts/smoke_env.py` plays random agents through it: about 2.3k agent
    steps/s (USA, 2 players), no seat bias over 200 4-player games. PettingZoo and
    Gymnasium are in the `[env]` extra.
  - Removed the 6-city test map (data, tests, the viewer's automatic layout); every test and
    script uses the USA map.
  - Moved to Python 3.14.3: `.venv` rebuilt, `requires-python >= 3.14`. Every
    dependency has 3.14 wheels.
- **Current: Phase 5, training and search agents.** Order decided: tier A (linear
  Q-learning / SARSA) first, as a first pass at hand-made features; the deep methods
  follow, then a second pass on tier A with what they reveal about which features
  matter.
  - Done: the policy adapter (`src/ttr/agents/policy.py`): `PolicyAgent` wraps any
    `policy(observation, mask) -> index` as an `Agent`, seeing exactly what the env
    shows that seat. Agent registry (`src/ttr/agents/registry.py`): `ttr-sim` and
    `ttr-view` take `random`, `greedy` or `linear:PATH`. Reward modes moved to
    `src/ttr/env/reward.py`, shared by the env and the hand-rolled learners.
  - Done: tier A, first pass (`src/ttr/learn/`, `ttr-train-linear`). Features in
    "Tier A features" below. The learner drives the engine directly, one seat against a
    scripted bot, reward = score margin / 100 between its decisions, γ = 1,
    normalized-LMS step α = 0.05, ε 0.2 → 0.02 over the first half.
  - Results, 2000 games each (~3 min), evaluated greedy over 100 games:

    | Run | vs random | vs greedy (win, margin) |
    | --- | --- | --- |
    | untrained (= random play) | 52% | 0%, −194 |
    | Q-learning vs random | 99% | 0%, −85 (best eval 9%, −53) |
    | Q-learning vs greedy | 100% | 0%, −83 |
    | SARSA vs greedy | 100% | 0%, −84 |

    It learns: random is beaten within ~100 games and the margin against greedy
    halves. It never beats greedy. Against greedy (60 games): about 21 claims a game
    with mean length 1.8 (greedy: 15.5 claims, 2.7), 0.9 tickets done / 1.1 failed
    (greedy 4.7 / 0.2), total 37 vs greedy's 108. Q-learning and SARSA end up nearly
    identical against greedy. Training against random was unstable: at 500 games it
    collapsed to 0% against random, then recovered.
  - Found and fixed during the first pass: route points are scored at the payment
    step, whose features first didn't describe the route, so Q(claim) could not tell
    routes apart; and nothing marked tickets kept on the last turn as lost (it drew
    tickets in the final round about once a game). Both fixed; the greedy-trained
    agents no longer draw tickets late.
  - Why it plateaus (read from the weights): the state features (the same in every
    action block) take nearly all the weight and fit V(s); the action features that
    rank one claim against another stay near 0 (route points ≈ 0.00, completes ticket
    ≈ +0.06). Ticket points arrive only at game end, about 100 decisions later.
  - Done: agent analysis dashboard (`ttr-dash`, `src/ttr/learn/dashboard.py`,
    matplotlib, `[analysis]` extra). Training runs now record per-game metrics
    (`src/ttr/learn/metrics.py`: score breakdown, tickets, claims and claim length,
    draw types, ticket draws and final-round ticket draws, ...), every evaluation's
    mean metrics, weight snapshots, and the bots' metrics on the final evaluation's
    games. Seven pages: overview, behavior over training, training games, action mix,
    weights, raw evaluations, raw games; one run or several compared. Chosen over an
    HTML page or a Pygame screen to stay on the stack's analysis tools (matplotlib).
  - Seen with it: the draw policy is nearly indifferent between face-up color and
    blind draws (biases −0.57 vs −0.54 after 2000 games against greedy), so small
    weight changes flip it wholesale: about 42 color / 0 blind draws a game in the
    evaluations at 1400–1800 games, 6 / 36 at 2000. Q-learning trained against
    random swings hard early (training margin +120 → −290 → +110 over games
    250–1000).
  - Analysis of the three recorded runs (2000 games each, one seed each):
    - Two strategies emerged. Trained against greedy (Q and SARSA alike): short
      routes (mean length 1.8, no 5–6 claims), keep the minimum 2 tickets, never draw
      more, complete 0.85 — score 38 vs greedy's 112. Trained against random: rush
      the end with long routes (length 3.7, triggers the end in 97% of games, 74
      turns vs 83) — route points 73, more than greedy's 61 — but keeps 3.7 tickets
      and completes 0.15, and still draws tickets in the final round (0.68 a game).
      Best margin against greedy of the three (−59).
    - The best policy of any run was Q-learning vs random at 200 games: 26% wins and
      −16 margin against greedy, claiming routes of mean length 5.5. It was lost by 400
      games and never regained.
    - Evaluation k of every run uses the same games (seed 10 000 + k), so runs are
      compared on paired games. The swings from one evaluation to the next move
      together across runs (all three at −71 to −72 or better at 1800), so they are
      mostly which games were drawn, not policy change. Against greedy, Q and SARSA
      plateau from about 1200 games; Q ends about 4.5 margin ahead over the last five
      evaluations, about 2 standard errors: weak evidence.
    - None of the agents interferes with greedy: greedy scores more against them
      (119–122) than against itself (106).
    - Ticket keep weights are the largest action weights and all say "fewer, cheaper
      tickets" (keep.count −0.16, keep.points −0.12): the agents learned tickets are
      a liability because they rarely finish them (39% of kept tickets completed vs
      greedy's 96%), not which tickets are finishable.
  - Done: tier A, second pass (2026-09-27).
    - Changes: Q(s, a) = v · state(s) + w[type] · ψ(s, a), one shared value and a
      per-type advantage (bias + that type's features, which now carry the state
      context that matters to them: hand size and endgame for draws, trains after
      and endgame for claims, last turn / open tickets / trains for ticket draws);
      ticket-choice features for finishability (share of path already on my other
      tickets' paths, costliest ticket vs uncommitted trains, points per train);
      `--opponent mixed` (random or greedy, a coin flip per game); the best
      checkpoint by margin against greedy is kept (`linear:PATH@best`);
      `ttr-dash --group` averages seeds.
    - Batch: 4 settings × 3 seeds, 2000 games each (`runs/linear/pass2/`).
      Against greedy (each seed's last 5 evaluations, mean over seeds):

      | Setting | Margin (seeds) | Win share | Pass 1 |
      | --- | --- | --- | --- |
      | Q vs random | −35 (−32, −37, −36) | 10% | −64, 2% |
      | Q vs greedy | −59 (−82, −36, −60) | 9% | −80, 0% |
      | Q vs mixed | −62 (−51, −83, −51) | 5% | — |
      | SARSA vs mixed | −66 (−79, −52, −68) | 6% | — |

      Every setting improved on pass 1 and all beat random ≥ 98%. The advantage
      weights now carry signal (pay.route_points +0.28, claim.points +0.15 for Q vs
      mixed; pass 1: ≈ 0).
    - Best checkpoints hold up on 200 fresh games against greedy: Q vs mixed s0
      36% wins, −12; Q vs greedy s2 36%, −14; Q vs greedy s1 33%, −15 — where the
      same runs' final weights score 12%, 0% and 10%. Evaluation margins inside one
      run swing between about −10 and −90 (Q vs greedy s2: −37, −14, −77, −79 over
      its last four), so the policy moves between strategies rather than
      converging. Keeping the best checkpoint is worth more than any other change.
    - The best policies are all the same strategy: race to the end. Long routes
      (mean length 3.9–4.2, 3.4–3.9 claims of 5–6), blind draws, the minimum 2
      tickets, trigger the end in 75–99% of games, which last about 72 turns vs 85.
      The greedy bot then scores 80–93 instead of its usual 106, because it runs
      out of time to finish tickets. Q vs mixed s0's final weights still claim long
      routes (8 claims of 5–6) but trigger the end only 27% of the time, and greedy
      scores 114: the tempo, not the long routes alone, is what works.
    - The mixed opponent did not help: Q vs random is best and most consistent
      (seed spread 5 vs 30). SARSA vs Q (mixed): no difference beyond seed noise.
    - Tickets are still the gap: every agent completes under 1 ticket a game
      (greedy 4.9). Rare final-round ticket choices still learn noisy weights
      (keep.doomed_points +0.12 for Q vs random, the wrong sign).
  - Done: live view. `ttr-train-linear --live [N]` rewrites the run file every N games
    (default 25) and after each evaluation (whole-file replace, so readers never see a
    partial file); `ttr-dash --live` rereads it every few seconds and redraws. Off by
    default: one write at the end, as before. A live save of a 2000-game run is about
    20 ms.
  - Decided (2026-09-27): keep doing tier-A passes until learning settles, before
    DQN / PPO.
  - Built for the third pass (all options, off by default so pass-2 runs reproduce):
    `--alpha-end` (linear α decay), `--average GAMES` (exponential weight average,
    evaluated and saved; raw weights saved too), `--lambda` (SARSA(λ) / Watkins
    Q(λ) eligibility traces, step scaled by 1 − λ), `--shaping POINTS`
    (potential-based, Φ = −POINTS × trains still needed for my incomplete tickets, a
    lost ticket counting 45; Φ = 0 at the start and end, so a game's shaping sums to
    0), `--opponent self` (greedy or the run's best checkpoint). New features,
    appended so older weight files still load (with these at 0): `ticket_share`
    (share of a served ticket's remaining trains a route covers; claim and pay),
    `tempo` ((fewest opponent trains − my trains) / 45; claim and draws),
    `near_opponent` (a route end is on an opponent's claimed routes; claim).
  - Found while building it: a game is about 85–100 of the learner's decisions, so
    traces need λ ≈ 0.97–0.99 to carry the final score back to the opening
    (0.9^100 ≈ 3·10⁻⁵, 0.99^100 ≈ 0.37); λ 0.8–0.9 would not reach it.
  - Third-pass ladder run (2026-09-27; `runs/linear/pass3/`). Raw numbers, analysis
    not finished. Against greedy, each seed's
    last 5 evaluations, mean over 3 seeds (pass-2 `q_random` reference: −35, 10% wins,
    within-run sd 7.7):

    | Setting | Margin (seeds) | Win | Within-run sd | Tickets done / failed | Claims × length |
    | --- | --- | --- | --- | --- | --- |
    | p3a avg (α 0.02, average 100) | −57 (−72, −66, −33) | 10% | 20.7 | 0.10 / 2.56 | 14.7 × 3.4 |
    | p3b + λ 0.98 | −164 (−174, −146, −172) | 0% | 14.0 | 0.00 / 3.78 | 0.2 × 0.5 |
    | p3c + shaping 1 | −98 (−100, −95, −101) | 2% | 23.7 | 0.15 / 2.71 | 6.6 × 4.3 |
    | p3d + self-play | −125 (−137, −96, −141) | 1% | 20.0 | 0.02 / 2.04 | 4.6 × 1.8 |

    First reading: every setting is worse than pass 2, and λ 0.98 collapsed — the
    agent almost stops claiming (0.2 claims a game, route points 3). Shaping recovers
    part of it; its curve was still rising at 2000 games (−142 → −79). Best
    checkpoints: p3a −50/−19/−17; p3d s1 −44. To check next: whether λ's collapse is
    the step scaling ((1 − λ) with accumulating traces) or credit reaching the wrong
    actions; λ 0.98 with a larger α; and whether α 0.02 alone (p3a) is simply slower
    than pass 2's 0.05.
  - Fourth pass (2026-09-27 overnight, `scripts/linear_pass4.ps1`, `runs/linear/pass4/`):
    Q vs random, 10000 games, 3 seeds, epsilon reaching 0.02 at game 1000 as before, to
    answer the pass-3 questions. 24 minutes for all 18 runs. Against greedy, last 5
    evaluations, mean over seeds; "mid" is the same at game 5000:

    | Setting | Margin (seeds) | Win | Within-run sd | Tickets done / failed | Mid |
    | --- | --- | --- | --- | --- | --- |
    | p4a α 0.05 (pass 2, longer) | −20 (−11, −29, −19) | 27% | 13.8 | 1.21 / 0.79 | −22 |
    | p4b + average 100 | −6 (−2, −6, −9) | 41% | 4.3 | 1.55 / 0.45 | −6 |
    | p4c α 0.02 + average (p3a, longer) | −10 (−13, −3, −14) | 35% | 5.0 | 0.36 / 1.65 | −30 |
    | p4d α 0.05 + average + shaping 1 | −7 (−13, −5, −2) | 39% | 4.2 | 1.60 / 0.40 | −11 |
    | p4e α 0.05 + average + λ 0.9 | −5 (+2, −7, −11) | 41% | 5.0 | 0.91 / 1.09 | −11 |
    | p4f α 0.2 + average + λ 0.98 | −8 (−10, −4, −11) | 40% | 8.4 | 1.38 / 0.63 | −29 |

    Readings: pass 3 was mostly too short. Averaging is the clearest gain (−20 → −6,
    evaluation noise sd 13.8 → 4.3); pass 3 blamed it for what was α 0.02's slowness.
    α 0.02 is slower, not worse (still rising). λ 0.98 learns once α is larger, so the
    pass-3 collapse was the step α(1 − λ) being too small. Shaping makes every seed
    complete its tickets without improving the margin. The fast settings plateau near
    −5 against greedy from mid-run: parity with greedy, not beyond, against random.
    Seeds split into ticket completers and ticket dumpers (keep 2, complete none).
  - Fifth pass (`scripts/linear_pass5.ps1`, `runs/linear/pass5/`): p4b's setting
    (α 0.05, average 100), 30000 games, 5 seeds, evaluations every 500, at most 12 runs
    at a time (~28 min a run). Does the training opponent get past parity, and Q vs
    SARSA? Against greedy, last 5 evaluations, mean over seeds:

    | Setting | Margin (seeds) | Win | Tickets done / failed | Curve, games 5k → 30k |
    | --- | --- | --- | --- | --- |
    | Q vs random | −14 (−19, −9, −10, −14, −16) | 29% | 1.68 / 0.32 | −13 → −12, flat |
    | SARSA vs random | −7 (−10, −2, −10, −10, −2) | 36% | 1.63 / 0.37 | −15 → −8, rising |
    | Q vs greedy | −14 (−15, −11, −30, −10, −4) | 33% | 1.36 / 0.64 | −5 → −15, peaks early |
    | Q vs mixed | −18 (−15, −15, −24, −18, −19) | 30% | 0.01 / 2.02 | −13 → −19 |
    | Q vs self | −58 (−38, −75, −73, −74, −30) | 7% | 1.60 / 0.40 | +2 → −64, collapses |

    Readings: longer training against random doesn't pass parity (Q flat from 5000 to
    30000 games; linear on these features tops out near −5 to −15 vs greedy). Every
    Q-learning run against a real opponent peaks near game 5000 and then declines at
    constant α; self-play collapses against greedy while still beating random by 130+.
    SARSA is steadier (evaluation sd 2.9, the lowest) and still rising: consistent with
    off-policy Q-learning chattering or diverging under function approximation where
    on-policy SARSA doesn't. Mixed opponents turn every seed into a ticket dumper.
    Best checkpoints reach +2 to +9 against greedy but are picked by the same
    evaluation that scores them, so they need re-scoring on fresh games.
  - Sixth pass (`scripts/linear_pass6.ps1`, `runs/linear/pass6/`): α decay 0.05 →
    0.005 as the fix for the late decline, over {Q, SARSA} × {random, greedy, self},
    plus SARSA at constant α against greedy and self, completing the algorithm ×
    opponent × α-schedule grid with pass 5. 8 settings × 5 seeds, 30000 games, 12 at a
    time (~1 h 45 min). Mixed dropped. Same measure as pass 5:

    | Setting | Constant α (pass 5 / 6) | α decay (pass 6) |
    | --- | --- | --- |
    | Q vs random | −14 | −17 (−14, −24, −18, −19, −11) |
    | SARSA vs random | −7 | −12 (−19, −18, −6, +3, −22) |
    | Q vs greedy | −14 | −12 (−21, −1, −7, −22, −10) |
    | SARSA vs greedy | −29 (−29, −27, −24, −31, −33) | −36 (−50, −23, −33, −36, −38) |
    | Q vs self | −58 | −35 (−27, −16, −22, −82, −26) |
    | SARSA vs self | −56 (−74, −45, −42, −74, −44) | −66 (−74, −81, −68, −78, −30) |

    Readings: α decay doesn't stop the decline (it helps Q against greedy and self a
    little and hurts SARSA). SARSA declines against greedy and self too, and does worse
    than Q there, so the decline is not off-policy divergence. The weights stay small
    (norm about 1), and the training margin falls with the evaluation margin: the
    policy drifts. Near game 5000 the agents claim about 12 routes of length ~3.7 and
    end the game themselves 85–90% of the time. Later they claim about 20 routes of
    length 2 and end it about 30% of the time. Every run against greedy or self peaks
    near game 5000 (−1 to −10). This looks like a limit of the linear features holding
    a good policy against a strong opponent: a case for DQN (tier B).
  - Re-scoring (`scripts/rescore.py`; `runs/linear/rescore_pass4-5.*`,
    `rescore_pass6.*`): final weights and best checkpoint of every pass 4–6 run against
    greedy on the same 1000 fresh games (batch seed 9001, never used in training;
    standard error about ±1 per agent). Mean margin over seeds, best checkpoint's seeds
    in brackets:

    | Setting | Final | Best (seeds) | Best win |
    | --- | --- | --- | --- |
    | p4a Q random α 0.05 | −12 | −7 (−8, −4, −9) | 38% |
    | p4b + average | −8 | −4 (−3, −1, −8) | 41% |
    | p4c α 0.02 + average | −13 | −5 (−9, −3, −1) | 42% |
    | p4d + shaping | −8 | −6 (−14, −1, −3) | 39% |
    | p4e + λ 0.9 | −7 | −3 (+0, +2, −11) | 43% |
    | p4f α 0.2 + λ 0.98 | −5 | +5 (−7, +13, +8) | 56% |
    | p5 Q random | −14 | −6 (−10, −6, −10, −11, +6) | 39% |
    | p5 SARSA random | −9 | −5 (−10, +2, −9, −7, +2) | 41% |
    | p5 Q greedy | −18 | +3 (+1, +1, +5, +3, +7) | 54% |
    | p5 Q mixed | −20 | −10 (+0, −3, −17, −13, −17) | 39% |
    | p5 Q self | −52 | +3 (+7, +3, +4, +5, −4) | 53% |
    | p6 Q random, decay | −16 | −6 (+1, −12, −8, −10, −0) | 40% |
    | p6 SARSA random, decay | −16 | −5 (−7, −7, +4, +2, −14) | 42% |
    | p6 Q greedy, decay | −12 | **+7 (+9, +6, +9, +6, +7)** | **59%** |
    | p6 SARSA greedy, decay | −36 | −5 (−8, −1, −3, −2, −13) | 43% |
    | p6 Q self, decay | −35 | +3 (+6, +7, +4, −6, +6) | 53% |
    | p6 SARSA self, decay | −67 | +0 (+9, +8, +6, −15, −8) | 49% |
    | p6 SARSA greedy | −32 | −8 (−9, −5, −3, −7, −15) | 39% |
    | p6 SARSA self | −51 | −3 (−7, +2, −6, +3, −7) | 45% |

    Readings: early stopping works. Linear Q trained against greedy, stopped at its best
    evaluation, beats greedy on fresh games: +3 at constant α, +7 with α decay, where
    every seed is between +6 and +9 with 59% wins. The recorded best margins were only
    mildly optimistic. The strongest single agent is `p4f_lam98_s1@best`: +13.4 ± 1.0,
    66% wins. Q beats SARSA against greedy and self; SARSA is only competitive against
    random. The checkpoints that beat greedy complete fewer tickets than the final
    weights (1.1–1.2 against 1.4–1.7): they win on routes and by ending the game.
  - Seventh pass (`scripts/linear_pass7.ps1`, `runs/linear/pass7/`): p4f's setting
    (α 0.2, λ 0.98, average 100) for 30000 games against random, greedy and self (Q),
    and SARSA(λ) against random; 4 settings × 5 seeds, 12 at a time (~1 h). Training
    evaluations (last 5, mean over seeds) and the fresh-game re-score
    (`runs/linear/rescore_pass7.*`):

    | Setting | Training eval | Curve 10k → 30k | Final, fresh | Best, fresh (seeds) | Best win |
    | --- | --- | --- | --- | --- | --- |
    | Q(λ) vs random | −23 | −15 → −25 | −24 | +4 (+5, +6, +5, +3, −0) | 55% |
    | SARSA(λ) vs random | −21 | −14 → −23 | −30 | +7 (+12, +5, +1, +17, −2) | 59% |
    | Q(λ) vs greedy | **+3** | **+1 → +2, no decline** | −8 | **+11 (+12, +15, +15, +10, +4)** | **64%** |
    | Q(λ) vs self | −12 | −28 → −12 | −15 | +10 (+12, +12, +15, +5, +7) | 63% |

    Readings: λ 0.98 against greedy is the first setting whose training curve holds
    above parity instead of peaking and declining, and its best checkpoints are the
    strongest setting so far (every seed positive, 64% wins). λ self-play also gives
    reliable greedy-beaters (+10). Final weights are noisy at α 0.2 (within-run sd
    10–15), so a final can be anywhere from −30 to +15. The strongest single agents
    on fresh games: `p7b_sarsa_lam98_random_s3@best` +17.4 ± 0.8 (73% wins);
    `p7c_q_lam98_greedy_s1` final weights +15.5 (70%) and its best +15.3;
    `p7d_q_lam98_self_s2@best` +14.9; `p7c_q_lam98_greedy_s2@best` +14.6.
  - Eighth pass (`scripts/linear_pass8.ps1`, `runs/linear/pass8/`): λ 0.98 with α decay
    0.2 → 0.02, Q against greedy and self, 5 seeds, all 10 at once: whether decay
    steadies the noisy finals of pass 7. The launcher died at about 10:08, before
    finishing (no exit lines in `pass8.log`; cause unverified). Every run against greedy
    and self-play s0 finished; self-play s1–s4 stopped at 27000–28000 games, so their
    "final" is the last live save. Their best checkpoints (games 10500–18500) are
    unaffected. Training evaluations and the fresh-game re-score
    (`runs/linear/rescore_pass8.*`), pass 7 without decay for reference:

    | Setting | Training eval (sd) | Final, fresh (seeds) | Best, fresh (seeds) | Best win |
    | --- | --- | --- | --- | --- |
    | p7c Q(λ) vs greedy | +3 (15.1) | −8 (−30, +15, +1, −7, −19) | +11 (+12, +15, +15, +10, +4) | 64% |
    | p8a + α decay | +3 (6.4) | **+6 (+11, +13, +11, +8, −14)** | +10 (+10, +14, +11, +10, +3) | 64% |
    | p7d Q(λ) vs self | −12 (11.4) | −15 (−6, −9, +4, −59, −4) | +10 (+12, +12, +15, +5, +7) | 63% |
    | p8b + α decay | −11 (7.0) | −12 (−6, −47, +9, −12, −6) | +9 (+14, −8, +10, +14, +15) | 62% |

    Readings: decay halves the evaluation noise and makes the final weights usable:
    four of five p8a seeds beat greedy by +8 to +13 with no checkpoint selection, the
    first setting whose final weights beat greedy on average (60% wins). The ceiling
    does not move: best checkpoints are +9 to +10, as in pass 7. One seed per setting
    still drifts (p8a s4 −14; p8b s1 −47 turned ticket-heavy, completing 5.0 tickets and
    failing 2.1 a game, and its best was only −8). The self-play checkpoints that beat
    greedy by +14 (s3, s4) complete almost no tickets: race to the end again. Strongest
    single agent is still `p7b_sarsa_lam98_random_s3@best` (+17.4); pass 8's best are
    p8b s4@best +14.9, s3@best +14.4, p8a s1@best +14.2.
  - Decided (2026-09-28): tier A is settled for now. Across passes 7–8 every λ 0.98
    setting's best checkpoints average +9 to +11 against greedy, the strongest single
    agents +14 to +17, and the remaining changes move noise, not the ceiling. The
    pass-6 reading stands: linear features cap how good a policy they can hold (ticket
    value is all-or-nothing over a whole path, which a sum of features can't express).
    Feature work is the untested lever; it waits for what the deep methods show.
    p8b s1–s4 are not rerun.
  - Done: tier B, DQN (`src/ttr/learn/dqn.py`, `ttr-train-dqn`; design in "Methods to
    compare"). PyTorch 2.14 (CUDA 13.0 build, RTX 3080) in a new `[deep]` extra.
    `dqn:PATH[@best]` in the agent registry, so `ttr-sim`, `ttr-view` and
    `scripts/rescore.py` play DQN agents; `ttr-dash` and `scripts/run_status.py` read
    DQN runs (no weight pages). `linear.evaluate` takes any agent spec as the opponent.
    - Speed: a gradient step costs ~4 ms on the GPU whatever the batch up to 2048
      (kernel-launch bound), and asking the network about one observation is faster on
      the CPU (0.16 ms). So games are played by CPU copies of the network and only the
      gradient steps use the GPU, with batches of 256 every 8 decisions: 120 ms a game
      alone, ~165 ms with 6 runs at once, plus ~35 s per evaluation (300 games).
    - Smoke run, 1500 games against greedy, 3 seeds: n-step 8 reaches −13, −20, −43
      (linear pass 2 was −35 to −66 at 2000 games); n-step 1 −106 to −121. The n = 8
      agents already race to the end like the best linear ones (2 tickets kept, almost
      none completed).
  - First DQN pass (`scripts/dqn_pass1.ps1`, `runs/dqn/pass1/`, 2026-09-28, 1 h 57 min):
    against greedy, averaging 100, n-step 1 / 8 / 32, 3 seeds, 30000 games, all 9 at
    once. Re-scored on 1000 fresh games per opponent (`runs/dqn/rescore_pass1.*`;
    wary and racer are the new bots below). Mean over seeds, per-seed in brackets:

    | Setting | vs greedy, final | vs greedy, best | Best win | vs wary (final / best) | vs racer (final / best) |
    | --- | --- | --- | --- | --- | --- |
    | n = 1 | **+12.5 (+10, +14, +13)** | +14.1 (+12, +16, +14) | 73% | +6.9 / +8.4 | −3.4 / +4.6 |
    | n = 8 | −49.8 | −26.4 | 22% | −51.0 / −31.4 | −60.7 / −51.6 |
    | n = 32 | −68.3 | −34.2 | 14% | −68.1 / −38.1 | −74.0 / −52.1 |

    Readings: n-step returns sped up the start and then hurt. n = 8 and 32 led at 3000
    games (−26, −34), peaked there, and drifted to −47 / −66 with shorter routes (2.9 /
    2.5) and fewer self-ended games, the drift linear passes 5–6 showed. n = 1 started
    slowest (−144 at 1000 games), passed 0 near 10k and was still rising (best
    checkpoints at 22–24.5k games). Why n-step hurts is untested: the n-step return mixes
    in the opponent's card luck over n decisions, and uncorrected n-step learns the
    exploring policy's value. So λ's lesson from tier A did not transfer.
    - n = 1 plays pure racing: 10 claims of mean length 4.4, 97 route points, keeps the
      minimum 2 tickets and completes 0.02, ends the game itself in 99% of games (69
      turns), longest-path bonus 87%. Against the best linear agent in training
      evaluations: −20.
    - Against tier A: every n = 1 seed's final network (+10 to +14) beats every linear
      setting's finals (best p8a +5.6) and matches linear best checkpoints (+9 to +11),
      without checkpoint picking. The strongest single agent is still linear (below).
  - Done: sparring bots and an opponent pool (2026-09-28). Every learner so far found
    the one exploit of greedy (which plans its tickets and never looks at the
    opponent): race to the end. Two causes: the opponent rewards racing, and tickets
    are hard to learn (the reward comes at the end, after dozens of right claims).
    - `wary` (`WaryAgent`, an option on `GreedyAgent`): once an opponent is down to 15
      trains or the final round starts, no more ticket draws, drop tickets it can't
      finish in the turns likely left, cash cards into the longest claims.
    - `racer` (`src/ttr/agents/racer.py`): the racing strategy scripted: fewest
      tickets, only 6-routes until it can't (15 points for 6 trains), extend its own
      network. Minimum length tuned on 300 games vs greedy: 4 → −13, 5 → −3, 6 → +12.
    - Wary doesn't beat racer: alert at 15–30 trains, or grabbing 5- or 6-routes
      itself, all lose −7 to −14. Pure racing beats both simple ticket planners in
      2-player games.
    - `collector` (`CollectorAgent`, the user's design): greedy for tickets. Keeps 2–3
      open, draws more as soon as fewer than 2 are far from done (8+ trains left),
      finishes the nearest ticket first, tempo-aware like wary. It completes the most
      tickets (about 5 a game) and loses to every bot (200–300 games each: −17 to −19
      vs greedy, −24 vs wary, −24 to −27 vs racer), failing 1.0 a game. Loosening it
      toward greedy (1–2 open, draw only with 15+ trains) brings it to parity with
      greedy and −12 vs racer, where greedy sits. Past the first few tickets, each one
      costs more trains and turns than it scores in 2-player games. Kept as the
      ticket-heavy style for the pool and evaluations.
    - `ttr-train-dqn --opponent pool` (`--pool greedy wary racer self`, one per game;
      `self` = the best network or one of the 5 latest evaluated ones), `--shaping`
      (potential-based, Φ = −POINTS × trains still needed for my tickets; sums to 0
      over a game), evaluations against every bot, best checkpoint on the mean margin
      over greedy, wary and racer. `rescore.py --opponents`. `ttr-dash` cycles through
      a run's opponents.
  - Round robin (`scripts/round_robin.py`, `runs/round_robin_2026-09-28.*`): 400 fresh
    games per pair, Elo with greedy = 1000:

    | Agent | Elo | vs linear best | vs DQN best | vs racer | vs wary | vs greedy |
    | --- | --- | --- | --- | --- | --- | --- |
    | linear `p7b_sarsa_lam98_random_s3@best` | 1233 | — | +19.4 | +14.6 | +10.8 | +17.4 |
    | DQN `d1a_n1_s1@best` | 1133 | −19.4 | — | +4.2 | +8.6 | +14.9 |
    | racer | 1077 | −14.6 | −4.2 | — | +7.0 | +12.9 |
    | wary | 1055 | −10.8 | −8.6 | −7.0 | — | +4.1 |
    | greedy | 1000 | −17.4 | −14.9 | −12.9 | −4.1 | — |

    The strongest single linear agent beats everything, racer included (+14.6), so it
    does something beyond pure racing; worth a look in the viewer. DQN's best beats
    every bot, narrowly racer.
  - DQN pass 2 (`scripts/dqn_pass2.ps1`, `runs/dqn/pass2/`, 2026-09-28, 1 h 53 min): n = 1,
    averaging 100, 3 seeds, 30000 games: vs greedy (pass-1 reference), vs pool (greedy,
    wary, racer, self; a quarter each), vs pool + shaping 1. Does the pool break racing,
    and does shaping bring ticket play? Re-scored on 1000 fresh games per opponent
    (`runs/dqn/rescore_pass2.*`), final / best, mean over seeds:

    | Arm | vs greedy | vs wary | vs racer | vs collector | Tickets done |
    | --- | --- | --- | --- | --- | --- |
    | vs greedy | +12.5 / +14.7 | +6.9 / +9.0 | −3.4 / +5.0 | +30.1 / +31.8 | 0.02 |
    | vs pool | +8.2 / +9.0 | +3.1 / +3.8 | **+22.2 / +22.3** | +24.8 / +26.0 | 0.01 |
    | pool + shaping | +10.8 / +13.2 | +6.0 / +8.1 | **+21.7 / +22.8** | +27.7 / +30.5 | 0.02 |

    Readings: no arm plays tickets. Every seed of every arm keeps the minimum 2 and
    completes 0.01–0.02 (−19 ticket points), claims ~10 routes of mean length 4.3–4.5,
    and ends the game itself 98–99% of the time. What the pool changed is the racing
    itself: pool-trained agents beat racer +22 (90–92% wins) where the greedy-trained
    ones tie it, at a cost of 2–6 against greedy and wary. Shaping didn't bring
    tickets (a game's shaping sums to 0, so abandoning a ticket costs the same as
    before) but sped learning up: −91 vs greedy at 1000 games against −174 / −202, and
    positive from ~12k games against ~22k for the pool without it. The greedy arm
    repeats pass 1 (same numbers). The pool arms went through a ticket-hoarding phase
    early (12 tickets kept at 5–6k games) and left it by ~10k.
  - Round robin 2 (`runs/round_robin_2026-09-28b.*`, 10 agents, 400 games per pair), Elo
    with greedy = 1000: linear `p7b_sarsa_lam98_random_s3@best` 1246, DQN pool + shaping
    `d2c_pool_shape_s1@best` 1235, DQN pool `d2b_pool_s1@best` 1172, DQN pass-1
    `d1a_n1_s1@best` 1106, DQN vs greedy `d2a_greedy_s2@best` 1103, wary 1057, racer 1013,
    greedy 1000, collector 829. Pool + shaping loses to the linear agent by only 6.4
    (pass 1's DQN: 19.4) and beats both greedy-trained DQNs by 18–21: training against
    a pool made a more robust racer, not a different strategy.
  - Behavior analysis (200 instrumented games per matchup; per claim: 6-routes, claims
    on the opponent's cheapest ticket paths, tempo):
    - **Racer has a flaw the pool agents exploit.** It claims only 6-routes while it has
      6+ trains and has no fallback once the 9 six-routes are gone. Against the pool +
      shaping DQN it ends holding 54 cards with 11 trains unused, having made 5.8
      claims; the DQN takes the 6-routes, lets the game run to 96 turns (vs ~68) and
      gets the longest-path bonus 98% of the time. Most of the pool agents' +22 vs
      racer is this exploit (the greedy-trained DQN, which hurries, gets +5).
    - **Linear vs DQN play.** The best linear agent claims fewer, longer routes (8.5 ×
      5.1 vs DQN's 10 × 4.4), draws only blind (0 face-up cards; DQN takes 15–22 face-up
      colors and 1–2 face-up Locomotives), reaches 22 trains left sooner (turn 42 vs
      47–51), and completes a few tickets (0.24–0.30 a game vs 0.01; −14 ticket points
      vs −19). DQN gets the longest-path bonus more (0.8–0.9 vs 0.2–0.7).
    - **Nobody blocks.** Claims on the opponent's cheapest ticket paths: 0.2–0.7 a game
      for every learner, all consistent with coincidence (greedy, which claims many
      short routes, hits 1.3–1.4).
  - Linear vs DQN cross-match (fresh seeds, random seats): the 4 strongest linear agents
    against the 3 pool + shaping seeds, 1000 games per pair: three of them beat every
    DQN seed by +3 to +7, the fourth (`p7c_q_lam98_greedy_s1`, +15.5 vs greedy) loses to
    every one by 14–17; mean −0.4. Setting against setting (every seed's best, 500
    games per pair): linear p7c (the best linear setting vs greedy) −7.0, 3 of 15 pairs
    won, 37% wins; linear p7d (self-play) −4.4, 7 of 15, 40%. **DQN beats linear as a
    method**; "linear beats DQN" came from the single best of ~150 linear networks.
    Linear seeds range from +3.7 to −18.5 against the same DQN; the three DQN seeds
    are within 2 of each other. Ranking against greedy doesn't transfer to ranking
    against other agents.
  - Racer fixed (2026-09-28): once no route of its minimum length (6) is open to it,
    it lowers the minimum to the longest route left instead of drawing to the end. Its
    results against the bots are unchanged (+11.6 vs greedy, +8.8 vs wary, +22.0 vs
    collector, 200 games each), and **it beats every trained agent**: +15.6 vs the pool
    + shaping DQN (76% wins), +22.5 vs the greedy-trained DQN, +1.8 vs the best linear
    agent. The DQNs had learned to beat broken racer, not racing; a correct scripted
    racer is the strongest agent so far.
  - Decided (2026-09-28): the observation gets an optional ticket-plan block
    (`ObservationEncoder(ticket_plan=True)`, 224 numbers appended; the 765-number
    default is unchanged): per route, the points of my open tickets whose cheapest path
    uses it and whether it alone completes one; per offered ticket, trains to connect,
    too big for my uncommitted trains, overlap with my plan, points per train; cards
    my plan still needs per color; trains committed / uncommitted, tickets lost. It is
    what the linear learner's ticket features give it (its one edge over DQN: ticket
    choice), as exact computations on the viewer's own information, in line with the
    other computed values. `ttr-train-dqn --ticket-plan`.
  - DQN pass 3 (`scripts/dqn_pass3.ps1`, `runs/dqn/pass3/`, 2026-09-29, 2 h): pool =
    greedy, wary, racer (fixed) twice, collector, self, and two strong linear agents
    (`p7d_q_lam98_self_s2@best`, `p8b_q_lam98_self_decay_s4@best`; the strongest,
    `p7b…s3@best`, held out for evaluation only). Shaping 1, n = 1, averaging 100,
    30000 games, 4 seeds; arms without and with `--ticket-plan`. Re-scored on 1000
    fresh games per opponent (`runs/dqn/rescore_pass3.*`), final / best, mean over seeds:

    | Arm | vs greedy | vs wary | vs racer (fixed) | vs collector | Tickets done vs racer |
    | --- | --- | --- | --- | --- | --- |
    | pool | +3.3 / +5.1 | −0.8 / +0.4 | **+2.8 / +1.9** | +20.5 / +23.2 | 0.16 |
    | pool + ticket plan | +5.9 / +5.9 | +1.5 / +2.0 | **+2.5 / +2.6** | +23.3 / +25.1 | 0.32 |

    Readings: with a real racer in the pool the agents hold even with it (pass 2's lost
    to the fixed racer by 15–22), and against the held-out linear agent they are at −3
    to −4 in training evaluations. The price is exploiting greedy less (+3 to +6, pass
    2: +11 to +15). Still racing: 2 tickets kept, ~10 claims of length 4.1–4.4, ends
    the game itself in 62–98% of games depending on the opponent. The ticket plan
    doubles ticket completion (0.07 vs 0.03 against greedy, 0.32 vs 0.16 against racer)
    and is slightly ahead on every re-score, but tickets stay far below the ~2 kept.
    Both arms flatten from ~14k games.
  - DQN trained against random only (`scripts/dqn_random.ps1`, `runs/dqn/random/`,
    2026-09-29): pass 2's greedy-arm settings with random as the only opponent, 3 seeds,
    30000 games (`runs/dqn/rescore_random.*`). Final weights: −93 vs greedy, −90 vs
    racer, 0% wins; +128 vs random. It never learns racing: 16–17 claims of mean length
    2.2–2.6, ends the game itself 59% of the time against random and 9% against greedy,
    and scores only 48 against random where the greedy-trained agent scores 91. Its
    curve against random flattens at ~+130 by game 7000. So random-only training fails
    for DQN, while linear learners trained against random transfer well (the strongest
    linear agent is one). Untested explanations: against random the margin is swamped
    by random's own score (it keeps and fails ~9 tickets; −83 on average, widely
    spread), which the learner barely affects, so the +40 available from long routes is
    lost in the noise; the linear features hand it "route points" directly. Test:
    `--reward score` against random, which removes the opponent's score.
  - Round robin 3 (`runs/round_robin_2026-09-29.*`, 10 agents, 400 games per pair), Elo
    with greedy = 1000: linear `p7b…s3@best` 1149, DQN pass-3 pool `d3a_pool_s1@best`
    1118, racer 1114, DQN pass-2 pool + shaping 1104, DQN pass-3 pool + ticket plan
    `d3b_pool_plan_s0@best` 1071, wary 1051, greedy 1000, collector 835, random-only DQN
    1, random −746. The top four are within a few points: the pass-3 agent beats racer
    +1.5, pass 2's best +2.7, wary +1.5, greedy +4.8, and loses to the linear agent by
    2.8 (pass 2's: 6.4). Racer beats pass 2's best by 11.1; pass 3 closed that gap.
    Every strong agent is a racer; they differ in how well they race.
  - Done: tier C, PPO (`src/ttr/learn/ppo.py`, `ttr-train-ppo`; design in "Methods to
    compare"). `ppo:PATH[@best]` in the registry; the dashboard, `rescore.py` and
    `run_status.py` read PPO runs. Smoke run (3 seeds, 3000 games, pass-3 pool, shaping,
    learning rate annealed over those 3000): it learns (−97 → −70 to −84 vs greedy,
    beats random ~+115), slower than DQN early (DQN pass 3: about −45 at 3000 games);
    policy entropy 1.8 → 1.0–1.25; KL 0.006–0.008 and clip fraction 0.10–0.13 early.
    0.11 s a game with 3 runs at once.
  - PPO pass 1 (`scripts/ppo_pass1.ps1`, `runs/ppo/pass1/`, 2026-09-29, 2 h 56 min): DQN
    pass 3's setting (pool with the fixed racer twice, collector, self, two strong linear
    agents; shaping 1), default PPO settings, 50000 games, 4 seeds; arms without and with
    `--ticket-plan`. Re-scored on 1000 fresh games per opponent (`runs/ppo/rescore_pass1.*`),
    final / best, mean over seeds:

    | Arm | vs greedy | vs wary | vs racer | vs collector |
    | --- | --- | --- | --- | --- |
    | pool | **+24.2 / +24.1** (86%) | +17.5 / +17.5 | **+4.9 / +4.1** (62%) | +37.8 / +37.8 |
    | pool + ticket plan | +18.5 / +19.1 | +12.0 / +12.6 | +1.7 / +1.3 | +33.8 / +34.3 |

    Readings: **PPO is the strongest method so far**, by a clear margin, and its final
    policies need no checkpoint picking (final ≈ best; seeds +19 to +28 vs greedy).
    Against the held-out linear agent in training evaluations: +1.6 (pool), −3.6 (ticket
    plan). Slow start (−58 vs greedy at 5k games, DQN pass 3 −23), then steady: positive
    from ~17k, +20 at 30k, +24 at 50k, still creeping up when the learning rate reached 0
    (KL → 0). Policy entropy 1.5 → 0.37. The ticket plan hurt PPO (the reverse of DQN).
  - Round robin 4 (`runs/round_robin_2026-09-29b.*`, 10 agents, 400 games per pair), Elo
    with greedy = 1000: PPO `p1a_pool_s0` 1323, PPO + ticket plan `p1b_pool_plan_s3` 1284,
    PPO `p1a_pool_s1` 1245, linear `p7b…s3@best` 1162, racer 1130, DQN pass-3
    `d3a_pool_s1@best` 1117, wary 1058, greedy 1000, collector 840. Every PPO agent beats
    every non-PPO agent; the best beats the linear agent +6.6, racer +8.5, DQN +13.5,
    greedy +27.1.
  - What PPO does differently (200 instrumented games per matchup): the same race (2
    tickets abandoned, ~10 claims of mean length 4.4–4.8, almost only blind draws: 42 blind,
    1–2 face-up a game), plus **one connected network for the longest-path bonus**: it takes
    the bonus in 96–98% of games against racer and the linear agent, who get it 4–6% (10
    points a game), and 85% against DQN (16%). It ends the game itself 81–91% of the time
    against the other racers. Route points are about equal to theirs (93–94 vs 89–90).
    One inefficiency left: its abandoned tickets cost −22 a game against racer's −19 (it
    doesn't keep the cheapest two).
  - PPO pass 2 (`scripts/ppo_pass2.ps1`, `runs/ppo/pass2/`, 2026-09-29, 4 h 13 min): pass 1's
    pool arm for 100000 games (learning rate annealed over all of them), with and without
    shaping, 4 seeds each. Re-scored on 1000 fresh games per opponent
    (`runs/ppo/rescore_pass2.*`), final policies, mean over seeds (best checkpoints within 0.6):

    | Arm | vs greedy | vs wary | vs racer | vs collector | Seeds vs greedy |
    | --- | --- | --- | --- | --- | --- |
    | shaping 1 | +25.4 (87%) | +18.6 | +6.5 (65%) | +38.2 | +23 to +28 |
    | no shaping | **+26.1** (88%) | **+19.5** | **+7.2** (66%) | **+39.6** | +25 to +27 |

    Readings: PPO has plateaued. Twice the training added 1–2 points over pass 1 and made
    the seeds consistent (within 4 of each other, pass 1: 8); the curves flatten from
    ~70k games. **Shaping isn't needed for PPO**: without it the arm learned faster early
    (−42 vs −54 against greedy at 10k games, +15 vs +5 at 30k) and ends equal or slightly
    ahead; DQN was the reverse. The slower learning-rate decay made the early phase
    slower (pass 1 was at +20 at 30k games) for little gain at the end: the schedule
    mattered more than the length. The strategy is unchanged: race, one connected network
    (longest-path bonus in 91–97% of games), 2 tickets abandoned (−23 a game), 0.01–0.02
    completed.
  - Round robin 5 (`runs/round_robin_2026-09-29c.*`, 400 games per pair), Elo with greedy =
    1000: PPO pass 2 no shaping `p2b_pool_s3` 1311, PPO pass 1 `p1a_pool_s0` 1306, PPO pass
    2 shaping `p2a_pool_shape_s1` 1295, linear `p7b…s3@best` 1158, racer 1126, DQN pass 3
    1113, wary 1057, greedy 1000, collector 842. The three PPO agents are within 1.4 of
    each other head to head (a tie); each beats the linear agent by 5–7, racer by 7–9,
    DQN by 10–14, greedy by 27–28. `p2b_pool_s3` is the reference strongest agent.
- **Next:**
  - Within 2-player games every method has converged on racing (PPO adding the connected
    network); tickets and blocking never appeared. Options: 3–5 players (routes contested,
    blocking matters, tickets may pay), Phase 6 strategy analysis of the agents we have,
    MCTS (tier D) as a non-learning contrast, or the win/loss reward mode.
  - Ticket choice: PPO abandons pricier tickets than racer does (−23 vs −19 a game).
  - Phase 6 strategy analysis on the agents we have (PLAN "Roadmap"): the longest-path
    bonus as an emergent sub-strategy, tempo, blocking (none so far).
  - Tickets never appeared in any learner. With the collector result (more tickets
    loses in 2-player games), racing may be close to right for 2 players on this map;
    3–5 players is where tickets could matter more.
  - DQN has converged on racing across three passes and seven settings, and its best
    agents now tie racer and the best linear agent. Next method: PPO with self-play on
    the same pool (tier C), as planned. The race-to-the-end strategy is the finding for
    the tempo question.
  - Cheap side experiment: DQN against random with `--reward score`, to test the
    noise explanation for random-only training failing.

## Roadmap

0. **Scaffolding**: package layout, `pyproject.toml`, test runner, `.gitignore`.
1. **Game engine**: pure Python with no RL dependencies. It covers board data (USA map),
   game state, legal-move generation, rule enforcement, and scoring,
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
   plugged into the same agent interface and evaluated the same way, on the USA map.
6. **Strategy analysis**: metrics that capture play style (blocking rate — a route claimed
   by an opponent while on a player's ticket path — route-length
   distribution, ticket draw/keep behavior, tempo), compared across methods, plus
   pandas/matplotlib notebooks and the viewer's overlays.
7. **Later / optional**: 3–5 players, the AlphaZero-style stretch agent, and Unity +
   ML-Agents for presentation.

## Visualization (Phase 3)

A robust viewer before any training, alongside the ASCII board view (kept). The
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
   - **Built as:** `HumanControl` in `viz/app.py`, given the seats a person plays
     (`--human 0`). A live timeline produces nothing when a human seat is to act, so it
     stalls there until a click supplies the move; bots play themselves as before.
   - Moves with no board target (payments, ticket keeps, draw tickets, pass) are chips in
     the bottom panel's right side, which also answers what to do with that empty space.
     Keeping tickets is the one sub-step with no single click: chips toggle a selection
     and the last chip confirms it, enabled only when the selection is legal.
   - **Decided:** acting after stepping back forks. The states after the current one are
     dropped, as an undo would, rather than refusing the move.
   - Backing out of a claim is step-back (`,`), not a cancel button: the engine has no
     action for it, since `CHOOSE_PAYMENT` only offers payments.
7. **Analysis overlays:** from a folder of records, color routes by a statistic (claim
   rate, claim rate per agent, average turn claimed, how often contested).
   - The statistics live in `src/ttr/analysis.py` (no Pygame), one `replay()` per record,
     read from the `claim_route` events in `game.log`. They return plain data so Phase 6
     can reuse them from pandas; the viewer maps them to colors.
   - **"Contested" means a blocked double** (decided 2026-09-26): one side of a double
     route was claimed and the other side was closed by it (2–3 players, §6). It is read
     straight off the log. The richer meaning, "claimed by an opponent while on a player's
     ticket path", needs ticket-path computation and is Phase 6's blocking-rate metric.
   - *(Done.)* Colors run dark blue → light blue → light yellow → light red → dark red,
     least to most significant (ColorBrewer RdYlBu/RdBu stops). The average turn is
     reversed: an early claim marks a route in demand, so early is red. Rates start at 0, so 0% reads as measured (blue), not as no
     data. The average turn spans its observed range. Each statistic shares one scale
     across everyone, each agent and each seat, so switching compares like with like.
     Routes with nothing to measure are dark gray. Filters: agent, seat, or both.
   - While an overlay is on, the seat panels show each seat's averages over the records
     (`Summary.player`: win %, score breakdown, ticket completion %, longest bonus,
     claims, trains)
     and its top routes for the statistic, in place of the replayed game's hands and
     tickets. `Summary.player_rows()` has the same results per player per game. The overlay hides claimed trains and the scoreboard; the panels still show
     the replayed game (the folder's first record unless `--record` names one).

**The ASCII board view stays** (decided 2026-09-26; this was once milestone 8, "retire
it"). It costs nothing to keep and runs without Pygame, e.g. `ttr-sim --show board`.
The `rich` text log stays too.

### Viewer backlog (tweaks and fixes, in no fixed order)

Noticed while using the viewer; none of them blocks Phase 4.

- **Full screen.** *(Done.)* `F11` / `--fullscreen`: `Screen.fit` scales to the display's
  height and widens the side panels to fill the width (no bars on 16:9); the window loop
  centers the screen and shifts mouse positions by the offset. `ttr-shot --fit WxH`
  renders the same layout headlessly. Not handled: Windows display scaling above 100%
  (SDL is not made DPI-aware, so a scaled display shows a blurred upscale).

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
- **The bottom panel's empty space.** *(Partly done, milestone 6:* the right side now
  holds a human seat's move chips.*)* It is still empty while a bot is to act. Candidates
  for that case: a longest-path / ticket-progress summary, the last few actions in full
  rather than the one-line ticker, or a larger hand and ticket display.
- **Several live games in a row.** *(Done.)* `ttr-view --games N` (`Series` in
  `viz/app.py`): game g uses seed + g with the agents dealt into random seats, as `ttr-sim`
  batches do (with a human seat nobody moves); the scoreboard stays up `--advance`
  seconds, then the next game starts; `n` / `p` move between games, each kept as left.
  Still open: `--record DIR` to replay a folder of saved games the same way.
- **Random seating, and colors that follow the agent.** *(Done.)* `simulate.seating`
  deals the agents into a random seat order per game (seeded by batch seed and game
  number) and the first seat is drawn from the game's seed, for `ttr-sim` batches,
  `ttr-sim --show` and `ttr-view` live games. Rotation only ever produced the cyclic
  orders (with 3 agents, A→B→C but never A→C→B); now every order comes up. A seat's
  color is its agent's position in `--agents` (first red, then blue, green, yellow,
  black) on its trains, panel and scoreboard, wherever it sits; records store the
  seating (`slots`). Records also store `first_player_drawn`: drawing the first seat
  uses one random number before the shuffle, so a replay has to draw it again rather
  than pass the seat in. Older records lack both fields and replay as before.
- **Ticket markers on the map.** *(Done.)* One seat's open tickets (the viewed seat, else
  the human seat, else P0) get a (shape, color) pair each, drawn beside both cities and
  leading the ticket in the panel (`viz/tickets.py`). The five base pairs (pink moon,
  yellow star, light green square, light blue circle, orange triangle) go first, then the
  shapes with colors rotated: 25 pairs, never two open tickets on the same one. A ticket
  keeps its pair until completed, which removes its markers and frees the pair. An offer
  reserves pairs as it is dealt, so ticked tickets show (hollow) the pair they will keep.
  Cities on marked tickets have their dots turned yellow (`BoardView.draw(city_fill=)`).
  Assignment is rebuilt from the game log; for that the engine now logs each player's
  initial ticket deal (`deal_initial_tickets`, private), as it already did later draws.
- **Better ticket display.** *(Partly done.)* Seat panels write full city names
  (`Montreal – New Orleans 13`); P0's wide panel lists tickets in columns of three rows,
  each as wide as its longest line, narrowing (smaller type, then an ellipsis) only when
  the columns would pass the panel's edge or a human's move chips. Side panels fit each
  line to the panel width the same way. Completed tickets get a drawn check mark (the
  panel font has no ✓ glyph). Still open: small ticket cards with the route drawn on
  them, and points that don't read as a card count.
- **End-of-game results popup.** *(Done, milestone 6.)* `panels.draw_result` overlays the
  scoreboard on the board: winner, then every seat's route points, ticket points with
  completed/failed counts, longest path, bonus and total.
- **Charts for the strategy analysis (Phase 6).** Beyond the per-route overlays in
  milestone 7: claim rate over time, which tickets get completed and when, route length
  distribution, blocking frequency, how the methods differ on each. These belong with the
  analysis work (matplotlib/pandas out of a folder of records), not in the Pygame viewer,
  but the records already hold everything they need.
- **In-place replacement of the face-up pile.** *(Done.)* Taking a face-up card used to
  slide the cards to its right down to the left, with the replacement always landing in
  the rightmost slot. `Game._refill_market(at=slot)` now puts the replacement back in the
  slot that was taken, as at a real table. It matters beyond looks: a human clicking a
  face-up card should not have the rest of the row jump under the cursor, and Phase 4
  encodes the market by position, so a shifting row is noise the agent has to learn
  around. When the deck is spent the row closes up instead (§9 #3).

**Library:** `pygame-ce` (decided), the actively maintained drop-in fork of Pygame
(same `import pygame`), with Python 3.14 wheels.

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

- **Tier A features (first pass, `src/ttr/learn/features.py`).** Q(s, a) =
  w[type(a)] · φ(s, a), one weight vector per action type (claim, pay, draw a color,
  draw a Locomotive, draw blind, draw tickets, keep tickets, pass), each over the same
  state features plus that type's action features. State: trains left (mine, fewest
  opponent's), final round, hand size, incomplete tickets, trains to finish them,
  tickets that can no longer be finished, route-point margin. Claim / pay: route
  points and length, on the cheapest path of one of my tickets, those tickets'
  points, completes a ticket, triggers the end (pay adds Locomotives spent and
  cards other path routes still need). Draws: color needed by my paths, second draw,
  a useful face-up card on offer. Ticket choice: count, points, trains to finish,
  fits the uncommitted trains, unreachable, opening choice, points lost if kept on
  the last turn. All from the acting seat's view.
- **Tier B design (`src/ttr/learn/dqn.py`).** Double DQN on the env's observation
  vector (765 numbers, memory level 2), an MLP 765 → 512 → 256 → 168 with a dueling
  head (value + advantage, the advantage centered over the legal actions: tier A's
  second pass showed the value/advantage split matters). Illegal actions are −inf when
  acting and in the target's argmax. The setting is tier A's, so the two compare
  directly: one seat against an opponent (random, greedy, or self = greedy or its own
  best checkpoint), reward = margin / 100 between the learner's decisions, γ = 1.
  - **n-step returns** play λ's part: a game is ~85 decisions and ticket points arrive
    at the end. γ = 1, uncorrected (exploration at ε 0.02 is rare enough).
  - Replay: 200k decisions, float16 observations, filled a game at a time (so the
    n-step returns are known), uniform sampling; target copied every 1000 steps; Adam
    1e-4, Huber loss, gradient norm clipped at 10; ε 1.0 → 0.02 over the first 5% of
    games.
  - Weight averaging (`--average`), the best checkpoint by margin against greedy, and
    the evaluation schedule all as in tier A. Evaluations add a third opponent, the
    strongest linear agent.
  - The learner drives the engine directly, as tier A does, rather than stepping the
    PettingZoo env: the same encoder, mask and action decoding the env uses, so it sees
    what the env would show that seat, without an env step for the opponent's moves.
- **Tier C design (`src/ttr/learn/ppo.py`).** PPO (CleanRL's recipe) on the same observation
  and the same setting as tier B, so the three tiers compare directly: separate actor and
  critic MLPs (512 → 256, tanh, orthogonal init), illegal logits set to −1e8 (invalid
  action masking), 32 games per update, GAE (γ = 1, λ = 0.95), 4 epochs × 4 minibatches,
  clip 0.2 (value loss clipped too), entropy 0.01, value 0.5, gradient norm 0.5, Adam
  2.5e-4 annealed to 0. Opponents, league self-play, shaping, the ticket-plan block,
  evaluations and best checkpoints are DQN's. The policy samples in training (its only
  exploration); evaluations and `ppo:PATH` take its most likely move.
- **Why linear rather than tabular for tier A:** tabular methods are only feasible on a
  tiny map, and there is none. Linear features keep the same update rules and the same
  Q-learning vs. SARSA comparison on the full map.
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
  765 numbers for 2 players; in general 100(N+1) + 4N + 10(N−1) + 447. Built by
  `ObservationEncoder` in `src/ttr/env/observation.py`, which documents each block's
  slice and scale:

  | Block | Encoding | Size (2p) |
  | --- | --- | --- |
  | Current sub-step | one-hot over the 5 phases | 5 |
  | Route ownership | per route: unclaimed / mine / each opponent | 300 |
  | Route open to me | per route: claimable by me (double-route rule, trains left) | 100 |
  | Route being paid for | one-hot, payment step only | 100 |
  | Market | counts per color | 9 |
  | Pile sizes | train deck, discard, ticket deck | 3 |
  | Per player | trains left, public score, hand size, ticket count | 8 |
  | Endgame | final round started; I still have a turn in it | 2 |
  | Card memory (Level 2) | per opponent: 9 known + 1 unknown; unseen pool: 9 | 19 |
  | My hand | counts per color | 9 |
  | My tickets | held, multi-hot by ticket ID | 30 |
  | Ticket completed | per held ticket: already connected by my routes | 30 |
  | Trains to finish | per held ticket: fewest trains still needed + an "impossible" flag | 60 |
  | Tickets on offer | 3 offer slots × one-hot ticket ID (matches `KeepTickets` action order) | 90 |

  - **The two endgame flags are equal for the player to act** (every final-round turn is
    that player's last). They differ only in a waiting seat's view, after it has played.
  - **Tickets are identified by ID.** The board never changes, so the network can learn
    what each ticket means.
  - **Computed values** ("open to me", "completed", "trains to finish", endgame flags) are
    exact computations on information the agent already has. The endgame flags are
    required because the final-turn ticket draw is unmasked.
  - **Trains to finish** is the fewest trains needed to connect a held ticket's cities,
    counting the player's own routes as free and using only routes still open to them.
    It changes only when a route is claimed, so compute it then and cache it.
    "Impossible" means no such path, or more trains needed than the player has left. It leans
    furthest toward strategy of the computed values; switching it off is a possible
    experiment.
  - **Tier A doesn't use this vector.** Linear Q-learning/SARSA needs hand-made features
    of state *and* action, designed with that tier. The vector is for DQN, PPO and a
    later AlphaZero-style agent.
  - **A graph network** (cities as nodes, routes as edges) is a possible later experiment.
  - **Optional ticket-plan block** (decided 2026-09-28, off by default): 224 more numbers
    with which routes serve my tickets (and their points), which complete one, how
    offered tickets fit, and the cards my plan needs. Added because DQN never linked
    ticket IDs to routes on its own; the details are in `src/ttr/env/observation.py`.
- **Reward: an env setting, defaulting to dense score margin.** The reward decides which
  strategies are worth learning at all, so it is an experiment variable, like memory level.

  | Mode | Signal | Rewards blocking? |
  | --- | --- | --- |
  | Own score | + route points on each claim; tickets and longest route at game end | No |
  | **Score margin (default)** | my points − mean of opponents' points, same schedule | Yes |
  | Win/loss | +1 / −1 at game end, 0 for a shared win | Yes |

  - **Dense isn't shaping:** in the score modes, rewards summed over a game equal exactly
    the final score or margin. Earlier feedback, same objective.
  - **Default is margin** because it rewards blocking, learns faster than sparse win/loss,
    and suits every tier (DQN and linear TD struggle with win/loss alone). Scale by about
    1/100.
  - **Margin with 3+ players** uses the mean of the opponents' scores, so the margins of
    all seats sum to 0 and the scale doesn't grow with the player count.
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

- **Engine speed**: *(settled for now.)* Legal moves are cached per state; 12.6k
  sub-steps/s engine-only, 11.2k with the env's mask, about 4.4k adding the observation
  (USA, 2 players, random play). Revisit if
  training throughput needs more, e.g. incremental claimability per route.

## Notes

- **Run data is local only** (decided 2026-09-28: development is on one machine).
  `runs/` (run JSON, network `.pt` files, logs) is gitignored; only the small summaries
  behind this file's tables are tracked (`runs/*/rescore_*`, `runs/round_robin_*`,
  `runs/linear/overnight-*.md`). Linear passes 1–8 and DQN pass 1 were committed
  before that and remain in git history (up to commit 4859cc8), so an old run can be
  restored with `git checkout 4859cc8 -- runs/<path>`.
- **Testing an old commit needs `PYTHONPATH`.** The venv installs `ttr` editable, so its
  `.pth` entry points at this working tree's `src/` whatever is checked out elsewhere. A
  `git worktree` or a bisected checkout therefore runs its own tests against the *current*
  renderer and engine, which fails or passes for the wrong reason. Run those with
  `PYTHONPATH=<worktree>/src` so the checkout's own source wins:

      git worktree add /tmp/wt <commit>
      cd /tmp/wt && PYTHONPATH=/tmp/wt/src <repo>/.venv/Scripts/python -m pytest -q

- **Leftover branch `phase3-analysis-overlays`** (local and on `origin`). Work happens on
  `main` (see CLAUDE.md); this branch was made by mistake for Phase 3 milestone 7 and was
  fast-forwarded into `main` on 2026-09-26, so it holds nothing `main` lacks. Safe to
  delete: `git branch -d phase3-analysis-overlays` and
  `git push origin --delete phase3-analysis-overlays`.
