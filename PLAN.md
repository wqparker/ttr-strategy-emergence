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

*Updated at the end of each session. Last updated: 2026-09-24.*

- **Just completed: engine ready for Phase 3.**
  - Phases 0–2 (engine, random/greedy bots, `rich` log, ASCII board view, seat-rotated
    match runner, stress test) merged into `main`.
  - USA map data hand-verified against the physical board (`"verified": true`).
  - Moved to Python 3.11.
- **Current: settle the open questions Phase 3 depends on**: observation encoding,
  action-space encoding, reward, RL library.
- **Next: Phase 3, the PettingZoo environment.** An AEC wrapper around the engine, with
  action masks built incrementally (see "Engine speed" under open questions).

## Roadmap

0. **Scaffolding**: package layout, `pyproject.toml`, test runner, `.gitignore`.
1. **Game engine**: pure Python with no RL dependencies. It covers board data (USA map plus
   the toy map), game state, legal-move generation, rule enforcement, and scoring,
   including the longest route. Unit tests cover the rule edge cases.
2. **Debugging tools and baselines**: `rich` terminal rendering of the game state.
   Random and simple greedy/heuristic agents serve as sanity checks and as evaluation
   opponents.
3. **PettingZoo environment**: an AEC wrapper around the engine, with action masking and
   the observation design (see open questions).
4. **Training**: self-play (e.g. PPO with action masking), evaluated against the baselines.
   Smoke-test on the toy map first, then train on the full map.
5. **Strategy analysis**: metrics that capture play style (blocking rate, route-length
   distribution, ticket draw/keep behavior, tempo), plus pandas/matplotlib notebooks.
6. **Later / optional**: 3–5 players, live Pygame visualization, an MCTS agent, and Unity +
   ML-Agents for presentation.

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
  - **Undecided:** whether the trained agent should get the final-turn guard as an action
    mask, or have to learn to avoid that move. Leaving it unmasked is also an interesting
    thing to measure.
- **Choosing the payment is a separate sub-step.** Claiming a route is two steps: choose
  the route, then choose the payment (the color for a gray route, how many Locomotives).
  This keeps each action mask small and the logic organized.

## Open questions

- **Observation encoding**: card memory is settled as fixed-size count vectors, not
  the event log: per opponent, 9 known counts + 1 unknown count, and 9 unseen-pool
  counts (see "Card memory"). Still open: how to encode routes, tickets, the market and
  the rest.
- **Action-space encoding**: the game has multi-step turns (drawing two cards, choosing
  which tickets to keep, choosing which cards to pay with). Payment is already decided
  as a sub-step. Should the rest be flat actions or sub-steps too? Also, how to keep the
  mask size manageable.
- **Reward**: raw final score or win/loss, vs. shaping toward specific behaviors. Shaping
  risks building in the very strategies being studied.
- **Engine speed**: about 10k steps/s now, and `step()` recomputes the legal-move list to
  validate each action. Revisit when legal moves become action masks in phase 3,
  probably with a cached mask per state.
- **Algorithm / library**: Stable-Baselines3 (sb3-contrib MaskablePPO), CleanRL, or RLlib.
  The choice depends on how easy multi-agent self-play is to set up.
