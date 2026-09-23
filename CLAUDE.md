# TTR Strategy Emergence — Project Instructions

## What this project is

Recreating Ticket to Ride as a multi-agent RL environment in Python, training
agents on it, and studying what strategies/play styles emerge — route-hoarding
vs. blocking opponents, risk tolerance on long routes vs. short safe ones, etc.
The analysis of what trained agents converge on matters as much as the
engineering.

## Language / stack decisions (read before suggesting alternatives)

- **Python, not Java** — despite already having comparable Uno (MCTS) and Risk
  (Q-learning) frameworks in Java from CS440, the Python RL tooling ecosystem
  outweighs reusing that code. Don't suggest porting the Java game logic
  directly; it's reference material for edge cases already worked through, not
  a starting point.
- **Multi-agent env structure**: PettingZoo
- **Single-agent / self-play wrapping**: Gymnasium (if needed)
- **RL algorithms**: Stable-Baselines3 / CleanRL / RLlib
- **Tree-search agents** (if any MCTS-based agents are added): hand-rolled,
  same approach as the CS440 Uno agent but lighter in Python — no external
  MCTS library
- **Live visualization during dev**: Pygame (board/route rendering)
- **Terminal debugging**: `rich` for formatted state/tables
- **Training & strategy analysis**: matplotlib/pandas, Jupyter for
  interactive exploration
- **Presentation layer**: deferred. Unity + ML-Agents is a possible later
  addition — Python-trained agents could plug in without redoing the RL work.
  Not a current priority.

## Open design questions (still being decided — check DEVLOG.md for the latest)

- Full TTR ruleset (route claiming, ticket cards, longest-route bonus) vs. a
  simplified subset to start
- State/observation representation — how much of the board + hidden info
  (opponents' hands, remaining tickets) agents see
- Start with 2-agent games before scaling to full 2–5 player structure?
- Reward shaping — raw score vs. shaping toward specific behaviors to study

## Conventions

- Build/test commands: TBD once the project scaffolding exists — update this
  section as soon as there's a real entry point.
- Progress and reasoning behind decisions are logged in `DEVLOG.md`, not in
  this file. Check there for the "why" behind anything that looks like an
  in-progress decision.