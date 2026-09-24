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
- **RL algorithms**: our own CleanRL-style single-file scripts (DQN, PPO), and
  hand-rolled linear Q-learning/SARSA. Not SB3 or RLlib (see PLAN.md
  "Methods to compare").
- **Tree-search agents** (MCTS with determinization is in the method roster): hand-rolled,
  same approach as the CS440 Uno agent but lighter in Python — no external
  MCTS library
- **Live visualization during dev**: Pygame (board/route rendering)
- **Terminal debugging**: `rich` for formatted state/tables
- **Training & strategy analysis**: matplotlib/pandas, Jupyter for
  interactive exploration
- **Presentation layer**: deferred. Unity + ML-Agents is a possible later
  addition — Python-trained agents could plug in without redoing the RL work.
  Not a current priority.

## Design decisions & open questions

The roadmap, settled decisions (with rationale), and open questions live in
`PLAN.md` — check there before proposing design changes. Settled so far:

- **2 players first**, scaling to 3–5 later (engine should support N players).
- **Full USA map and full ruleset** for bulk development. Don't propose a
  simplified ruleset — a tiny toy map exists only as a smoke test, loaded as a
  different board data file with the same rules.

- **Trained agents get Level 2 memory of public information**: known opponent cards
  plus unseen-pool counts, computed by the env. Memory level is a configurable knob
  (see PLAN.md). Scripted bots can be weakened on purpose (memory level, final-turn
  guards) to make easier opponents.
- **Claiming is two steps:** choose the route, then choose the payment.
- **Methods to compare:** linear Q-learning vs. SARSA, DQN, PPO self-play, MCTS
  (AlphaZero-style as a stretch goal). The aim is comparing methods, not only finding
  the strongest.

Still open: observation encoding (beyond card memory), the rest of the action-space
encoding, reward shaping.

## Conventions

- Setup: `py -3.11 -m venv .venv` then `.venv/Scripts/python -m pip install -e ".[dev]"`
  (Python 3.11+).
- Tests: `.venv/Scripts/python -m pytest`.
- Layout: engine in `src/ttr/` (`game.py` rules/state, `board.py` + `data/*.json`
  board data, `scoring.py`, `actions.py`), tests in `tests/`. Code comments like
  `§9 #N` point to the rulings in `docs/RULES.md`.
- `src/ttr/data/usa.json` has been hand-verified against the physical board
  (`"verified": true`). If a route or ticket seems wrong, check the board before
  changing the data.
- `PLAN.md` holds the plan, roadmap, and reasoning behind decisions — update it
  when a decision is made or the roadmap changes. At the end of each session, update
  its **Current progress** section (just completed / current / next) and its date.
- `DEVLOG.md` is the user's own handwritten session notes. **Do not edit it.**
- **Game rules:** use `docs/RULES.md`, a Markdown transcription of the official rules
  with this project's rulings for edge cases the PDF doesn't cover. The original is
  `docs/tt_rules_2015_en.pdf` and is the final authority. If the two disagree, fix
  RULES.md. Don't implement rules from memory.
