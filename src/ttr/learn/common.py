"""What the learners share (PLAN.md "Methods to compare"): evaluations on paired
games, the scripted baselines and whole-file run writing for every tier; and for
the network learners (ttr.learn.dqn, ttr.learn.ppo) the observation and mask,
the training-game loop with its reward and shaping, how opponents are drawn, the
best-checkpoint bookkeeping, saving and loading networks, and the command line.

Evaluation k of every run plays the same games (batch seed 10 000 + k), so runs
are compared on paired games. Needs the `[env]` extra (numpy); the few torch
helpers import torch when called, so the linear learner runs without `[deep]`.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ttr.actions import Action
from ttr.agents import GreedyAgent, RandomAgent
from ttr.agents.registry import make_agent
from ttr.board import Board
from ttr.env import actions as A
from ttr.env.observation import ObservationEncoder
from ttr.env.reward import REWARD_MODES, reward_values
from ttr.game import Game
from ttr.learn.metrics import game_metrics, mean_metrics

# Scripted bots by name: the baselines, and the linear learner's opponents.
OPPONENTS = {
    "random": lambda seed: RandomAgent(seed),
    "greedy": lambda seed: GreedyAgent(seed),
}
# The network learners' training opponents; "pool" draws from `pool` for each opponent seat.
TRAIN_OPPONENTS = ("random", "greedy", "wary", "racer", "self", "pool")
# Evaluation opponents; the best checkpoint is chosen on the mean margin over all but random.
EVAL_BOTS = ("random", "greedy", "wary", "racer")
# The strongest linear agent on fresh games (PLAN.md, pass 7 re-score: +17.4 vs greedy).
BEST_LINEAR = "linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best"
# Opponent kinds a training game's row flags as vs_<kind> (plus vs_linear and vs_dqn).
VS_KINDS = ("random", "greedy", "wary", "racer", "collector", "self")


# ------------------------------------------------------------------ files


def swap_in(tmp: Path, path: Path) -> None:
    """Replace `path` with the finished file `tmp` in one step, so a reader
    (ttr-dash --live) never sees half of it."""
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader has the file open for a moment
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def write_json(path: Path, data: Any) -> None:
    """Write `data` as JSON, whole: to a temporary file, then swapped in."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    swap_in(tmp, path)


# ------------------------------------------------------------------ evaluation


def eval_seed(i: int) -> int:
    """The batch seed of evaluation i, the same in every run."""
    return 10_000 + i


def evaluation_due(done: int, every: int, total: int) -> bool:
    """Evaluate after `done` games: every `every` games (0 = never) and at the end."""
    return bool(every and done % every == 0) or done == total


def evaluate(agent, opponent: str, games: int, board: Board, seed: int = 0,
             max_turns: int = 1000, num_players: int = 2) -> Dict[str, float]:
    """Mean game metrics (ttr.learn.metrics) of `agent` over `games` against a
    scripted bot, random seats. A LinearAgent plays greedily (epsilon 0). Any
    `Agent` works, so the scripted bots get the same numbers as baselines.
    `opponent` is a bot name, or any agent spec (ttr.agents.registry). With
    `num_players` > 2, every other seat is its own copy of `opponent`."""
    make = OPPONENTS.get(opponent) or (lambda s: make_agent(opponent, s))
    saved = getattr(agent, "epsilon", None)
    if saved is not None:
        agent.epsilon = 0.0
    rng = random.Random(seed)
    rows = []
    try:
        for g in range(games):
            game = Game(board, num_players=num_players, seed=seed * 100_000 + g, max_turns=max_turns)
            seat = rng.randrange(num_players)
            # the first copy's seed is the 2-player one, so 2-player evaluations are unchanged
            opps = [make(seed * 1000 + g + 100_000_000 * k) for k in range(num_players - 1)]
            by_seat = {q: opps.pop(0) for q in range(num_players) if q != seat}
            while not game.game_over:
                p = game.current_player
                game.step(agent.act(game, p) if p == seat else by_seat[p].act(game, p))
            rows.append(game_metrics(game, seat))
    finally:
        if saved is not None:
            agent.epsilon = saved
    return mean_metrics(rows)


def evaluations(agent, specs: Dict[str, str], games: int, board: Board, index: int, max_turns: int = 1000,
                num_players: int = 2) -> Dict[str, Dict[str, float]]:
    """Evaluation `index`: `agent` against each of `specs` (name -> opponent) on that evaluation's games."""
    return {name: evaluate(agent, spec, games, board, seed=eval_seed(index), max_turns=max_turns,
                           num_players=num_players)
            for name, spec in specs.items()}


def baseline_evaluations(opponents: Sequence[str], games: int, board: Board, index: int, max_turns: int = 1000,
              num_players: int = 2) -> Dict[str, Dict[str, Dict[str, float]]]:
    """The random and greedy bots on evaluation `index`'s games, for comparison: bot -> opponent -> metrics."""
    seed = eval_seed(index)
    return {bot: {opp: evaluate(OPPONENTS[bot](seed), opp, games, board, seed=seed, max_turns=max_turns,
                                num_players=num_players)
                  for opp in opponents}
            for bot in ("random", "greedy")}


# ------------------------------------------------------------------ network learners: playing


def observe(encoder: ObservationEncoder, game: Game, player: int) -> Tuple[np.ndarray, np.ndarray, List[Action]]:
    """What a learner sees and may do: the observation, the mask (bool) and the
    engine action behind each legal index (None elsewhere)."""
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    by_index: List[Optional[Action]] = [None] * A.N_ACTIONS
    for a in game.legal_actions():
        i = A.encode(game, a)
        mask[i] = True
        by_index[i] = a
    return encoder.encode(game, player), mask, by_index


def ticket_trains(encoder: ObservationEncoder, game: Game, p: int) -> int:
    """Trains `p` still needs for its incomplete tickets, a ticket that can no
    longer be finished counting a full supply (ttr.learn.features `ticket_trains`)."""
    full = game.board.trains_per_player
    trains = game.players[p].trains
    total = 0
    for tid in game.players[p].tickets:
        d = encoder.trains_to_finish(game, p, tid)
        total += full if d is None or d > trains else d
    return total


def play_learner_game(encoder: ObservationEncoder, decide: Callable[[np.ndarray, np.ndarray], int], opponent,
                      game: Game, seat: int, cfg) -> Tuple[List[float], float]:
    """Play `game` to the end: `seat` moves by `decide(observation, mask)` -> action
    index, every other seat by `opponent` (one agent, or a dict seat -> agent).

    Returns one reward per decision, the change in `reward_values` (cfg.reward_mode,
    x cfg.reward_scale; win/loss unscaled) from it to the next, opponent moves
    included; with `cfg.shaping`, plus Phi(next) - Phi(this) for Phi = -shaping x
    reward_scale x ticket_trains (Ng et al., 1999). Phi is 0 before any ticket is
    kept and at the end, so a game's shaping sums to 0 and the best policy doesn't
    change; that sum is returned too."""
    scale = 1.0 if cfg.reward_mode == "win" else cfg.reward_scale
    agent_at = opponent.__getitem__ if isinstance(opponent, dict) else (lambda p: opponent)
    rewards: List[float] = []
    shaping = 0.0
    decided = False
    last = reward_values(game, cfg.reward_mode)[seat]
    last_phi = 0.0
    while not game.game_over:
        p = game.current_player
        if p != seat:
            game.step(agent_at(p).act(game, p))
            continue
        value = reward_values(game, cfg.reward_mode)[seat]
        phi = -cfg.shaping * cfg.reward_scale * ticket_trains(encoder, game, p) if cfg.shaping else 0.0
        if decided:
            rewards.append(scale * (value - last) + phi - last_phi)
            shaping += phi - last_phi
        last, last_phi = value, phi
        obs, mask, by_index = observe(encoder, game, p)
        decided = True
        game.step(by_index[decide(obs, mask)])
    if decided:
        rewards.append(scale * (reward_values(game, cfg.reward_mode)[seat] - last) - last_phi)  # Phi(end) = 0
        shaping -= last_phi
    return rewards, shaping


def draw_opponent(cfg, seed: int, best, league: Iterable, frozen: Callable[[Any, int], Any]):
    """One opponent seat's agent for a training game, from its own `seed`: the bot
    `cfg.opponent`; "self" = a coin flip between greedy and `frozen(best, seed)`;
    or "pool" = one of `cfg.pool`, where "self" means a frozen copy of the learner,
    the best network or one of the `league` latest evaluated ones (greedy until the
    first evaluation)."""
    if cfg.opponent == "self":
        if best is not None and random.Random(seed).random() < 0.5:
            return frozen(best, seed)
        return GreedyAgent(seed)
    if cfg.opponent == "pool":
        name = random.Random(seed).choice(cfg.pool)
        if name != "self":
            return make_agent(name, seed)
        nets = [best, *league] if best is not None else []
        return frozen(random.Random(seed).choice(nets), seed) if nets else GreedyAgent(seed)
    return make_agent(cfg.opponent, seed)


def opponent_shares(opponents: Iterable) -> Dict[str, float]:
    """A training game's opponent flags: vs_<kind> is the share of the opponent
    seats of that kind (0 or 1 with one opponent)."""
    names = [getattr(o, "name", "") for o in opponents]

    def share(hit: Callable[[str], bool]) -> float:
        return sum(1 for n in names if hit(n)) / len(names)

    return {**{f"vs_{kind}": share(lambda n, kind=kind: n == kind) for kind in VS_KINDS},
            "vs_linear": share(lambda n: n.startswith("linear:")), "vs_dqn": share(lambda n: n.startswith("dqn:"))}


# ------------------------------------------------------------------ network learners: runs


def check_config(cfg, eval_opponents: Sequence[str]) -> None:
    """Fail on a bad setting before training, not mid-run."""
    if cfg.opponent not in TRAIN_OPPONENTS:
        raise ValueError(f"opponent must be one of {TRAIN_OPPONENTS}")
    if cfg.reward_mode not in REWARD_MODES:
        raise ValueError(f"reward mode must be one of {REWARD_MODES}")
    if "greedy" not in eval_opponents:
        raise ValueError("the evaluations must include greedy (runs are compared on it)")
    for name in cfg.pool if cfg.opponent == "pool" else ():
        if name != "self":
            make_agent(name, 0)  # an unknown name fails here, not mid-run


def record_evaluation(out, entry: dict, selected_on: Sequence[str], net, league) -> None:
    """Add evaluation `entry` to `out.history`. If its mean margin over
    `selected_on` beats the best so far, a copy of `net` becomes the best
    checkpoint (`out.best`, `out.best_net`); a copy joins `league` either way."""
    out.history.append(entry)
    selection = statistics.mean(entry["eval"][o]["margin"] for o in selected_on)
    if out.best is None or selection > out.best["selection"]:
        out.best = {"games": entry["games"], "margin_vs_greedy": entry["eval"]["greedy"]["margin"],
                    "selection": selection, "selected_on": list(selected_on)}
        out.best_net = copy.deepcopy(net)  # on the CPU, where the self-play opponent plays
    league.append(copy.deepcopy(net))


def progress_line(e: dict) -> str:
    t, ev = e["train"], e["eval"]
    line = (f"{e['games']:6d} games  eps {e['epsilon']:.3f}  steps {e['grad_steps']:7d}  "
            f"train win {t['won']:.2f} margin {t['margin']:+6.1f}  |td| {t['mean_abs_td']:.3f}  eval")
    for name, m in ev.items():
        line += f"  vs {name} win {m['win_share']:.2f} ({m['margin']:+.1f})"
    return line + f"  [{e['seconds']:.0f}s]"


def resolve_device(name: str):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def save_run(path: Path, nets: Dict[str, Any], method: str, config: dict, network: dict, **fields) -> None:
    """RUN.pt holds `nets` (name -> network, or None), RUN.json the run: method,
    config, the network's description, the weights file's name, then `fields` in
    order. Each is written whole, so a reader (ttr-dash --live) never sees half."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu = lambda net: None if net is None else {k: v.detach().cpu() for k, v in net.state_dict().items()}
    buf = io.BytesIO()
    torch.save({k: cpu(net) for k, net in nets.items()}, buf)
    weights = path.with_suffix(".pt")
    tmp = weights.with_name(weights.name + ".tmp")
    tmp.write_bytes(buf.getvalue())
    swap_in(tmp, weights)
    write_json(path, {"method": method, "config": config, "network": network, "weights_file": weights.name,
                      **fields})


def load_run(path: Path, method: str, best: bool = False, device: str = "cpu") -> Tuple[dict, dict]:
    """The network description and final (or best) weights of a run saved by `save_run`."""
    import torch

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("method") != method:
        raise ValueError(f"{path} is not a {method.upper()} run")
    nets = torch.load(path.with_name(data["weights_file"]), map_location=device, weights_only=True)
    state = nets["best" if best else "final"]
    if state is None:
        raise ValueError(f"{path} has no best checkpoint")
    return data["network"], state


# ------------------------------------------------------------------ network learners: command line


def add_run_arguments(parser: argparse.ArgumentParser, d) -> None:
    """The options DQN and PPO share; `d` is the config holding their defaults."""
    parser.add_argument("--opponent", choices=TRAIN_OPPONENTS, default=d.opponent,
                        help="training opponent; self = greedy or the learner's own best checkpoint so far, "
                             "a coin flip each game; pool = one of --pool for each opponent seat")
    parser.add_argument("--pool", nargs="+", default=list(d.pool), metavar="SPEC",
                        help="with --opponent pool: agent specs to draw from; 'self' = the best network or one "
                             "of the --league latest evaluated ones (greedy until the first evaluation)")
    parser.add_argument("--league", type=int, default=d.league, help="latest evaluated networks 'self' draws from")
    parser.add_argument("--shaping", type=float, default=d.shaping, metavar="POINTS",
                        help="potential-based shaping: POINTS per train still needed for my tickets (0 = off)")
    parser.add_argument("--ticket-plan", action="store_true",
                        help="add the observation's ticket-plan block: which routes serve my tickets, offered "
                             "tickets' cost and fit, cards my ticket paths still need (ttr.env.observation)")
    parser.add_argument("--games", type=int, default=d.games)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(d.hidden), metavar="UNITS",
                        help="hidden layer widths")
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--reward", choices=REWARD_MODES, default=d.reward_mode)
    parser.add_argument("--memory-level", type=int, choices=(0, 1, 2), default=d.memory_level)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--device", default=d.device, help="auto (cuda if available), cpu, cuda, cuda:1, ...")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch CPU threads (default 1: several runs share the CPU)")
    parser.add_argument("--eval-every", type=int, default=1000, help="games between evaluations (0 = at the end)")
    parser.add_argument("--eval-games", type=int, default=100, help="games against each evaluation opponent")
    parser.add_argument("--eval-opponents", nargs="+", default=list(EVAL_BOTS), metavar="SPEC",
                        help="evaluation opponents (must include greedy); the best checkpoint is picked on the "
                             "mean margin over all but random")
    parser.add_argument("--eval-linear", default=BEST_LINEAR, metavar="SPEC",
                        help="also evaluate against this agent (ttr.agents.registry spec); 'none' to skip")
    parser.add_argument("--out", type=Path, required=True,
                        help="run JSON (config, per-game rows, evaluations, baselines); networks go to the .pt beside it")
    parser.add_argument("--live", type=int, nargs="?", const=100, default=0, metavar="N",
                        help="also write the run files every N games (default 100) and after every evaluation")


def run_from_cli(args: argparse.Namespace, cfg, train: Callable, method: str, header: str) -> None:
    """Train `cfg` as the command line asks (`add_run_arguments`) and save the run."""
    import torch

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    torch.set_num_threads(args.threads)
    print(header)
    print(f"(best checkpoint kept by mean margin vs {', '.join(o for o in args.eval_opponents if o != 'random')}; "
          f"play it as {method}:{args.out}@best)")
    live = None
    if args.live:
        print(f"live: {args.out} rewritten every {args.live} games (watch with ttr-dash --live {args.out})")
        live = lambda result: result.save(args.out, finished=False)
    eval_linear = None if args.eval_linear.lower() == "none" else args.eval_linear
    result = train(cfg, eval_every=args.eval_every, eval_games=args.eval_games, eval_linear=eval_linear,
                   eval_opponents=tuple(args.eval_opponents), on_progress=live, progress_every=args.live)
    result.save(args.out)
    print(f"saved {args.out} and {args.out.with_suffix('.pt')}  (ttr-dash {args.out} to analyze)")
