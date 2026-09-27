"""Linear Q-learning and SARSA on hand-made features (PLAN.md "Methods to
compare", tier A).

    ttr-train-linear --algo q --opponent mixed --games 2000 --out runs/linear/q_mixed.json
    ttr-sim --agents linear:runs/linear/q_mixed.json greedy --games 200
    ttr-sim --agents linear:runs/linear/q_mixed.json@best greedy       # the best checkpoint

Q(s, a) = v . state(s) + w[type(a)] . psi(s, a): a value shared by every action
plus an advantage per action type (features in ttr.learn.features). The learner
plays one seat against a scripted bot (random, greedy, or a coin flip between
them each game: "mixed"); its seat is random each game. It decides
at every one of its engine sub-steps, and the reward between two of its
decisions is the change in `reward_values` (ttr.env.reward, default the score
margin, scaled by 1/100) over everything that happened in between, opponent
moves included. gamma = 1: every game ends.

    Q-learning  target = r + max_a' Q(s', a')          (off-policy)
    SARSA       target = r + Q(s', a'), a' the action actually taken, exploration included
    terminal    target = r

    delta = target - Q(s, a);  n = state . state + psi . psi
    v         += alpha * delta * state / n
    w[type(a)] += alpha * delta * psi / n

The update is normalized (normalized LMS), so alpha is the share of the error
corrected on that sample whatever the number of features. Exploration is
epsilon-greedy, epsilon decaying linearly over the first part of training.

Every evaluation that beats the best margin against greedy so far keeps a copy
of the weights ("best" in the saved file, `linear:PATH@best` to play it). The
same evaluation both picks and scores it, so its recorded score is optimistic.

The run file is written once, when training ends. `--live [N]` also rewrites it
every N games (default 25) and after every evaluation, so `ttr-dash --live`
can follow the run while it trains; each write replaces the file whole.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ttr.actions import Action
from ttr.agents import GreedyAgent, RandomAgent
from ttr.board import Board, load_board
from ttr.env.reward import REWARD_MODES, reward_values
from ttr.game import Game
from ttr.learn.features import ACTION_TYPES, STATE_FEATURES, all_features, feature_names
from ttr.learn.metrics import METRICS, game_metrics, mean_metrics

ALGOS = ("q", "sarsa")
OPPONENTS = {
    "random": lambda seed: RandomAgent(seed),
    "greedy": lambda seed: GreedyAgent(seed),
}
# Training opponents: the bots, or either at random each game.
TRAIN_OPPONENTS = {
    **OPPONENTS,
    "mixed": lambda seed: (GreedyAgent if random.Random(seed).random() < 0.5 else RandomAgent)(seed),
}
VALUE = "value"  # the weight block over the state features


class LinearAgent:
    """Acts greedily on Q (ties at random), or epsilon-greedily while training."""

    def __init__(self, weights: Optional[Dict[str, np.ndarray]] = None, epsilon: float = 0.0,
                 seed: Optional[int] = None, name: str = "linear") -> None:
        self.weights = weights if weights is not None else {
            VALUE: np.zeros(len(STATE_FEATURES)),
            **{k: np.zeros(len(feature_names(k))) for k in ACTION_TYPES},
        }
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.name = name

    def evaluate(self, game: Game, p: int
                 ) -> Tuple[List[Action], np.ndarray, List[Tuple[str, np.ndarray]], List[float]]:
        """Legal actions, the state vector, each action's (type, psi) and Q values."""
        actions = game.legal_actions()
        state, feats = all_features(game, p, actions)
        value = float(self.weights[VALUE] @ state)
        qs = [value + float(self.weights[kind] @ psi) for kind, psi in feats]
        return actions, state, feats, qs

    def choose(self, qs: Sequence[float]) -> int:
        if len(qs) > 1 and self.epsilon > 0 and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(qs))
        best = max(qs)
        return self.rng.choice([i for i, q in enumerate(qs) if q >= best - 1e-12])

    def act(self, game: Game, player: int) -> Action:
        actions, _, _, qs = self.evaluate(game, player)
        return actions[self.choose(qs)]

    # ------------------------------------------------------------ files

    def to_dict(self) -> dict:
        return {
            "features": {VALUE: list(STATE_FEATURES), **{k: list(feature_names(k)) for k in ACTION_TYPES}},
            "weights": {k: [round(float(x), 6) for x in w] for k, w in self.weights.items()},
        }

    def save(self, path: Path, **meta) -> None:
        """Write the file whole: to a temporary file, then swapped in, so a reader
        (ttr-dash --live) never sees half of it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({**meta, **self.to_dict()}, indent=1), encoding="utf-8")
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # Windows: a reader has the file open for a moment
                time.sleep(0.05 * (attempt + 1))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path, seed: Optional[int] = None, name: Optional[str] = None,
             best: bool = False) -> "LinearAgent":
        """The final weights, or with `best` the best checkpoint's."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {VALUE: list(STATE_FEATURES), **{k: list(feature_names(k)) for k in ACTION_TYPES}}
        if data.get("features") != expected:
            raise ValueError(f"{path}: its features differ from this version's; retrain")
        if best and "best" not in data:
            raise ValueError(f"{path} has no best checkpoint")
        source = data["best"]["weights"] if best else data["weights"]
        weights = {k: np.array(source[k], dtype=np.float64) for k in expected}
        return cls(weights, seed=seed, name=name or f"linear:{path}{'@best' if best else ''}")


# ------------------------------------------------------------------ training


@dataclass
class TrainConfig:
    algo: str = "q"
    opponent: str = "random"
    games: int = 2000
    alpha: float = 0.05
    epsilon_start: float = 0.2
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.5  # share of the games over which epsilon falls to its end value
    reward_mode: str = "margin"
    reward_scale: float = 0.01
    seed: int = 0
    max_turns: int = 1000
    board: str = "usa"


def epsilon_at(cfg: TrainConfig, g: int) -> float:
    span = max(1, int(cfg.games * cfg.epsilon_decay))
    f = min(1.0, g / span)
    return cfg.epsilon_start + f * (cfg.epsilon_end - cfg.epsilon_start)


def _update(agent: LinearAgent, kind: str, psi: np.ndarray, state: np.ndarray, target: float,
            alpha: float) -> float:
    v, w = agent.weights[VALUE], agent.weights[kind]
    delta = target - float(v @ state) - float(w @ psi)
    step = alpha * delta / (float(state @ state) + float(psi @ psi))
    v += step * state
    w += step * psi
    return delta


def train_game(agent: LinearAgent, opponent, game: Game, seat: int, cfg: TrainConfig) -> Dict[str, float]:
    """Play one game, learning at every decision of `seat`. Returns the seat's
    game metrics (ttr.learn.metrics) plus `decisions` and `mean_abs_td`."""
    scale = 1.0 if cfg.reward_mode == "win" else cfg.reward_scale
    last_value = reward_values(game, cfg.reward_mode)[seat]
    prev: Optional[Tuple[str, np.ndarray, np.ndarray]] = None  # (type, psi, state)
    deltas: List[float] = []
    while not game.game_over:
        p = game.current_player
        if p != seat:
            game.step(opponent.act(game, p))
            continue
        value = reward_values(game, cfg.reward_mode)[seat]
        r, last_value = scale * (value - last_value), value
        actions, state, feats, qs = agent.evaluate(game, p)
        i = agent.choose(qs)
        if prev is not None:
            nxt = max(qs) if cfg.algo == "q" else qs[i]
            deltas.append(_update(agent, *prev, r + nxt, cfg.alpha))
        prev = (feats[i][0], feats[i][1], state)
        game.step(actions[i])
    r = scale * (reward_values(game, cfg.reward_mode)[seat] - last_value)
    if prev is not None:
        deltas.append(_update(agent, *prev, r, cfg.alpha))
    row = game_metrics(game, seat)
    row["vs_greedy"] = float(getattr(opponent, "name", "") == "greedy")
    row["decisions"] = float(len(deltas))
    row["mean_abs_td"] = statistics.mean(abs(d) for d in deltas) if deltas else 0.0
    return row


def evaluate(agent, opponent: str, games: int, board: Board, seed: int = 0,
             max_turns: int = 1000) -> Dict[str, float]:
    """Mean game metrics (ttr.learn.metrics) of `agent` over `games` against a
    scripted bot, random seats. A LinearAgent plays greedily (epsilon 0). Any
    `Agent` works, so the scripted bots get the same numbers as baselines."""
    saved = getattr(agent, "epsilon", None)
    if saved is not None:
        agent.epsilon = 0.0
    rng = random.Random(seed)
    rows = []
    try:
        for g in range(games):
            game = Game(board, num_players=2, seed=seed * 100_000 + g, max_turns=max_turns)
            seat = rng.randrange(2)
            opp = OPPONENTS[opponent](seed * 1000 + g)
            while not game.game_over:
                p = game.current_player
                game.step(agent.act(game, p) if p == seat else opp.act(game, p))
            rows.append(game_metrics(game, seat))
    finally:
        if saved is not None:
            agent.epsilon = saved
    return mean_metrics(rows)


@dataclass
class TrainResult:
    """Everything a run records; `save` writes it next to the weights.

    games      one row per training game: metrics + game, epsilon, seat, decisions, mean_abs_td
    history    one entry per evaluation: games so far, epsilon, the training means
               since the previous one, and eval[opponent] = mean metrics
    snapshots  the weights at each evaluation (games -> type -> list)
    baselines  the scripted bots on the final evaluation's games: bot -> opponent -> metrics
    best       the weights at the evaluation with the best margin against greedy
    """

    agent: LinearAgent
    games: List[Dict[str, float]] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)
    snapshots: List[dict] = field(default_factory=list)
    baselines: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    eval_games: int = 0
    best: Optional[dict] = None

    def save(self, path: Path, cfg: TrainConfig, finished: bool = True, **meta) -> None:
        """`finished` False marks a snapshot taken mid-run (--live)."""
        progress = {"games": len(self.games), "of": cfg.games, "finished": finished}
        self.agent.save(path, config=asdict(cfg), eval_games=self.eval_games, metrics=METRICS, progress=progress,
                        history=self.history, snapshots=self.snapshots, baselines=self.baselines, games=self.games,
                        best=self.best, **meta)


EVAL_OPPONENTS = ("random", "greedy")


def _eval_seed(i: int) -> int:
    return 10_000 + i


def train(cfg: TrainConfig, eval_every: int = 0, eval_games: int = 100,
          log=print, agent: Optional[LinearAgent] = None, baselines: bool = True,
          on_progress: Optional[Callable[["TrainResult"], None]] = None, progress_every: int = 0) -> TrainResult:
    """Train for cfg.games games. Every `eval_every` games (0 = only at the end)
    the greedy policy plays `eval_games` against random and against greedy, and
    the weights are snapshotted. With `baselines`, the random and greedy bots then
    play the final evaluation's games too, for comparison. `on_progress(result)`
    is called every `progress_every` games and after every evaluation (live runs)."""
    if cfg.algo not in ALGOS:
        raise ValueError(f"algo must be one of {ALGOS}")
    if cfg.opponent not in TRAIN_OPPONENTS:
        raise ValueError(f"opponent must be one of {sorted(TRAIN_OPPONENTS)}")
    if cfg.reward_mode not in REWARD_MODES:
        raise ValueError(f"reward mode must be one of {REWARD_MODES}")
    board = load_board(cfg.board)
    out = TrainResult(agent or LinearAgent(seed=cfg.seed), eval_games=eval_games)
    agent = out.agent
    rng = random.Random(cfg.seed)
    window: List[Dict[str, float]] = []
    start = time.perf_counter()
    for g in range(cfg.games):
        agent.epsilon = epsilon_at(cfg, g)
        game = Game(board, num_players=2, seed=rng.getrandbits(32), max_turns=cfg.max_turns)
        opponent = TRAIN_OPPONENTS[cfg.opponent](rng.getrandbits(32))
        seat = rng.randrange(2)
        row = train_game(agent, opponent, game, seat, cfg)
        row.update(game=float(g + 1), epsilon=agent.epsilon, seat=float(seat))
        out.games.append(row)
        window.append(row)
        done = g + 1
        if (eval_every and done % eval_every == 0) or done == cfg.games:
            entry = {
                "games": done,
                "epsilon": round(agent.epsilon, 4),
                "seconds": round(time.perf_counter() - start, 1),
                "train": {k: statistics.mean(r[k] for r in window) for k in ("won", "margin", "mean_abs_td")},
                "eval": {
                    opp: evaluate(agent, opp, eval_games, board, seed=_eval_seed(len(out.history)),
                                  max_turns=cfg.max_turns)
                    for opp in EVAL_OPPONENTS
                },
            }
            out.history.append(entry)
            weights = agent.to_dict()["weights"]
            out.snapshots.append({"games": done, "weights": weights})
            selection = entry["eval"]["greedy"]["margin"]
            if out.best is None or selection > out.best["margin_vs_greedy"]:
                out.best = {"games": done, "margin_vs_greedy": selection, "weights": weights}
            window = []
            log(_progress_line(entry))
            if on_progress is not None:
                on_progress(out)
        elif on_progress is not None and progress_every and done % progress_every == 0:
            on_progress(out)
    if baselines:
        seed = _eval_seed(len(out.history) - 1)
        for bot in ("random", "greedy"):
            out.baselines[bot] = {
                opp: evaluate(OPPONENTS[bot](seed), opp, eval_games, board, seed=seed, max_turns=cfg.max_turns)
                for opp in EVAL_OPPONENTS
            }
    return out


def _progress_line(e: dict) -> str:
    t, ev = e["train"], e["eval"]
    return (f"{e['games']:6d} games  eps {e['epsilon']:.3f}  train win {t['won']:.2f} "
            f"margin {t['margin']:+6.1f}  |td| {t['mean_abs_td']:.3f}  "
            f"eval vs random win {ev['random']['win_share']:.2f} ({ev['random']['margin']:+.1f})  "
            f"vs greedy win {ev['greedy']['win_share']:.2f} ({ev['greedy']['margin']:+.1f})  "
            f"[{e['seconds']:.0f}s]")


def main(argv: Optional[Sequence[str]] = None) -> None:
    d = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=ALGOS, default=d.algo)
    parser.add_argument("--opponent", choices=sorted(TRAIN_OPPONENTS), default=d.opponent,
                        help="training opponent; mixed = random or greedy, a coin flip each game")
    parser.add_argument("--games", type=int, default=d.games)
    parser.add_argument("--alpha", type=float, default=d.alpha)
    parser.add_argument("--epsilon-start", type=float, default=d.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=d.epsilon_end)
    parser.add_argument("--epsilon-decay", type=float, default=d.epsilon_decay,
                        help="share of the games over which epsilon falls to its end value")
    parser.add_argument("--reward", choices=REWARD_MODES, default=d.reward_mode)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--eval-every", type=int, default=200, help="games between evaluations (0 = at the end)")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--init", type=Path, help="start from these weights instead of zeros")
    parser.add_argument("--out", type=Path, required=True,
                        help="JSON: weights, config, per-game rows, evaluations, weight snapshots, baselines")
    parser.add_argument("--live", type=int, nargs="?", const=25, default=0, metavar="N",
                        help="also write the run file every N games (default 25) and after every evaluation, "
                             "for ttr-dash --live; off by default (one write at the end is faster)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = TrainConfig(
        algo=args.algo, opponent=args.opponent, games=args.games, alpha=args.alpha,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_decay=args.epsilon_decay,
        reward_mode=args.reward, seed=args.seed,
    )
    agent = LinearAgent.load(args.init, seed=cfg.seed) if args.init else None
    print(f"(best checkpoint kept by margin vs greedy; play it as linear:{args.out}@best)")
    print(f"training {cfg.algo} vs {cfg.opponent}: {cfg.games} games, alpha {cfg.alpha}, "
          f"epsilon {cfg.epsilon_start} -> {cfg.epsilon_end}, reward {cfg.reward_mode}")
    init = str(args.init) if args.init else None
    live = None
    if args.live:
        print(f"live: {args.out} rewritten every {args.live} games (watch with ttr-dash --live {args.out})")
        live = lambda result: result.save(args.out, cfg, finished=False, init=init)
    result = train(cfg, eval_every=args.eval_every, eval_games=args.eval_games, agent=agent,
                   on_progress=live, progress_every=args.live)
    result.save(args.out, cfg, init=init)
    print(f"saved {args.out}  (ttr-dash {args.out} to analyze)")


if __name__ == "__main__":
    main()
