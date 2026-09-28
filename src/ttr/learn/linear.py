"""Linear Q-learning and SARSA on hand-made features (PLAN.md "Methods to
compare", tier A).

    ttr-train-linear --algo q --opponent mixed --games 2000 --out runs/linear/q_mixed.json
    ttr-sim --agents linear:runs/linear/q_mixed.json greedy --games 200
    ttr-sim --agents linear:runs/linear/q_mixed.json@best greedy       # the best checkpoint

Q(s, a) = v . state(s) + w[type(a)] . psi(s, a): a value shared by every action
plus an advantage per action type (features in ttr.learn.features). The learner
plays one seat against an opponent (random, greedy, "mixed" = a coin flip
between them each game, or "self" = a coin flip between greedy and the
learner's own best checkpoint so far); its seat is random each game. It decides
at every one of its engine sub-steps, and the reward between two of its
decisions is the change in `reward_values` (ttr.env.reward, default the score
margin, scaled by 1/100) over everything that happened in between, opponent
moves included. gamma = 1: every game ends.

    Q-learning  target = r + max_a' Q(s', a')          (off-policy)
    SARSA       target = r + Q(s', a'), a' the action actually taken, exploration included
    terminal    target = r

    delta = target - Q(s, a);  n = x . x  (x = [state(s), psi(s, a)], the sample's features)
    z = lambda * z + x                      (eligibility trace; lambda 0 gives z = x)
    weights += alpha * (1 - lambda) * delta * z / n     ((1 - lambda) floored at 0.01)

The step is normalized by the sample's features (normalized LMS), so alpha is
the share of the error corrected on that sample. With `lam` > 0 the error also
reaches earlier decisions through the trace: SARSA(lambda), or Watkins's
Q(lambda) for Q-learning, which clears the trace after an exploratory action.
A trace sums about 1/(1 - lambda) feature vectors, so the step is scaled by
(1 - lambda) to keep `alpha` comparable across lambda. A game is about 85-100
of the learner's decisions, so lambda must be near 1 for the end-of-game
reward to reach the opening: 0.9^100 is 3e-5, 0.97^100 is 0.05, 0.99^100 is 0.37.
Exploration is epsilon-greedy, epsilon decaying linearly over the first part of
training.

Options, all off by default (the second-pass behavior):
    alpha_end   alpha falls linearly from `alpha` to this over the run
    lam         eligibility traces (TD(lambda)); try 0.97-0.99 (see above)
    average     keep an exponential average of the weights over about this many
                games; evaluations, checkpoints and the saved weights use it
                (the raw weights still learn and act in training)
    shaping     potential-based shaping, Phi = -shaping * trains still needed for
                my incomplete tickets (a lost ticket counts 45) * reward_scale,
                reward += Phi(s') - Phi(s), Phi = 0 at the end (Ng et al., 1999:
                the best policy doesn't change). Evaluations still report the true score.

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
from ttr.learn.features import ACTION_TYPES, STATE_FEATURES, Context, feature_names, features_with_context
from ttr.learn.metrics import METRICS, game_metrics, mean_metrics

ALGOS = ("q", "sarsa")
OPPONENTS = {
    "random": lambda seed: RandomAgent(seed),
    "greedy": lambda seed: GreedyAgent(seed),
}
# Training opponents: the bots, either at random each game, or "self" (built in train()).
TRAIN_OPPONENTS = {
    **OPPONENTS,
    "mixed": lambda seed: (GreedyAgent if random.Random(seed).random() < 0.5 else RandomAgent)(seed),
    "self": None,
}
VALUE = "value"  # the weight block over the state features


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


def expected_features() -> Dict[str, List[str]]:
    return {VALUE: list(STATE_FEATURES), **{k: list(feature_names(k)) for k in ACTION_TYPES}}


def zero_weights() -> Dict[str, np.ndarray]:
    return {k: np.zeros(len(names)) for k, names in expected_features().items()}


class LinearAgent:
    """Acts greedily on Q (ties at random), or epsilon-greedily while training."""

    def __init__(self, weights: Optional[Dict[str, np.ndarray]] = None, epsilon: float = 0.0,
                 seed: Optional[int] = None, name: str = "linear") -> None:
        self.weights = weights if weights is not None else zero_weights()
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.name = name
        self.last_context: Optional[Context] = None

    def evaluate(self, game: Game, p: int
                 ) -> Tuple[List[Action], np.ndarray, List[Tuple[str, np.ndarray]], List[float]]:
        """Legal actions, the state vector, each action's (type, psi) and Q values.
        The decision's Context is kept in `last_context`."""
        actions = game.legal_actions()
        ctx, feats = features_with_context(game, p, actions)
        self.last_context = ctx
        value = float(self.weights[VALUE] @ ctx.state)
        qs = [value + float(self.weights[kind] @ psi) for kind, psi in feats]
        return actions, ctx.state, feats, qs

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
            "features": expected_features(),
            "weights": {k: [round(float(x), 6) for x in w] for k, w in self.weights.items()},
        }

    def save(self, path: Path, **meta) -> None:
        """Write the file whole: to a temporary file, then swapped in, so a reader
        (ttr-dash --live) never sees half of it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({**meta, **self.to_dict()}, indent=1), encoding="utf-8")
        swap_in(tmp, path)

    @classmethod
    def load(cls, path: Path, seed: Optional[int] = None, name: Optional[str] = None,
             best: bool = False) -> "LinearAgent":
        """The final weights, or with `best` the best checkpoint's. Weights are
        matched by feature name: a feature added since the file was saved starts
        at 0, which plays exactly as the saved agent did."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        saved_names = data.get("features", {})
        if best and not data.get("best"):
            raise ValueError(f"{path} has no best checkpoint")
        source = data["best"]["weights"] if best else data["weights"]
        weights = {}
        for k, names in expected_features().items():
            if k not in saved_names:
                raise ValueError(f"{path}: no {k!r} weights; saved by an older layout, retrain")
            known = dict(zip(saved_names[k], source[k]))
            unknown = set(known) - set(names)
            if unknown:
                raise ValueError(f"{path}: {k} features {sorted(unknown)} are unknown to this version")
            weights[k] = np.array([known.get(n, 0.0) for n in names], dtype=np.float64)
        return cls(weights, seed=seed, name=name or f"linear:{path}{'@best' if best else ''}")


# ------------------------------------------------------------------ training


@dataclass
class TrainConfig:
    algo: str = "q"
    opponent: str = "random"
    games: int = 2000
    alpha: float = 0.05
    alpha_end: Optional[float] = None  # None: constant alpha
    lam: float = 0.0  # eligibility-trace decay; 0 = one-step TD
    average: int = 0  # weight-averaging window in games; 0 = off
    shaping: float = 0.0  # potential-based shaping, points per train still needed; 0 = off
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


def alpha_at(cfg: TrainConfig, g: int) -> float:
    if cfg.alpha_end is None:
        return cfg.alpha
    f = min(1.0, g / max(1, cfg.games - 1))
    return cfg.alpha + f * (cfg.alpha_end - cfg.alpha)


def _update(agent: LinearAgent, kind: str, psi: np.ndarray, state: np.ndarray, target: float,
            alpha: float) -> float:
    """One-step update of Q(s, a) toward `target` (the lambda = 0 case)."""
    v, w = agent.weights[VALUE], agent.weights[kind]
    delta = target - float(v @ state) - float(w @ psi)
    step = alpha * delta / (float(state @ state) + float(psi @ psi))
    v += step * state
    w += step * psi
    return delta


class Trace:
    """Eligibility trace over the weight blocks: z = lambda * z + x after each
    decision, and an update moves every block by step * z."""

    def __init__(self, weights: Dict[str, np.ndarray], lam: float) -> None:
        self.lam = lam
        self.z = {k: np.zeros_like(w) for k, w in weights.items()}

    def visit(self, kind: str, psi: np.ndarray, state: np.ndarray) -> None:
        for z in self.z.values():
            z *= self.lam
        self.z[VALUE] += state
        self.z[kind] += psi

    def clear(self) -> None:
        for z in self.z.values():
            z[:] = 0.0

    def apply(self, weights: Dict[str, np.ndarray], step: float) -> None:
        for k, z in self.z.items():
            if z.any():
                weights[k] += step * z


def _potential(ctx: Optional[Context], cfg: TrainConfig) -> float:
    if not cfg.shaping or ctx is None:
        return 0.0
    return -cfg.shaping * ctx.ticket_trains * cfg.reward_scale


def train_game(agent: LinearAgent, opponent, game: Game, seat: int, cfg: TrainConfig,
               alpha: Optional[float] = None) -> Dict[str, float]:
    """Play one game, learning at every decision of `seat`. Returns the seat's
    game metrics (ttr.learn.metrics) plus `decisions`, `mean_abs_td` and
    `shaping` (the shaping reward summed over the game)."""
    alpha = cfg.alpha if alpha is None else alpha
    scale = 1.0 if cfg.reward_mode == "win" else cfg.reward_scale
    last_value = reward_values(game, cfg.reward_mode)[seat]
    last_phi = 0.0
    shaped = 0.0
    trace = Trace(agent.weights, cfg.lam) if cfg.lam else None
    prev: Optional[Tuple[str, np.ndarray, np.ndarray]] = None  # (type, psi, state)
    deltas: List[float] = []

    def learn(target: float) -> None:
        kind, psi, state = prev
        if trace is None:
            deltas.append(_update(agent, kind, psi, state, target, alpha))
            return
        delta = target - float(agent.weights[VALUE] @ state) - float(agent.weights[kind] @ psi)
        step = alpha * max(1.0 - cfg.lam, 0.01) * delta / (float(state @ state) + float(psi @ psi))
        trace.apply(agent.weights, step)
        deltas.append(delta)

    while not game.game_over:
        p = game.current_player
        if p != seat:
            game.step(opponent.act(game, p))
            continue
        value = reward_values(game, cfg.reward_mode)[seat]
        r, last_value = scale * (value - last_value), value
        actions, state, feats, qs = agent.evaluate(game, p)
        phi = _potential(agent.last_context, cfg)
        if prev is not None:
            r += phi - last_phi
            shaped += phi - last_phi
        last_phi = phi
        i = agent.choose(qs)
        if prev is not None:
            learn(r + (max(qs) if cfg.algo == "q" else qs[i]))
            if trace is not None and cfg.algo == "q" and qs[i] < max(qs) - 1e-12:
                trace.clear()  # Watkins: no credit through an exploratory step
        kind, psi = feats[i]
        if trace is not None:
            trace.visit(kind, psi, state)
        prev = (kind, psi, state)
        game.step(actions[i])
    r = scale * (reward_values(game, cfg.reward_mode)[seat] - last_value) - last_phi  # Phi(end) = 0
    shaped -= last_phi
    if prev is not None:
        learn(r)
    row = game_metrics(game, seat)
    row["vs_greedy"] = float(getattr(opponent, "name", "") == "greedy")
    row["vs_self"] = float(getattr(opponent, "name", "") == "self")
    row["decisions"] = float(len(deltas))
    row["mean_abs_td"] = statistics.mean(abs(d) for d in deltas) if deltas else 0.0
    row["shaping"] = shaped
    row["alpha"] = alpha
    return row


def evaluate(agent, opponent: str, games: int, board: Board, seed: int = 0,
             max_turns: int = 1000) -> Dict[str, float]:
    """Mean game metrics (ttr.learn.metrics) of `agent` over `games` against a
    scripted bot, random seats. A LinearAgent plays greedily (epsilon 0). Any
    `Agent` works, so the scripted bots get the same numbers as baselines.
    `opponent` is a bot name, or any agent spec (ttr.agents.registry)."""
    make = OPPONENTS.get(opponent)
    if make is None:
        from ttr.agents.registry import make_agent

        make = lambda s: make_agent(opponent, s)
    saved = getattr(agent, "epsilon", None)
    if saved is not None:
        agent.epsilon = 0.0
    rng = random.Random(seed)
    rows = []
    try:
        for g in range(games):
            game = Game(board, num_players=2, seed=seed * 100_000 + g, max_turns=max_turns)
            seat = rng.randrange(2)
            opp = make(seed * 1000 + g)
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

    games      one row per training game: metrics + game, epsilon, alpha, seat, decisions,
               mean_abs_td, shaping, vs_greedy, vs_self
    history    one entry per evaluation: games so far, epsilon, the training means
               since the previous one, and eval[opponent] = mean metrics
    snapshots  the evaluated weights at each evaluation (games -> type -> list)
    baselines  the scripted bots on the final evaluation's games: bot -> opponent -> metrics
    best       the evaluated weights at the evaluation with the best margin against greedy

    `agent` learns; `policy` is what is evaluated and saved: the same agent, or
    the averaged weights with `average`.
    """

    agent: LinearAgent
    policy: Optional[LinearAgent] = None
    games: List[Dict[str, float]] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)
    snapshots: List[dict] = field(default_factory=list)
    baselines: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    eval_games: int = 0
    best: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = self.agent

    def save(self, path: Path, cfg: TrainConfig, finished: bool = True, **meta) -> None:
        """`finished` False marks a snapshot taken mid-run (--live)."""
        progress = {"games": len(self.games), "of": cfg.games, "finished": finished}
        if self.policy is not self.agent:
            meta["raw_weights"] = self.agent.to_dict()["weights"]
        self.policy.save(path, config=asdict(cfg), eval_games=self.eval_games, metrics=METRICS, progress=progress,
                         history=self.history, snapshots=self.snapshots, baselines=self.baselines, games=self.games,
                         best=self.best, **meta)


EVAL_OPPONENTS = ("random", "greedy")


def _eval_seed(i: int) -> int:
    return 10_000 + i


def train(cfg: TrainConfig, eval_every: int = 0, eval_games: int = 100,
          log=print, agent: Optional[LinearAgent] = None, baselines: bool = True,
          on_progress: Optional[Callable[["TrainResult"], None]] = None, progress_every: int = 0) -> TrainResult:
    """Train for cfg.games games. Every `eval_every` games (0 = only at the end)
    the evaluated policy plays `eval_games` against random and against greedy, and
    its weights are snapshotted. With `baselines`, the random and greedy bots then
    play the final evaluation's games too, for comparison. `on_progress(result)`
    is called every `progress_every` games and after every evaluation (live runs)."""
    if cfg.algo not in ALGOS:
        raise ValueError(f"algo must be one of {ALGOS}")
    if cfg.opponent not in TRAIN_OPPONENTS:
        raise ValueError(f"opponent must be one of {sorted(TRAIN_OPPONENTS)}")
    if cfg.reward_mode not in REWARD_MODES:
        raise ValueError(f"reward mode must be one of {REWARD_MODES}")
    if not 0 <= cfg.lam <= 1:
        raise ValueError("lambda must be in [0, 1]")
    board = load_board(cfg.board)
    agent = agent or LinearAgent(seed=cfg.seed)
    policy = None
    if cfg.average:
        policy = LinearAgent({k: w.copy() for k, w in agent.weights.items()}, seed=cfg.seed, name="averaged")
    out = TrainResult(agent, policy, eval_games=eval_games)
    rng = random.Random(cfg.seed)
    window: List[Dict[str, float]] = []
    start = time.perf_counter()
    for g in range(cfg.games):
        agent.epsilon = epsilon_at(cfg, g)
        game = Game(board, num_players=2, seed=rng.getrandbits(32), max_turns=cfg.max_turns)
        opp_seed = rng.getrandbits(32)
        if cfg.opponent == "self":
            # Greedy, or the best checkpoint so far (greedy until the first evaluation).
            if out.best is not None and random.Random(opp_seed).random() < 0.5:
                opponent = LinearAgent({k: np.array(v) for k, v in out.best["weights"].items()},
                                       seed=opp_seed, name="self")
            else:
                opponent = GreedyAgent(opp_seed)
        else:
            opponent = TRAIN_OPPONENTS[cfg.opponent](opp_seed)
        seat = rng.randrange(2)
        row = train_game(agent, opponent, game, seat, cfg, alpha=alpha_at(cfg, g))
        if policy is not None:
            rate = 1.0 / cfg.average
            for k, w in agent.weights.items():
                policy.weights[k] += rate * (w - policy.weights[k])
        row.update(game=float(g + 1), epsilon=agent.epsilon, seat=float(seat))
        out.games.append(row)
        window.append(row)
        done = g + 1
        if (eval_every and done % eval_every == 0) or done == cfg.games:
            entry = {
                "games": done,
                "epsilon": round(agent.epsilon, 4),
                "alpha": round(row["alpha"], 5),
                "seconds": round(time.perf_counter() - start, 1),
                "train": {k: statistics.mean(r[k] for r in window) for k in ("won", "margin", "mean_abs_td")},
                "eval": {
                    opp: evaluate(out.policy, opp, eval_games, board, seed=_eval_seed(len(out.history)),
                                  max_turns=cfg.max_turns)
                    for opp in EVAL_OPPONENTS
                },
            }
            out.history.append(entry)
            weights = out.policy.to_dict()["weights"]
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
                        help="training opponent; mixed = random or greedy, self = greedy or the learner's own "
                             "best checkpoint so far; a coin flip each game")
    parser.add_argument("--games", type=int, default=d.games)
    parser.add_argument("--alpha", type=float, default=d.alpha)
    parser.add_argument("--alpha-end", type=float, default=None,
                        help="let alpha fall linearly to this by the last game (default: constant)")
    parser.add_argument("--lambda", dest="lam", type=float, default=d.lam,
                        help="eligibility traces, TD(lambda); 0 (default) = one-step. A game is ~90 decisions, "
                             "so try 0.97-0.99")
    parser.add_argument("--average", type=int, default=d.average, metavar="GAMES",
                        help="evaluate and save an exponential average of the weights over about GAMES games (0 = off)")
    parser.add_argument("--shaping", type=float, default=d.shaping, metavar="POINTS",
                        help="potential-based shaping: POINTS per train still needed for my tickets (0 = off)")
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
        algo=args.algo, opponent=args.opponent, games=args.games, alpha=args.alpha, alpha_end=args.alpha_end,
        lam=args.lam, average=args.average, shaping=args.shaping,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_decay=args.epsilon_decay,
        reward_mode=args.reward, seed=args.seed,
    )
    agent = LinearAgent.load(args.init, seed=cfg.seed) if args.init else None
    extras = [f"{k} {getattr(cfg, k)}" for k in ("alpha_end", "lam", "average", "shaping") if getattr(cfg, k)]
    print(f"training {cfg.algo} vs {cfg.opponent}: {cfg.games} games, alpha {cfg.alpha}, "
          f"epsilon {cfg.epsilon_start} -> {cfg.epsilon_end}, reward {cfg.reward_mode}"
          + (f"; {', '.join(extras)}" if extras else ""))
    print(f"(best checkpoint kept by margin vs greedy; play it as linear:{args.out}@best)")
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
