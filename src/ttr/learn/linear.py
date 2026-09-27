"""Linear Q-learning and SARSA on hand-made features (PLAN.md "Methods to
compare", tier A).

    ttr-train-linear --algo q --opponent random --games 2000 --out runs/linear/q_random.json
    ttr-sim --agents linear:runs/linear/q_random.json greedy --games 200

Q(s, a) = w[type(a)] . phi(s, a), features in ttr.learn.features. The learner
plays one seat against a scripted bot; its seat is random each game. It decides
at every one of its engine sub-steps, and the reward between two of its
decisions is the change in `reward_values` (ttr.env.reward, default the score
margin, scaled by 1/100) over everything that happened in between, opponent
moves included. gamma = 1: every game ends.

    Q-learning  target = r + max_a' Q(s', a')          (off-policy)
    SARSA       target = r + Q(s', a'), a' the action actually taken, exploration included
    terminal    target = r

    w[type(a)] += alpha * (target - Q(s, a)) * phi(s, a) / (phi . phi)

The update is normalized by phi . phi (normalized LMS), so alpha is the share of
the error corrected on that sample whatever the number of features. Exploration
is epsilon-greedy, epsilon decaying linearly over the first part of training.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ttr.actions import Action
from ttr.agents import GreedyAgent, RandomAgent
from ttr.board import Board, load_board
from ttr.env.reward import REWARD_MODES, reward_values
from ttr.game import Game
from ttr.learn.features import ACTION_TYPES, all_features, feature_names

ALGOS = ("q", "sarsa")
OPPONENTS = {
    "random": lambda seed: RandomAgent(seed),
    "greedy": lambda seed: GreedyAgent(seed),
}


class LinearAgent:
    """Acts greedily on Q (ties at random), or epsilon-greedily while training."""

    def __init__(self, weights: Optional[Dict[str, np.ndarray]] = None, epsilon: float = 0.0,
                 seed: Optional[int] = None, name: str = "linear") -> None:
        self.weights = weights if weights is not None else {
            k: np.zeros(len(feature_names(k))) for k in ACTION_TYPES
        }
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.name = name

    def evaluate(self, game: Game, p: int) -> Tuple[List[Action], List[Tuple[str, np.ndarray]], List[float]]:
        """Legal actions, their (type, phi) and Q values."""
        actions = game.legal_actions()
        feats = all_features(game, p, actions)
        qs = [float(self.weights[kind] @ phi) for kind, phi in feats]
        return actions, feats, qs

    def choose(self, qs: Sequence[float]) -> int:
        if len(qs) > 1 and self.epsilon > 0 and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(qs))
        best = max(qs)
        return self.rng.choice([i for i, q in enumerate(qs) if q >= best - 1e-12])

    def act(self, game: Game, player: int) -> Action:
        actions, _, qs = self.evaluate(game, player)
        return actions[self.choose(qs)]

    # ------------------------------------------------------------ files

    def to_dict(self) -> dict:
        return {
            "features": {k: list(feature_names(k)) for k in ACTION_TYPES},
            "weights": {k: [round(float(x), 6) for x in w] for k, w in self.weights.items()},
        }

    def save(self, path: Path, **meta) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**meta, **self.to_dict()}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, seed: Optional[int] = None, name: Optional[str] = None) -> "LinearAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = {}
        for k in ACTION_TYPES:
            if data["features"].get(k) != list(feature_names(k)):
                raise ValueError(f"{path}: {k} features differ from this version's; retrain")
            weights[k] = np.array(data["weights"][k], dtype=np.float64)
        return cls(weights, seed=seed, name=name or f"linear:{path}")


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


@dataclass
class GameStats:
    won: bool
    margin: float
    score: int
    decisions: int
    mean_abs_td: float


def epsilon_at(cfg: TrainConfig, g: int) -> float:
    span = max(1, int(cfg.games * cfg.epsilon_decay))
    f = min(1.0, g / span)
    return cfg.epsilon_start + f * (cfg.epsilon_end - cfg.epsilon_start)


def _update(agent: LinearAgent, kind: str, phi: np.ndarray, target: float, alpha: float) -> float:
    w = agent.weights[kind]
    delta = target - float(w @ phi)
    w += alpha * delta * phi / float(phi @ phi)
    return delta


def train_game(agent: LinearAgent, opponent, game: Game, seat: int, cfg: TrainConfig) -> GameStats:
    """Play one game, learning at every decision of `seat`."""
    scale = 1.0 if cfg.reward_mode == "win" else cfg.reward_scale
    last_value = reward_values(game, cfg.reward_mode)[seat]
    prev: Optional[Tuple[str, np.ndarray]] = None
    deltas: List[float] = []
    while not game.game_over:
        p = game.current_player
        if p != seat:
            game.step(opponent.act(game, p))
            continue
        value = reward_values(game, cfg.reward_mode)[seat]
        r, last_value = scale * (value - last_value), value
        actions, feats, qs = agent.evaluate(game, p)
        i = agent.choose(qs)
        if prev is not None:
            nxt = max(qs) if cfg.algo == "q" else qs[i]
            deltas.append(_update(agent, *prev, r + nxt, cfg.alpha))
        prev = feats[i]
        game.step(actions[i])
    r = scale * (reward_values(game, cfg.reward_mode)[seat] - last_value)
    if prev is not None:
        deltas.append(_update(agent, *prev, r, cfg.alpha))
    return _stats(game, seat, len(deltas), deltas)


def _stats(game: Game, seat: int, decisions: int, deltas: Sequence[float] = ()) -> GameStats:
    result = game.result
    scores = [r.total for r in result.players]
    others = [s for q, s in enumerate(scores) if q != seat]
    return GameStats(
        won=seat in result.winners and len(result.winners) == 1,
        margin=scores[seat] - statistics.mean(others),
        score=scores[seat],
        decisions=decisions,
        mean_abs_td=statistics.mean(abs(d) for d in deltas) if deltas else 0.0,
    )


def evaluate(agent: LinearAgent, opponent: str, games: int, board: Board, seed: int = 0,
             max_turns: int = 1000) -> Dict[str, float]:
    """Greedy play (epsilon 0) against a scripted bot, random seats. Returns the
    win rate (shared wins count half), mean margin and mean score."""
    saved, agent.epsilon = agent.epsilon, 0.0
    wins = margins = scores = 0.0
    rng = random.Random(seed)
    try:
        for g in range(games):
            game = Game(board, num_players=2, seed=seed * 100_000 + g, max_turns=max_turns)
            seat = rng.randrange(2)
            opp = OPPONENTS[opponent](seed * 1000 + g)
            while not game.game_over:
                p = game.current_player
                game.step(agent.act(game, p) if p == seat else opp.act(game, p))
            result = game.result
            if seat in result.winners:
                wins += 1 / len(result.winners)
            s = _stats(game, seat, 0)
            margins += s.margin
            scores += s.score
    finally:
        agent.epsilon = saved
    return {"win_rate": wins / games, "margin": margins / games, "score": scores / games}


def train(cfg: TrainConfig, eval_every: int = 0, eval_games: int = 100,
          log=print, agent: Optional[LinearAgent] = None) -> Tuple[LinearAgent, List[dict]]:
    """Train for cfg.games games. Every `eval_every` games (0 = only at the end)
    the greedy policy plays `eval_games` against random and against greedy; the
    history holds those results and the training averages since the last one."""
    if cfg.algo not in ALGOS:
        raise ValueError(f"algo must be one of {ALGOS}")
    if cfg.opponent not in OPPONENTS:
        raise ValueError(f"opponent must be one of {sorted(OPPONENTS)}")
    if cfg.reward_mode not in REWARD_MODES:
        raise ValueError(f"reward mode must be one of {REWARD_MODES}")
    board = load_board(cfg.board)
    agent = agent or LinearAgent(seed=cfg.seed)
    rng = random.Random(cfg.seed)
    history: List[dict] = []
    window: List[GameStats] = []
    start = time.perf_counter()
    for g in range(cfg.games):
        agent.epsilon = epsilon_at(cfg, g)
        game = Game(board, num_players=2, seed=rng.getrandbits(32), max_turns=cfg.max_turns)
        opponent = OPPONENTS[cfg.opponent](rng.getrandbits(32))
        window.append(train_game(agent, opponent, game, rng.randrange(2), cfg))
        done = g + 1
        if (eval_every and done % eval_every == 0) or done == cfg.games:
            entry = {
                "games": done,
                "epsilon": round(agent.epsilon, 4),
                "train_win_rate": statistics.mean(s.won for s in window),
                "train_margin": statistics.mean(s.margin for s in window),
                "mean_abs_td": statistics.mean(s.mean_abs_td for s in window),
                "seconds": round(time.perf_counter() - start, 1),
            }
            for opp in ("random", "greedy"):
                res = evaluate(agent, opp, eval_games, board, seed=10_000 + len(history), max_turns=cfg.max_turns)
                entry.update({f"vs_{opp}_{k}": v for k, v in res.items()})
            history.append(entry)
            window = []
            log(_progress_line(entry))
    return agent, history


def _progress_line(e: dict) -> str:
    return (f"{e['games']:6d} games  eps {e['epsilon']:.3f}  train win {e['train_win_rate']:.2f} "
            f"margin {e['train_margin']:+6.1f}  |td| {e['mean_abs_td']:.3f}  "
            f"eval vs random win {e['vs_random_win_rate']:.2f} ({e['vs_random_margin']:+.1f})  "
            f"vs greedy win {e['vs_greedy_win_rate']:.2f} ({e['vs_greedy_margin']:+.1f})  "
            f"[{e['seconds']:.0f}s]")


def main(argv: Optional[Sequence[str]] = None) -> None:
    d = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=ALGOS, default=d.algo)
    parser.add_argument("--opponent", choices=sorted(OPPONENTS), default=d.opponent)
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
    parser.add_argument("--out", type=Path, required=True, help="weights + config + history JSON")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = TrainConfig(
        algo=args.algo, opponent=args.opponent, games=args.games, alpha=args.alpha,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_decay=args.epsilon_decay,
        reward_mode=args.reward, seed=args.seed,
    )
    agent = LinearAgent.load(args.init, seed=cfg.seed) if args.init else None
    print(f"training {cfg.algo} vs {cfg.opponent}: {cfg.games} games, alpha {cfg.alpha}, "
          f"epsilon {cfg.epsilon_start} -> {cfg.epsilon_end}, reward {cfg.reward_mode}")
    agent, history = train(cfg, eval_every=args.eval_every, eval_games=args.eval_games, agent=agent)
    agent.save(args.out, config=asdict(cfg), init=str(args.init) if args.init else None, history=history)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
