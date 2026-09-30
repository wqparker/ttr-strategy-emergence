"""Double DQN with action masking on the env's observation vector (PLAN.md
"Methods to compare", tier B). One file, CleanRL-style.

    ttr-train-dqn --opponent greedy --games 30000 --out runs/dqn/greedy_s0.json
    ttr-sim --agents dqn:runs/dqn/greedy_s0.json greedy --games 200
    ttr-sim --agents dqn:runs/dqn/greedy_s0.json@best greedy       # the best checkpoint

The setting is the linear learner's (ttr.learn.linear): one seat against an
opponent, a random seat each game. The opponent is a scripted bot (random,
greedy, wary, racer), "self" (a coin flip between greedy and the learner's own
best checkpoint so far), or "pool": one of `pool` drawn each game, where "self"
means a frozen copy of the learner, the best network or one of the `league`
latest evaluated ones (a small league, so no single trick beats the whole
pool). It decides at
every one of its engine sub-steps and sees exactly what the PettingZoo env
shows that seat: the ObservationEncoder vector (765 numbers for 2 players,
memory level 2) and the Discrete(168) mask. The reward between two of its
decisions is the change in `reward_values` (default the score margin, x 1/100)
over everything in between, opponent moves included. gamma = 1: every game ends.

    Q(s, .) = network(observation), 168 outputs; illegal actions are -inf when
              acting and in targets
    dueling:  Q(s, a) = V(s) + A(s, a) - mean of A(s, a') over the legal a'
    target:   G = r_t + ... + r_(t+n-1)                  (n-step, fewer if the game ends)
              y = G + Q_target(s', argmax over legal a' of Q_online(s', a'))
                  with s' = s_(t+n), and y = G if the game ended within n (Double DQN)
    loss:     Huber(Q_online(s_t, a_t) - y), Adam, gradient norm clipped

n-step returns do the job lambda did for the linear learner: a game is about
85-100 of the learner's decisions and ticket points arrive only at the end, so
one-step targets pass the end-of-game score back one decision per target
update. n = 1 is plain Double DQN.

A game's transitions go into the replay buffer when it ends (its n-step returns
are known then); then the network takes one gradient step per `train_every`
decisions of that game, on batches sampled uniformly from the buffer, once the
buffer holds `learning_starts` transitions. The target network is copied from
the online one every `target_every` gradient steps. Exploration is
epsilon-greedy over the legal actions, epsilon falling linearly over the first
`epsilon_decay` share of the games.

Speed (RTX 3080, i9-11900K): a gradient step costs about 4 ms on the GPU
whatever the batch size up to 2048 (kernel-launch bound), so the default is
fewer, larger batches (256 every 8 decisions). Choosing a move asks the network
about one observation, which takes 0.16 ms on one CPU thread and longer on the
GPU, so games are played by CPU copies of the networks and only the gradient
steps use `device`.

`average` keeps an exponential average of the network's weights over about
that many games (updated after each game); evaluations, checkpoints and the
saved "final" network use it, while the raw network learns and acts in training.
The linear learner's clearest gain came from this.

`shaping` adds potential-based shaping (Ng et al., 1999): Phi = -shaping x
reward_scale x the trains still needed for my incomplete tickets (a lost one
counting a full supply), reward += Phi(next) - Phi(this). Every claim that
advances a ticket pays at once instead of at the end; a game's shaping sums to
0, so the best policy doesn't change. Evaluations report the true score.

Evaluations play every opponent in `eval_opponents` (default random, greedy,
wary, racer) plus the best linear agent. Every evaluation that beats the best
mean margin so far over the scripted opponents other than random keeps a copy
of the evaluated network ("best"; DQN pass 1 picked on greedy alone). The same
evaluation picks and scores it, so its recorded score is optimistic;
scripts/rescore.py re-scores on fresh games.

Files: `--out RUN.json` holds everything the dashboard reads (ttr-dash: config,
per-game rows, evaluations, baselines; no weight pages), in the linear run
layout with "method": "dqn"; the networks go next to it in RUN.pt (final, best,
and the raw network when averaging). Both are rewritten whole, so a reader
never sees half a file. Needs the `[env]` and `[deep]` extras (numpy, torch).
"""

from __future__ import annotations

import argparse
import copy
import random
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ttr.actions import Action
from ttr.board import load_board
from ttr.env import actions as A
from ttr.env.observation import ObservationEncoder
from ttr.game import Game
from ttr.learn.common import (BEST_LINEAR, EVAL_BOTS, TRAIN_OPPONENTS, add_run_arguments,  # noqa: F401 (re-exported)
                              baseline_evaluations, check_config, draw_opponent, evaluation_due, evaluations,
                              load_run, observe, opponent_shares, play_learner_game, progress_line,
                              record_evaluation, resolve_device, run_from_cli, save_run)
from ttr.learn.metrics import METRICS, game_metrics


# ------------------------------------------------------------------ network


class QNetwork(nn.Module):
    """An MLP from the observation to 168 Q values, masked to -inf where illegal."""

    def __init__(self, obs_size: int, hidden: Sequence[int] = (512, 256), dueling: bool = True) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        width = obs_size
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.body = nn.Sequential(*layers)
        self.dueling = dueling
        self.head = nn.Linear(width, A.N_ACTIONS)
        self.value = nn.Linear(width, 1) if dueling else None

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """obs (B, obs_size) float, mask (B, 168) bool -> Q (B, 168)."""
        x = self.body(obs)
        q = self.head(x)
        if self.value is not None:
            m = mask.to(q.dtype)
            mean = (q * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1.0)
            q = self.value(x) + q - mean
        return q.masked_fill(~mask, float("-inf"))


class DQNAgent:
    """Acts greedily on Q over the legal actions, or epsilon-greedily while
    training. Keeps its own observation encoder, so it plays through the match
    runner, the viewer and the evaluations like any `Agent`."""

    def __init__(self, net: QNetwork, num_players: int = 2, memory_level: int = 2, ticket_plan: bool = False,
                 epsilon: float = 0.0,
                 seed: Optional[int] = None, name: str = "dqn") -> None:
        self.net = net
        self.encoder = ObservationEncoder(num_players, memory_level, ticket_plan)
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.name = name

    @property
    def device(self) -> torch.device:
        return next(self.net.parameters()).device

    def observe(self, game: Game, player: int) -> Tuple[np.ndarray, np.ndarray, List[Action]]:
        return observe(self.encoder, game, player)

    def q_values(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            return self.net(o, m)[0].cpu().numpy()

    def choose(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[int, float]:
        """(index, Q of the greedy choice)."""
        q = self.q_values(obs, mask)
        greedy = int(q.argmax())
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            legal = np.flatnonzero(mask)
            return int(legal[self.rng.randrange(len(legal))]), float(q[greedy])
        return greedy, float(q[greedy])

    def act(self, game: Game, player: int) -> Action:
        if player != game.current_player:
            raise ValueError(f"seat {player} asked to act on seat {game.current_player}'s turn")
        obs, mask, by_index = self.observe(game, player)
        return by_index[self.choose(obs, mask)[0]]


# ------------------------------------------------------------------ replay


def n_step_returns(rewards: Sequence[float], n: int) -> Tuple[np.ndarray, np.ndarray]:
    """For one game's rewards r_0..r_(T-1) (r_t follows decision t, the last one
    ends the game), gamma = 1: G_t = r_t + ... + r_(t+n-1) cut at the end, and
    whether the game ended within those n (then there is nothing to bootstrap)."""
    r = np.asarray(rewards, dtype=np.float64)
    t = np.arange(len(r))
    ends = np.minimum(t + n, len(r))
    c = np.concatenate([[0.0], np.cumsum(r)])
    return (c[ends] - c[t]).astype(np.float32), t + n >= len(r)


class Replay:
    """A ring buffer of the learner's decisions, filled a whole game at a time.
    Each decision's observation and mask are stored once; a transition points
    at the decision it bootstraps from (n later in the same game), which is
    newer and therefore overwritten after it. Observations are kept as float16."""

    def __init__(self, capacity: int, obs_size: int, n_step: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.n_step = n_step
        self.obs = np.zeros((capacity, obs_size), dtype=np.float16)
        self.mask = np.zeros((capacity, A.N_ACTIONS), dtype=bool)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.ret = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.boot = np.zeros(capacity, dtype=np.int64)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add_game(self, obs: Sequence[np.ndarray], masks: Sequence[np.ndarray], actions: Sequence[int],
                 rewards: Sequence[float]) -> None:
        count = len(actions)
        if not count:
            return
        if count > self.capacity:
            raise ValueError(f"a game of {count} decisions doesn't fit a buffer of {self.capacity}")
        idx = (self.pos + np.arange(count)) % self.capacity
        ret, done = n_step_returns(rewards, self.n_step)
        self.obs[idx] = np.asarray(obs)
        self.mask[idx] = np.asarray(masks)
        self.action[idx] = actions
        self.ret[idx] = ret
        self.done[idx] = done
        self.boot[idx] = np.where(done, idx, (idx + self.n_step) % self.capacity)  # done: unused, points at itself
        self.pos = (self.pos + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        i = self.rng.integers(0, self.size, batch)
        both = np.concatenate([i, self.boot[i]])  # each decision, then the one it bootstraps from
        obs = torch.as_tensor(self.obs[both], device=device).float()  # sent as float16
        mask = torch.as_tensor(self.mask[both], device=device)
        put = lambda x: torch.as_tensor(x, device=device)
        return (obs[:batch], mask[:batch], put(self.action[i]), put(self.ret[i]),
                obs[batch:], mask[batch:], put(self.done[i]))


def td_step(online: QNetwork, target: QNetwork, opt: torch.optim.Optimizer, batch: Tuple[torch.Tensor, ...],
            grad_clip: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """One Double DQN gradient step. Returns (loss, mean |TD error|) as tensors,
    left on the device so a game's steps need no synchronization."""
    obs, mask, action, ret, next_obs, next_mask, done = batch
    q = online(obs, mask).gather(1, action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        best = online(next_obs, next_mask).argmax(1, keepdim=True)
        q_next = target(next_obs, next_mask).gather(1, best).squeeze(1)
        y = ret + torch.where(done, torch.zeros_like(q_next), q_next)
    loss = F.smooth_l1_loss(q, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
    opt.step()
    return loss.detach(), (q.detach() - y).abs().mean()


# ------------------------------------------------------------------ training


@dataclass
class DQNConfig:
    opponent: str = "greedy"
    pool: Tuple[str, ...] = ("greedy", "wary", "racer", "self")  # with opponent "pool": one drawn per game
    league: int = 5  # "self" in a pool: the best network or one of this many latest evaluated ones
    shaping: float = 0.0  # potential-based shaping, points per train still needed; 0 = off
    games: int = 30000
    hidden: Tuple[int, ...] = (512, 256)
    dueling: bool = True
    n_step: int = 1
    lr: float = 1e-4
    lr_end: Optional[float] = None  # None: constant
    batch: int = 256
    buffer: int = 200_000  # decisions (about 2000 games)
    learning_starts: int = 10_000  # decisions in the buffer before the first gradient step
    train_every: int = 8  # learner decisions per gradient step (32 samples per decision)
    target_every: int = 1000  # gradient steps between target-network copies
    grad_clip: float = 10.0
    average: int = 0  # weight-averaging window in games; 0 = off
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.05  # share of the games over which epsilon falls to its end value
    reward_mode: str = "margin"
    reward_scale: float = 0.01
    memory_level: int = 2
    ticket_plan: bool = False  # add the observation's ticket-plan block (ttr.env.observation)
    seed: int = 0
    max_turns: int = 1000
    board: str = "usa"
    device: str = "auto"


def epsilon_at(cfg: DQNConfig, g: int) -> float:
    f = min(1.0, g / max(1, int(cfg.games * cfg.epsilon_decay)))
    return cfg.epsilon_start + f * (cfg.epsilon_end - cfg.epsilon_start)


def lr_at(cfg: DQNConfig, g: int) -> float:
    if cfg.lr_end is None:
        return cfg.lr
    return cfg.lr + min(1.0, g / max(1, cfg.games - 1)) * (cfg.lr_end - cfg.lr)


@dataclass
class Episode:
    """One training game from the learner's side: per decision the observation,
    mask, action index, the reward that followed it and the greedy Q."""

    obs: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    qs: List[float] = field(default_factory=list)
    shaping: float = 0.0  # the shaping reward summed over the game (0 up to rounding)


def play_training_game(learner: DQNAgent, opponent, game: Game, seat: int, cfg: DQNConfig) -> Episode:
    """One training game (rewards and shaping: ttr.learn.common.play_learner_game),
    the learner playing epsilon-greedily; `ep.shaping` is the shaping's sum (0)."""
    ep = Episode()

    def decide(obs: np.ndarray, mask: np.ndarray) -> int:
        i, q = learner.choose(obs, mask)
        ep.obs.append(obs)
        ep.masks.append(mask)
        ep.actions.append(i)
        ep.qs.append(q)
        return i

    ep.rewards, ep.shaping = play_learner_game(learner.encoder, decide, opponent, game, seat, cfg)
    return ep


@dataclass
class DQNResult:
    """Everything a run records, in the linear run layout (ttr.learn.linear
    TrainResult) so ttr-dash reads it; `snapshots` stays empty (no weights to
    chart). `best` holds the games and margin in the JSON and the network in
    the .pt file."""

    cfg: DQNConfig
    online: QNetwork
    policy: QNetwork
    games: List[Dict[str, float]] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)
    baselines: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    eval_games: int = 0
    best: Optional[dict] = None
    best_net: Optional[QNetwork] = None
    eval_linear: Optional[str] = None

    def save(self, path: Path, finished: bool = True) -> None:
        cfg = self.cfg
        nets = {"final": self.policy, "best": self.best_net}
        if self.policy is not self.online:
            nets["raw"] = self.online
        save_run(path, nets, "dqn", {**asdict(cfg), "hidden": list(cfg.hidden)},
                 {"obs_size": ObservationEncoder(2, cfg.memory_level, cfg.ticket_plan).size,
                  "hidden": list(cfg.hidden), "dueling": cfg.dueling, "memory_level": cfg.memory_level,
                  "ticket_plan": cfg.ticket_plan},
                 eval_linear=self.eval_linear, eval_games=self.eval_games, metrics=METRICS,
                 progress={"games": len(self.games), "of": cfg.games, "finished": finished},
                 history=self.history, snapshots=[], baselines=self.baselines, games=self.games, best=self.best)


def load_network(path: Path, best: bool = False, device: str = "cpu") -> Tuple[QNetwork, dict]:
    """The final (or best) network of a run saved by `DQNResult.save`, and what its
    observation needs: {"memory_level", "ticket_plan"} (DQNAgent's arguments)."""
    spec, state = load_run(path, "dqn", best, device)
    net = QNetwork(spec["obs_size"], spec["hidden"], spec["dueling"]).to(device)
    net.load_state_dict(state)
    net.eval()
    return net, {"memory_level": spec["memory_level"], "ticket_plan": spec.get("ticket_plan", False)}


def train(cfg: DQNConfig, eval_every: int = 0, eval_games: int = 100, eval_linear: Optional[str] = BEST_LINEAR,
          eval_opponents: Sequence[str] = EVAL_BOTS, log=print, baselines: bool = True,
          on_progress: Optional[Callable[[DQNResult], None]] = None, progress_every: int = 0) -> DQNResult:
    """Train for cfg.games games. Every `eval_every` games (0 = only at the end)
    the evaluated network plays `eval_games` against each of `eval_opponents`
    and (if given) the `eval_linear` agent, on the same games the linear runs
    used (seed 10 000 + k for evaluation k). The best checkpoint is the
    evaluation with the best mean margin over the scripted opponents other than
    random. With `baselines`, the random and greedy bots then play the final
    evaluation's games too. `on_progress(result)` is called every
    `progress_every` games and after every evaluation (live runs)."""
    check_config(cfg, eval_opponents)
    if cfg.n_step < 1:
        raise ValueError("n_step must be at least 1")
    selected_on = [o for o in eval_opponents if o != "random"]
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    board = load_board(cfg.board)
    obs_size = ObservationEncoder(2, cfg.memory_level, cfg.ticket_plan).size
    view = {"memory_level": cfg.memory_level, "ticket_plan": cfg.ticket_plan}  # every DQNAgent's observation
    online = QNetwork(obs_size, cfg.hidden, cfg.dueling).to(device)
    target = copy.deepcopy(online).requires_grad_(False)
    policy = copy.deepcopy(online).requires_grad_(False) if cfg.average else online
    # Playing asks for one observation at a time, which the CPU answers faster than
    # a GPU (launch latency), so games are played by CPU copies: `actor` follows the
    # online network after each game's gradient steps, `eval_net` takes the
    # evaluated network at each evaluation.
    cpu = torch.device("cpu")
    actor = online if device == cpu else copy.deepcopy(online).to(cpu).requires_grad_(False)
    eval_net = copy.deepcopy(policy).to(cpu).requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    replay = Replay(cfg.buffer, obs_size, cfg.n_step, seed=cfg.seed)
    learner = DQNAgent(actor, **view, seed=cfg.seed, name="dqn")
    evaluated = DQNAgent(eval_net, **view, seed=cfg.seed, name="dqn")
    out = DQNResult(cfg, online, policy, eval_games=eval_games, eval_linear=eval_linear)
    eval_specs = {**{o: o for o in eval_opponents}, **({"linear": eval_linear} if eval_linear else {})}
    league: Deque[QNetwork] = deque(maxlen=cfg.league)  # the latest evaluated networks, on the CPU
    frozen = lambda net, seed: DQNAgent(net, **view, seed=seed, name="self")  # a copy of the learner for self-play

    rng = random.Random(cfg.seed)
    window: List[Dict[str, float]] = []
    steps = 0  # gradient steps
    owed = 0.0  # decisions not yet paid for with a gradient step
    start = time.perf_counter()
    for g in range(cfg.games):
        learner.epsilon = epsilon_at(cfg, g)
        lr = lr_at(cfg, g)
        for group in opt.param_groups:
            group["lr"] = lr
        game = Game(board, num_players=2, seed=rng.getrandbits(32), max_turns=cfg.max_turns)
        opponent = draw_opponent(cfg, rng.getrandbits(32), out.best_net, league, frozen)
        seat = rng.randrange(2)
        ep = play_training_game(learner, opponent, game, seat, cfg)
        replay.add_game(ep.obs, ep.masks, ep.actions, ep.rewards)

        losses, tds = [], []
        if len(replay) >= cfg.learning_starts:
            owed += len(ep.actions) / cfg.train_every
            while owed >= 1:
                owed -= 1
                loss, td = td_step(online, target, opt, replay.sample(cfg.batch, device), cfg.grad_clip)
                losses.append(loss)
                tds.append(td)
                steps += 1
                if steps % cfg.target_every == 0:
                    target.load_state_dict(online.state_dict())
            if actor is not online:
                actor.load_state_dict(online.state_dict())
        if policy is not online:
            with torch.no_grad():
                for p_avg, p in zip(policy.parameters(), online.parameters()):
                    p_avg.lerp_(p, 1.0 / cfg.average)

        row = game_metrics(game, seat)
        row.update(
            game=float(g + 1), epsilon=learner.epsilon, lr=lr, seat=float(seat), **opponent_shares([opponent]),
            decisions=float(len(ep.actions)),
            shaping=ep.shaping,
            mean_abs_td=float(torch.stack(tds).mean()) if tds else 0.0,
            loss=float(torch.stack(losses).mean()) if losses else 0.0,
            q_mean=statistics.mean(ep.qs) if ep.qs else 0.0,
            grad_steps=float(steps),
        )
        out.games.append(row)
        window.append(row)
        done = g + 1
        if evaluation_due(done, eval_every, cfg.games):
            eval_net.load_state_dict(policy.state_dict())
            entry = {
                "games": done,
                "epsilon": round(learner.epsilon, 4),
                "lr": lr,
                "grad_steps": steps,
                "seconds": round(time.perf_counter() - start, 1),
                "train": {k: statistics.mean(r[k] for r in window) for k in ("won", "margin", "mean_abs_td")},
                "eval": evaluations(evaluated, eval_specs, eval_games, board, len(out.history), cfg.max_turns),
            }
            record_evaluation(out, entry, selected_on, eval_net, league)
            window = []
            log(progress_line(entry))
            if on_progress is not None:
                on_progress(out)
        elif on_progress is not None and progress_every and done % progress_every == 0:
            on_progress(out)
    if baselines:
        out.baselines = baseline_evaluations(eval_opponents, eval_games, board, len(out.history) - 1, cfg.max_turns)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    d = DQNConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser, d)
    parser.add_argument("--no-dueling", dest="dueling", action="store_false", help="a plain Q head")
    parser.add_argument("--n-step", type=int, default=d.n_step, help="n-step returns (1 = one-step Double DQN)")
    parser.add_argument("--lr-end", type=float, default=None, help="let lr fall linearly to this by the last game")
    parser.add_argument("--batch", type=int, default=d.batch)
    parser.add_argument("--buffer", type=int, default=d.buffer, help="replay capacity in decisions")
    parser.add_argument("--learning-starts", type=int, default=d.learning_starts, metavar="DECISIONS")
    parser.add_argument("--train-every", type=int, default=d.train_every, metavar="DECISIONS")
    parser.add_argument("--target-every", type=int, default=d.target_every, metavar="STEPS")
    parser.add_argument("--grad-clip", type=float, default=d.grad_clip, help="0 = off")
    parser.add_argument("--average", type=int, default=d.average, metavar="GAMES",
                        help="evaluate and save an exponential average of the weights over about GAMES games (0 = off)")
    parser.add_argument("--epsilon-start", type=float, default=d.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=d.epsilon_end)
    parser.add_argument("--epsilon-decay", type=float, default=d.epsilon_decay,
                        help="share of the games over which epsilon falls to its end value")
    args = parser.parse_args(argv)

    cfg = DQNConfig(
        opponent=args.opponent, pool=tuple(args.pool), league=args.league, shaping=args.shaping, games=args.games,
        hidden=tuple(args.hidden), dueling=args.dueling, n_step=args.n_step, lr=args.lr, lr_end=args.lr_end,
        batch=args.batch, buffer=args.buffer, learning_starts=args.learning_starts, train_every=args.train_every,
        target_every=args.target_every, grad_clip=args.grad_clip, average=args.average,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_decay=args.epsilon_decay,
        reward_mode=args.reward, memory_level=args.memory_level, ticket_plan=args.ticket_plan, seed=args.seed,
        device=args.device,
    )
    header = (f"training dqn vs {cfg.opponent} on {resolve_device(cfg.device)}: {cfg.games} games, "
              f"hidden {list(cfg.hidden)}{' dueling' if cfg.dueling else ''}, n-step {cfg.n_step}, lr {cfg.lr}"
              f"{f' -> {cfg.lr_end}' if cfg.lr_end is not None else ''}, batch {cfg.batch}, "
              f"epsilon {cfg.epsilon_start} -> {cfg.epsilon_end}, reward {cfg.reward_mode}"
              + (f", average {cfg.average}" if cfg.average else "") + (f", shaping {cfg.shaping}" if cfg.shaping else "")
              + (", ticket plan" if cfg.ticket_plan else "")
              + (f"; pool {' '.join(cfg.pool)}" if cfg.opponent == "pool" else ""))
    run_from_cli(args, cfg, train, "dqn", header)


if __name__ == "__main__":
    main()
