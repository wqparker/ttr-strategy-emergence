"""Proximal Policy Optimization with action masking on the env's observation
vector (PLAN.md "Methods to compare", tier C). One file, CleanRL-style.

    ttr-train-ppo --opponent pool --shaping 1 --games 30000 --out runs/ppo/pool_s0.json
    ttr-sim --agents ppo:runs/ppo/pool_s0.json@best racer --games 200

The setting is the DQN learner's (ttr.learn.dqn), so the two compare directly:
one seat against an opponent (a scripted bot, "self" = greedy or the learner's
best checkpoint, or "pool": one of `pool` per game, where "self" is a frozen
copy of the learner, the best policy or one of the `league` latest evaluated
ones); a random seat each game; the ObservationEncoder vector and the
Discrete(168) mask; reward = change in the score margin x 1/100 between its
decisions, opponent moves included; optional potential-based `shaping` and the
`ticket_plan` observation block, exactly as in DQN. With `players` > 2 (`--players`)
every other seat draws its own opponent the same way, the margin is against the
opponents' mean, and evaluations fill every other seat with the evaluation opponent.
A network plays only the player count it was trained for (the observation's size
depends on it).

    policy  pi(a | s) = softmax of actor(s) over the legal actions (illegal logits
            set to -1e8: invalid action masking, Huang & Ontanon 2020)
    value   V(s) = critic(s)                       (separate MLPs, orthogonal init)

Each update plays `games_per_update` games with the current policy, sampling its
moves, and computes per game GAE advantages (gamma, lambda; the value after the
last decision is 0: the game is over) and returns G = A + V. Then
`update_epochs` passes over the batch in `minibatches` shuffled minibatches:

    ratio   = pi_new(a|s) / pi_old(a|s)
    policy  = -min(ratio * A, clip(ratio, 1 - clip, 1 + clip) * A)     (A normalized per minibatch)
    value   = 1/2 max((V - G)^2, (V_old + clip(V - V_old, +-clip) - G)^2)
    loss    = policy - ent_coef * entropy + vf_coef * value

Adam, the learning rate annealed linearly to 0, gradient norm clipped.
Exploration is the policy's own randomness (the entropy bonus keeps it from
collapsing early), not epsilon-greedy.

As in DQN, games are played by a CPU copy of the network (one observation at a
time is faster there) and the gradient steps use `device`.

Evaluations play the policy's most likely legal action (argmax; `eval_sample`
samples instead), against each of `eval_opponents` plus the best linear agent,
on the same paired games as every other learner; the best checkpoint is the
evaluation with the best mean margin over the scripted opponents other than
random. `ppo:PATH` in the agent registry plays the same way (argmax).

Files: RUN.json in the DQN / linear run layout ("method": "ppo"; ttr-dash reads
it), plus "updates" (per update: losses, entropy, approximate KL, clip
fraction); RUN.pt holds the final and best networks. Per-game rows: `epsilon` is
0 (no epsilon-greedy), `mean_abs_td` is the value error |G - V(s)|, the PPO
analog of the TD error, and `entropy` is the policy's mean entropy over the
game's decisions. Needs the `[env]` and `[deep]` extras.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import random
import statistics
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ttr.actions import Action
from ttr.agents import GreedyAgent
from ttr.agents.registry import make_agent
from ttr.board import load_board
from ttr.env import actions as A
from ttr.env.observation import ObservationEncoder
from ttr.env.reward import REWARD_MODES, reward_values
from ttr.game import Game
from ttr.learn.dqn import (BEST_LINEAR, EVAL_BOTS, TRAIN_OPPONENTS, _progress_line, observe, resolve_device,
                           ticket_trains)
from ttr.learn.linear import OPPONENTS, _eval_seed, evaluate, swap_in
from ttr.learn.metrics import METRICS, game_metrics

MASKED = -1e8  # logit of an illegal action: probability 0 without the NaNs -inf brings


# ------------------------------------------------------------------ network


def _layer(n_in: int, n_out: int, std: float = 2 ** 0.5) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp(n_in: int, hidden: Sequence[int], n_out: int, out_std: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    width = n_in
    for h in hidden:
        layers += [_layer(width, h), nn.Tanh()]
        width = h
    layers.append(_layer(width, n_out, out_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs over the observation (CleanRL's layout)."""

    def __init__(self, obs_size: int, hidden: Sequence[int] = (512, 256)) -> None:
        super().__init__()
        self.actor = _mlp(obs_size, hidden, A.N_ACTIONS, out_std=0.01)
        self.critic = _mlp(obs_size, hidden, 1, out_std=1.0)

    def logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.actor(obs).masked_fill(~mask, MASKED)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    return z - np.log(np.exp(z).sum())


class PPOAgent:
    """Plays the policy: samples a move (`sample`, as in training) or takes the
    most likely legal one (evaluations, the registry). Keeps its own encoder."""

    def __init__(self, net: ActorCritic, num_players: int = 2, memory_level: int = 2, ticket_plan: bool = False,
                 sample: bool = False, seed: Optional[int] = None, name: str = "ppo") -> None:
        self.net = net
        self.encoder = ObservationEncoder(num_players, memory_level, ticket_plan)
        self.sample = sample
        self.rng = np.random.default_rng(seed)
        self.name = name

    def step(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[int, float, float, float]:
        """(index, its log-probability, V(s), the policy's entropy) with sampling."""
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.net.logits(o, torch.as_tensor(mask).unsqueeze(0))[0].numpy().astype(np.float64)
            value = float(self.net.value(o)[0])
        logp = _log_softmax(logits)
        p = np.exp(logp)
        i = int(self.rng.choice(len(p), p=p / p.sum())) if self.sample else int(logits.argmax())
        return i, float(logp[i]), value, float(-(p * logp).sum())

    def act(self, game: Game, player: int) -> Action:
        if player != game.current_player:
            raise ValueError(f"seat {player} asked to act on seat {game.current_player}'s turn")
        obs, mask, by_index = observe(self.encoder, game, player)
        return by_index[self.step(obs, mask)[0]]


# ------------------------------------------------------------------ rollouts


@dataclass
class Episode:
    """One game from the learner's side, per decision."""

    obs: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    shaping: float = 0.0


def play_rollout_game(learner: PPOAgent, opponent, game: Game, seat: int, cfg: "PPOConfig") -> Episode:
    """Rewards and shaping exactly as ttr.learn.dqn.play_training_game. `opponent`
    plays every other seat, or is a dict of seat -> agent (one agent per seat)."""
    scale = 1.0 if cfg.reward_mode == "win" else cfg.reward_scale
    agent_at = opponent.__getitem__ if isinstance(opponent, dict) else (lambda p: opponent)
    ep = Episode()
    last = reward_values(game, cfg.reward_mode)[seat]
    last_phi = 0.0
    while not game.game_over:
        p = game.current_player
        if p != seat:
            game.step(agent_at(p).act(game, p))
            continue
        value = reward_values(game, cfg.reward_mode)[seat]
        phi = -cfg.shaping * cfg.reward_scale * ticket_trains(learner.encoder, game, p) if cfg.shaping else 0.0
        if ep.actions:
            ep.rewards.append(scale * (value - last) + phi - last_phi)
            ep.shaping += phi - last_phi
        last, last_phi = value, phi
        obs, mask, by_index = observe(learner.encoder, game, p)
        i, logp, v, entropy = learner.step(obs, mask)
        ep.obs.append(obs)
        ep.masks.append(mask)
        ep.actions.append(i)
        ep.logprobs.append(logp)
        ep.values.append(v)
        ep.entropies.append(entropy)
        game.step(by_index[i])
    if ep.actions:
        ep.rewards.append(scale * (reward_values(game, cfg.reward_mode)[seat] - last) - last_phi)  # Phi(end) = 0
        ep.shaping -= last_phi
    return ep


def gae(rewards: Sequence[float], values: Sequence[float], gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimates and returns for one finished game: the
    value after the last decision is 0."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(values, dtype=np.float32)


def ppo_update(net: ActorCritic, opt: torch.optim.Optimizer, episodes: Sequence[Episode], cfg: "PPOConfig",
               device: torch.device, rng: np.random.Generator) -> Dict[str, float]:
    """`update_epochs` passes of clipped PPO over the games in `episodes`."""
    advs, rets = zip(*(gae(ep.rewards, ep.values, cfg.gamma, cfg.gae_lambda) for ep in episodes))
    put = lambda x, dtype=None: torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
    obs = put(np.concatenate([np.asarray(ep.obs) for ep in episodes]), torch.float32)
    mask = put(np.concatenate([np.asarray(ep.masks) for ep in episodes]))
    act = put(np.concatenate([ep.actions for ep in episodes]), torch.int64)
    old_logp = put(np.concatenate([ep.logprobs for ep in episodes]), torch.float32)
    old_v = put(np.concatenate([ep.values for ep in episodes]), torch.float32)
    adv, ret = put(np.concatenate(advs), torch.float32), put(np.concatenate(rets), torch.float32)
    n = len(act)
    size = max(1, n // cfg.minibatches)
    stats: Dict[str, List[float]] = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")}
    for _ in range(cfg.update_epochs):
        order = torch.as_tensor(rng.permutation(n), device=device)
        for start in range(0, n, size):
            i = order[start:start + size]
            logits = net.logits(obs[i], mask[i])
            logp_all = torch.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, act[i].unsqueeze(1)).squeeze(1)
            entropy = -(logp_all.exp() * logp_all).sum(-1)
            log_ratio = logp - old_logp[i]
            ratio = log_ratio.exp()
            a = adv[i]
            if cfg.norm_adv and len(a) > 1:
                a = (a - a.mean()) / (a.std() + 1e-8)
            policy_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
            v = net.value(obs[i])
            if cfg.clip_vloss:
                v_clipped = old_v[i] + (v - old_v[i]).clamp(-cfg.clip, cfg.clip)
                value_loss = 0.5 * torch.max((v - ret[i]) ** 2, (v_clipped - ret[i]) ** 2).mean()
            else:
                value_loss = 0.5 * ((v - ret[i]) ** 2).mean()
            loss = policy_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * value_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["policy_loss"].append(float(policy_loss))
                stats["value_loss"].append(float(value_loss))
                stats["entropy"].append(float(entropy.mean()))
                stats["approx_kl"].append(float(((ratio - 1) - log_ratio).mean()))
                stats["clipfrac"].append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
    return {k: statistics.mean(v) for k, v in stats.items()}


# ------------------------------------------------------------------ training


@dataclass
class PPOConfig:
    opponent: str = "pool"
    pool: Tuple[str, ...] = ("greedy", "wary", "racer", "self")
    league: int = 5
    shaping: float = 0.0
    games: int = 30000
    games_per_update: int = 32  # about 2700 decisions per update
    hidden: Tuple[int, ...] = (512, 256)
    lr: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 1.0
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    eval_sample: bool = False  # evaluations sample moves instead of taking the most likely
    reward_mode: str = "margin"
    reward_scale: float = 0.01
    memory_level: int = 2
    ticket_plan: bool = False
    players: int = 2  # seats per game: the learner and players - 1 opponents
    seed: int = 0
    max_turns: int = 1000
    board: str = "usa"
    device: str = "auto"


@dataclass
class PPOResult:
    """Everything a run records, in the DQN / linear run layout."""

    cfg: PPOConfig
    net: ActorCritic
    games: List[Dict[str, float]] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)
    updates: List[dict] = field(default_factory=list)
    baselines: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    eval_games: int = 0
    best: Optional[dict] = None
    best_net: Optional[ActorCritic] = None
    eval_linear: Optional[str] = None

    def save(self, path: Path, finished: bool = True) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cpu = lambda net: {k: v.detach().cpu() for k, v in net.state_dict().items()}
        buf = io.BytesIO()
        torch.save({"final": cpu(self.net), "best": cpu(self.best_net) if self.best_net is not None else None}, buf)
        weights = path.with_suffix(".pt")
        tmp = weights.with_name(weights.name + ".tmp")
        tmp.write_bytes(buf.getvalue())
        swap_in(tmp, weights)

        cfg = self.cfg
        data = {
            "method": "ppo",
            "config": {**asdict(cfg), "hidden": list(cfg.hidden), "pool": list(cfg.pool)},
            "network": {"obs_size": ObservationEncoder(cfg.players, cfg.memory_level, cfg.ticket_plan).size,
                        "hidden": list(cfg.hidden), "memory_level": cfg.memory_level, "ticket_plan": cfg.ticket_plan,
                        "num_players": cfg.players},
            "weights_file": weights.name,
            "eval_linear": self.eval_linear,
            "eval_games": self.eval_games,
            "metrics": METRICS,
            "progress": {"games": len(self.games), "of": cfg.games, "finished": finished},
            "history": self.history,
            "snapshots": [],
            "baselines": self.baselines,
            "games": self.games,
            "updates": self.updates,
            "best": self.best,
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        swap_in(tmp, path)


def load_policy(path: Path, best: bool = False) -> Tuple[ActorCritic, dict]:
    """The final (or best) network of a run saved by `PPOResult.save` (on the CPU),
    and its observation settings {"num_players", "memory_level", "ticket_plan"}
    (runs from before multiplayer training are 2-player)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("method") != "ppo":
        raise ValueError(f"{path} is not a PPO run")
    spec = data["network"]
    nets = torch.load(path.with_name(data["weights_file"]), map_location="cpu", weights_only=True)
    state = nets["best" if best else "final"]
    if state is None:
        raise ValueError(f"{path} has no best checkpoint")
    net = ActorCritic(spec["obs_size"], spec["hidden"])
    net.load_state_dict(state)
    net.eval()
    return net, {"num_players": spec.get("num_players", 2), "memory_level": spec["memory_level"],
                 "ticket_plan": spec.get("ticket_plan", False)}


def train(cfg: PPOConfig, eval_every: int = 0, eval_games: int = 100, eval_linear: Optional[str] = BEST_LINEAR,
          eval_opponents: Sequence[str] = EVAL_BOTS, log=print, baselines: bool = True,
          on_progress: Optional[Callable[[PPOResult], None]] = None, progress_every: int = 0) -> PPOResult:
    """Train for cfg.games games, one PPO update per `games_per_update` games.
    Evaluations, best checkpoint and live saving as in ttr.learn.dqn.train."""
    if cfg.opponent not in TRAIN_OPPONENTS:
        raise ValueError(f"opponent must be one of {TRAIN_OPPONENTS}")
    if cfg.reward_mode not in REWARD_MODES:
        raise ValueError(f"reward mode must be one of {REWARD_MODES}")
    if "greedy" not in eval_opponents:
        raise ValueError("the evaluations must include greedy (runs are compared on it)")
    if not 2 <= cfg.players <= 5:
        raise ValueError("players must be 2-5")
    for name in cfg.pool if cfg.opponent == "pool" else ():
        if name != "self":
            make_agent(name, 0)  # an unknown name fails here, not mid-run
    selected_on = [o for o in eval_opponents if o != "random"]
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    board = load_board(cfg.board)
    view = {"num_players": cfg.players, "memory_level": cfg.memory_level, "ticket_plan": cfg.ticket_plan}
    net = ActorCritic(ObservationEncoder(**view).size, cfg.hidden).to(device)
    cpu = torch.device("cpu")
    actor = net if device == cpu else copy.deepcopy(net).to(cpu).requires_grad_(False)
    eval_net = copy.deepcopy(net).to(cpu).requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    learner = PPOAgent(actor, **view, sample=True, seed=cfg.seed, name="ppo")
    evaluated = PPOAgent(eval_net, **view, sample=cfg.eval_sample, seed=cfg.seed, name="ppo")
    out = PPOResult(cfg, net, eval_games=eval_games, eval_linear=eval_linear)
    eval_specs = {**{o: o for o in eval_opponents}, **({"linear": eval_linear} if eval_linear else {})}
    league: Deque[ActorCritic] = deque(maxlen=cfg.league)
    update_rng = np.random.default_rng(cfg.seed)

    def own_policy(seed: int):
        """A frozen copy of the learner, sampling as in training: the best policy or
        a recent one; greedy until the first evaluation."""
        nets = [out.best_net, *league] if out.best_net is not None else []
        if not nets:
            return GreedyAgent(seed)
        return PPOAgent(random.Random(seed).choice(nets), **view, sample=True, seed=seed, name="self")

    def opponent_for(seed: int):
        """One opponent seat's agent for this game: each seat is drawn on its own."""
        if cfg.opponent == "self" and out.best_net is not None and random.Random(seed).random() < 0.5:
            return PPOAgent(out.best_net, **view, sample=True, seed=seed, name="self")
        if cfg.opponent == "self":
            return GreedyAgent(seed)
        if cfg.opponent == "pool":
            name = random.Random(seed).choice(cfg.pool)
            return own_policy(seed) if name == "self" else make_agent(name, seed)
        return make_agent(cfg.opponent, seed)

    rng = random.Random(cfg.seed)
    window: List[Dict[str, float]] = []
    batch: List[Episode] = []
    n_updates = max(1, cfg.games // cfg.games_per_update)
    start = time.perf_counter()
    for g in range(cfg.games):
        game = Game(board, num_players=cfg.players, seed=rng.getrandbits(32), max_turns=cfg.max_turns)
        # one seed per opponent seat, drawn before the learner's seat, so 2-player runs are unchanged
        opp_seeds = [rng.getrandbits(32) for _ in range(cfg.players - 1)]
        seat = rng.randrange(cfg.players)
        opponents = dict(zip([q for q in range(cfg.players) if q != seat], map(opponent_for, opp_seeds)))
        ep = play_rollout_game(learner, opponents, game, seat, cfg)
        if ep.actions:
            batch.append(ep)
        _, returns = gae(ep.rewards, ep.values, cfg.gamma, cfg.gae_lambda) if ep.actions else (None, [])

        lr = cfg.lr * (1.0 - len(out.updates) / n_updates) if cfg.anneal_lr else cfg.lr
        row = game_metrics(game, seat)
        opp_names = [getattr(o, "name", "") for o in opponents.values()]
        share = lambda pred: sum(1 for n in opp_names if pred(n)) / len(opp_names)  # of the opponent seats
        row.update(
            game=float(g + 1), epsilon=0.0, lr=lr, seat=float(seat),
            **{f"vs_{name}": share(lambda n, name=name: n == name)
               for name in ("random", "greedy", "wary", "racer", "collector", "self")},
            vs_linear=share(lambda n: n.startswith("linear:")), vs_dqn=share(lambda n: n.startswith("dqn:")),
            decisions=float(len(ep.actions)),
            mean_abs_td=float(np.mean(np.abs(np.asarray(returns) - np.asarray(ep.values)))) if ep.actions else 0.0,
            entropy=statistics.mean(ep.entropies) if ep.entropies else 0.0,
            shaping=ep.shaping,
        )
        out.games.append(row)
        window.append(row)

        if len(batch) >= cfg.games_per_update:
            for group in opt.param_groups:
                group["lr"] = lr
            stats = ppo_update(net, opt, batch, cfg, device, update_rng)
            out.updates.append({"games": g + 1, "lr": lr, **stats})
            batch = []
            if actor is not net:
                actor.load_state_dict(net.state_dict())

        done = g + 1
        if (eval_every and done % eval_every == 0) or done == cfg.games:
            eval_net.load_state_dict(net.state_dict())
            entry = {
                "games": done,
                "epsilon": 0.0,
                "lr": lr,
                "updates": len(out.updates),
                "grad_steps": len(out.updates) * cfg.update_epochs * cfg.minibatches,
                "seconds": round(time.perf_counter() - start, 1),
                "train": {k: statistics.mean(r[k] for r in window) for k in ("won", "margin", "mean_abs_td", "entropy")},
                "eval": {name: evaluate(evaluated, spec, eval_games, board, seed=_eval_seed(len(out.history)),
                                        max_turns=cfg.max_turns, num_players=cfg.players)
                         for name, spec in eval_specs.items()},
            }
            out.history.append(entry)
            selection = statistics.mean(entry["eval"][o]["margin"] for o in selected_on)
            if out.best is None or selection > out.best["selection"]:
                out.best = {"games": done, "margin_vs_greedy": entry["eval"]["greedy"]["margin"],
                            "selection": selection, "selected_on": selected_on}
                out.best_net = copy.deepcopy(eval_net)
            league.append(copy.deepcopy(eval_net))
            window = []
            log(_progress_line(entry) + f"  entropy {entry['train']['entropy']:.2f}")
            if on_progress is not None:
                on_progress(out)
        elif on_progress is not None and progress_every and done % progress_every == 0:
            on_progress(out)
    if baselines:
        seed = _eval_seed(len(out.history) - 1)
        for bot in ("random", "greedy"):
            out.baselines[bot] = {opp: evaluate(OPPONENTS[bot](seed), opp, eval_games, board, seed=seed,
                                                max_turns=cfg.max_turns, num_players=cfg.players)
                                  for opp in eval_opponents}
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    d = PPOConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opponent", choices=TRAIN_OPPONENTS, default=d.opponent,
                        help="training opponent; pool = one of --pool each game (default)")
    parser.add_argument("--pool", nargs="+", default=list(d.pool), metavar="SPEC",
                        help="with --opponent pool: agent specs to draw from; 'self' = the best policy or one of the "
                             "--league latest evaluated ones (greedy until the first evaluation)")
    parser.add_argument("--league", type=int, default=d.league)
    parser.add_argument("--shaping", type=float, default=d.shaping, metavar="POINTS",
                        help="potential-based shaping: POINTS per train still needed for my tickets (0 = off)")
    parser.add_argument("--ticket-plan", action="store_true", help="add the observation's ticket-plan block")
    parser.add_argument("--games", type=int, default=d.games)
    parser.add_argument("--games-per-update", type=int, default=d.games_per_update)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(d.hidden), metavar="UNITS")
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    parser.add_argument("--gamma", type=float, default=d.gamma)
    parser.add_argument("--gae-lambda", type=float, default=d.gae_lambda)
    parser.add_argument("--update-epochs", type=int, default=d.update_epochs)
    parser.add_argument("--minibatches", type=int, default=d.minibatches)
    parser.add_argument("--clip", type=float, default=d.clip)
    parser.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false")
    parser.add_argument("--ent-coef", type=float, default=d.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=d.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=d.max_grad_norm)
    parser.add_argument("--eval-sample", action="store_true", help="evaluations sample moves (default: argmax)")
    parser.add_argument("--reward", choices=REWARD_MODES, default=d.reward_mode)
    parser.add_argument("--memory-level", type=int, choices=(0, 1, 2), default=d.memory_level)
    parser.add_argument("--players", type=int, choices=(2, 3, 4, 5), default=d.players,
                        help="seats per game: the learner and PLAYERS - 1 opponents, each drawn on its own; "
                             "evaluations fill every other seat with the evaluation opponent")
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--device", default=d.device, help="auto (cuda if available), cpu, cuda, ...")
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads (default 1)")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=100, help="games against each evaluation opponent")
    parser.add_argument("--eval-opponents", nargs="+", default=list(EVAL_BOTS), metavar="SPEC")
    parser.add_argument("--eval-linear", default=BEST_LINEAR, metavar="SPEC", help="'none' to skip")
    parser.add_argument("--out", type=Path, required=True, help="run JSON; networks go to the .pt beside it")
    parser.add_argument("--live", type=int, nargs="?", const=100, default=0, metavar="N",
                        help="also write the run files every N games and after every evaluation")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    torch.set_num_threads(args.threads)

    cfg = PPOConfig(
        opponent=args.opponent, pool=tuple(args.pool), league=args.league, shaping=args.shaping,
        games=args.games, games_per_update=args.games_per_update, hidden=tuple(args.hidden), lr=args.lr,
        anneal_lr=args.anneal_lr, gamma=args.gamma, gae_lambda=args.gae_lambda, update_epochs=args.update_epochs,
        minibatches=args.minibatches, clip=args.clip, clip_vloss=args.clip_vloss, ent_coef=args.ent_coef,
        vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm, eval_sample=args.eval_sample,
        reward_mode=args.reward, memory_level=args.memory_level, ticket_plan=args.ticket_plan, players=args.players,
        seed=args.seed, device=args.device,
    )
    eval_linear = None if args.eval_linear.lower() == "none" else args.eval_linear
    print(f"training ppo vs {cfg.opponent} on {resolve_device(cfg.device)}: {cfg.players} players, {cfg.games} games, "
          f"{cfg.games_per_update} per update, hidden {list(cfg.hidden)}, lr {cfg.lr}{' annealed' if cfg.anneal_lr else ''}, "
          f"gamma {cfg.gamma}, lambda {cfg.gae_lambda}, clip {cfg.clip}, entropy {cfg.ent_coef}, reward {cfg.reward_mode}"
          + (f", shaping {cfg.shaping}" if cfg.shaping else "") + (", ticket plan" if cfg.ticket_plan else "")
          + (f"; pool {' '.join(cfg.pool)}" if cfg.opponent == "pool" else ""))
    print(f"(best checkpoint kept by mean margin vs {', '.join(o for o in args.eval_opponents if o != 'random')}; "
          f"play it as ppo:{args.out}@best)")
    live = None
    if args.live:
        live = lambda result: result.save(args.out, finished=False)
    result = train(cfg, eval_every=args.eval_every, eval_games=args.eval_games, eval_linear=eval_linear,
                   eval_opponents=tuple(args.eval_opponents), on_progress=live, progress_every=args.live)
    result.save(args.out)
    print(f"saved {args.out} and {args.out.with_suffix('.pt')}  (ttr-dash {args.out} to analyze)")


if __name__ == "__main__":
    main()
