"""Expert iteration, step one: a search's moves as training data for a network
(PLAN.md "Methods to compare", the AlphaZero-style stretch goal).

The search (tier D) plays tickets and no learner ever found them; the question is
whether a network can hold that strategy when shown it. `SearchRecorder` wraps a
search agent and records each of its decisions that had a choice, from its seat's
view; `scripts/eval_agents.py --data DIR` plays the search on match games and
saves every chunk of games as one `.npz` file in DIR:

    obs        float16 [n, size]  ObservationEncoder vector with the ticket-plan block
                                  (the plain vector is its first part)
    mask       bool    [n, 168]   legal moves
    action     int16   [n]        the move played
    searched   bool    [n]        a search chose it (else the rollout policy chose a
                                  second draw or a payment: steps=main)
    visits     int16   [n, 168]   searched: the root's visit count per legal move
    q          float16 [n, 168]   searched: the root's mean value per visited move
    phase      int8    [n]        index in ttr.env.observation.PHASES
    turn, seat, game              the decision's turn, the searcher's seat, the game number
    score, margin   float32 [n]   reward_values at the decision (points)
    final_score, final_margin, won   float32 [n]   the game's result for the searcher
    agent, opponent, players, seed   the batch, as eval_agents.py names it

Moves with a single legal option are not recorded.

`ttr-train-distill` then trains PPO's network (ttr.learn.ppo.ActorCritic) on the
recorded decisions, supervised:

    ttr-train-distill --data runs/exit/pass1/data --epochs 30 --out runs/exit/pass1/x1a_s0.json
    ttr-sim --agents ppo:runs/exit/pass1/x1a_s0.json greedy --games 200

    policy  cross-entropy between the masked policy and the search's visit counts
            (`target` visits, sharpened by `temperature`: N^(1/T)) or its move (`target`
            move); a second draw or payment the rollout policy chose is its move
    value   squared error between V(s) and the return still to come, in PPO's units:
            (final - now) x reward_scale for the reward mode (own score by default)

Games numbered g % `val_every` == `val_every` - 1 are held out for validation in every
run (0: none), so arms compare on the same held-out games; each epoch reports the validation
cross-entropy, the share of decisions where the network's most likely move is the
search's, by sub-step, and the value error. Every `eval_every` epochs the network
plays (its most likely move) against the evaluation opponents on the learners'
paired evaluation games; the best checkpoint is the evaluation with the best mean
margin over the scripted opponents other than random, as in every learner.

The run is saved as a PPO run ("method": "ppo"), so `ppo:PATH[@best]`, rescore.py,
`mcts:guide=ppo:PATH` and `ttr-train-ppo --init PATH` take it as they are. Its
"history" holds the evaluations ("games" = the epoch, for the tools that read it),
"epochs" the per-epoch losses, "data" what was trained on. Needs the `[env]` and
`[deep]` extras for training; recording needs only `[env]`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ttr.board import load_board
from ttr.env import actions as A
from ttr.env.observation import PHASES, ObservationEncoder
from ttr.env.reward import reward_values
from ttr.game import Game
from ttr.learn.common import BEST_LINEAR, evaluations, observe, resolve_device, save_run
from ttr.learn.metrics import METRICS

RECORDED = ("obs", "mask", "action", "searched", "visits", "q", "phase", "turn", "seat", "score", "margin")
DTYPES = {"obs": np.float16, "mask": bool, "action": np.int16, "searched": bool, "visits": np.int16,
          "q": np.float16, "phase": np.int8, "turn": np.int16, "seat": np.int8, "score": np.float32,
          "margin": np.float32}
EVAL_OPPONENTS = ("random", "greedy", "wary", "racer", "collector")


# ------------------------------------------------------------------ recording


class SearchRecorder:
    """Plays a search agent's moves (an MCTSAgent) and records each of its decisions
    that had a choice: the observation from its seat (ticket-plan block included), the
    mask, the move, and for a searched decision the root's visit counts and mean values
    per legal move. `take` hands over one finished game's records."""

    def __init__(self, agent, num_players: int, memory_level: int = 2) -> None:
        self.agent = agent
        self.name = getattr(agent, "name", "search")
        self.encoder = ObservationEncoder(num_players, memory_level, ticket_plan=True)
        self.rows: List[dict] = []

    @property
    def searches(self) -> int:
        return self.agent.searches

    @property
    def search_seconds(self) -> float:
        return self.agent.search_seconds

    def act(self, game: Game, player: int):
        if len(game.legal_actions()) == 1:
            return self.agent.act(game, player)
        obs, mask, by_index = observe(self.encoder, game, player)
        index = {a: i for i, a in enumerate(by_index) if a is not None}
        before = self.agent.searches
        action = self.agent.act(game, player)
        visits = np.zeros(A.N_ACTIONS, np.int16)
        q = np.zeros(A.N_ACTIONS, np.float16)
        searched = self.agent.searches > before
        if searched:
            for a, node in self.agent.last_root.children.items():
                i = index.get(a)
                if i is not None and node.visits:
                    visits[i] = node.visits
                    q[i] = node.mean
        self.rows.append({"obs": obs, "mask": mask, "action": index[action], "searched": searched,
                          "visits": visits, "q": q, "phase": PHASES.index(game.phase), "turn": game.turn,
                          "seat": player, "score": reward_values(game, "score")[player],
                          "margin": reward_values(game, "margin")[player]})
        return action

    def take(self, game: Game, seat: int, g: int) -> Dict[str, np.ndarray]:
        """The finished game's records as arrays (RECORDED plus game and the result), cleared here."""
        rows, self.rows = self.rows, []
        out = {k: np.asarray([r[k] for r in rows], dtype=DTYPES[k]) for k in RECORDED}
        n = len(rows)
        result = game.result
        out["game"] = np.full(n, g, np.int32)
        out["final_score"] = np.full(n, reward_values(game, "score")[seat], np.float32)
        out["final_margin"] = np.full(n, reward_values(game, "margin")[seat], np.float32)
        out["won"] = np.full(n, float(result is not None and result.winners == [seat]), np.float32)
        return out


def chunk_path(directory: Path, agent: str, opponent: str, players: int, seed: int, games: List[int]) -> Path:
    """Where a chunk of games is saved: a short hash of the batch, then the game numbers."""
    key = hashlib.sha1(f"{agent}|{opponent}|{players}|{seed}".encode()).hexdigest()[:10]
    return Path(directory) / f"{key}_g{games[0]:05d}-{games[-1]:05d}.npz"


def save_chunk(path: Path, games: List[Dict[str, np.ndarray]], agent: str, opponent: str, players: int,
               seed: int) -> None:
    """One chunk's games in one compressed file, written whole (temporary file, then swapped in)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.concatenate([g[k] for g in games]) for k in games[0]}
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays, agent=np.array(agent), opponent=np.array(opponent),
                            players=np.array(players), seed=np.array(seed))
    os.replace(tmp, path)


# ------------------------------------------------------------------ data


def load_data(directory, players: int = 2, opponents: Optional[Sequence[str]] = None,
              first_games: int = 0) -> Dict[str, np.ndarray]:
    """Every recorded decision in `directory` (`save_chunk` files; or in each of several
    directories) at `players` players, concatenated; a game saved twice (a chunk replayed
    after a stop) counts once. With
    `opponents`, only the games against those; with `first_games`, only games numbered
    below it in each batch. "batch" is each row's index in "batches", the list of
    (agent, opponent, players, seed)."""
    parts: List[Dict[str, np.ndarray]] = []
    batches: List[Tuple[str, str, int, int]] = []
    seen = set()
    directories = [directory] if isinstance(directory, (str, Path)) else list(directory)
    for path in [f for d in directories for f in sorted(Path(d).glob("*.npz"))]:
        with np.load(path) as f:
            batch = (str(f["agent"]), str(f["opponent"]), int(f["players"]), int(f["seed"]))
            if batch[2] != players or (opponents and batch[1] not in opponents):
                continue
            arrays = {k: f[k] for k in f.files if k not in ("agent", "opponent", "players", "seed")}
        if batch not in batches:
            batches.append(batch)
        fresh = [int(g) for g in np.unique(arrays["game"])
                 if (batch, int(g)) not in seen and (not first_games or g < first_games)]
        seen.update((batch, g) for g in fresh)
        keep = np.isin(arrays["game"], fresh)
        if keep.any():
            part = {k: v[keep] for k, v in arrays.items()}
            part["batch"] = np.full(int(keep.sum()), batches.index(batch), np.int16)
            parts.append(part)
    if not parts:
        raise ValueError(f"no recorded {players}-player decisions in {directory}")
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["batches"] = batches
    return data


def policy_targets(data: Dict[str, np.ndarray], target: str = "visits", temperature: float = 1.0) -> np.ndarray:
    """[n, 168] float32: per decision, the distribution the policy is trained toward. A
    searched decision: its visit counts ^ (1 / temperature), normalized (`target`
    visits), or its move (`target` move); anything else: the move played."""
    if target not in ("visits", "move"):
        raise ValueError("target must be visits or move")
    n = len(data["action"])
    out = np.zeros((n, A.N_ACTIONS), np.float32)
    out[np.arange(n), data["action"]] = 1.0
    if target == "visits":
        s = data["searched"]
        v = data["visits"][s].astype(np.float64) ** (1.0 / temperature)
        out[s] = v / v.sum(1, keepdims=True)
    return out


def value_targets(data: Dict[str, np.ndarray], reward_mode: str = "score", reward_scale: float = 0.01) -> np.ndarray:
    """The return still to come at each decision, in PPO's units: (final - now) x
    reward_scale for score and margin; for win, the result (+1 sole win, else -1)."""
    if reward_mode == "win":
        return (2 * data["won"] - 1).astype(np.float32)
    if reward_mode not in ("score", "margin"):
        raise ValueError("reward mode must be score, margin or win")
    return ((data["final_" + reward_mode] - data[reward_mode]) * reward_scale).astype(np.float32)


def summarize_data(data: Dict[str, np.ndarray], val: np.ndarray) -> dict:
    """What a run was trained on: games and decisions per opponent, and the split."""
    by_opponent = {}
    for i, (_, opponent, _, _) in enumerate(data["batches"]):
        rows = data["batch"] == i
        by_opponent[opponent] = {"games": int(len(np.unique(data["game"][rows]))), "decisions": int(rows.sum()),
                                 "searched": int(data["searched"][rows].sum())}
    return {"teachers": sorted({b[0] for b in data["batches"]}), "decisions": int(len(val)),
            "validation": int(val.sum()), "by_opponent": by_opponent}


# ------------------------------------------------------------------ training


@dataclass
class DistillConfig:
    data: Tuple[str, ...] = ("runs/exit/pass1/data",)  # one directory or several
    opponents: Tuple[str, ...] = ()  # train only on the games against these (empty: all)
    first_games: int = 0  # only games numbered below this in each batch (0: all)
    ticket_plan: bool = False
    target: str = "visits"
    temperature: float = 1.0
    reward_mode: str = "score"  # what the value head predicts
    reward_scale: float = 0.01
    epochs: int = 30
    batch: int = 512
    lr: float = 3e-4
    anneal_lr: bool = True
    weight_decay: float = 0.0
    max_grad_norm: float = 0.5
    hidden: Tuple[int, ...] = (512, 256)
    val_every: int = 10
    players: int = 2
    memory_level: int = 2
    seed: int = 0
    max_turns: int = 1000
    board: str = "usa"
    device: str = "auto"


@dataclass
class DistillResult:
    cfg: DistillConfig
    net: object = None
    best_net: Optional[object] = None
    best: Optional[dict] = None
    history: List[dict] = field(default_factory=list)
    epochs: List[dict] = field(default_factory=list)
    data: Optional[dict] = None
    eval_games: int = 0
    eval_linear: Optional[str] = None

    def save(self, path: Path, finished: bool = True) -> None:
        cfg = self.cfg
        save_run(path, {"final": self.net, "best": self.best_net}, "ppo",
                 {**asdict(cfg), "hidden": list(cfg.hidden), "opponents": list(cfg.opponents), "distill": True},
                 {"obs_size": ObservationEncoder(cfg.players, cfg.memory_level, cfg.ticket_plan).size,
                  "hidden": list(cfg.hidden), "memory_level": cfg.memory_level, "ticket_plan": cfg.ticket_plan,
                  "num_players": cfg.players},
                 eval_linear=self.eval_linear, eval_games=self.eval_games, metrics=METRICS,
                 progress={"games": len(self.epochs), "of": cfg.epochs, "finished": finished},
                 history=self.history, best=self.best, epochs=self.epochs, data=self.data)


def train(cfg: DistillConfig, eval_every: int = 5, eval_games: int = 100, eval_linear: Optional[str] = BEST_LINEAR,
          eval_opponents: Sequence[str] = EVAL_OPPONENTS, log=print,
          on_progress: Optional[Callable[[DistillResult], None]] = None) -> DistillResult:
    """Supervised training of an ActorCritic on the recorded search decisions in `cfg.data`."""
    import torch
    from torch import nn

    from ttr.learn.ppo import ActorCritic, PPOAgent

    if "greedy" not in eval_opponents:
        raise ValueError("the evaluations must include greedy (runs are compared on it)")
    if cfg.memory_level != 2:
        raise ValueError("the decisions were recorded at memory level 2")
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    view = {"num_players": cfg.players, "memory_level": cfg.memory_level, "ticket_plan": cfg.ticket_plan}
    size = ObservationEncoder(**view).size

    data = load_data(cfg.data, cfg.players, cfg.opponents, cfg.first_games)
    val = data["game"] % cfg.val_every == cfg.val_every - 1 if cfg.val_every else np.zeros(len(data["game"]), bool)
    targets = policy_targets(data, cfg.target, cfg.temperature)
    returns = value_targets(data, cfg.reward_mode, cfg.reward_scale)
    put = lambda x, dtype=None: torch.as_tensor(np.ascontiguousarray(x), device=device, dtype=dtype)
    split = {}
    for name, rows in (("train", ~val), ("val", val)):
        split[name] = {"obs": put(data["obs"][rows, :size]), "mask": put(data["mask"][rows]),
                       "target": put(targets[rows].astype(np.float16)),  # half the GPU memory; 3 digits suffice
                       "action": put(data["action"][rows], torch.int64),
                       "ret": put(returns[rows]), "phase": data["phase"][rows], "n": int(rows.sum())}
    out = DistillResult(cfg, data=summarize_data(data, val), eval_games=eval_games, eval_linear=eval_linear)
    del data, targets
    log(f"{out.data['decisions']} decisions ({split['train']['n']} train, {split['val']['n']} validation) from "
        + ", ".join(f"{o.split('/')[-1]} {d['games']}" for o, d in out.data["by_opponent"].items()) + " games")

    net = out.net = ActorCritic(size, cfg.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5, weight_decay=cfg.weight_decay)
    eval_net = copy.deepcopy(net).to("cpu").requires_grad_(False)
    evaluated = PPOAgent(eval_net, **view, seed=cfg.seed, name="distill")
    eval_specs = {**{o: o for o in eval_opponents}, **({"linear": eval_linear} if eval_linear else {})}
    selected_on = [o for o in eval_opponents if o != "random"]
    board = load_board(cfg.board)

    def losses(part: dict, i):
        """Per decision: cross-entropy to the target, value loss, the most likely move."""
        obs = part["obs"][i].float()
        logp = torch.log_softmax(net.logits(obs, part["mask"][i]), -1)
        ce = -(part["target"][i].float() * logp).sum(-1)  # illegal moves: target 0, so they add 0
        vloss = 0.5 * (net.value(obs) - part["ret"][i]) ** 2
        return ce, vloss, logp.argmax(-1)

    def validate() -> dict:
        part = split["val"]
        if not part["n"]:
            return {"val_ce": float("nan"), "val_value_loss": float("nan"), "val_match": float("nan"),
                    "val_match_by_phase": {}}
        ce, vl, hit = [], [], []
        with torch.no_grad():
            for start in range(0, part["n"], 4096):
                i = slice(start, start + 4096)
                c, v, move = losses(part, i)
                ce.append(c)
                vl.append(v)
                hit.append(move == part["action"][i])
        ce, vl, hit = (torch.cat(x).cpu().numpy() for x in (ce, vl, hit))
        by_phase = {PHASES[k].name.lower(): float(hit[part["phase"] == k].mean())
                    for k in range(len(PHASES)) if (part["phase"] == k).any()}
        return {"val_ce": float(ce.mean()), "val_value_loss": float(vl.mean()), "val_match": float(hit.mean()),
                "val_match_by_phase": by_phase}

    start_time = time.perf_counter()
    part = split["train"]
    steps_per_epoch = max(1, part["n"] // cfg.batch)
    total_steps = cfg.epochs * steps_per_epoch
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        order = torch.as_tensor(rng.permutation(part["n"]), device=device)
        ce_sum = v_sum = 0.0
        for b in range(steps_per_epoch):
            if cfg.anneal_lr:
                for group in opt.param_groups:
                    group["lr"] = cfg.lr * (1.0 - step / total_steps)
            ce, vloss, _ = losses(part, order[b * cfg.batch:(b + 1) * cfg.batch])
            loss = ce.mean() + vloss.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            ce_sum += ce.detach().mean().item()
            v_sum += vloss.detach().mean().item()
            step += 1
        row = {"epoch": epoch, "train_ce": ce_sum / steps_per_epoch, "train_value_loss": v_sum / steps_per_epoch,
               **validate(), "seconds": round(time.perf_counter() - start_time, 1)}
        out.epochs.append(row)
        line = (f"epoch {epoch:3d}  train ce {row['train_ce']:.3f} v {row['train_value_loss']:.4f}  "
                f"val ce {row['val_ce']:.3f} v {row['val_value_loss']:.4f} match {row['val_match']:.3f} ("
                + ", ".join(f"{k} {v:.2f}" for k, v in row["val_match_by_phase"].items()) + ")")
        if (eval_every and epoch % eval_every == 0) or epoch == cfg.epochs:
            eval_net.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()})
            entry = {"games": epoch, "epoch": epoch, "seconds": row["seconds"],
                     "train": {"val_ce": row["val_ce"], "val_match": row["val_match"]},
                     "eval": evaluations(evaluated, eval_specs, eval_games, board, len(out.history), cfg.max_turns,
                                         cfg.players)}
            out.history.append(entry)
            selection = statistics.mean(entry["eval"][o]["margin"] for o in selected_on)
            if out.best is None or selection > out.best["selection"]:
                out.best = {"games": epoch, "epoch": epoch, "margin_vs_greedy": entry["eval"]["greedy"]["margin"],
                            "selection": selection, "selected_on": list(selected_on)}
                out.best_net = copy.deepcopy(eval_net)
            g = entry["eval"]["greedy"]
            line += ("\n    eval " + "  ".join(f"vs {k.split('/')[-1]} {m['margin']:+.1f} ({m['win_share']:.0%})"
                                             for k, m in entry["eval"].items())
                     + f"  | vs greedy: tickets kept {g['tickets_kept']:.2f} done {g['tickets_completed']:.2f}"
                     f" failed {g['tickets_failed']:.2f}, claims {g['claims']:.1f} x {g['mean_claim_length']:.2f}"
                     f"  [{time.perf_counter() - start_time:.0f}s]")
            if on_progress is not None:
                on_progress(out)
        log(line)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    import torch

    d = DistillConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", default=list(d.data), metavar="DIR",
                        help="directories of recorded decisions (eval_agents.py --data), trained on together")
    parser.add_argument("--opponents", nargs="+", default=[], metavar="SPEC",
                        help="train only on the games against these opponents (default: all)")
    parser.add_argument("--first-games", type=int, default=0, metavar="N",
                        help="only games numbered below N in each batch (a smaller data set; default all)")
    parser.add_argument("--ticket-plan", action="store_true", help="observe with the ticket-plan block")
    parser.add_argument("--target", choices=("visits", "move"), default=d.target,
                        help="policy target of a searched decision: its visit counts or the move played")
    parser.add_argument("--temperature", type=float, default=d.temperature, help="visit counts ^ (1 / T)")
    parser.add_argument("--reward", choices=("score", "margin", "win"), default=d.reward_mode,
                        help="what the value head predicts: the return still to come under this reward")
    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--batch", type=int, default=d.batch)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    parser.add_argument("--weight-decay", type=float, default=d.weight_decay)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(d.hidden), metavar="UNITS")
    parser.add_argument("--val-every", type=int, default=d.val_every,
                        help="games numbered g %% N == N - 1 are held out for validation (0: none)")
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--device", default=d.device, help="auto (cuda if available), cpu, cuda, ...")
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    parser.add_argument("--eval-every", type=int, default=5, help="epochs between evaluations (0 = at the end)")
    parser.add_argument("--eval-games", type=int, default=100, help="games against each evaluation opponent")
    parser.add_argument("--eval-opponents", nargs="+", default=list(EVAL_OPPONENTS), metavar="SPEC",
                        help="evaluation opponents (must include greedy); the best checkpoint is picked on the "
                             "mean margin over all but random")
    parser.add_argument("--eval-linear", default=BEST_LINEAR, metavar="SPEC", help="'none' to skip")
    parser.add_argument("--out", type=Path, required=True,
                        help="run JSON; the networks go to the .pt beside it")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    torch.set_num_threads(args.threads)
    cfg = DistillConfig(data=tuple(args.data), opponents=tuple(args.opponents), first_games=args.first_games,
                        ticket_plan=args.ticket_plan,
                        target=args.target, temperature=args.temperature, reward_mode=args.reward,
                        epochs=args.epochs, batch=args.batch, lr=args.lr, anneal_lr=args.anneal_lr,
                        weight_decay=args.weight_decay, hidden=tuple(args.hidden), val_every=args.val_every,
                        seed=args.seed, device=args.device)
    print(f"distilling {' + '.join(cfg.data)} on {resolve_device(cfg.device)}: target {cfg.target}"
          + (f" ^ (1/{cfg.temperature})" if cfg.target == "visits" and cfg.temperature != 1 else "")
          + f", value {cfg.reward_mode}, {cfg.epochs} epochs of {cfg.batch}, lr {cfg.lr}"
          + (" annealed" if cfg.anneal_lr else "") + f", hidden {list(cfg.hidden)}"
          + (", ticket plan" if cfg.ticket_plan else "")
          + (f"; games vs {' '.join(cfg.opponents)}" if cfg.opponents else "")
          + (f"; games numbered below {cfg.first_games}" if cfg.first_games else ""))
    eval_linear = None if args.eval_linear.lower() == "none" else args.eval_linear
    result = train(cfg, eval_every=args.eval_every, eval_games=args.eval_games, eval_linear=eval_linear,
                   eval_opponents=tuple(args.eval_opponents), on_progress=lambda r: r.save(args.out, finished=False))
    result.save(args.out)
    print(f"saved {args.out} and {args.out.with_suffix('.pt')}  (plays as ppo:{args.out}, best as ppo:{args.out}@best)")


if __name__ == "__main__":
    main()
