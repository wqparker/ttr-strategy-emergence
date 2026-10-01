"""Agents by name, for the command lines (`ttr-sim --agents`, `ttr-view --agents`).

    random                     RandomAgent
    greedy                     GreedyAgent
    wary                       WaryAgent: greedy that watches the opponents' trains
    racer                      RacerAgent: minimum tickets, long routes, end fast
    collector                  CollectorAgent: greedy for tickets, 2-3 open at a time, finished one by one
    linear:PATH                trained linear Q / SARSA weights (ttr.learn.linear; `[env]` extra)
    linear:PATH@best           the same run's best checkpoint
    dqn:PATH                   a trained DQN run's final network (ttr.learn.dqn; `[deep]` extra)
    dqn:PATH@best              the same run's best checkpoint
    ppo:PATH                   a trained PPO run's final policy, most likely move (ttr.learn.ppo; `[deep]` extra)
    ppo:PATH@best              the same run's best checkpoint
    mcts                       MCTSAgent: search with determinization, no training (ttr.agents.mcts)
    mcts:OPTIONS               the same with options, e.g. mcts:iterations=800,rollout=racer

`make_agent(spec, seed)` builds one; `agent_spec` validates a spec for argparse.
Weight files are read once per path and shared.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

from ttr.agents.base import Agent
from ttr.agents.greedy import CollectorAgent, GreedyAgent, WaryAgent
from ttr.agents.racer import RacerAgent
from ttr.agents.random_agent import RandomAgent

KINDS = ("random", "greedy", "wary", "racer", "collector", "linear:PATH", "linear:PATH@best", "dqn:PATH", "dqn:PATH@best", "ppo:PATH", "ppo:PATH@best", "mcts",
         "mcts:OPTIONS")
_linear_weights: Dict[str, dict] = {}
_dqn_nets: Dict[str, Tuple[object, dict]] = {}  # spec argument -> (network, its observation settings)
_ppo_nets: Dict[str, Tuple[object, dict]] = {}


def make_agent(spec: str, seed: int) -> Agent:
    kind, _, arg = spec.partition(":")
    if kind == "random" and not arg:
        return RandomAgent(seed)
    if kind == "greedy" and not arg:
        return GreedyAgent(seed)
    if kind == "wary" and not arg:
        return WaryAgent(seed)
    if kind == "racer" and not arg:
        return RacerAgent(seed)
    if kind == "collector" and not arg:
        return CollectorAgent(seed)
    if kind == "mcts":
        from ttr.agents.mcts import MCTSAgent

        return MCTSAgent.from_spec(arg, seed, name=spec)
    if kind == "linear" and arg:
        from ttr.learn.linear import LinearAgent

        path, best = (arg[:-len("@best")], True) if arg.endswith("@best") else (arg, False)
        if arg not in _linear_weights:
            _linear_weights[arg] = LinearAgent.load(Path(path), best=best).weights
        weights = {k: w.copy() for k, w in _linear_weights[arg].items()}
        return LinearAgent(weights, seed=seed, name=spec)
    if kind == "dqn" and arg:
        import torch

        from ttr.learn.dqn import DQNAgent, load_network

        path, best = (arg[:-len("@best")], True) if arg.endswith("@best") else (arg, False)
        if arg not in _dqn_nets:
            torch.set_num_threads(1)  # one observation at a time; matches run several processes
            _dqn_nets[arg] = load_network(Path(path), best=best)
        net, view = _dqn_nets[arg]
        return DQNAgent(net, **view, seed=seed, name=spec)
    if kind == "ppo" and arg:
        import torch

        from ttr.learn.ppo import PPOAgent, load_policy

        path, best = (arg[:-len("@best")], True) if arg.endswith("@best") else (arg, False)
        if arg not in _ppo_nets:
            torch.set_num_threads(1)
            _ppo_nets[arg] = load_policy(Path(path), best=best)
        net, view = _ppo_nets[arg]
        return PPOAgent(net, **view, seed=seed, name=spec)
    raise ValueError(f"unknown agent {spec!r}; expected one of {', '.join(KINDS)}")


def agent_spec(spec: str) -> str:
    """argparse type: the spec, checked (and for files, loaded) up front."""
    try:
        make_agent(spec, 0)
    except (ValueError, OSError, KeyError, ImportError) as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return spec
