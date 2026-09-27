"""Agents by name, for the command lines (`ttr-sim --agents`, `ttr-view --agents`).

    random                     RandomAgent
    greedy                     GreedyAgent
    linear:PATH                trained linear Q / SARSA weights (ttr.learn.linear; `[env]` extra)

`make_agent(spec, seed)` builds one; `agent_spec` validates a spec for argparse.
Weight files are read once per path and shared.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from ttr.agents.base import Agent
from ttr.agents.greedy import GreedyAgent
from ttr.agents.random_agent import RandomAgent

KINDS = ("random", "greedy", "linear:PATH")
_linear_weights: Dict[str, dict] = {}


def make_agent(spec: str, seed: int) -> Agent:
    kind, _, arg = spec.partition(":")
    if kind == "random" and not arg:
        return RandomAgent(seed)
    if kind == "greedy" and not arg:
        return GreedyAgent(seed)
    if kind == "linear" and arg:
        from ttr.learn.linear import LinearAgent

        if arg not in _linear_weights:
            _linear_weights[arg] = LinearAgent.load(Path(arg)).weights
        weights = {k: w.copy() for k, w in _linear_weights[arg].items()}
        return LinearAgent(weights, seed=seed, name=spec)
    raise ValueError(f"unknown agent {spec!r}; expected one of {', '.join(KINDS)}")


def agent_spec(spec: str) -> str:
    """argparse type: the spec, checked (and for files, loaded) up front."""
    try:
        make_agent(spec, 0)
    except (ValueError, OSError, KeyError) as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return spec
