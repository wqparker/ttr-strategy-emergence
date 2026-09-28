"""Scripted baseline agents (PLAN.md roadmap phase 2).

Agents get the Game object for convenience but must only use information their
seat could see: their own hand/tickets plus public state (see RULES.md §9 #17).
"""

from ttr.agents.base import Agent
from ttr.agents.greedy import CollectorAgent, GreedyAgent, WaryAgent
from ttr.agents.racer import RacerAgent
from ttr.agents.random_agent import RandomAgent

__all__ = ["Agent", "CollectorAgent", "GreedyAgent", "RacerAgent", "RandomAgent", "WaryAgent"]
