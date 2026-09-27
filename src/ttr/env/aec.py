"""The PettingZoo AEC environment around the engine (PLAN.md Phase 4).

    from ttr.env.aec import env
    e = env(num_players=2, board="usa", reward_mode="margin")
    e.reset(seed=0)
    for agent in e.agent_iter():
        obs, reward, terminated, truncated, info = e.last()
        action = None if terminated or truncated else policy(obs["observation"], obs["action_mask"])
        e.step(action)

Agents are "player_0".."player_{N-1}", one per seat; the engine draws the first
seat at random (RULES.md §9 #8). Each agent step is one engine sub-step
(PLAN.md "Action space"), so the same agent often acts several times in a row:
draw then second draw, claim then pay.

Observation: {"observation": ObservationEncoder vector from that seat's view,
"action_mask": 168 int8 entries}. The mask is all 0 for a seat that is not
acting. An illegal action raises ValueError rather than being penalized.

Reward (PLAN.md "Reward"), per sub-step, to every seat:

    score   change in my score
    margin  change in (my score - mean of opponents' scores)      [default]
    win     +1 / -1 at game end; 0 to each winner of a shared win (§9 #12)

In the score modes the score is route points during the game and the final total
(tickets and longest route included) at the end, so the rewards over a game sum
to exactly the final score or margin, times `reward_scale` (default 1/100; the
win mode is not scaled). With 2 players the mean of opponents is the opponent.

A game cut off by `max_turns` is scored as it stands and ends as a truncation.
"""

from __future__ import annotations

import io
import random
from typing import Dict, List, Optional, Union

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from ttr.board import Board, load_board
from ttr.env import actions as A
from ttr.env.observation import ObservationEncoder
from ttr.game import Game

REWARD_MODES = ("score", "margin", "win")
OBS_HIGH = 3.0  # scaled counts are about [0, 1]; a 105-card hand is 2.1


def env(**kwargs) -> AECEnv:
    """The environment with PettingZoo's order checks (step before reset,
    acting out of turn)."""
    return wrappers.OrderEnforcingWrapper(raw_env(**kwargs))


class raw_env(AECEnv):
    metadata = {"name": "ticket_to_ride_v0", "render_modes": ["human", "ansi"], "is_parallelizable": False}

    def __init__(
        self,
        num_players: int = 2,
        board: Union[str, Board] = "usa",
        reward_mode: str = "margin",
        reward_scale: float = 0.01,
        memory_level: int = 2,
        max_turns: Optional[int] = 1000,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.board = load_board(board) if isinstance(board, str) else board
        if len(self.board.routes) > A.MAX_ROUTES:
            raise A.ActionIndexError(f"board {self.board.name!r} has more than {A.MAX_ROUTES} routes")
        if len(self.board.tickets) < A.OFFER * num_players:
            # A seat dealt an empty opening offer could only "keep nothing", which
            # has no index. Only the toy map (12 tickets) at 5 players gets here.
            raise ValueError(f"board {self.board.name!r} has too few tickets to deal {num_players} players")
        self.num_players = num_players
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.encoder = ObservationEncoder(num_players, memory_level)

        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_seat = {a: i for i, a in enumerate(self.possible_agents)}
        obs_space = spaces.Dict({
            "observation": spaces.Box(0.0, OBS_HIGH, (self.encoder.size,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, (A.N_ACTIONS,), dtype=np.int8),
        })
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: spaces.Discrete(A.N_ACTIONS) for a in self.possible_agents}
        self._seeds = random.Random()
        self.game: Optional[Game] = None

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    # ---------------------------------------------------------------- reset

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        if seed is not None:
            self._seeds = random.Random(seed)
        self.game = Game(self.board, self.num_players, seed=self._seeds.getrandbits(64), max_turns=self.max_turns)
        self.encoder.reset()
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._values = self._reward_values()
        self.agent_selection = self.possible_agents[self.game.current_player]

    # ---------------------------------------------------------------- step

    def step(self, action: Optional[int]) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        game = self.game
        index = int(action)
        if not 0 <= index < A.N_ACTIONS or not self._mask()[index]:
            raise ValueError(f"action {action} is not legal for {agent} in phase {game.phase.value}")
        self._cumulative_rewards[agent] = 0.0
        game.step(A.decode(game, index))

        values = self._reward_values()
        scale = 1.0 if self.reward_mode == "win" else self.reward_scale
        for a, seat in self.agent_seat.items():
            self.rewards[a] = scale * (values[seat] - self._values[seat])
        self._values = values

        if game.game_over:
            done = self.truncations if game.result.truncated else self.terminations
            for a, seat in self.agent_seat.items():
                done[a] = True
                r = game.result.players[seat]
                self.infos[a] = {
                    "score": r.total,
                    "winner": seat in game.result.winners,
                    "tickets_completed": r.tickets_completed,
                    "tickets_failed": r.tickets_failed,
                    "longest_path_bonus": r.longest_path_bonus,
                }
        else:
            self.agent_selection = self.possible_agents[game.current_player]
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def _reward_values(self) -> List[float]:
        """Per seat, the quantity whose change is the reward."""
        game = self.game
        result = game.result
        if self.reward_mode == "win":
            if result is None:
                return [0.0] * self.num_players
            win = 0.0 if len(result.winners) > 1 else 1.0
            return [win if p in result.winners else -1.0 for p in range(self.num_players)]
        scores = [r.total for r in result.players] if result else [p.route_points for p in game.players]
        if self.reward_mode == "score":
            return [float(s) for s in scores]
        total = sum(scores)
        others = self.num_players - 1
        return [s - (total - s) / others for s in scores]

    # ------------------------------------------------------------ observe

    def _mask(self) -> np.ndarray:
        mask = np.zeros(A.N_ACTIONS, dtype=np.int8)
        mask[A.legal_indices(self.game)] = 1
        return mask

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        seat = self.agent_seat[agent]
        game = self.game
        acting = not game.game_over and game.current_player == seat
        return {
            "observation": self.encoder.encode(game, seat),
            "action_mask": self._mask() if acting else np.zeros(A.N_ACTIONS, dtype=np.int8),
        }

    # ------------------------------------------------------------- render

    def render(self) -> Optional[str]:
        if self.render_mode is None or self.game is None:
            return None
        from rich.console import Console

        from ttr.render import render_board_view

        if self.render_mode == "human":
            console = Console()
            console.print(render_board_view(self.game, width=console.width))
            return None
        console = Console(file=io.StringIO(), width=160, color_system=None)
        console.print(render_board_view(self.game, width=160))
        return console.file.getvalue()

    def close(self) -> None:
        pass
