"""Tier D: Monte Carlo tree search with determinization (PLAN.md "Methods to compare").

No training: every decision is a fresh search from the current state, and the search
sees only what its seat may see (RULES.md §9 #17). Each iteration deals the hidden
information afresh (`determinize`) and plays that sampled world forward:

- The tree holds only the searcher's own decisions (Information Set MCTS with a single
  observer; Cowling, Powley & Whitehouse 2012). A node is a sequence of its own actions,
  shared by every sampled world in which they were legal. Among the actions legal in the
  current world PUCT picks (as in AlphaZero), with the rollout policy's move as the prior:
  most visits go to the policy's move, and another needs better results to draw visits
  away from it. With `prior=0`, UCB1 with availability counts picks instead, every action
  tried once first. Plain UCB1 loses to greedy: a rollout's final margin varies by 30+
  points while most moves differ by a few, so a few hundred iterations spread over ~30
  moves choose nearly at random (PLAN.md "Tier D design").
- Opponents' moves, in the tree and below it, come from an opponent model: a scripted
  bot playing the sampled hand and tickets. They are chance events, not tree nodes.
- Ticket offers drawn inside the search differ from world to world, so the rollout
  policy chooses what to keep from them; only an offer already in hand (at the root) is
  searched.
- Below the tree the rollout policy plays the searcher's seat to the end of the game.
- The value is the final reward of `ttr.env.reward` for the searcher's seat (margin by
  default), / 100 in the score modes.

The move is the root's most visited action. With `steps=all`, a claim's payment
continues the claim's subtree, whose statistics still hold: claiming reveals nothing.

Options (also as a registry spec, `mcts:iterations=800,rollout=racer`), defaults first:

    iterations      400: search iterations per decision (forced moves take none)
    c               0.5: exploration constant (PUCT's, or UCB1's), on the reward's scale
    prior           0.5: PUCT, the prior probability being `prior` on the rollout policy's
                    move plus an even share of the rest on every move. 0: UCB1
    steps           main: search only the main action of a turn and ticket choices; second
                    draws and payments are the rollout policy's. all: search every decision
    rollout         greedy: scripted bot playing the searcher's seat below the tree
    opponent        greedy: scripted bot modelling every opponent seat
    reward          margin | score | win (ttr.env.reward)
    memory          2: card memory level (ttr.memory). 0 deals opponents' whole hands from
                    the unseen cards; 1 and 2 keep the cards known to be in their hands
    known_tickets   0: 1 keeps the opponents' true tickets instead of sampling them.
                    Cheating: a diagnostic upper bound on what inferring them could be worth
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from typing import Dict, List, Optional

from ttr.actions import Action, ClaimRoute
from ttr.agents.greedy import CollectorAgent, GreedyAgent, WaryAgent
from ttr.agents.racer import RacerAgent
from ttr.agents.random_agent import RandomAgent
from ttr.cards import ALL_COLORS
from ttr.env.reward import REWARD_MODES, reward_values
from ttr.game import Game, Phase
from ttr.memory import FULL_DECK, CardMemory

# Policies the search can play with: scripted bots, fast enough for thousands of rollouts.
POLICIES = {"random": RandomAgent, "greedy": GreedyAgent, "wary": WaryAgent, "racer": RacerAgent,
            "collector": CollectorAgent}
TICKET_PHASES = (Phase.CHOOSE_INITIAL_TICKETS, Phase.KEEP_TICKETS)
MINOR_PHASES = (Phase.DRAW_SECOND_CARD, Phase.CHOOSE_PAYMENT)  # left to the policy with steps=main


def determinize(game: Game, viewer: int, rng: random.Random, memory_level: int = 2,
                known_tickets: bool = False) -> Game:
    """A copy of `game` with everything `viewer` can't see dealt at random, consistently
    with what it can: the opponents' cards outside its card memory and the draw deck from
    the unseen cards, the opponents' tickets (held and on offer), tickets awaiting return
    and the ticket deck from the tickets it doesn't hold, and a fresh random state for
    later shuffles. Only public information and the viewer's own cards and tickets are
    read, in a fixed order, so games that differ only in what the viewer can't see give
    the same world for the same `rng`."""
    world = game.clone()
    world.rng = random.Random(rng.getrandbits(64))
    me = game.players[viewer]
    opponents = [q for q in range(game.num_players) if q != viewer]

    known = CardMemory.at(game, viewer, memory_level).view(game).known  # empty at level 0
    unseen = Counter(FULL_DECK)
    unseen.subtract(me.hand)
    unseen.subtract(game.market)
    unseen.subtract(game.discard)
    for cards in known.values():
        unseen.subtract(cards)
    pool = [c for c in ALL_COLORS for _ in range(unseen[c])]
    rng.shuffle(pool)
    for q in opponents:
        hand = Counter(known.get(q, ()))
        missing = game.players[q].hand_size - sum(hand.values())
        hand.update(pool[len(pool) - missing:])
        del pool[len(pool) - missing:]
        world.players[q].hand = hand
    assert len(pool) == len(game.deck)
    world.deck = pool

    if known_tickets:
        pool = sorted(game.ticket_deck)
        rng.shuffle(pool)
        world.ticket_deck = pool
    else:
        mine = set(me.tickets) | set(me.pending_tickets)
        pool = [t.id for t in game.board.tickets if t.id not in mine]
        rng.shuffle(pool)
        for q in opponents:
            player = world.players[q]
            for attr in ("tickets", "pending_tickets"):
                n = len(getattr(player, attr))
                setattr(player, attr, pool[:n])
                del pool[:n]
        n = len(world._initial_returns)
        world._initial_returns = pool[:n]
        del pool[:n]
        assert len(pool) == len(game.ticket_deck)
        world.ticket_deck = pool
    world.invalidate()
    return world


class Node:
    """One of the searcher's decision points: the statistics of the action leading here."""

    __slots__ = ("children", "visits", "total", "available")

    def __init__(self) -> None:
        self.children: Dict[Action, Node] = {}
        self.visits = 0
        self.total = 0.0  # summed values, searcher's view
        self.available = 0  # iterations in which the action leading here was legal

    @property
    def mean(self) -> float:
        return self.total / self.visits if self.visits else 0.0


class MCTSAgent:
    def __init__(self, seed: Optional[int] = None, iterations: int = 400, c: float = 0.5, prior: float = 0.5,
                 steps: str = "main", rollout: str = "greedy", opponent: str = "greedy", reward: str = "margin",
                 memory: int = 2, known_tickets: bool = False, name: str = "mcts") -> None:
        if rollout not in POLICIES or opponent not in POLICIES:
            raise ValueError(f"rollout and opponent must be one of {', '.join(POLICIES)}")
        if reward not in REWARD_MODES:
            raise ValueError(f"reward must be one of {', '.join(REWARD_MODES)}")
        if memory not in (0, 1, 2):
            raise ValueError("memory must be 0, 1 or 2")
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if not 0 <= prior < 1:
            raise ValueError("prior must be in [0, 1)")
        if steps not in ("all", "main"):
            raise ValueError("steps must be all or main")
        self.rng = random.Random(seed)
        self.iterations = iterations
        self.c = c
        self.prior = prior
        self.steps = steps
        self._delegated = MINOR_PHASES if steps == "main" else ()
        self._policy = POLICIES[rollout](self.rng.getrandbits(32))  # plays the delegated decisions
        self.rollout = rollout
        self.opponent = opponent
        self.reward = reward
        self.memory = memory
        self.known_tickets = known_tickets
        self.name = name
        self.last_root: Optional[Node] = None  # the latest search, for inspection
        self.searches = 0
        self.search_seconds = 0.0
        self._carry = None  # (state signature, subtree) of the claim just chosen, for its payment

    @classmethod
    def from_spec(cls, options: str, seed: Optional[int] = None, name: str = "mcts") -> "MCTSAgent":
        """`iterations=800,rollout=racer,...` (the part of a registry spec after `mcts:`)."""
        types = {"iterations": int, "c": float, "prior": float, "steps": str, "rollout": str, "opponent": str, "reward": str,
                 "memory": int, "known_tickets": lambda v: {"1": True, "true": True, "0": False,
                                                           "false": False}[v.lower()]}
        kwargs = {}
        for item in filter(None, options.split(",")):
            key, _, value = item.partition("=")
            if key not in types or not value:
                raise ValueError(f"bad mcts option {item!r}; options: {', '.join(types)}")
            try:
                kwargs[key] = types[key](value)
            except (KeyError, ValueError):
                raise ValueError(f"bad value in mcts option {item!r}") from None
        return cls(seed, name=name, **kwargs)

    def act(self, game: Game, player: int) -> Action:
        carry, self._carry = self._carry, None
        legal = game.legal_actions()
        if len(legal) == 1:
            return legal[0]
        if game.phase in self._delegated:
            return self._policy.act(game, player)
        start = time.perf_counter()
        root = Node()
        if carry is not None and carry[0] == self._signature(game):
            root = carry[1]
        models = [POLICIES[self.rollout if q == player else self.opponent](self.rng.getrandbits(32))
                  for q in range(game.num_players)]
        while root.visits < self.iterations:
            self._iterate(root, game, player, models)
        action = max((a for a in legal if a in root.children), key=lambda a: root.children[a].visits)
        if isinstance(action, ClaimRoute):
            self._carry = (self._signature(game, action.route_id), root.children[action])
        self.last_root = root
        self.searches += 1
        self.search_seconds += time.perf_counter() - start
        return action

    @staticmethod
    def _signature(game: Game, route: Optional[int] = None):
        """Identifies the payment step that follows claiming `route` here (or, without
        `route`, the payment step `game` is at)."""
        if route is None:
            if game.phase is not Phase.CHOOSE_PAYMENT:
                return None
            route = game.pending_route
        return id(game), game.turn, route, len(game.log)

    def _iterate(self, root: Node, game: Game, me: int, models: List) -> None:
        world = determinize(game, me, self.rng, self.memory, self.known_tickets)
        node, path, in_tree = root, [root], True
        while not world.game_over:
            p = world.current_player
            if (not in_tree or p != me or world.phase in self._delegated
                    or (world.phase in TICKET_PHASES and node is not root)):
                world.step(models[p].act(world, p))
                continue
            legal = world.legal_actions()
            if self.prior:
                action = self._select_puct(node, legal, models[me].act(world, me))
            else:
                action = self._select_ucb(node, legal, models[me], world)
            child = node.children.get(action)
            if child is None:
                child = node.children[action] = Node()
                child.available = 1
                in_tree = False  # the rest of the game is the rollout
            world.step(action)
            node = child
            path.append(child)
        value = reward_values(world, self.reward)[me]
        if self.reward != "win":
            value /= 100
        for n in path:
            n.visits += 1
            n.total += value

    def _select_ucb(self, node: Node, legal: List[Action], policy, world: Game) -> Action:
        """UCB1 with availability counts; untried actions first, the policy's before the rest."""
        children = node.children
        untried = []
        for a in legal:
            child = children.get(a)
            if child is None:
                untried.append(a)
            else:
                child.available += 1
        if untried:
            preferred = policy.act(world, world.current_player)
            return preferred if preferred in untried else self.rng.choice(untried)
        return max(legal, key=lambda a: self._ucb(children[a]))

    def _ucb(self, node: Node) -> float:
        return node.total / node.visits + self.c * math.sqrt(math.log(node.available) / node.visits)

    def _select_puct(self, node: Node, legal: List[Action], preferred: Action) -> Action:
        """PUCT: mean value + c * prior * sqrt(parent visits) / (1 + visits); an untried
        action counts at the parent's mean value."""
        children = node.children
        share = (1 - self.prior) / len(legal)
        scale = self.c * math.sqrt(max(node.visits, 1))
        best, best_score = None, -math.inf
        for a in legal:
            prior = share + self.prior if a == preferred else share
            child = children.get(a)
            if child is None:
                score = node.mean + scale * prior
            else:
                child.available += 1
                score = child.total / child.visits + scale * prior / (1 + child.visits)
            if score > best_score:
                best, best_score = a, score
        return best
