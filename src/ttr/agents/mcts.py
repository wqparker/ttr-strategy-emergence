"""Tier D: Monte Carlo tree search with determinization (PLAN.md "Methods to compare").

No training: every decision is a fresh search from the current state, and the search
sees only what its seat may see (RULES.md §9 #17). Each iteration deals the hidden
information afresh (`determinize`) and plays that sampled world forward:

- The tree holds only the searcher's own decisions (Information Set MCTS with a single
  observer; Cowling, Powley & Whitehouse 2012). A node is a sequence of its own actions,
  shared by every sampled world in which they were legal. Among the actions legal in the
  current world PUCT picks (as in AlphaZero), with the rollout policy's move as the prior:
  most visits go to the policy's move, and another needs better results to draw visits
  away from it. With `guide`, a trained PPO policy's probabilities are the prior instead.
  With `prior=0`, UCB1 with availability counts picks, every action tried once first. Plain UCB1 loses to greedy: a rollout's final margin varies by 30+
  points while most moves differ by a few, so a few hundred iterations spread over ~30
  moves choose nearly at random (PLAN.md "Tier D design").
- Opponents' moves, in the tree and below it, come from an opponent model: a scripted
  bot playing the sampled hand and tickets (with several, `greedy+racer`, each sampled
  world draws one per opponent seat). They are chance events, not tree nodes. With
  `opponent=infer` the bot is part of what each world samples: racer with the probability
  that the opponent plays like a racer, given its public play so far (`racer_belief`),
  else greedy.
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
                    move (or spread as the guide's probabilities) plus an even share of the
                    rest on every move. 0: UCB1
    guide           none: a PPO run (ppo:PATH[@best]) whose policy gives the prior instead of
                    the rollout policy's move (needs the [env] and [deep] extras)
    steps           main: search only the main action of a turn and ticket choices; second
                    draws and payments are the rollout policy's. all: search every decision
    rollout         greedy: scripted bot playing the searcher's seat below the tree
    opponent        greedy: scripted bot modelling every opponent seat. Several joined by
                    `+` (greedy+racer): each sampled world draws one per seat, evenly.
                    infer: racer or greedy per world and seat, by `racer_belief`
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

# The two play styles `opponent=infer` tells apart, each named for the bot that plays it in
# the sampled worlds: ticket players (greedy, wary, collector) and racers (racer, PPO p2b,
# linear p7b). From their public record against greedy and racer, 200 games per agent and
# opponent, add-one smoothed (scripts/opponent_styles.py, 2026-10-02): the chance of keeping
# 3 opening tickets, of drawing tickets on a later turn, and the share of claims by length.
STYLES = {
    "greedy": {"keep3": 0.555, "ticket_draw": 0.0348,
               "length": {1: 0.1354, 2: 0.4582, 3: 0.2140, 4: 0.1212, 5: 0.0489, 6: 0.0223}},
    "racer": {"keep3": 0.000832, "ticket_draw": 2.53e-05,
              "length": {1: 0.0388, 2: 0.0393, 3: 0.0723, 4: 0.1084, 5: 0.1914, 6: 0.5496}},
}


def racer_belief(game: Game, q: int) -> float:
    """The probability that seat q plays like a racer rather than a ticket player, given
    its public play so far (its opening keep, its claims' lengths, whether each of its turns
    drew tickets), from STYLES with even odds before it has played. Naive Bayes: each
    observation counts independently."""
    racer, greedy = STYLES["racer"], STYLES["greedy"]
    odds = 0.0  # log(P(play | racer) / P(play | greedy))
    turns = draws = 0
    card_turns = set()
    for e in game.log:
        if e.player != q:
            continue
        if e.kind == "keep_initial_tickets":
            three = e.public["count"] == 3
            odds += math.log((racer["keep3"] if three else 1 - racer["keep3"])
                             / (greedy["keep3"] if three else 1 - greedy["keep3"]))
        elif e.kind == "claim_route":
            n = game.board.routes[e.public["route"]].length
            odds += math.log(racer["length"][n] / greedy["length"][n])
            turns += 1
        elif e.kind == "draw_tickets":
            draws += 1
            turns += 1
        elif e.kind in ("draw_face_up", "draw_blind"):
            card_turns.add(e.turn)
    turns += len(card_turns)
    odds += draws * math.log(racer["ticket_draw"] / greedy["ticket_draw"])
    odds += (turns - draws) * math.log((1 - racer["ticket_draw"]) / (1 - greedy["ticket_draw"]))
    return 1 / (1 + math.exp(-max(-50.0, min(50.0, odds))))


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
                 memory: int = 2, known_tickets: bool = False, guide=None, name: str = "mcts") -> None:
        """`guide` is a PPO agent spec (or an agent with `net` and `encoder`, as PPOAgent has)."""
        if rollout not in POLICIES or not (opponent == "infer" or all(k in POLICIES for k in opponent.split("+"))):
            raise ValueError(f"rollout and opponent must be one of {', '.join(POLICIES)} "
                             "(opponent: several joined by +)")
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
        if guide is not None and not prior:
            raise ValueError("a guide gives the PUCT prior: it needs prior > 0")
        if isinstance(guide, str) and not guide.startswith("ppo:"):
            raise ValueError("guide must be a PPO run, ppo:PATH[@best]")
        self.rng = random.Random(seed)
        self.iterations = iterations
        self.c = c
        self.prior = prior
        self.steps = steps
        self._delegated = MINOR_PHASES if steps == "main" else ()
        self._policy = POLICIES[rollout](self.rng.getrandbits(32))  # plays the delegated decisions
        self.rollout = rollout
        self.opponent = opponent
        self._infer = opponent == "infer"
        self._opponent_kinds = ["greedy", "racer"] if self._infer else opponent.split("+")
        self.guide = guide if guide is None or isinstance(guide, str) else getattr(guide, "name", "guide")
        self._guide = guide
        if isinstance(guide, str):
            from ttr.agents.registry import make_agent

            self._guide = make_agent(guide, self.rng.getrandbits(32))
        self.last_beliefs: Dict[int, float] = {}  # opponent seat -> racer_belief, with opponent=infer
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
        types = {"iterations": int, "c": float, "prior": float, "steps": str, "rollout": str, "opponent": str,
                 "guide": str, "reward": str,
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
        # per seat, the bots a sampled world can play it with: the rollout policy for mine
        choices = [[POLICIES[k](self.rng.getrandbits(32))
                    for k in ([self.rollout] if q == player else self._opponent_kinds)]
                   for q in range(game.num_players)]
        if self._infer:
            self.last_beliefs = {q: racer_belief(game, q) for q in range(game.num_players) if q != player}
        # the guide sees only this seat's view, the same in every sampled world: once for the root
        self._root_weights = self._guide_weights(game, player) if self._guide is not None and self.prior else None
        while root.visits < self.iterations:
            self._iterate(root, game, player, choices)
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

    def _iterate(self, root: Node, game: Game, me: int, choices: List[List]) -> None:
        world = determinize(game, me, self.rng, self.memory, self.known_tickets)
        models = [bots[0] if len(bots) == 1 else
                  (bots[1] if self.rng.random() < self.last_beliefs[q] else bots[0]) if self._infer else
                  self.rng.choice(bots)
                  for q, bots in enumerate(choices)]
        node, path, in_tree = root, [root], True
        while not world.game_over:
            p = world.current_player
            if (not in_tree or p != me or world.phase in self._delegated
                    or (world.phase in TICKET_PHASES and node is not root)):
                world.step(models[p].act(world, p))
                continue
            legal = world.legal_actions()
            if self.prior:
                if self._guide is None:
                    weights = {models[me].act(world, me): 1.0}
                else:
                    weights = self._root_weights if node is root else self._guide_weights(world, me)
                action = self._select_puct(node, legal, weights)
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

    def _guide_weights(self, world: Game, me: int) -> Dict[Action, float]:
        """The guide policy's probability of each legal move, as it would play `me` here."""
        import numpy as np
        import torch

        from ttr.learn.common import observe

        obs, mask, by_index = observe(self._guide.encoder, world, me)
        with torch.no_grad():
            logits = self._guide.net.logits(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                                            torch.as_tensor(mask).unsqueeze(0))[0].double()
            p = torch.softmax(logits, 0).numpy()
        return {by_index[i]: float(p[i]) for i in np.flatnonzero(mask)}

    def _select_puct(self, node: Node, legal: List[Action], weights: Dict[Action, float]) -> Action:
        """PUCT: mean value + c * prior * sqrt(parent visits) / (1 + visits), the prior being
        `prior` spread by `weights` (the rollout policy's move, or the guide's probabilities)
        plus an even share of the rest; an untried action counts at the parent's mean value."""
        children = node.children
        share = (1 - self.prior) / len(legal)
        scale = self.c * math.sqrt(max(node.visits, 1))
        best, best_score = None, -math.inf
        for a in legal:
            w = weights.get(a)
            prior = share + self.prior * w if w else share
            child = children.get(a)
            if child is None:
                score = node.mean + scale * prior
            else:
                child.available += 1
                score = child.total / child.visits + scale * prior / (1 + child.visits)
            if score > best_score:
                best, best_score = a, score
        return best
