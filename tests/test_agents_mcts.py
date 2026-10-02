import argparse
import dataclasses
import random
from collections import Counter

import pytest

from helpers import set_hand, started_game
from ttr.actions import ClaimRoute, DrawTickets, KeepTickets, Pay
from ttr.agents import mcts
from ttr.agents.greedy import GreedyAgent
from ttr.agents.mcts import MCTSAgent, determinize
from ttr.agents.registry import agent_spec, make_agent
from ttr.board import load_board
from ttr.cards import Color
from ttr.game import Game, Phase
from ttr.memory import FULL_DECK, CardMemory
from ttr.simulate import play_game


def greedy_game(players: int, seed: int, steps: int) -> Game:
    """A game `steps` sub-steps into greedy play: the cards known to the others
    (face-up takes) and hidden ones (blind draws) are both in play."""
    game = Game(load_board("usa"), num_players=players, seed=seed)
    bots = [GreedyAgent(seed + q) for q in range(players)]
    for _ in range(steps):
        p = game.current_player
        game.step(bots[p].act(game, p))
    return game


def hidden(world: Game, viewer: int):
    """Everything in a world that `viewer` can't see."""
    others = [q for q in range(world.num_players) if q != viewer]
    return ([(world.players[q].hand, world.players[q].tickets, world.players[q].pending_tickets) for q in others],
            world.deck, world.ticket_deck, world._initial_returns)


def scramble(game: Game, viewer: int, seed: int) -> Game:
    """A copy that differs from `game` only in what `viewer` can't see: the
    opponents' unknown cards swapped with the deck, the opponents' tickets dealt
    again from those the viewer doesn't hold, the deck orders and random state."""
    rng = random.Random(seed)
    other = game.clone()
    known = CardMemory.at(game, viewer).view(game).known
    others = [q for q in range(game.num_players) if q != viewer]
    pool = list(other.deck)
    for q in others:
        unknown = other.players[q].hand - known[q]
        pool += list(unknown.elements())
    rng.shuffle(pool)
    for q in others:
        n = other.players[q].hand_size - sum(known[q].values())
        other.players[q].hand = known[q] + Counter(pool[:n])
        del pool[:n]
    other.deck = pool
    tickets = list(other.ticket_deck) + list(other._initial_returns)
    for q in others:
        tickets += other.players[q].tickets + other.players[q].pending_tickets
    rng.shuffle(tickets)
    for q in others:
        for attr in ("tickets", "pending_tickets"):
            n = len(getattr(other.players[q], attr))
            setattr(other.players[q], attr, tickets[:n])
            del tickets[:n]
    n = len(other._initial_returns)
    other._initial_returns, other.ticket_deck = tickets[:n], tickets[n:]
    other.rng = random.Random(seed)
    # the log's private entries (blind draws, ticket identities) of everyone else
    other.log = [e if e.player == viewer else dataclasses.replace(e, private={"scrambled": seed}) for e in other.log]
    other.invalidate()
    return other


def initial_choice_game() -> Game:
    """3 players, the first seat has chosen its opening tickets: its returns wait
    in `_initial_returns` and the third seat's offer is still pending."""
    game = Game(load_board("usa"), num_players=3, seed=11, first_player=0)
    pending = game.players[0].pending_tickets
    game.step(next(a for a in game.legal_actions() if len(a.ticket_ids) == 2))
    assert game.current_player == 1 and game._initial_returns and pending
    return game


STATES = [("2p mid-game", lambda: greedy_game(2, 3, 60)), ("3p mid-game", lambda: greedy_game(3, 4, 90)),
          ("opening choice", initial_choice_game)]


@pytest.mark.parametrize("label,make", STATES, ids=[s[0] for s in STATES])
def test_determinize_keeps_what_the_viewer_sees(label, make):
    game = make()
    viewer = game.current_player
    known = CardMemory.at(game, viewer).view(game).known
    for seed in range(5):
        world = determinize(game, viewer, random.Random(seed))
        assert world.route_owner == game.route_owner and world.market == game.market
        assert world.discard == game.discard and world.phase is game.phase
        assert world.current_player == game.current_player and world.turn == game.turn
        assert len(world.deck) == len(game.deck) and len(world.ticket_deck) == len(game.ticket_deck)
        assert len(world._initial_returns) == len(game._initial_returns)
        assert world.legal_actions() == game.legal_actions()
        for q, (w, g) in enumerate(zip(world.players, game.players)):
            assert (w.trains, w.route_points, w.routes) == (g.trains, g.route_points, g.routes)
            assert (w.hand_size, len(w.tickets), len(w.pending_tickets)) == (
                g.hand_size, len(g.tickets), len(g.pending_tickets))
            if q == viewer:
                assert (w.hand, w.tickets, w.pending_tickets) == (g.hand, g.tickets, g.pending_tickets)
            else:
                assert not known[q] - w.hand  # known cards stay where they are
        cards = Counter(world.deck) + Counter(world.discard) + Counter(world.market)
        for p in world.players:
            cards += p.hand
        assert cards == FULL_DECK
        tickets = world.ticket_deck + world._initial_returns
        for p in world.players:
            tickets += p.tickets + p.pending_tickets
        assert sorted(tickets) == list(range(len(game.board.tickets)))


@pytest.mark.parametrize("label,make", STATES, ids=[s[0] for s in STATES])
def test_determinize_ignores_what_the_viewer_cannot_see(label, make):
    game = make()
    viewer = game.current_player
    other = scramble(game, viewer, seed=99)
    assert hidden(other, viewer) != hidden(game, viewer)
    for seed in range(3):
        a = determinize(game, viewer, random.Random(seed))
        b = determinize(other, viewer, random.Random(seed))
        assert hidden(a, viewer) == hidden(b, viewer)
        assert a.rng.getstate() == b.rng.getstate()


def test_determinize_varies_the_hidden_information():
    game = greedy_game(2, 3, 60)
    worlds = {repr(hidden(determinize(game, game.current_player, random.Random(s)), game.current_player))
              for s in range(5)}
    assert len(worlds) == 5


def test_memory_level_0_deals_whole_hands_and_known_tickets_keeps_them():
    game = greedy_game(2, 5, 80)
    viewer = game.current_player
    opp = 1 - viewer
    known = CardMemory.at(game, viewer).view(game).known[opp]
    assert known  # the opponent has taken face-up cards it still holds
    dealt = [determinize(game, viewer, random.Random(s), memory_level=0).players[opp].hand for s in range(20)]
    assert any(known - hand for hand in dealt)  # level 0 forgets them
    world = determinize(game, viewer, random.Random(0), known_tickets=True)
    assert world.players[opp].tickets == game.players[opp].tickets
    assert sorted(world.ticket_deck) == sorted(game.ticket_deck)


def test_search_ignores_what_it_cannot_see():
    game = greedy_game(2, 3, 60)
    viewer = game.current_player
    other = scramble(game, viewer, seed=7)
    a, b = MCTSAgent(seed=1, iterations=25), MCTSAgent(seed=1, iterations=25)
    assert a.act(game, viewer) == b.act(other, viewer)
    stats = lambda agent: {k: (n.visits, round(n.total, 9)) for k, n in agent.last_root.children.items()}
    assert stats(a) == stats(b)


def last_turn_with_cards(color: str, count: int, locomotives: int = 0) -> Game:
    """Player 0's last turn of the game, holding `count` cards of one color."""
    game = started_game(seed=3)
    set_hand(game, 0, **{color: count, "locomotive": locomotives})
    game.final_turns_remaining = 2  # player 0, then player 1, then the game ends
    game.invalidate()
    return game


def test_search_takes_the_most_points_on_its_last_turn():
    game = last_turn_with_cards("yellow", 6)
    agent = MCTSAgent(seed=0, iterations=300, reward="score", prior=0, steps="all")
    action = agent.act(game, 0)
    assert isinstance(action, ClaimRoute) and game.board.routes[action.route_id].length == 6
    root = agent.last_root
    assert root.visits == 300 and sum(n.visits for n in root.children.values()) == 300
    assert set(root.children) == set(game.legal_actions())  # every root action tried


def test_payment_continues_the_claims_subtree():
    game = last_turn_with_cards("yellow", 6, locomotives=1)
    agent = MCTSAgent(seed=0, iterations=200, reward="score", prior=0, steps="all")
    claim = agent.act(game, 0)
    assert isinstance(claim, ClaimRoute)
    subtree = agent.last_root.children[claim]
    before = subtree.visits
    game.step(claim)
    assert len(game.legal_actions()) > 1
    pay = agent.act(game, 0)
    assert isinstance(pay, Pay) and agent.last_root is subtree
    assert 0 < before < subtree.visits == 200


def test_offers_drawn_inside_the_search_are_left_to_the_rollout_policy():
    game = started_game(seed=4)
    agent = MCTSAgent(seed=2, iterations=60, prior=0)
    agent.act(game, 0)
    node = agent.last_root.children[DrawTickets()]
    assert node.visits > 1 and node.children  # the next node is its next turn, not the keep choice
    assert not any(isinstance(a, KeepTickets) for a in node.children)

    game.step(DrawTickets())  # an offer in hand is searched
    agent.act(game, 0)
    assert set(agent.last_root.children) == set(game.legal_actions())


def test_a_prior_sends_most_visits_to_the_rollout_policys_move():
    game = greedy_game(2, 3, 60)
    me = game.current_player
    agent = MCTSAgent(seed=0, iterations=120)
    agent.act(game, me)
    children = agent.last_root.children
    policy = GreedyAgent(0).act(game, me)
    assert children[policy].visits == max(n.visits for n in children.values()) > 120 / 3
    assert sum(n.visits for n in children.values()) == 120


def test_main_steps_leave_second_draws_and_payments_to_the_policy():
    game = last_turn_with_cards("yellow", 6, locomotives=1)
    agent = MCTSAgent(seed=0, iterations=50, reward="score")
    claim = agent.act(game, 0)
    game.step(claim)
    assert len(game.legal_actions()) > 1
    assert agent.act(game, 0) == GreedyAgent().act(game, 0) and agent.searches == 1


@pytest.mark.parametrize("opponent,played", [("racer", {"racer"}), ("greedy+racer", {"greedy", "racer"})])
def test_opponent_models(monkeypatch, opponent, played):
    """Every sampled world plays the opponent with one of the listed bots; with
    several, each of them in some worlds."""
    calls = Counter()

    def counted(kind):
        class Counted(mcts.POLICIES[kind]):
            def act(self, game, player):
                calls[kind] += 1
                return super().act(game, player)
        return Counted

    for kind in ("greedy", "racer"):
        monkeypatch.setitem(mcts.POLICIES, kind, counted(kind))
    game = greedy_game(2, 3, 60)
    MCTSAgent(seed=0, iterations=40, rollout="random", opponent=opponent).act(game, game.current_player)
    assert set(calls) == played


@pytest.mark.parametrize("players", [2, 3])
def test_plays_whole_games(players):
    game = Game(load_board("usa"), num_players=players, seed=players, max_turns=1000)
    agents = [MCTSAgent(seed=0, iterations=3)] + [GreedyAgent(q) for q in range(1, players)]
    result = play_game(game, agents)  # step() rejects illegal moves
    assert result.winners and agents[0].searches > 20


def test_registry_specs():
    agent = make_agent("mcts:iterations=7,c=0.3,prior=0.4,steps=main,rollout=racer,opponent=wary,reward=score,"
                       "memory=0,known_tickets=1", 5)
    assert isinstance(agent, MCTSAgent) and agent.name.startswith("mcts:")
    assert (agent.iterations, agent.c, agent.prior, agent.steps, agent.rollout, agent.opponent, agent.reward,
            agent.memory, agent.known_tickets) == (7, 0.3, 0.4, "main", "racer", "wary", "score", 0, True)
    assert (make_agent("mcts", 0).iterations, make_agent("mcts", 0).prior, make_agent("mcts", 0).steps) == (400, 0.5, "main")
    assert make_agent("mcts:opponent=greedy+racer", 0).opponent == "greedy+racer"
    for bad in ("mcts:depth=3", "mcts:rollout=ppo", "mcts:opponent=greedy+ppo", "mcts:iterations=x", "mcts:reward=points", "mcts:iterations=",
                "mcts:prior=1", "mcts:steps=some"):
        with pytest.raises(ValueError):
            make_agent(bad, 0)
        with pytest.raises(argparse.ArgumentTypeError):
            agent_spec(bad)
