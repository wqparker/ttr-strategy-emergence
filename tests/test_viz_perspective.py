"""What each viewer may see (ttr.viz.perspective). No display needed."""

import random
from collections import Counter

import pytest

from ttr.actions import ClaimRoute, DrawBlind, DrawFaceUp, Pay
from ttr.board import load_board
from ttr.cards import Color
from ttr.game import Game
from ttr.memory import CardMemory
from ttr.viz.perspective import Perspective, code_map

from helpers import play_random, route_id, set_hand, set_market, started_game


def two_reds_taken(seed: int = 5):
    """P0 takes two face-up reds, which P1 may count (RULES.md §9 #17)."""
    game = started_game(seed=seed)
    set_market(game, Color.RED, Color.RED, Color.BLUE, Color.GREEN, Color.WHITE)
    game.step(DrawFaceUp(Color.RED))
    game.step(DrawFaceUp(Color.RED))
    return game


def test_all_seeing_reveals_every_hand_and_ticket():
    game = started_game(seed=3)
    vm = Perspective().view(game)
    assert vm.all_seeing and vm.viewer is None
    for seat, facts in enumerate(vm.seats):
        assert facts.revealed and facts.hand == game.players[seat].hand
        assert facts.tickets is not None
        assert {t.id for t in facts.tickets} == set(game.players[seat].tickets)
        assert facts.known is None and facts.unknown is None
    assert vm.table.unseen is None  # no estimating needed when everything shows


def test_player_view_hides_opponents_but_keeps_public_facts():
    game = started_game(seed=3)
    vm = Perspective(0).view(game)
    me, them = vm.seats[0], vm.seats[1]
    assert me.is_viewer and me.revealed and me.hand == game.players[0].hand
    assert me.tickets is not None
    # Hidden: the opponent's cards and the contents of their tickets.
    assert them.hand is None and them.tickets is None
    # Public (RULES.md §9 #13, #17): sizes, trains, score.
    assert them.hand_size == sum(game.players[1].hand.values())
    assert them.ticket_count == len(game.players[1].tickets)
    assert (them.trains, them.route_points) == (game.players[1].trains, game.players[1].route_points)


def test_player_view_matches_card_memory():
    game = two_reds_taken()
    vm = Perspective(1).view(game)
    truth = CardMemory.at(game, 1, 2).view(game)
    assert vm.seats[0].known == Counter({Color.RED: 2}) == truth.known[0]
    assert vm.seats[0].unknown == truth.unknown[0]
    assert vm.table.unseen == truth.unseen
    # A lower bound, never more than the opponent really holds.
    for color, n in vm.seats[0].known.items():
        assert n <= game.players[0].hand[color]


@pytest.mark.parametrize("level", [0, 1, 2])
def test_memory_levels(level):
    game = two_reds_taken()
    vm = Perspective(1, level).view(game)
    them = vm.seats[0]
    assert them.hand is None  # hidden at every level
    if level == 0:
        assert them.known is None and them.unknown is None
    else:
        assert them.known == Counter({Color.RED: 2})
    assert (vm.table.unseen is not None) == (level == 2)


def test_known_cards_drop_when_spent():
    game = two_reds_taken()
    perspective = Perspective(1)
    assert perspective.view(game).seats[0].known == Counter({Color.RED: 2})

    game.step(DrawBlind())  # P1's turn, then back to P0
    game.step(DrawBlind())
    assert game.current_player == 0
    set_hand(game, 0, red=3)
    rid = route_id(game, "Nashville", "Atlanta")  # gray, 1
    game.step(ClaimRoute(rid))
    game.step(Pay(Color.RED, 0))  # one of the two known reds is spent
    assert perspective.view(game).seats[0].known == Counter({Color.RED: 1})


def test_tickets_carry_completion_and_pending_flags():
    game = Game(load_board("usa"), num_players=2, seed=1, first_player=0)
    vm = Perspective().view(game)  # still choosing the initial tickets
    assert all(t.pending for t in vm.seats[0].tickets)
    assert vm.seats[0].ticket_count == 0

    game = started_game(seed=1)
    ticket = game.board.tickets[game.players[0].tickets[0]]
    rid = route_id(game, "Denver", "Santa Fe")
    set_hand(game, 0, red=3)
    game.step(ClaimRoute(rid))
    game.step(Pay(Color.RED, 0))
    vm = Perspective().view(game)
    facts = {t.id: t for t in vm.seats[0].tickets}
    assert not any(t.pending for t in facts.values())
    assert facts[ticket.id].done is False  # one 2-space route completes nothing
    assert facts[ticket.id].points == ticket.points


def test_perspective_survives_replay_scrubbing_backwards():
    """A reused Perspective rebuilds its tracker when the log shrinks, so a
    replay can step backwards without inflating the known counts."""
    game = started_game(seed=7)
    rng = random.Random(7)
    states = [game.clone()]
    for _ in range(60):
        if game.game_over:
            break
        game.step(rng.choice(game.legal_actions()))
        states.append(game.clone())

    perspective = Perspective(1)
    perspective.view(states[-1])
    for state in states[::-7]:  # jump around, newest to oldest
        vm = perspective.view(state)
        truth = CardMemory.at(state, 1, 2).view(state)
        assert vm.seats[0].known == truth.known[0]
        assert vm.table.unseen == truth.unseen


def test_events_are_public_only():
    game = started_game(seed=2)
    rng = random.Random(2)
    play_random(game, rng)
    vm = Perspective(0).view(game, events=5)
    assert len(vm.events) <= 5
    blind = [line for line in vm.events if "from the deck" in line]
    for line in blind:  # a blind draw never names the card
        assert not any(c.value in line for c in Color)


def test_names_and_codes():
    game = started_game()
    vm = Perspective().view(game, names=["greedy", "random"])
    assert [s.name for s in vm.seats] == ["greedy", "random"]
    codes = code_map(game.board)
    assert codes["Los Angeles"] == "LAX" and len(codes) == len(game.board.cities)
