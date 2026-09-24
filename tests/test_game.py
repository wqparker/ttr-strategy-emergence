import random

import pytest

from helpers import play_random, route_id, set_hand, set_market, started_game, total_cards
from ttr.actions import ClaimRoute, DrawBlind, DrawFaceUp, DrawTickets, KeepTickets, Pass, Pay
from ttr.board import load_board
from ttr.cards import Color
from ttr.game import Game, IllegalAction, Phase

L = Color.LOCOMOTIVE


# ----------------------------------------------------------------- setup (§2)


def test_setup_deals_hands_market_and_tickets():
    game = Game(load_board("usa"), num_players=3, seed=1)
    assert all(sum(p.hand.values()) == 4 for p in game.players)
    assert len(game.market) == 5
    assert game.market.count(L) < 3
    assert all(len(p.pending_tickets) == 3 for p in game.players)
    assert len(game.ticket_deck) == 30 - 9
    assert total_cards(game) == 110
    assert game.phase is Phase.CHOOSE_INITIAL_TICKETS


def test_initial_tickets_keep_at_least_two():
    game = Game(load_board("usa"), num_players=2, seed=2, first_player=1)
    assert game.current_player == 1
    sizes = sorted(len(a.ticket_ids) for a in game.legal_actions())
    assert sizes == [2, 2, 2, 3]


def test_initial_returns_go_to_bottom_after_everyone_chooses():  # §9 #7
    game = Game(load_board("usa"), num_players=2, seed=3, first_player=0)
    p0 = game.players[0].pending_tickets
    game.step(KeepTickets(frozenset(p0[:2])))
    assert p0[2] not in game.ticket_deck  # not returned yet
    assert game.current_player == 1
    p1 = game.players[1].pending_tickets
    game.step(KeepTickets(frozenset(p1[1:])))
    assert set(game.ticket_deck[-2:]) == {p0[2], p1[0]}
    assert len(game.ticket_deck) == 26
    assert game.phase is Phase.CHOOSE_ACTION and game.current_player == 0


def test_first_player_is_seeded_random():
    firsts = {Game(load_board("toy"), num_players=4, seed=s).first_player for s in range(40)}
    assert firsts == {0, 1, 2, 3}


# ------------------------------------------------------ drawing cards (§3A, §4)


def test_draw_two_blind_cards_ends_turn():
    game = started_game()
    game.step(DrawBlind())
    assert game.phase is Phase.DRAW_SECOND_CARD and game.current_player == 0
    game.step(DrawBlind())
    assert game.phase is Phase.CHOOSE_ACTION and game.current_player == 1
    assert sum(game.players[0].hand.values()) == 6


def test_no_option_to_stop_after_one_card():  # §9 #14
    game = started_game()
    game.step(DrawBlind())
    assert all(isinstance(a, (DrawBlind, DrawFaceUp)) for a in game.legal_actions())


def test_face_up_locomotive_first_ends_turn():
    game = started_game()
    set_market(game, L, Color.RED, Color.RED, Color.BLUE, Color.BLUE)
    game.step(DrawFaceUp(L))
    assert game.current_player == 1
    assert game.players[0].hand[L] >= 1


def test_face_up_locomotive_not_allowed_as_second_draw():  # §9 #1
    game = started_game()
    set_market(game, L, Color.RED, Color.RED, Color.BLUE, Color.BLUE)
    game.step(DrawFaceUp(Color.RED))
    assert DrawFaceUp(L) not in game.legal_actions()
    with pytest.raises(IllegalAction):
        game.step(DrawFaceUp(L))


def test_blind_locomotive_counts_as_one_card():
    game = started_game()
    game.deck.remove(L)
    game.deck.append(L)  # top of deck
    game.step(DrawBlind())
    assert game.phase is Phase.DRAW_SECOND_CARD


def test_face_up_card_is_replaced():
    game = started_game()
    set_market(game, Color.RED, Color.RED, Color.BLUE, Color.BLUE, Color.GREEN)
    game.deck.remove(Color.WHITE)
    game.deck.append(Color.WHITE)
    game.step(DrawFaceUp(Color.GREEN))
    assert sorted(game.market) == sorted([Color.RED, Color.RED, Color.BLUE, Color.BLUE, Color.WHITE])


def test_three_locomotives_face_up_resets_market():
    game = started_game()
    set_market(game, L, L, Color.RED, Color.BLUE, Color.GREEN)
    game.deck.remove(L)
    game.deck.append(L)  # replacement for the taken card makes 3 Locomotives
    before = game.market_resets
    game.step(DrawFaceUp(Color.RED))
    assert game.market_resets > before
    assert game.market.count(L) < 3
    assert total_cards(game) == 110


def test_market_reset_reshuffles_discards_when_deck_runs_out():  # §9 #2
    game = started_game()
    # Deck: nothing but a few cards; discard holds the rest. Market has 3 Locos.
    set_market(game, L, L, L, Color.RED, Color.BLUE)
    game.discard.extend(game.deck[:-2])
    del game.deck[:-2]
    game._check_market_reset()
    assert game.market.count(L) < 3
    assert total_cards(game) == 110


def test_market_reset_guard_when_legal_market_impossible():  # §9 #2 hard guard
    game = started_game()
    # Put every card in player 1's hand except 3 Locos + 2 others in the market.
    set_market(game, L, L, L, Color.RED, Color.BLUE)
    hand = game.players[1].hand
    for card in game.deck + game.discard:
        hand[card] += 1
    game.deck, game.discard = [], []
    game._check_market_reset()  # must return, not loop forever
    assert game.market.count(L) == 3


def test_empty_deck_and_discard_disables_blind_draw():  # §4
    game = started_game()
    hand = game.players[1].hand
    for card in game.deck + game.discard:
        hand[card] += 1
    game.deck, game.discard = [], []
    actions = game.legal_actions()
    assert DrawBlind() not in actions
    assert any(isinstance(a, DrawFaceUp) for a in actions)


def test_turn_ends_after_one_card_if_nothing_left():  # §9 #4
    game = started_game()
    set_market(game, Color.RED)
    hand = game.players[1].hand
    for card in game.deck + game.discard:
        hand[card] += 1
    game.deck, game.discard = [], []
    game.step(DrawFaceUp(Color.RED))
    assert game.current_player == 1


# ---------------------------------------------------------- claiming (§3B, §6)


def test_claim_colored_route_scores_and_spends():
    game = started_game()
    rid = route_id(game, "Montreal", "New York")  # blue, 3
    set_hand(game, 0, blue=2, locomotive=1)
    game.step(ClaimRoute(rid))
    assert game.phase is Phase.CHOOSE_PAYMENT
    assert game.legal_actions() == [Pay(Color.BLUE, 1)]
    game.step(Pay(Color.BLUE, 1))
    p0 = game.players[0]
    assert game.route_owner[rid] == 0
    assert p0.trains == 42 and p0.route_points == 4
    assert sum(p0.hand.values()) == 0
    assert sorted(game.discard) == sorted([Color.BLUE, Color.BLUE, L])
    assert game.current_player == 1


def test_gray_route_payment_options():
    game = started_game()
    rid = route_id(game, "Toronto", "Montreal")  # gray, 3
    set_hand(game, 0, red=3, green=2, locomotive=3)
    game.step(ClaimRoute(rid))
    options = set(game.legal_actions())
    assert options == {
        Pay(Color.RED, 0), Pay(Color.RED, 1), Pay(Color.RED, 2),
        Pay(Color.GREEN, 1), Pay(Color.GREEN, 2),
        Pay(None, 3),
    }


def test_cannot_claim_without_matching_cards():
    game = started_game()
    rid = route_id(game, "Montreal", "New York")  # blue, 3
    set_hand(game, 0, red=5, blue=1, locomotive=1)
    assert ClaimRoute(rid) not in game.legal_actions()


def test_cannot_claim_route_longer_than_remaining_trains():  # §9 #10
    game = started_game()
    rid = route_id(game, "Seattle", "Helena")  # yellow, 6
    set_hand(game, 0, yellow=6)
    game.players[0].trains = 5
    assert ClaimRoute(rid) not in game.legal_actions()


def test_claimed_route_unavailable():
    game = started_game()
    rid = route_id(game, "Nashville", "Atlanta")
    game.route_owner[rid] = 1
    set_hand(game, 0, red=1)
    assert ClaimRoute(rid) not in game.legal_actions()


@pytest.mark.parametrize("num_players", [2, 3])
def test_double_route_closed_in_small_games(num_players):
    game = started_game(num_players=num_players)
    a = route_id(game, "Boston", "New York", "yellow")
    b = route_id(game, "Boston", "New York", "red")
    game.route_owner[a] = 1
    set_hand(game, 0, red=2)
    assert ClaimRoute(b) not in game.legal_actions()


@pytest.mark.parametrize("num_players", [4, 5])
def test_double_route_open_to_others_in_large_games(num_players):
    game = started_game(num_players=num_players)
    a = route_id(game, "Boston", "New York", "yellow")
    b = route_id(game, "Boston", "New York", "red")
    set_hand(game, 0, red=2)
    game.route_owner[a] = 1
    assert ClaimRoute(b) in game.legal_actions()
    game.route_owner[a] = 0  # same player may never own both halves
    assert ClaimRoute(b) not in game.legal_actions()


# ------------------------------------------------------------- tickets (§3C)


def test_draw_tickets_keep_at_least_one_and_return_to_bottom():
    game = started_game()
    top3 = game.ticket_deck[:3]
    game.step(DrawTickets())
    assert game.phase is Phase.KEEP_TICKETS
    assert min(len(a.ticket_ids) for a in game.legal_actions()) == 1
    assert len(game.legal_actions()) == 7
    game.step(KeepTickets(frozenset([top3[0]])))
    assert top3[0] in game.players[0].tickets
    assert set(game.ticket_deck[-2:]) == set(top3[1:])
    assert game.current_player == 1


def test_draw_tickets_with_fewer_than_three_left():  # §9 #6
    game = started_game()
    game.ticket_deck = game.ticket_deck[:2]
    game.step(DrawTickets())
    assert sorted(len(a.ticket_ids) for a in game.legal_actions()) == [1, 1, 2]


def test_no_ticket_draw_when_deck_empty():  # §9 #5
    game = started_game()
    game.ticket_deck = []
    assert DrawTickets() not in game.legal_actions()


def test_pass_only_when_nothing_else_legal():  # §9 #11
    game = started_game()
    assert Pass() not in game.legal_actions()
    game.ticket_deck = []
    set_hand(game, 0)
    hand = game.players[1].hand
    for card in game.deck + game.discard + game.market:
        hand[card] += 1
    game.deck, game.discard, game.market = [], [], []
    assert game.legal_actions() == [Pass()]


# ------------------------------------------------------------ game end (§7)


def test_final_round_gives_everyone_one_more_turn_trigger_last():  # §9 #15
    game = started_game(num_players=3)
    game.players[0].trains = 3
    rid = route_id(game, "Nashville", "Atlanta")  # length 1
    set_hand(game, 0, red=1)
    game.step(ClaimRoute(rid))
    game.step(Pay(Color.RED, 0))
    assert game.players[0].trains == 2
    assert game.final_turns_remaining == 3
    order = []
    while not game.game_over:
        order.append(game.current_player)
        game.players[game.current_player].trains = 0  # another trigger changes nothing
        game.step(DrawBlind())
        if not game.game_over and game.phase is Phase.DRAW_SECOND_CARD:
            game.step(DrawBlind())
    assert order == [1, 2, 0]


def test_player_with_zero_trains_still_takes_final_turn():
    game = started_game()
    game.players[0].trains = 0
    game.step(DrawBlind())
    game.step(DrawBlind())  # triggers
    assert game.current_player == 1
    game.step(DrawBlind())
    game.step(DrawBlind())
    assert game.current_player == 0 and not game.game_over
    assert not any(isinstance(a, ClaimRoute) for a in game.legal_actions())
    game.step(DrawTickets())
    game.step(game.legal_actions()[0])
    assert game.game_over


# --------------------------------------------------------- final scoring (§8)


def _force_finish(game):
    game._finish(truncated=False)
    return game.result


def test_final_scoring_tickets_and_longest_path():
    game = started_game()
    p0, p1 = game.players
    boston_ny = route_id(game, "Boston", "New York", "yellow")  # 2
    ny_wash = route_id(game, "New York", "Washington", "orange")  # 2
    seattle_helena = route_id(game, "Seattle", "Helena")  # 6
    for rid, owner in [(boston_ny, 0), (ny_wash, 0), (seattle_helena, 1)]:
        game.route_owner[rid] = owner
        game.players[owner].routes.append(rid)
        game.players[owner].route_points += game.board.routes[rid].points
    tickets = {(t.a, t.b): t.id for t in game.board.tickets}
    p0.tickets = [tickets[("New York", "Atlanta")]]  # not completed: -6
    p1.tickets = []
    result = _force_finish(game)
    r0, r1 = result.players
    assert (r0.route_points, r0.ticket_points, r0.longest_path) == (4, -6, 4)
    assert (r1.route_points, r1.longest_path, r1.longest_path_bonus) == (15, 6, True)
    assert r0.total == -2 and r1.total == 25
    assert result.winners == [1]


def _give_route(game, rid, owner, route_points):
    game.route_owner[rid] = owner
    game.players[owner].routes.append(rid)
    game.players[owner].route_points = route_points


def test_longest_path_tie_both_get_bonus_and_full_tie_is_shared_win():
    game = started_game()
    _give_route(game, route_id(game, "Boston", "New York", "yellow"), 0, 2)
    _give_route(game, route_id(game, "New York", "Pittsburgh", "white"), 1, 2)
    game.players[0].tickets = []
    game.players[1].tickets = []
    result = _force_finish(game)
    assert all(r.longest_path_bonus for r in result.players)
    assert result.winners == [0, 1]  # §9 #12 shared win


def test_tiebreak_most_completed_tickets():
    game = started_game(board="toy")
    _give_route(game, route_id(game, "A", "B", "red"), 0, 4)
    _give_route(game, route_id(game, "C", "D"), 1, 2)
    t_cd = next(t for t in game.board.tickets if {t.a, t.b} == {"C", "D"})  # 2 pts
    game.players[0].tickets = []
    game.players[1].tickets = [t_cd.id]
    result = _force_finish(game)
    # Both: longest path 2 -> both get the bonus. p0: 4 + 10 = 14; p1: 2 + 2 + 10 = 14.
    assert result.players[0].total == result.players[1].total == 14
    assert result.winners == [1]


def test_tiebreak_longest_path_holder():
    game = started_game(board="toy")
    _give_route(game, route_id(game, "B", "D"), 0, 17)  # length 4 -> bonus
    _give_route(game, route_id(game, "C", "D"), 1, 27)  # length 2
    game.players[0].tickets = []
    game.players[1].tickets = []
    result = _force_finish(game)
    assert result.players[0].total == result.players[1].total == 27
    assert result.winners == [0]


# ------------------------------------------------------- whole-game invariants


@pytest.mark.parametrize("board,num_players", [("toy", 2), ("usa", 2), ("usa", 3), ("usa", 5)])
def test_random_games_terminate_and_conserve_cards(board, num_players):
    rng = random.Random(123)
    for seed in range(5):
        game = Game(load_board(board), num_players=num_players, seed=seed, max_turns=3000)

        def check(g):
            assert total_cards(g) == 110
            assert all(p.trains >= 0 for p in g.players)
            owned = sum(len(p.routes) for p in g.players)
            assert owned == len(g.route_owner)

        play_random(game, rng, check)
        assert game.result is not None
        n_tickets = sum(len(p.tickets) for p in game.players) + len(game.ticket_deck)
        assert n_tickets == len(game.board.tickets)


def test_same_seed_same_game():
    def run(seed):
        game = Game(load_board("usa"), seed=seed)
        rng = random.Random(seed)
        play_random(game, rng)
        return [(e.kind, e.player, dict(e.public)) for e in game.log]

    assert run(7) == run(7)
