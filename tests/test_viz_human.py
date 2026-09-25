"""Human play in the viewer (ttr.viz.app), driven by synthesized clicks.

The rule under test throughout: the UI only ever offers moves that are in
`game.legal_actions()`, and a click that is not one of them changes nothing.
"""

import pytest

pygame = pytest.importorskip("pygame")

from ttr.actions import ClaimRoute, DrawBlind, DrawFaceUp, DrawTickets, KeepTickets, Pay  # noqa: E402
from ttr.agents import GreedyAgent  # noqa: E402
from ttr.board import load_board  # noqa: E402
from ttr.game import Game, Phase  # noqa: E402
from ttr.viz.app import HumanControl, Viewer, live_timeline  # noqa: E402
from ttr.viz.screen import Screen  # noqa: E402

from helpers import route_id, set_hand  # noqa: E402


def human_game(num_players=2, seed=3, seat=0, keep_tickets=True):
    """A live game with one human seat, past the initial ticket choice."""
    agents = [GreedyAgent(seed + i) for i in range(num_players)]
    game = Game(load_board("usa"), num_players=num_players, seed=seed, first_player=0, max_turns=1000)
    timeline = live_timeline(game, agents, [seat])
    human = HumanControl([seat])
    viewer = Viewer(timeline, Screen(game.board, scale=0.5), playing=False, human=human)
    viewer.draw(pygame.Surface(viewer.screen.size))
    if keep_tickets:
        while timeline.current.phase is Phase.CHOOSE_INITIAL_TICKETS:
            if human.acts_now(timeline.current):
                click(viewer, "choice", -1)  # the confirm chip
            else:
                viewer.act("forward")
    return viewer, human, timeline


def click(viewer, kind, index=0):
    """Click what the last frame drew: a chip, a face-up slot or a pile."""
    viewer.draw(pygame.Surface(viewer.screen.size))
    hits = viewer.screen.hits
    rect = hits["choices"][index] if kind == "choice" else (
        hits["market"][index] if kind == "market" else hits[kind])
    return viewer.click(rect.center)


def click_route(viewer, rid):
    car = viewer.screen.board_view.layout.routes[rid].cars[0]
    ox, oy = viewer.screen.board_origin
    x, y = viewer.screen.board_view.to_screen(car.center)
    viewer.draw(pygame.Surface(viewer.screen.size))
    return viewer.click((x + ox, y + oy))


# ------------------------------------------------------------------ the seat


def test_bots_play_until_the_human_is_to_act():
    viewer, human, timeline = human_game(num_players=3, seat=1)
    timeline.to_end()
    game = timeline.current
    assert game.current_player == 1 and human.acts_now(game)
    assert not timeline.forward()  # stalls: the seat is not a bot's to play
    assert not game.game_over


def test_a_bot_only_game_never_stalls():
    """The same timeline with no human seat plays itself to the end."""
    game = Game(load_board("usa"), num_players=2, seed=3, first_player=0, max_turns=1000)
    timeline = live_timeline(game, [GreedyAgent(1), GreedyAgent(2)])
    timeline.to_end()
    assert timeline.current.game_over


# ----------------------------------------------------------------- the board


def test_claiming_a_route_then_paying():
    viewer, human, timeline = human_game()
    set_hand(timeline.current, 0, red=6, locomotive=2)
    rid = route_id(timeline.current, "Denver", "Santa Fe")  # gray, 2

    assert click_route(viewer, rid) == ClaimRoute(rid)
    assert timeline.current.phase is Phase.CHOOSE_PAYMENT
    labels = [label for label, _, _ in human.choices(timeline.current)]
    assert labels[0] == "2 red" and "1 red + 1 loco" in labels

    assert isinstance(click(viewer, "choice", 0), Pay)
    game = timeline.current
    assert game.route_owner[rid] == 0 and game.players[0].trains == 43
    assert game.current_player == 1  # the claim ended the turn


def test_an_unclaimable_route_does_nothing():
    viewer, human, timeline = human_game()
    set_hand(timeline.current, 0, blue=1)  # cannot pay for anything
    rid = route_id(timeline.current, "Denver", "Santa Fe")
    before = len(timeline.states)
    assert click_route(viewer, rid) is None
    assert len(timeline.states) == before and not timeline.current.route_owner


def test_hover_only_highlights_claimable_routes():
    viewer, human, timeline = human_game()
    set_hand(timeline.current, 0, red=6)
    claimable = human.claimable(timeline.current)
    assert claimable and all(
        ClaimRoute(rid) in timeline.current.legal_actions() for rid in claimable
    )
    # In a payment sub-step nothing on the board is claimable.
    click_route(viewer, claimable[0])
    assert human.claimable(timeline.current) == []


# ----------------------------------------------------------------- the table


def test_market_deck_and_ticket_clicks():
    viewer, human, timeline = human_game()
    color = timeline.current.market[1]
    action = click(viewer, "market", 1)
    assert action == DrawFaceUp(color)
    assert timeline.current.players[0].hand[color] >= 1

    if timeline.current.current_player == 0:  # a non-locomotive take leaves a second draw
        assert click(viewer, "deck") == DrawBlind()

    viewer, human, timeline = human_game()
    assert click(viewer, "tickets") == DrawTickets()
    assert timeline.current.phase is Phase.KEEP_TICKETS


def test_clicks_do_nothing_on_a_bot_seat():
    viewer, human, timeline = human_game(seat=1)
    timeline.to_start()
    game = timeline.current
    assert game.current_player == 0 and not human.acts_now(game)
    before = len(timeline.states)
    assert click(viewer, "market", 0) is None
    assert click(viewer, "deck") is None
    assert human.choices(game) == []
    assert len(timeline.states) == before


# --------------------------------------------------------------- the tickets


def test_ticket_chips_toggle_and_confirm():
    viewer, human, timeline = human_game(keep_tickets=False)
    game = timeline.current
    assert game.phase is Phase.CHOOSE_INITIAL_TICKETS
    pending = list(game.players[0].pending_tickets)
    assert [label for label, _, _ in human.choices(game)][-1] == "keep 3"

    click(viewer, "choice", 0)  # untick one
    assert human.keep == set(pending[1:])
    assert [sel for _, sel, _ in human.choices(game)] == [False, True, True, False]

    action = click(viewer, "choice", -1)
    assert action == KeepTickets(frozenset(pending[1:]))
    assert game.players[0].tickets == []  # the click applied to a new state
    assert timeline.current.players[0].tickets == pending[1:]


def test_confirm_is_disabled_below_the_minimum_keep():
    viewer, human, timeline = human_game(keep_tickets=False)
    game = timeline.current
    for i in range(3):  # untick all three; the initial minimum is 2
        click(viewer, "choice", i)
    assert human.keep == set()
    assert human.choices(game)[-1][2] is False
    before = len(timeline.states)
    assert click(viewer, "choice", -1) is None
    assert len(timeline.states) == before


# ------------------------------------------------------------------ timeline


def test_acting_after_stepping_back_forks():
    viewer, human, timeline = human_game()
    set_hand(timeline.current, 0, red=6)
    rid = route_id(timeline.current, "Denver", "Santa Fe")
    click_route(viewer, rid)
    click(viewer, "choice", 0)
    claimed = len(timeline.states)

    timeline.back()
    timeline.back()
    assert timeline.index == claimed - 3
    other = route_id(timeline.current, "Helena", "Denver")  # green, 4
    set_hand(timeline.current, 0, green=4)
    click_route(viewer, other)
    assert len(timeline.states) == timeline.index + 1  # the old branch is gone
    assert timeline.current.pending_route == other


def test_every_offered_chip_is_a_legal_move():
    viewer, human, timeline = human_game(num_players=3)
    for _ in range(40):
        game = timeline.current
        if game.game_over:
            break
        if not human.acts_now(game):
            timeline.forward()
            continue
        chips = human.choices(game)
        for i, (_, _, enabled) in enumerate(chips):
            if not enabled:
                continue
            # Asking for a chip either plays a legal move or only toggles state.
            action = human.click_choice(game, i)
            assert action is None or action in game.legal_actions()
        if not click(viewer, "deck"):
            click(viewer, "market", 0)


def test_result_box_draws_at_the_end():
    game = Game(load_board("usa"), num_players=3, seed=8, first_player=0, max_turns=1000)
    timeline = live_timeline(game, [GreedyAgent(i) for i in range(3)])
    human = HumanControl([0])
    viewer = Viewer(timeline, Screen(game.board, scale=0.5), playing=False, human=human)
    timeline.to_end()
    assert timeline.current.game_over and timeline.current.result is not None
    viewer.draw(pygame.Surface(viewer.screen.size))  # the scoreboard, over the board
    assert human.claimable(timeline.current) == []  # nothing to click once it is over
    assert human.choices(timeline.current) == []
