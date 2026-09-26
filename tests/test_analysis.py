"""Per-route statistics over game records (ttr.analysis)."""

import pytest

from ttr.actions import ClaimRoute, Pay
from ttr.analysis import STATS, AnalysisError, Summary, analyze, record_paths, summarize
from ttr.board import load_board
from ttr.cards import Color
from ttr.record import GameRecord
from ttr.simulate import run_matches

from helpers import set_hand, started_game


def claim(game, route_id):
    """The current player claims a route, paying in its color (red if gray)."""
    route = game.board.routes[route_id]
    color = route.color or Color.RED
    set_hand(game, game.current_player, **{color.value: route.length})
    game.step(ClaimRoute(route_id))
    game.step(Pay(color, 0))


def double_route(game):
    return next(r for r in game.board.routes if r.sibling is not None)


def single_route(game):
    return next(r for r in game.board.routes if r.sibling is None)


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    folder = tmp_path_factory.mktemp("records")
    run_matches(["greedy", "random"], 8, load_board("usa"), seed=3, record_dir=folder, board_ref="usa")
    return folder


# ------------------------------------------------------------ hand-made games


def test_known_claims_give_known_values():
    game = started_game(seed=1)
    double, single = double_route(game), single_route(game)
    claim(game, double.id)  # P0, turn 0
    claim(game, single.id)  # P1, turn 1
    summary = Summary(game.board)
    summary.add(game, ["alpha", "beta"])

    assert summary.games == 1 and summary.agents == ["alpha", "beta"]
    assert summary.value(double.id, "claim_rate") == 1.0
    assert summary.value(double.sibling, "claim_rate") == 0.0
    assert summary.value(double.id, "avg_turn") == 0
    assert summary.value(single.id, "avg_turn") == 1
    assert summary.value(double.sibling, "avg_turn") is None  # never claimed
    # The claimed side closed its sibling (2 players); a single route has no pair.
    assert summary.value(double.sibling, "contested") == 1.0
    assert summary.value(double.id, "contested") == 0.0
    assert summary.value(single.id, "contested") is None

    assert summary.value(double.id, "claim_rate", agent="alpha") == 1.0
    assert summary.value(double.id, "claim_rate", agent="beta") == 0.0
    assert summary.value(single.id, "avg_turn", agent="beta") == 1
    assert summary.value(double.sibling, "contested", agent="alpha") == 1.0
    assert summary.value(double.sibling, "contested", agent="beta") == 0.0


def test_seat_filter():
    game = started_game(seed=1)
    double, single = double_route(game), single_route(game)
    claim(game, double.id)  # P0
    claim(game, single.id)  # P1
    summary = Summary(game.board)
    summary.add(game, ["greedy", "greedy"])  # same agent both seats: only seats tell them apart
    summary.add(started_game(seed=2), ["greedy", "greedy"])

    assert summary.seats == [0, 1] and summary.games_for(seat=0) == 2
    assert summary.value(double.id, "claim_rate", seat=0) == 0.5
    assert summary.value(double.id, "claim_rate", seat=1) == 0.0
    assert summary.value(single.id, "claim_rate", seat=1) == 0.5
    assert summary.value(double.id, "claim_rate", agent="greedy") == 0.5
    assert summary.value(single.id, "avg_turn", seat=0) is None
    assert summary.value(double.sibling, "contested", seat=0) == 0.5
    assert summary.value(double.sibling, "contested", seat=1) == 0.0
    # Agent and seat together: that agent in that seat.
    assert summary.value(single.id, "claim_rate", agent="greedy", seat=1) == 0.5
    assert summary.value(single.id, "claim_rate", agent="random", seat=1) is None  # never sat


def test_rates_average_over_games():
    board = load_board("usa")
    summary = Summary(board)
    first = started_game(seed=1)
    route = double_route(first)
    claim(first, route.id)
    summary.add(first, ["a", "b"])
    summary.add(started_game(seed=2), ["a", "b"])  # nothing claimed
    assert summary.value(route.id, "claim_rate") == 0.5
    assert summary.value(route.sibling, "contested") == 0.5


def test_four_players_close_nothing():
    game = started_game(num_players=4, seed=1)
    route = double_route(game)
    claim(game, route.id)
    summary = Summary(game.board)
    summary.add(game, ["a", "b", "c", "d"])
    assert summary.value(route.sibling, "contested") == 0.0  # still open to others (§6)


def test_missing_agent_names_are_unknown():
    game = started_game()
    summary = Summary(game.board)
    summary.add(game, [])
    assert summary.agents == ["unknown"]


def test_unknown_stat_is_rejected():
    summary = Summary(load_board("usa"))
    with pytest.raises(AnalysisError):
        summary.value(0, "nonsense")


def test_game_on_another_board_is_rejected():
    summary = Summary(load_board("toy"))
    with pytest.raises(AnalysisError):
        summary.add(started_game(), ["a", "b"])


# --------------------------------------------------------------- record folders


def test_folder_matches_the_replayed_games(records):
    summary = analyze(records)
    games = [GameRecord.load(p).replay() for p in record_paths(records)]
    assert summary.games == len(games) == 8
    assert summary.games_by_agent == {"greedy": 8, "random": 8}
    owned = sum(len(g.route_owner) for g in games)
    assert sum(len(s.claims) for s in summary.routes.values()) == owned
    for rid in summary.routes:
        rate = summary.value(rid, "claim_rate")
        claimed = sum(rid in g.route_owner for g in games)
        assert rate == claimed / 8
        # Distinct agents split the claims between them; so do the seats.
        by_agent = sum(summary.value(rid, "claim_rate", a) for a in summary.agents)
        assert by_agent == pytest.approx(rate)
        by_seat = sum(summary.value(rid, "claim_rate", seat=s) for s in summary.seats)
        assert by_seat == pytest.approx(rate)


def test_player_results_match_the_replayed_games(records):
    summary = analyze(records)
    games = [GameRecord.load(p) for p in record_paths(records)]
    finals = [r.replay() for r in games]
    assert len(summary.seat_games) == 2 * len(games)

    seat0 = summary.player(seat=0)
    assert seat0.games == len(games)
    assert seat0.total == pytest.approx(sum(g.result.players[0].total for g in finals) / len(finals))
    assert seat0.claims == pytest.approx(sum(len(g.players[0].routes) for g in finals) / len(finals))
    done = sum(g.result.players[0].tickets_completed for g in finals)
    failed = sum(g.result.players[0].tickets_failed for g in finals)
    assert seat0.completion_rate == pytest.approx(done / (done + failed))
    # Wins over all seats add up to one per game (shared wins are split).
    wins = sum(summary.player(seat=s).win_rate * summary.player(seat=s).games for s in summary.seats)
    assert wins == pytest.approx(len(games))
    greedy = summary.player(agent="greedy")
    assert greedy.games == len(games) and greedy.agents == {"greedy": len(games)}
    assert summary.player(agent="nobody") is None


def test_player_rows(records):
    rows = analyze(records).player_rows()
    assert len(rows) == 16
    assert {"agent", "seat", "won", "total", "claims", "trains"} <= set(rows[0])


def test_unfinished_games_count_claims_but_not_results():
    game = started_game(seed=1)
    claim(game, single_route(game).id)
    summary = Summary(game.board)
    summary.add(game, ["a", "b"])
    assert summary.seat_games == [] and summary.player() is None
    assert summary.value(single_route(game).id, "claim_rate") == 1.0


def test_single_record_file(records):
    path = record_paths(records)[0]
    assert analyze(path).games == 1


def test_rows_cover_every_route_and_stat(records):
    summary = analyze(records)
    rows = summary.rows()
    assert len(rows) == len(summary.board.routes)
    for stat in STATS:
        assert stat in rows[0]
        for agent in summary.agents:
            assert f"{stat}[{agent}]" in rows[0]
        for seat in summary.seats:
            assert f"{stat}[P{seat}]" in rows[0]


def test_rows_load_into_pandas(records):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(analyze(records).rows())
    assert len(df) == 100 and df["claim_rate"].between(0, 1).all()


def test_empty_or_missing_folder(tmp_path):
    with pytest.raises(AnalysisError):
        analyze(tmp_path)
    with pytest.raises(AnalysisError):
        analyze(tmp_path / "nowhere")
    with pytest.raises(AnalysisError):
        summarize([])


def test_mixed_boards_are_rejected(records, tmp_path):
    usa = GameRecord.load(record_paths(records)[0])
    run_matches(["random", "random"], 1, load_board("toy"), seed=0, record_dir=tmp_path, board_ref="toy")
    toy = GameRecord.load(record_paths(tmp_path)[0])
    with pytest.raises(AnalysisError):
        summarize([usa, toy])
