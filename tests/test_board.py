import pytest

from ttr.board import BoardError, board_from_dict, load_board
from ttr.cards import Color, standard_deck


def test_standard_deck():
    deck = standard_deck()
    assert len(deck) == 110
    assert deck.count(Color.LOCOMOTIVE) == 14
    for color in Color:
        if color is not Color.LOCOMOTIVE:
            assert deck.count(color) == 12


def test_usa_board_shape():
    board = load_board("usa")
    assert len(board.cities) == 36
    assert len(board.routes) == 100
    assert len(board.tickets) == 30
    assert board.trains_per_player == 45
    doubles = [r for r in board.routes if r.sibling is not None]
    assert len(doubles) == 44  # 22 double routes
    for r in doubles:
        sib = board.routes[r.sibling]
        assert sib.sibling == r.id and {sib.a, sib.b} == {r.a, r.b}


def test_usa_every_city_is_used():
    board = load_board("usa")
    used = {c for r in board.routes for c in (r.a, r.b)}
    assert used == set(board.cities)


def test_toy_board_loads():
    board = load_board("toy")
    assert board.verified
    assert any(r.sibling is not None and r.is_gray for r in board.routes)
    assert any(r.sibling is not None and not r.is_gray for r in board.routes)


def _minimal(**overrides):
    data = {
        "name": "t",
        "cities": ["A", "B"],
        "routes": [{"a": "A", "b": "B", "length": 2, "color": "red"}],
        "tickets": [],
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    "overrides",
    [
        {"routes": [{"a": "A", "b": "Z", "length": 2, "color": "red"}]},
        {"routes": [{"a": "A", "b": "B", "length": 7, "color": "red"}]},
        {"routes": [{"a": "A", "b": "B", "length": 2, "color": "pink"}]},
        {"routes": [{"a": "A", "b": "B", "length": 2, "color": "red"}] * 3},
        {"cities": ["A", "B", "C"]},  # C unreachable
        {"tickets": [{"a": "A", "b": "Q", "points": 3}]},
    ],
)
def test_invalid_boards_rejected(overrides):
    with pytest.raises(BoardError):
        board_from_dict(_minimal(**overrides))


def test_layout_loaded_for_bundled_boards():
    for name in ("usa", "toy"):
        board = load_board(name)
        assert set(board.layout) == set(board.cities)
    codes = [p.code for p in load_board("usa").layout.values()]
    assert len(set(codes)) == len(codes)


def test_layout_must_cover_every_city():
    with pytest.raises(BoardError):
        board_from_dict(_minimal(layout={"A": {"x": 0, "y": 0}}))
