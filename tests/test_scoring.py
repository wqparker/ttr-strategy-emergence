from ttr.board import Route
from ttr.scoring import connected, longest_path


def r(i, a, b, length):
    return Route(id=i, a=a, b=b, length=length, color=None, sibling=None)


def test_longest_path_empty():
    assert longest_path([]) == 0


def test_longest_path_counts_train_spaces_not_routes():
    assert longest_path([r(0, "A", "B", 6), r(1, "C", "D", 1), r(2, "D", "E", 1)]) == 6


def test_longest_path_chain():
    assert longest_path([r(0, "A", "B", 2), r(1, "B", "C", 3), r(2, "C", "D", 1)]) == 6


def test_longest_path_branch_takes_two_longest_arms():
    # Y shape centred on B: arms of 5, 4, 1 -> best trail uses the 5 and 4 arms.
    routes = [r(0, "B", "A", 5), r(1, "B", "C", 4), r(2, "B", "D", 1)]
    assert longest_path(routes) == 9


def test_longest_path_loop_can_revisit_city():
    # Triangle A-B-C plus tail C-D: a trail can go around the loop and exit.
    routes = [r(0, "A", "B", 2), r(1, "B", "C", 2), r(2, "C", "A", 2), r(3, "C", "D", 3)]
    assert longest_path(routes) == 9


def test_longest_path_figure_eight_passes_city_twice():
    # Two loops sharing city X; a trail can use all routes (X visited twice).
    routes = [
        r(0, "X", "A", 1), r(1, "A", "B", 1), r(2, "B", "X", 1),
        r(3, "X", "C", 1), r(4, "C", "D", 1), r(5, "D", "X", 1),
    ]
    assert longest_path(routes) == 6


def test_longest_path_never_reuses_a_route():
    # Star with 4 arms from X: only two arms can be in one trail.
    routes = [r(i, "X", c, 3) for i, c in enumerate("ABCD")]
    assert longest_path(routes) == 6


def test_connected():
    routes = [r(0, "A", "B", 1), r(1, "B", "C", 1), r(2, "D", "E", 1)]
    assert connected(routes, "A", "C")
    assert not connected(routes, "A", "E")
    assert not connected(routes, "A", "Z")
