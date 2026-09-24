"""Board data: cities, routes, destination tickets. Loaded from JSON in ttr/data/.

A double route is simply two routes joining the same pair of cities; pairing is
derived from the data rather than stored, so the data file can't get it wrong.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ttr.cards import Color

# RULES.md §5
ROUTE_POINTS: Dict[int, int] = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}


@dataclass(frozen=True)
class Route:
    id: int
    a: str
    b: str
    length: int
    color: Optional[Color]  # None = gray (any one color)
    sibling: Optional[int]  # id of the other half of a double route, if any

    @property
    def points(self) -> int:
        return ROUTE_POINTS[self.length]

    @property
    def is_gray(self) -> bool:
        return self.color is None


@dataclass(frozen=True)
class Ticket:
    id: int
    a: str
    b: str
    points: int


@dataclass(frozen=True)
class Board:
    name: str
    verified: bool
    trains_per_player: int
    cities: Tuple[str, ...]
    routes: Tuple[Route, ...]
    tickets: Tuple[Ticket, ...]

    def routes_between(self, a: str, b: str) -> List[Route]:
        return [r for r in self.routes if {r.a, r.b} == {a, b}]


class BoardError(ValueError):
    pass


def _parse_color(raw: str) -> Optional[Color]:
    if raw == "gray":
        return None
    try:
        color = Color(raw)
    except ValueError:
        raise BoardError(f"unknown route color {raw!r}") from None
    if color is Color.LOCOMOTIVE:
        raise BoardError("routes can't be locomotive-colored")
    return color


def board_from_dict(data: dict) -> Board:
    cities = tuple(data["cities"])
    city_set = set(cities)
    if len(city_set) != len(cities):
        raise BoardError("duplicate city names")

    raw_routes = data["routes"]
    by_pair: Dict[frozenset, List[int]] = defaultdict(list)
    for i, r in enumerate(raw_routes):
        for end in (r["a"], r["b"]):
            if end not in city_set:
                raise BoardError(f"route {i} references unknown city {end!r}")
        if r["a"] == r["b"]:
            raise BoardError(f"route {i} is a loop")
        if not 1 <= r["length"] <= 6:
            raise BoardError(f"route {i} has invalid length {r['length']}")
        by_pair[frozenset((r["a"], r["b"]))].append(i)

    sibling: Dict[int, Optional[int]] = {}
    for pair, ids in by_pair.items():
        if len(ids) > 2:
            raise BoardError(f"more than two routes between {sorted(pair)}")
        if len(ids) == 2:
            first, second = ids
            if raw_routes[first]["length"] != raw_routes[second]["length"]:
                raise BoardError(f"double route {sorted(pair)} has mismatched lengths")
            sibling[first], sibling[second] = second, first

    routes = tuple(
        Route(
            id=i,
            a=r["a"],
            b=r["b"],
            length=r["length"],
            color=_parse_color(r["color"]),
            sibling=sibling.get(i),
        )
        for i, r in enumerate(raw_routes)
    )

    tickets = []
    for i, t in enumerate(data["tickets"]):
        for end in (t["a"], t["b"]):
            if end not in city_set:
                raise BoardError(f"ticket {i} references unknown city {end!r}")
        tickets.append(Ticket(id=i, a=t["a"], b=t["b"], points=t["points"]))

    board = Board(
        name=data["name"],
        verified=bool(data.get("verified", False)),
        trains_per_player=data.get("trains_per_player", 45),
        cities=cities,
        routes=routes,
        tickets=tuple(tickets),
    )
    _check_connected(board)
    return board


def _check_connected(board: Board) -> None:
    adj: Dict[str, List[str]] = defaultdict(list)
    for r in board.routes:
        adj[r.a].append(r.b)
        adj[r.b].append(r.a)
    seen = {board.cities[0]}
    stack = [board.cities[0]]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    missing = set(board.cities) - seen
    if missing:
        raise BoardError(f"board is not connected; unreachable: {sorted(missing)}")


def load_board(name_or_path: Union[str, Path] = "usa") -> Board:
    """Load a bundled board by name ("usa", "toy") or a JSON file by path."""
    path = Path(name_or_path)
    if path.suffix == ".json":
        text = path.read_text(encoding="utf-8")
    else:
        text = resources.files("ttr").joinpath("data", f"{name_or_path}.json").read_text(
            encoding="utf-8"
        )
    return board_from_dict(json.loads(text))
