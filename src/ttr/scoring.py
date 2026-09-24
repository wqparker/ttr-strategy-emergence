"""Scoring helpers: ticket completion and longest continuous path (RULES.md §8)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

from ttr.board import Route, Ticket

LONGEST_PATH_BONUS = 10


def _adjacency(routes: Iterable[Route]) -> Dict[str, List[Route]]:
    adj: Dict[str, List[Route]] = defaultdict(list)
    for r in routes:
        adj[r.a].append(r)
        adj[r.b].append(r)
    return adj


def connected(routes: Iterable[Route], a: str, b: str) -> bool:
    """True if the given routes form a continuous path between cities a and b."""
    adj = _adjacency(routes)
    if a not in adj or b not in adj:
        return False
    seen = {a}
    stack = [a]
    while stack:
        city = stack.pop()
        if city == b:
            return True
        for r in adj[city]:
            nxt = r.b if r.a == city else r.a
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def ticket_results(routes: List[Route], tickets: Iterable[Ticket]) -> List[Tuple[Ticket, bool]]:
    return [(t, connected(routes, t.a, t.b)) for t in tickets]


def longest_path(routes: Iterable[Route]) -> int:
    """Length in train spaces of the longest trail through the routes.

    A trail may revisit cities (loops allowed) but never reuses a route.
    Exhaustive DFS from every city; fine for a single player's <=45 trains.
    """
    routes = list(routes)
    if not routes:
        return 0
    adj = _adjacency(routes)
    best = 0

    def dfs(city: str, used: Set[int], length: int) -> None:
        nonlocal best
        if length > best:
            best = length
        for r in adj[city]:
            if r.id in used:
                continue
            used.add(r.id)
            dfs(r.b if r.a == city else r.a, used, length + r.length)
            used.remove(r.id)

    for city in adj:
        dfs(city, set(), 0)
    return best
