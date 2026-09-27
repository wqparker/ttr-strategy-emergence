"""The observation: one flat float32 vector from one seat's view (PLAN.md
"Observation", Phase 4).

    encoder = ObservationEncoder(num_players=2, memory_level=2)
    obs = encoder.encode(game, viewer)     # numpy float32, encoder.size entries
    encoder.layout["route_owner"]           # the slice each block occupies

"Me" (the viewer) comes first and opponents follow in seat order after me, so
one network can play any seat. Only what the viewer may see is read: their own
hand, tickets and ticket offer, the table, and card memory from public events
(ttr.memory). Counts are scaled to about [0, 1]; a few can pass 1 in unusual
games (a hand over HAND_SCALE cards, more than 20 tickets).

| Block          | Encoding                                                     | Size (2p) |
| -------------- | ------------------------------------------------------------ | --------- |
| phase          | one-hot over the 5 sub-steps (all 0 once the game is over)   | 5         |
| route_owner    | per route slot: unclaimed / me / each opponent (N+1 one-hot) | 300       |
| route_open     | per route: open to me (§6 double-route rule) and trains left | 100       |
| route_paying   | one-hot of the route being paid for, payment step only       | 100       |
| market         | face-up counts per color, /5                                 | 9         |
| piles          | train deck /110, discard /110, ticket deck /board tickets    | 3         |
| players        | per player: trains left, route points, hand size, tickets    | 8         |
| endgame        | final round started; I still have a turn in it               | 2         |
| memory         | per opponent 9 known + 1 unknown, then the unseen pool (9)   | 19        |
| hand           | my cards per color, / that color's deck count                | 9         |
| tickets_held   | multi-hot by ticket ID                                       | 30        |
| tickets_done   | per held ticket: already connected by my routes              | 30        |
| tickets_trains | per held ticket: fewest trains still needed; then impossible | 60        |
| tickets_offer  | 3 offer slots x one-hot ticket ID (KeepTickets bit order)    | 90        |

Total 765 for 2 players; 100(N+1) + 4N + 10(N-1) + 447 in general.

Boards smaller than 100 routes / 30 tickets leave the extra slots at 0 (a
padded route slot is not "unclaimed"). The memory block is all 0 at level 0;
level 1 fills known and unknown cards, level 2 adds the unseen pool.

Trains to finish: the fewest trains needed to connect the ticket's cities,
counting my routes as free and using only routes still open to me, scaled by
trains per player; 1 if there is no such path. Impossible = no path, or more
trains needed than I have left. It depends only on the claimed routes, so it
is cached until the next claim. The cache is keyed on the game object and its
number of claims: code that edits a state by hand should call `reset()`.

For the player to act, both endgame flags are equal (every turn in the final
round is that player's last); they differ only for a waiting seat.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ttr.board import Board
from ttr.cards import ALL_COLORS
from ttr.env.actions import MAX_ROUTES, OFFER
from ttr.game import MARKET_SIZE, Game, Phase
from ttr.memory import FULL_DECK, CardMemory

MAX_TICKETS = 30
PHASES = (
    Phase.CHOOSE_INITIAL_TICKETS,
    Phase.CHOOSE_ACTION,
    Phase.DRAW_SECOND_CARD,
    Phase.CHOOSE_PAYMENT,
    Phase.KEEP_TICKETS,
)
DECK_SIZE = sum(FULL_DECK.values())  # 110
SCORE_SCALE = 100  # route points; 45 trains of 6-routes is about 112
HAND_SCALE = 50
TICKETS_HELD_SCALE = 20  # random play reaches 19; real games hold far fewer

_UNREACHABLE = 1 << 30
_PHASE_INDEX = {p: i for i, p in enumerate(PHASES)}
_COLOR_SCALE = np.array([FULL_DECK[c] for c in ALL_COLORS], dtype=np.float32)


class ObservationError(ValueError):
    pass


def layout(num_players: int) -> Dict[str, slice]:
    """The slice of the vector each block occupies, in order."""
    n = num_players
    sizes = [
        ("phase", len(PHASES)),
        ("route_owner", MAX_ROUTES * (n + 1)),
        ("route_open", MAX_ROUTES),
        ("route_paying", MAX_ROUTES),
        ("market", len(ALL_COLORS)),
        ("piles", 3),
        ("players", 4 * n),
        ("endgame", 2),
        ("memory", (len(ALL_COLORS) + 1) * (n - 1) + len(ALL_COLORS)),
        ("hand", len(ALL_COLORS)),
        ("tickets_held", MAX_TICKETS),
        ("tickets_done", MAX_TICKETS),
        ("tickets_trains", 2 * MAX_TICKETS),
        ("tickets_offer", OFFER * MAX_TICKETS),
    ]
    result: Dict[str, slice] = {}
    start = 0
    for name, size in sizes:
        result[name] = slice(start, start + size)
        start += size
    return result


@dataclass
class _ViewerCache:
    """What depends only on the claimed routes, for one viewer."""

    adjacency: List[List[Tuple[int, int]]]  # city -> (neighbor, trains): mine = 0, open = length
    open_to_me: np.ndarray  # route_open block
    distances: Dict[int, List[int]] = field(default_factory=dict)  # source -> city -> trains


class ObservationEncoder:
    """Builds observations for one game at a time. It keeps a card-memory
    tracker per viewer and the per-claim cache, and starts over whenever it is
    given a different game object."""

    def __init__(self, num_players: int, memory_level: int = 2) -> None:
        if not 2 <= num_players <= 5:
            raise ObservationError("Ticket to Ride supports 2-5 players")
        if memory_level not in (0, 1, 2):
            raise ObservationError("memory level must be 0, 1 or 2")
        self.num_players = num_players
        self.memory_level = memory_level
        self.layout = layout(num_players)
        self.size = self.layout["tickets_offer"].stop
        self._board: Optional[Board] = None
        self._city_index: Dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        """Forget every cache (new game, or a state edited by hand)."""
        self._game: Optional[Game] = None
        self._claims = -1
        self._memory: Dict[int, CardMemory] = {}
        self._viewers: Dict[int, _ViewerCache] = {}

    # -------------------------------------------------------------- caches

    def _sync(self, game: Game) -> None:
        if game.num_players != self.num_players:
            raise ObservationError(f"encoder is for {self.num_players} players, game has {game.num_players}")
        if game.board is not self._board:
            board = game.board
            if len(board.routes) > MAX_ROUTES or len(board.tickets) > MAX_TICKETS:
                raise ObservationError(f"board {board.name!r} exceeds {MAX_ROUTES} routes / {MAX_TICKETS} tickets")
            self._board = board
            self._city_index = {c: i for i, c in enumerate(board.cities)}
        if game is not self._game:
            self.reset()
            self._game = game
        if len(game.route_owner) != self._claims:
            self._claims = len(game.route_owner)
            self._viewers = {}

    def _viewer_cache(self, game: Game, viewer: int) -> _ViewerCache:
        cache = self._viewers.get(viewer)
        if cache is None:
            trains = game.players[viewer].trains
            index = self._city_index
            adjacency: List[List[Tuple[int, int]]] = [[] for _ in index]
            open_to_me = np.zeros(MAX_ROUTES, dtype=np.float32)
            for r in game.board.routes:
                if game.route_owner.get(r.id) == viewer:
                    w = 0
                elif game.route_open_to(viewer, r):
                    w = r.length
                    if trains >= w:
                        open_to_me[r.id] = 1.0
                else:
                    continue
                a, b = index[r.a], index[r.b]
                adjacency[a].append((b, w))
                adjacency[b].append((a, w))
            cache = self._viewers[viewer] = _ViewerCache(adjacency, open_to_me)
        return cache

    def _distance(self, cache: _ViewerCache, a: str, b: str) -> Optional[int]:
        """Fewest trains from city a to city b over my routes (free) and routes
        open to me, by Dijkstra from a; every distance from a is cached."""
        source = self._city_index[a]
        dist = cache.distances.get(source)
        if dist is None:
            adjacency = cache.adjacency
            dist = [_UNREACHABLE] * len(adjacency)
            dist[source] = 0
            heap = [(0, source)]
            pop, push = heapq.heappop, heapq.heappush
            while heap:
                d, city = pop(heap)
                if d > dist[city]:
                    continue
                for nxt, w in adjacency[city]:
                    nd = d + w
                    if nd < dist[nxt]:
                        dist[nxt] = nd
                        push(heap, (nd, nxt))
            cache.distances[source] = dist
        d = dist[self._city_index[b]]
        return None if d == _UNREACHABLE else d

    def trains_to_finish(self, game: Game, viewer: int, ticket_id: int) -> Optional[int]:
        """Fewest trains `viewer` still needs to connect the ticket's cities;
        None if no path through routes open to them."""
        self._sync(game)
        ticket = game.board.tickets[ticket_id]
        return self._distance(self._viewer_cache(game, viewer), ticket.a, ticket.b)

    # -------------------------------------------------------------- encode

    def encode(self, game: Game, viewer: int) -> np.ndarray:
        self._sync(game)
        L = self.layout
        n = self.num_players
        board = game.board
        me = game.players[viewer]
        seats = [(viewer + i) % n for i in range(n)]  # me first, then seat order
        slot = {p: i for i, p in enumerate(seats)}
        obs = np.zeros(self.size, dtype=np.float32)

        phase = _PHASE_INDEX.get(game.phase)
        if phase is not None:
            obs[L["phase"].start + phase] = 1.0

        owner = obs[L["route_owner"]].reshape(MAX_ROUTES, n + 1)
        owner[: len(board.routes), 0] = 1.0
        for rid, p in game.route_owner.items():
            owner[rid, 0] = 0.0
            owner[rid, 1 + slot[p]] = 1.0

        cache = self._viewer_cache(game, viewer)
        obs[L["route_open"]] = cache.open_to_me
        if game.phase is Phase.CHOOSE_PAYMENT and game.pending_route is not None:
            obs[L["route_paying"].start + game.pending_route] = 1.0

        market = Counter(game.market)
        obs[L["market"]] = [market[c] / MARKET_SIZE for c in ALL_COLORS]
        obs[L["piles"]] = [
            len(game.deck) / DECK_SIZE,
            len(game.discard) / DECK_SIZE,
            len(game.ticket_deck) / len(board.tickets),
        ]

        players = obs[L["players"]].reshape(n, 4)
        for i, p in enumerate(seats):
            ps = game.players[p]
            players[i] = (
                ps.trains / board.trains_per_player,
                ps.route_points / SCORE_SCALE,
                ps.hand_size / HAND_SCALE,
                len(ps.tickets) / TICKETS_HELD_SCALE,
            )

        remaining = game.final_turns_remaining
        if remaining is not None and not game.game_over:
            obs[L["endgame"].start] = 1.0
            if (viewer - game.current_player) % n < remaining:
                obs[L["endgame"].start + 1] = 1.0

        if self.memory_level:
            self._encode_memory(game, viewer, seats, obs[L["memory"]])

        obs[L["hand"]] = np.array([me.hand[c] for c in ALL_COLORS], dtype=np.float32) / _COLOR_SCALE

        held = obs[L["tickets_held"]]
        done = obs[L["tickets_done"]]
        trains = obs[L["tickets_trains"]]
        for tid in me.tickets:
            held[tid] = 1.0
            ticket = board.tickets[tid]
            need = self._distance(cache, ticket.a, ticket.b)
            if need == 0:
                done[tid] = 1.0
            elif need is None:
                trains[tid] = 1.0
                trains[MAX_TICKETS + tid] = 1.0
            else:
                trains[tid] = min(need, board.trains_per_player) / board.trains_per_player
                if need > me.trains:
                    trains[MAX_TICKETS + tid] = 1.0

        offer = obs[L["tickets_offer"]].reshape(OFFER, MAX_TICKETS)
        for i, tid in enumerate(me.pending_tickets):
            offer[i, tid] = 1.0
        return obs

    def _encode_memory(self, game: Game, viewer: int, seats: List[int], out: np.ndarray) -> None:
        memory = self._memory.get(viewer)
        if memory is None:
            memory = self._memory[viewer] = CardMemory(self.num_players, viewer, self.memory_level)
        view = memory.view(game)
        width = len(ALL_COLORS) + 1
        for i, p in enumerate(seats[1:]):
            known = view.known[p]
            block = out[i * width:(i + 1) * width]
            block[:-1] = np.array([known[c] for c in ALL_COLORS], dtype=np.float32) / _COLOR_SCALE
            block[-1] = view.unknown[p] / HAND_SCALE
        if view.unseen is not None:
            unseen = view.unseen
            out[-len(ALL_COLORS):] = np.array([unseen[c] for c in ALL_COLORS], dtype=np.float32) / _COLOR_SCALE
