"""Game engine: state, legal-move generation, rule enforcement, final scoring.

Rules follow docs/RULES.md; "§9 #N" comments point at this project's rulings for
cases the official rules don't cover. The engine is pure Python with no RL
dependencies. Environments and agents drive it through `legal_actions()` and
`step()`.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import combinations
from typing import Any, Dict, List, Optional

from ttr.actions import (
    Action,
    ClaimRoute,
    DrawBlind,
    DrawFaceUp,
    DrawTickets,
    KeepTickets,
    Pass,
    Pay,
)
from ttr.board import Board, Route, load_board
from ttr.cards import TRAIN_COLORS, Color, standard_deck
from ttr.scoring import LONGEST_PATH_BONUS, longest_path, ticket_results

MARKET_SIZE = 5
STARTING_HAND = 4
TICKETS_DEALT = 3
INITIAL_MIN_KEEP = 2
LATER_MIN_KEEP = 1
END_TRIGGER_TRAINS = 2


class Phase(str, Enum):
    CHOOSE_INITIAL_TICKETS = "choose_initial_tickets"
    CHOOSE_ACTION = "choose_action"
    DRAW_SECOND_CARD = "draw_second_card"
    CHOOSE_PAYMENT = "choose_payment"
    KEEP_TICKETS = "keep_tickets"
    GAME_OVER = "game_over"


class IllegalAction(ValueError):
    pass


@dataclass
class PlayerState:
    trains: int
    hand: Counter = field(default_factory=Counter)  # Color -> count
    tickets: List[int] = field(default_factory=list)  # kept ticket ids
    routes: List[int] = field(default_factory=list)  # claimed route ids
    route_points: int = 0  # scored immediately on claiming (§5)
    pending_tickets: List[int] = field(default_factory=list)  # drawn, not yet kept/returned

    @property
    def hand_size(self) -> int:
        """Cards held. Public information (RULES.md §9 #13), unlike the cards."""
        return sum(self.hand.values())


@dataclass
class Event:
    """One entry in the game log. `public` is what every player sees; `private`
    is only visible to `player` (blind-drawn card, ticket identities)."""

    turn: int
    player: Optional[int]
    kind: str
    public: Dict[str, Any] = field(default_factory=dict)
    private: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayerResult:
    route_points: int
    ticket_points: int  # net: completed minus failed
    tickets_completed: int
    tickets_failed: int
    longest_path: int
    longest_path_bonus: bool
    total: int


@dataclass
class GameResult:
    players: List[PlayerResult]
    winners: List[int]  # more than one = shared win (§9 #12)
    truncated: bool  # ended by max_turns rather than by the rules


class Game:
    def __init__(
        self,
        board: Optional[Board] = None,
        num_players: int = 2,
        seed: Optional[int] = None,
        first_player: Optional[int] = None,
        max_turns: Optional[int] = None,
    ) -> None:
        if not 2 <= num_players <= 5:
            raise ValueError("Ticket to Ride supports 2-5 players")
        self.board = board if board is not None else load_board("usa")
        self.num_players = num_players
        self.rng = random.Random(seed)
        self.max_turns = max_turns

        self.players = [PlayerState(trains=self.board.trains_per_player) for _ in range(num_players)]
        self.route_owner: Dict[int, int] = {}
        self.deck: List[Color] = standard_deck()  # top of deck = end of list
        self.discard: List[Color] = []
        self.market: List[Color] = []
        self.ticket_deck: List[int] = [t.id for t in self.board.tickets]  # top = index 0
        self.log: List[Event] = []

        self.turn = 0  # completed main turns
        self.market_resets = 0
        self.final_turns_remaining: Optional[int] = None
        self.pending_route: Optional[int] = None
        self.consecutive_passes = 0
        self.result: Optional[GameResult] = None

        # §9 #8: random first seat unless the caller fixes it (evaluation rotates it).
        self.first_player = (
            first_player if first_player is not None else self.rng.randrange(num_players)
        )
        self._setup()

    # ------------------------------------------------------------------ setup

    def _setup(self) -> None:
        self.rng.shuffle(self.deck)
        self.rng.shuffle(self.ticket_deck)
        for p in self.players:
            for _ in range(STARTING_HAND):
                p.hand[self._draw_from_deck()] += 1
        self._refill_market()
        # §9 #7: everyone is dealt from the same deck up front, then chooses in turn.
        for p in self.players:
            p.pending_tickets = self._draw_tickets(TICKETS_DEALT)
        self._initial_returns: List[int] = []
        self.current_player = self.first_player
        self._initial_choices_left = self.num_players
        self.phase = Phase.CHOOSE_INITIAL_TICKETS

    # ------------------------------------------------------------ card piles

    def _draw_from_deck(self) -> Optional[Color]:
        """Top card of the deck, reshuffling the discard pile if needed (§4)."""
        if not self.deck:
            if not self.discard:
                return None
            self.deck, self.discard = self.discard, []
            self.rng.shuffle(self.deck)
            self._log(None, "reshuffle", size=len(self.deck))
        return self.deck.pop()

    def _cards_available(self) -> bool:
        return bool(self.deck or self.discard)

    def _refill_market(self, at: Optional[int] = None) -> None:
        """Top the market back up to five. `at` is the slot a card was just taken
        from: its replacement goes back into that slot, as at a real table, so the
        row does not shift under the player who is looking at it. If the deck is
        spent the row closes up instead (§9 #3: the market may hold fewer than 5)."""
        if at is not None and len(self.market) < MARKET_SIZE:
            card = self._draw_from_deck()
            if card is not None:
                self.market.insert(at, card)
        while len(self.market) < MARKET_SIZE:
            card = self._draw_from_deck()
            if card is None:
                break
            self.market.append(card)
        self._check_market_reset()

    def _check_market_reset(self) -> None:
        """§4 / §9 #2: while 3+ Locomotives are face up, discard all and redeal."""
        while self.market.count(Color.LOCOMOTIVE) >= 3 and self._legal_market_possible():
            self.market_resets += 1
            self._log(None, "market_reset", discarded=[c.value for c in self.market])
            self.discard.extend(self.market)
            self.market = []
            while len(self.market) < MARKET_SIZE:
                card = self._draw_from_deck()
                if card is None:
                    break
                self.market.append(card)

    def _legal_market_possible(self) -> bool:
        """Can the cards outside players' hands form a market with <3 Locomotives?
        If not, redealing could never stop (§9 #2 hard guard)."""
        pool = self.deck + self.discard + self.market
        size = min(MARKET_SIZE, len(pool))
        non_locos = sum(1 for c in pool if c is not Color.LOCOMOTIVE)
        return non_locos >= size - 2

    def _draw_tickets(self, n: int) -> List[int]:
        drawn = self.ticket_deck[:n]
        del self.ticket_deck[:n]
        return drawn

    def _return_tickets(self, ticket_ids: List[int]) -> None:
        """Returned tickets go to the bottom in random order (§3C, §9 #18)."""
        ids = list(ticket_ids)
        self.rng.shuffle(ids)
        self.ticket_deck.extend(ids)

    # ---------------------------------------------------------- legal moves

    def legal_actions(self) -> List[Action]:
        if self.phase is Phase.GAME_OVER:
            return []
        player = self.players[self.current_player]

        if self.phase in (Phase.CHOOSE_INITIAL_TICKETS, Phase.KEEP_TICKETS):
            pending = player.pending_tickets
            base_min = INITIAL_MIN_KEEP if self.phase is Phase.CHOOSE_INITIAL_TICKETS else LATER_MIN_KEEP
            min_keep = min(base_min, len(pending))  # §9 #6
            return [
                KeepTickets(frozenset(combo))
                for k in range(min_keep, len(pending) + 1)
                for combo in combinations(pending, k)
            ]

        if self.phase is Phase.CHOOSE_PAYMENT:
            assert self.pending_route is not None
            return list(self._payments(self.current_player, self.board.routes[self.pending_route]))

        if self.phase is Phase.DRAW_SECOND_CARD:
            return self._card_draws(second=True)

        # CHOOSE_ACTION
        actions: List[Action] = self._card_draws(second=False)
        actions.extend(ClaimRoute(r.id) for r in self.board.routes if self._can_claim(self.current_player, r))
        if self.ticket_deck:  # §9 #5
            actions.append(DrawTickets())
        if not actions:
            actions.append(Pass())  # §9 #11
        return actions

    def _card_draws(self, second: bool) -> List[Action]:
        actions: List[Action] = []
        for color in dict.fromkeys(self.market):  # distinct, stable order
            if second and color is Color.LOCOMOTIVE:
                continue  # §3A / §9 #1
            actions.append(DrawFaceUp(color))
        if self._cards_available():
            actions.append(DrawBlind())
        return actions

    def route_open_to(self, p: int, route: Route) -> bool:
        """Unclaimed and not closed to player p by the double-route rules (§6).
        Ignores cards and trains; public information only."""
        if route.id in self.route_owner:
            return False
        if route.sibling is not None and route.sibling in self.route_owner:
            if self.num_players <= 3:  # §6: only one of a double route in 2-3p
                return False
            if self.route_owner[route.sibling] == p:  # §6: never both halves
                return False
        return True

    def _can_claim(self, p: int, route: Route) -> bool:
        if not self.route_open_to(p, route):
            return False
        if self.players[p].trains < route.length:  # §9 #10
            return False
        return next(iter(self._payments(p, route)), None) is not None

    def _payments(self, p: int, route: Route):
        """Yield every distinct legal Pay for this route from player p's hand."""
        hand = self.players[p].hand
        locos = hand[Color.LOCOMOTIVE]
        colors = TRAIN_COLORS if route.is_gray else [route.color]
        for color in colors:
            for k in range(0, min(locos, route.length - 1) + 1):
                if hand[color] >= route.length - k:
                    yield Pay(color, k)
        if locos >= route.length:  # §9 #9
            yield Pay(None, route.length)

    # ----------------------------------------------------------------- step

    def step(self, action: Action) -> None:
        if action not in self.legal_actions():
            raise IllegalAction(f"{action!r} is not legal in phase {self.phase.value}")
        p = self.current_player
        player = self.players[p]

        if isinstance(action, KeepTickets):
            self._keep_tickets(p, action.ticket_ids)
        elif isinstance(action, DrawFaceUp):
            slot = self.market.index(action.color)
            del self.market[slot]
            player.hand[action.color] += 1
            self._log(p, "draw_face_up", color=action.color.value)
            self._refill_market(slot)
            locomotive = action.color is Color.LOCOMOTIVE
            self._after_card_draw(ends_turn=locomotive)  # §3A: face-up Loco is the whole draw
        elif isinstance(action, DrawBlind):
            card = self._draw_from_deck()
            assert card is not None
            player.hand[card] += 1
            self._log(p, "draw_blind", private={"color": card.value})
            self._after_card_draw(ends_turn=False)  # blind Loco counts as a normal card
        elif isinstance(action, ClaimRoute):
            self.pending_route = action.route_id
            self.phase = Phase.CHOOSE_PAYMENT
        elif isinstance(action, Pay):
            self._pay_and_claim(p, action)
        elif isinstance(action, DrawTickets):
            player.pending_tickets = self._draw_tickets(TICKETS_DEALT)
            self._log(
                p,
                "draw_tickets",
                count=len(player.pending_tickets),
                private={"tickets": list(player.pending_tickets)},
            )
            self.phase = Phase.KEEP_TICKETS
        elif isinstance(action, Pass):
            self._log(p, "pass")
            self.consecutive_passes += 1
            if self.consecutive_passes >= self.num_players:
                # §9 #20: nobody can do anything, so the game can never progress.
                self._log(None, "stalemate")
                self.turn += 1
                self._finish(truncated=False)
                return
            self._end_turn()
            return
        self.consecutive_passes = 0

    def _after_card_draw(self, ends_turn: bool) -> None:
        if self.phase is Phase.DRAW_SECOND_CARD or ends_turn:
            self._end_turn()
            return
        self.phase = Phase.DRAW_SECOND_CARD
        if not self._card_draws(second=True):  # §9 #4: nothing left to draw
            self._end_turn()

    def _pay_and_claim(self, p: int, pay: Pay) -> None:
        assert self.pending_route is not None
        route = self.board.routes[self.pending_route]
        player = self.players[p]
        paid: List[Color] = [Color.LOCOMOTIVE] * pay.locomotives
        if pay.color is not None:
            paid += [pay.color] * (route.length - pay.locomotives)
        for card in paid:
            player.hand[card] -= 1
        player.hand += Counter()  # drop zero counts
        self.discard.extend(paid)
        # New discards may let a short market fill up, or a stuck 3-Locomotive
        # market (§9 #2 guard) finally reset.
        self._refill_market()
        self.route_owner[route.id] = p
        player.routes.append(route.id)
        player.trains -= route.length
        player.route_points += route.points
        self._log(p, "claim_route", route=route.id, paid=[c.value for c in paid])
        self.pending_route = None
        self._end_turn()

    def _keep_tickets(self, p: int, keep: frozenset) -> None:
        player = self.players[p]
        returned = [t for t in player.pending_tickets if t not in keep]
        kept = [t for t in player.pending_tickets if t in keep]
        player.tickets.extend(kept)
        player.pending_tickets = []

        if self.phase is Phase.CHOOSE_INITIAL_TICKETS:
            self._log(p, "keep_initial_tickets", count=len(kept), private={"tickets": kept})
            self._initial_returns.extend(returned)
            self._initial_choices_left -= 1
            if self._initial_choices_left:
                self.current_player = (p + 1) % self.num_players
            else:
                # §9 #7: returns go to the bottom only once everyone has chosen.
                self._return_tickets(self._initial_returns)
                self._initial_returns = []
                self.current_player = self.first_player
                self.phase = Phase.CHOOSE_ACTION
            return

        self._log(p, "keep_tickets", count=len(kept), private={"tickets": kept})
        self._return_tickets(returned)
        self._end_turn()

    def _end_turn(self) -> None:
        p = self.current_player
        self.turn += 1
        if self.final_turns_remaining is None:
            if self.players[p].trains <= END_TRIGGER_TRAINS:
                # §7 / §9 #15: every player, including p, gets exactly one more turn.
                self.final_turns_remaining = self.num_players
                self._log(p, "final_round_triggered")
        else:
            self.final_turns_remaining -= 1
            if self.final_turns_remaining == 0:
                self._finish(truncated=False)
                return
        if self.max_turns is not None and self.turn >= self.max_turns:
            self._finish(truncated=True)
            return
        self.current_player = (p + 1) % self.num_players
        self.phase = Phase.CHOOSE_ACTION

    # ------------------------------------------------------------- scoring

    def _finish(self, truncated: bool) -> None:
        results = []
        lengths = []
        for player in self.players:
            routes = [self.board.routes[r] for r in player.routes]
            tickets = [self.board.tickets[t] for t in player.tickets]
            outcomes = ticket_results(routes, tickets)
            completed = sum(1 for _, ok in outcomes if ok)
            net = sum(t.points if ok else -t.points for t, ok in outcomes)
            lengths.append(longest_path(routes))
            results.append(
                PlayerResult(
                    route_points=player.route_points,
                    ticket_points=net,
                    tickets_completed=completed,
                    tickets_failed=len(outcomes) - completed,
                    longest_path=lengths[-1],
                    longest_path_bonus=False,
                    total=player.route_points + net,
                )
            )
        best_len = max(lengths)
        if best_len > 0:  # nobody with zero routes earns the bonus
            for r in results:
                if r.longest_path == best_len:  # §8: all tied players score it
                    r.longest_path_bonus = True
                    r.total += LONGEST_PATH_BONUS

        # §8 winner + tiebreaks, then shared win (§9 #12)
        def key(i: int):
            r = results[i]
            return (r.total, r.tickets_completed, r.longest_path_bonus)

        best = max(key(i) for i in range(self.num_players))
        winners = [i for i in range(self.num_players) if key(i) == best]
        self.result = GameResult(players=results, winners=winners, truncated=truncated)
        self.phase = Phase.GAME_OVER
        self._log(None, "game_over", winners=winners, truncated=truncated)

    # ------------------------------------------------------------- helpers

    @property
    def game_over(self) -> bool:
        return self.phase is Phase.GAME_OVER

    def clone(self) -> "Game":
        """Independent copy of the full state (for replay snapshots and search).
        The board is immutable and shared; log events are never mutated, so the
        log list is copied but its events are shared."""
        g = copy.copy(self)
        g.players = [
            replace(
                p,
                hand=Counter(p.hand),
                tickets=list(p.tickets),
                routes=list(p.routes),
                pending_tickets=list(p.pending_tickets),
            )
            for p in self.players
        ]
        g.route_owner = dict(self.route_owner)
        g.deck = list(self.deck)
        g.discard = list(self.discard)
        g.market = list(self.market)
        g.ticket_deck = list(self.ticket_deck)
        g.log = list(self.log)
        g._initial_returns = list(self._initial_returns)
        g.rng = random.Random()
        g.rng.setstate(self.rng.getstate())
        return g

    def _log(self, player: Optional[int], kind: str, private: Optional[dict] = None, **public: Any) -> None:
        self.log.append(Event(self.turn, player, kind, public, private or {}))
