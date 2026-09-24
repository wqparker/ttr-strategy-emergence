"""`rich` terminal rendering of game state, for debugging (PLAN.md phase 2)."""

from __future__ import annotations

from typing import List, Optional, Sequence

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ttr.ascii_map import render_map
from ttr.cards import TRAIN_COLORS, Color
from ttr.game import Event, Game, GameResult
from ttr.scoring import connected

# Terminal styles for card colors ("purple" cards shown magenta).
STYLE = {
    Color.PURPLE: "magenta",
    Color.BLUE: "blue",
    Color.ORANGE: "dark_orange",
    Color.WHITE: "white",
    Color.GREEN: "green",
    Color.YELLOW: "yellow",
    Color.BLACK: "grey50",
    Color.RED: "red",
    Color.LOCOMOTIVE: "bold cyan",
}
SEAT_STYLE = ["bold blue", "bold red", "bold green", "bold yellow", "bold magenta"]
PANEL_WIDTH = 26  # player panels in the board view
# Card "chips" for the board view: colored background blocks.
CHIP = {
    Color.PURPLE: "bold white on magenta",
    Color.BLUE: "bold white on blue",
    Color.ORANGE: "bold black on dark_orange",
    Color.WHITE: "bold black on white",
    Color.GREEN: "bold black on green",
    Color.YELLOW: "bold black on yellow",
    Color.BLACK: "bold white on grey23",
    Color.RED: "bold white on red",
    Color.LOCOMOTIVE: "bold black on bright_cyan",
}


def card_text(color: Optional[Color], count: int = 1) -> Text:
    if color is None:
        return Text("gray" + (f"×{count}" if count > 1 else ""), style="grey70")
    label = "LOCO" if color is Color.LOCOMOTIVE else color.value
    return Text(label + (f"×{count}" if count > 1 else ""), style=STYLE[color])


def hand_text(hand) -> Text:
    out = Text()
    for color in TRAIN_COLORS + [Color.LOCOMOTIVE]:
        if hand[color]:
            out.append_text(card_text(color, hand[color]))
            out.append(" ")
    return out


def render_game(game: Game, reveal: bool = True) -> Group:
    """Full state. reveal=False hides hands and tickets (public view only)."""
    header = Text(
        f"Turn {game.turn}  ·  phase {game.phase.value}  ·  to act: P{game.current_player}"
        f"  ·  deck {len(game.deck)} / discard {len(game.discard)}"
        f"  ·  tickets left {len(game.ticket_deck)}"
    )
    if game.final_turns_remaining is not None:
        header.append(f"  ·  FINAL ROUND ({game.final_turns_remaining} turns left)", style="bold red")

    market = Text("Market: ")
    for c in game.market:
        market.append_text(card_text(c))
        market.append(" ")

    players = Table(show_header=True, header_style="bold", expand=False)
    for col in ("", "trains", "route pts", "cards", "tickets", "hand"):
        players.add_column(col)
    for i, p in enumerate(game.players):
        players.add_row(
            Text(f"P{i}", style=SEAT_STYLE[i]),
            str(p.trains),
            str(p.route_points),
            str(sum(p.hand.values())),
            str(len(p.tickets)),
            hand_text(p.hand) if reveal else Text("hidden", style="dim"),
        )

    parts = [header, market, players]
    if reveal:
        for i, p in enumerate(game.players):
            if p.tickets:
                names = ", ".join(
                    f"{t.a}–{t.b} ({t.points})" for t in (game.board.tickets[x] for x in p.tickets)
                )
                parts.append(Text(f"P{i} tickets: {names}", style="dim"))
    return Group(*parts)


def render_routes(game: Game) -> Table:
    table = Table(title="Claimed routes", show_header=True, header_style="bold")
    for col in ("owner", "route", "len", "color"):
        table.add_column(col)
    for rid, owner in sorted(game.route_owner.items(), key=lambda kv: kv[1]):
        r = game.board.routes[rid]
        table.add_row(
            Text(f"P{owner}", style=SEAT_STYLE[owner]), f"{r.a}–{r.b}", str(r.length), card_text(r.color)
        )
    return table


def render_result(result: GameResult, names=None) -> Panel:
    table = Table(show_header=True, header_style="bold")
    for col in ("", "routes", "tickets", "done/failed", "longest", "bonus", "total"):
        table.add_column(col)
    for i, r in enumerate(result.players):
        label = f"P{i}" + (f" {names[i]}" if names else "")
        table.add_row(
            Text(label + (" ★" if i in result.winners else ""), style=SEAT_STYLE[i]),
            str(r.route_points),
            f"{r.ticket_points:+d}",
            f"{r.tickets_completed}/{r.tickets_failed}",
            str(r.longest_path),
            "+10" if r.longest_path_bonus else "",
            Text(str(r.total), style="bold"),
        )
    title = "Final scores" + (" (TRUNCATED)" if result.truncated else "")
    return Panel(table, title=title, expand=False)


# ------------------------------------------------------------ event text


def _route_name(game: Game, rid: int) -> str:
    r = game.board.routes[rid]
    return f"{r.a}–{r.b} ({r.length}, {r.color.value if r.color else 'gray'})"


def describe_event(game: Game, e: Event) -> Optional[str]:
    """Plain-English line for a log event (None for events not worth showing)."""
    who = f"P{e.player}"
    if e.kind == "draw_face_up":
        return f"{who} took a face-up {e.public['color']}"
    if e.kind == "draw_blind":
        return f"{who} drew from the deck"
    if e.kind == "claim_route":
        paid = e.public["paid"]
        locos = paid.count("locomotive")
        extra = f", {locos} loco" if locos else ""
        return f"{who} claimed {_route_name(game, e.public['route'])}{extra}"
    if e.kind == "draw_tickets":
        return f"{who} drew {e.public['count']} tickets"
    if e.kind in ("keep_tickets", "keep_initial_tickets"):
        return f"{who} kept {e.public['count']} ticket(s)"
    if e.kind == "market_reset":
        return "3 locomotives face up: market discarded and redealt"
    if e.kind == "reshuffle":
        return "discard pile reshuffled into the deck"
    if e.kind == "final_round_triggered":
        return f"{who} is at 2 or fewer trains: FINAL ROUND, everyone gets one more turn"
    if e.kind == "pass":
        return f"{who} passed (no legal move)"
    if e.kind == "stalemate":
        return "nobody can move: game ends (stalemate)"
    if e.kind == "game_over":
        return "game over"
    return None


# ------------------------------------------------------------ board view


def chip(color: Color, count: int = 0) -> Text:
    label = "LOCO" if color is Color.LOCOMOTIVE else color.value
    if count:
        label += f" {count}"
    return Text(f" {label} ", style=CHIP[color])


def _player_panel(game: Game, i: int, name: str, reveal: bool) -> Panel:
    p = game.players[i]
    body = Text()
    body.append(f"trains {p.trains:>2}   points {p.route_points}\n")
    body.append(f"cards {sum(p.hand.values())}   tickets {len(p.tickets)}\n\n")
    if reveal:
        for color in TRAIN_COLORS + [Color.LOCOMOTIVE]:
            if p.hand[color]:
                body.append_text(chip(color, p.hand[color]))
                body.append("\n")
        if p.tickets:
            body.append("\n")
            mine = [game.board.routes[r] for r in p.routes]
            for t in (game.board.tickets[x] for x in p.tickets):
                done = connected(mine, t.a, t.b)
                codes = [game.board.layout[c].code if game.board.layout else c for c in (t.a, t.b)]
                body.append("✓ " if done else "· ", style="bold green" if done else "dim")
                body.append(f"{codes[0]}–{codes[1]} {t.points}\n", style=None if done else "dim")
    else:
        body.append("(hidden)", style="dim")
    marker = " ◀" if game.current_player == i and not game.game_over else ""
    return Panel(body, title=Text(f"P{i} {name}{marker}", style=SEAT_STYLE[i]), width=PANEL_WIDTH)


def render_board_view(
    game: Game,
    names: Optional[Sequence[str]] = None,
    recent: Sequence[str] = (),
    reveal: bool = True,
    width: int = 160,
) -> Group:
    """Market + deck info on top, ASCII map in the middle, players on either side.
    `width` is the terminal width; the map takes whatever the side panels leave."""
    names = list(names) if names else [""] * game.num_players
    map_width = max(60, min(110, width - 2 * PANEL_WIDTH - 4))

    top = Text("Face up: ")
    for c in game.market:
        top.append_text(chip(c))
        top.append(" ")
    top.append(
        f"   deck {len(game.deck)} · discard {len(game.discard)} · tickets {len(game.ticket_deck)}"
        f" · turn {game.turn}"
    )
    if game.final_turns_remaining is not None and not game.game_over:
        top.append(f"  FINAL ROUND ({game.final_turns_remaining} left)", style="bold red")

    left = [_player_panel(game, i, names[i], reveal) for i in range(0, game.num_players, 2)]
    right = [_player_panel(game, i, names[i], reveal) for i in range(1, game.num_players, 2)]
    middle = Table.grid(padding=(0, 1))
    middle.add_column()
    middle.add_column()
    middle.add_column()
    middle.add_row(Group(*left), render_map(game.board, game.route_owner, width=map_width), Group(*right))

    parts: List = [top, middle]
    for line in recent:
        parts.append(Text(f"  {line}", style="italic"))
    return Group(*parts)
