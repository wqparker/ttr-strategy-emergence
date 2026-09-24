"""`rich` terminal rendering of game state, for debugging (PLAN.md phase 2)."""

from __future__ import annotations

from typing import Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ttr.cards import TRAIN_COLORS, Color
from ttr.game import Game, GameResult

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
SEAT_STYLE = ["bold blue", "bold red", "bold green", "bold yellow", "bold white"]


def card_text(color: Optional[Color], count: int = 1) -> Text:
    if color is None:
        return Text(f"gray" + (f"×{count}" if count > 1 else ""), style="grey70")
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
