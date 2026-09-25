"""Run bot-vs-bot games and summarize results.

    ttr-sim --agents greedy random --games 200
    ttr-sim --agents greedy greedy --show                 # text log
    ttr-sim --agents greedy greedy --show board --step    # ASCII board
    ttr-sim --games 50 --record runs/records              # save replayable games

(or `python -m ttr.simulate ...`, which is the same entry point)

Seats are rotated across games (§9 #8) so first-player advantage averages out.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from rich.console import Console
from rich.table import Table

from ttr.actions import Action
from ttr.agents import Agent, GreedyAgent, RandomAgent
from ttr.board import Board, load_board
from ttr.game import Game, GameResult
from ttr.record import GameRecord
from ttr.render import describe_event, render_board_view, render_game, render_result, render_routes

AGENTS: Dict[str, Callable[[int], Agent]] = {
    "random": lambda seed: RandomAgent(seed),
    "greedy": lambda seed: GreedyAgent(seed),
}


def play_game(
    game: Game,
    agents: Sequence[Agent],
    on_step=None,
    actions_out: Optional[List[Action]] = None,
) -> GameResult:
    """Play to the end. If `actions_out` is given, every action is appended to it
    (for a GameRecord)."""
    while not game.game_over:
        p = game.current_player
        action = agents[p].act(game, p)
        game.step(action)
        if actions_out is not None:
            actions_out.append(action)
        if on_step:
            on_step(game)
    assert game.result is not None
    return game.result


def _save_record(
    record_dir: Optional[Path], game: Game, actions: List[Action], agents: Sequence[str],
    seed: int, board: str, name: str,
) -> None:
    if record_dir is None:
        return
    GameRecord.from_game(game, actions, agents=agents, board=board, seed=seed).save(
        record_dir / f"{name}.json"
    )


@dataclass
class AgentStats:
    games: int = 0
    wins: float = 0.0  # shared wins split evenly
    totals: List[int] = field(default_factory=list)
    route_points: List[int] = field(default_factory=list)
    ticket_points: List[int] = field(default_factory=list)
    tickets_completed: List[int] = field(default_factory=list)
    tickets_failed: List[int] = field(default_factory=list)
    longest_bonus: int = 0


def run_matches(
    agent_names: Sequence[str],
    games: int,
    board: Board,
    seed: int = 0,
    max_turns: Optional[int] = 1000,
    record_dir: Optional[Path] = None,
    board_ref: Optional[str] = None,
) -> Dict[str, object]:
    """`record_dir` saves every game as a GameRecord; `board_ref` is how the record
    names the board (defaults to the board's name)."""
    n = len(agent_names)
    stats = [AgentStats() for _ in range(n)]  # indexed by agent slot, not seat
    turns: List[int] = []
    truncated = 0
    for g in range(games):
        # Rotate seats: agent slot i sits in seat (i + g) % n and seat 0 goes first.
        seat_of = [(i + g) % n for i in range(n)]
        agents_by_seat: List[Optional[Agent]] = [None] * n
        names_by_seat: List[str] = [""] * n
        for slot, seat in enumerate(seat_of):
            agents_by_seat[seat] = AGENTS[agent_names[slot]](seed * 1000 + g * 10 + slot)
            names_by_seat[seat] = agent_names[slot]
        game_seed = seed * 100000 + g
        game = Game(board, num_players=n, seed=game_seed, first_player=0, max_turns=max_turns)
        actions: List[Action] = []
        result = play_game(game, agents_by_seat, actions_out=actions if record_dir else None)
        _save_record(
            record_dir, game, actions, names_by_seat, game_seed,
            board_ref or board.name, f"seed{seed}_game{g:04d}",
        )
        turns.append(game.turn)
        truncated += result.truncated
        for slot, seat in enumerate(seat_of):
            s, r = stats[slot], result.players[seat]
            s.games += 1
            if seat in result.winners:
                s.wins += 1 / len(result.winners)
            s.totals.append(r.total)
            s.route_points.append(r.route_points)
            s.ticket_points.append(r.ticket_points)
            s.tickets_completed.append(r.tickets_completed)
            s.tickets_failed.append(r.tickets_failed)
            s.longest_bonus += r.longest_path_bonus
    return {"stats": stats, "turns": turns, "truncated": truncated}


def _summary_table(agent_names: Sequence[str], summary: Dict[str, object]) -> Table:
    mean = statistics.mean
    table = Table(title="Match summary (per agent slot)", header_style="bold")
    for col in ("slot", "agent", "win %", "avg total", "avg routes", "avg tickets",
                "done/failed", "longest bonus %"):
        table.add_column(col)
    for i, (name, s) in enumerate(zip(agent_names, summary["stats"])):
        table.add_row(
            str(i),
            name,
            f"{100 * s.wins / s.games:.1f}",
            f"{mean(s.totals):.1f}",
            f"{mean(s.route_points):.1f}",
            f"{mean(s.ticket_points):+.1f}",
            f"{mean(s.tickets_completed):.2f}/{mean(s.tickets_failed):.2f}",
            f"{100 * s.longest_bonus / s.games:.0f}",
        )
    return table


def show_game(console: Console, board: Board, args: argparse.Namespace) -> None:
    """Play one game, printing each turn as a text log or as the ASCII board view."""
    n = len(args.agents)
    agents = [AGENTS[name](args.seed + i) for i, name in enumerate(args.agents)]
    game = Game(board, num_players=n, seed=args.seed, first_player=0, max_turns=args.max_turns)
    seen = [0]  # log entries already shown
    last_turn = [-1]

    def frame(g: Game) -> None:
        if g.turn == last_turn[0] and not g.game_over:
            return  # mid-turn sub-step; wait for the turn to finish
        last_turn[0] = g.turn
        lines = [text for text in (describe_event(g, e) for e in g.log[seen[0]:]) if text]
        seen[0] = len(g.log)
        if args.show == "board":
            if args.step or args.delay:
                console.clear()
            console.print(render_board_view(g, names=args.agents, recent=lines, width=console.width))
        else:
            console.rule(" · ".join(lines) or f"turn {g.turn}")
            console.print(render_game(g))
        if args.step and not g.game_over:
            input("  [Enter] next turn ")
        elif args.delay:
            time.sleep(args.delay)

    frame(game)  # starting position
    actions: List[Action] = []
    result = play_game(game, agents, on_step=frame, actions_out=actions)
    _save_record(args.record, game, actions, args.agents, args.seed, args.board, f"seed{args.seed}_show")
    if args.show == "log":
        console.print(render_routes(game))
    console.print(render_result(result, names=args.agents))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", default=["greedy", "random"], choices=sorted(AGENTS))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--board", default="usa")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="fix the seed to replay a game/batch exactly (default: random, printed at start)",
    )
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument(
        "--show", nargs="?", const="log", choices=["log", "board"],
        help="play one game and display every turn: 'log' (text tables, default) or 'board' (ASCII map)",
    )
    parser.add_argument("--step", action="store_true", help="with --show: press Enter between turns")
    parser.add_argument("--delay", type=float, default=0.0, help="with --show: seconds to pause per turn")
    parser.add_argument(
        "--record", type=Path, default=None, metavar="DIR",
        help="save every game as a replayable record (seed + actions) in DIR",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles/pipes default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    console = Console()
    if args.seed is None:
        args.seed = random.randrange(1_000_000)
    console.print(f"[dim]seed {args.seed}  (rerun with --seed {args.seed} to replay)[/]")
    board = load_board(args.board)
    if not board.verified:
        console.print(f"[yellow]warning: board '{board.name}' data is UNVERIFIED (see data file note)[/]")

    if args.show:
        show_game(console, board, args)
        return

    start = time.time()
    summary = run_matches(
        args.agents, args.games, board, seed=args.seed, max_turns=args.max_turns,
        record_dir=args.record, board_ref=args.board,
    )
    elapsed = time.time() - start
    console.print(_summary_table(args.agents, summary))
    if args.record:
        console.print(f"[dim]records saved to {args.record}[/]")
    turns = summary["turns"]
    console.print(
        f"{args.games} games in {elapsed:.1f}s · turns median {statistics.median(turns):.0f}"
        f" (max {max(turns)}) · truncated {summary['truncated']}"
    )


if __name__ == "__main__":
    main()
