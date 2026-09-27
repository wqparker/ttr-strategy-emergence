"""Per-game behavior and result metrics for one seat, read from a finished game's
log and result, so they apply to any agent (learner or scripted bot) alike. The
training runs record them for every game and evaluation, and the analysis
dashboard (ttr.learn.dashboard) plots them.

Pure Python. Every value is a number, so rows average and tabulate directly.
"""

from __future__ import annotations

import statistics
from typing import Dict, Iterable, List

from ttr.game import Game

# Name -> short description, in display order. The dashboard's tables and
# behavior charts follow this order.
METRICS: Dict[str, str] = {
    "won": "sole winner (1) or not (0)",
    "win_share": "1 for a win, 1/k for a k-way shared win, else 0",
    "margin": "my total - opponents' mean total",
    "score": "final total",
    "opp_score": "opponents' mean total",
    "route_points": "points from claimed routes",
    "ticket_points": "net ticket points (completed - failed)",
    "tickets_completed": "tickets completed",
    "tickets_failed": "tickets failed",
    "longest_bonus": "earned the longest-path bonus",
    "longest_path": "longest continuous path, in trains",
    "claims": "routes claimed",
    "mean_claim_length": "mean length of claimed routes",
    "long_claims": "claims of length 5 or 6",
    "trains_left": "trains left at the end",
    "locomotives_spent": "Locomotives spent on routes",
    "draw_color": "face-up colored cards taken",
    "draw_locomotive": "face-up Locomotives taken",
    "draw_blind": "blind draws",
    "ticket_draws": "ticket draws after the opening",
    "tickets_kept": "tickets kept in total (opening included)",
    "final_round_ticket_draws": "ticket draws during the final round",
    "passes": "passes",
    "triggered_end": "this seat triggered the final round",
    "turns": "turns in the game (all seats)",
}


def game_metrics(game: Game, seat: int) -> Dict[str, float]:
    """METRICS for `seat` in a finished game."""
    result = game.result
    if result is None:
        raise ValueError("the game isn't over")
    board = game.board
    mine = result.players[seat]
    opp_totals = [r.total for q, r in enumerate(result.players) if q != seat]
    lengths: List[int] = []
    counts = dict.fromkeys(("draw_color", "draw_locomotive", "draw_blind", "ticket_draws",
                            "final_round_ticket_draws", "passes", "locomotives_spent", "tickets_kept"), 0)
    final_round = triggered = False
    for e in game.log:
        if e.kind == "final_round_triggered":
            final_round = True
            triggered = triggered or e.player == seat
        if e.player != seat:
            continue
        if e.kind == "claim_route":
            lengths.append(board.routes[e.public["route"]].length)
            counts["locomotives_spent"] += e.public["paid"].count("locomotive")
        elif e.kind == "draw_face_up":
            counts["draw_locomotive" if e.public["color"] == "locomotive" else "draw_color"] += 1
        elif e.kind == "draw_blind":
            counts["draw_blind"] += 1
        elif e.kind == "draw_tickets":
            counts["ticket_draws"] += 1
            counts["final_round_ticket_draws"] += final_round
        elif e.kind in ("keep_tickets", "keep_initial_tickets"):
            counts["tickets_kept"] += e.public["count"]
        elif e.kind == "pass":
            counts["passes"] += 1
    won = seat in result.winners
    row = {
        "won": float(won and len(result.winners) == 1),
        "win_share": 1 / len(result.winners) if won else 0.0,
        "margin": mine.total - statistics.mean(opp_totals),
        "score": mine.total,
        "opp_score": statistics.mean(opp_totals),
        "route_points": mine.route_points,
        "ticket_points": mine.ticket_points,
        "tickets_completed": mine.tickets_completed,
        "tickets_failed": mine.tickets_failed,
        "longest_bonus": float(mine.longest_path_bonus),
        "longest_path": mine.longest_path,
        "claims": len(lengths),
        "mean_claim_length": statistics.mean(lengths) if lengths else 0.0,
        "long_claims": sum(1 for n in lengths if n >= 5),
        "trains_left": game.players[seat].trains,
        **counts,
        "triggered_end": float(triggered),
        "turns": game.turn,
    }
    return {k: float(row[k]) for k in METRICS}


def mean_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    return {k: statistics.mean(r[k] for r in rows) for k in METRICS} if rows else {}
