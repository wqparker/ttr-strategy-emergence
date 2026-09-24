"""ASCII map of the board: cities as short codes at their layout positions,
routes as character lines. Unclaimed routes are dim in their route color;
claimed routes are drawn as the owner's seat number on the seat color.

Double routes share one line on screen (a claimed half is drawn over the other).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rich.text import Text

from ttr.board import Board
from ttr.cards import Color

ROUTE_FG = {
    None: "grey50",
    Color.PURPLE: "magenta",
    Color.BLUE: "blue",
    Color.ORANGE: "dark_orange",
    Color.WHITE: "white",
    Color.GREEN: "green",
    Color.YELLOW: "yellow",
    Color.BLACK: "grey35",
    Color.RED: "red",
}
# Claimed routes: bold character on the owner's background.
SEAT_CLAIM = [
    "bold white on blue",
    "bold white on red",
    "bold black on green",
    "bold black on yellow",
    "bold white on magenta",
]


def _project(board: Board, width: int, height: int) -> Dict[str, Tuple[int, int]]:
    """Map layout x/y to (col, row) with room for 3-char labels."""
    xs = [p.x for p in board.layout.values()]
    ys = [p.y for p in board.layout.values()]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    span_x = (max_x - min_x) or 1
    span_y = (max_y - min_y) or 1
    pos = {}
    for city, p in board.layout.items():
        col = 1 + round((p.x - min_x) / span_x * (width - 5))
        row = round((max_y - p.y) / span_y * (height - 1))
        pos[city] = (col, row)
    return pos


def _line(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int, str]]:
    """Bresenham from a to b; each cell gets a char for its step direction."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells = []
    x, y = x0, y0
    while (x, y) != (x1, y1):
        e2 = 2 * err
        moved_x = moved_y = False
        if e2 >= dy:
            err += dy
            x += sx
            moved_x = True
        if e2 <= dx:
            err += dx
            y += sy
            moved_y = True
        if moved_x and moved_y:
            ch = "\\" if sx == sy else "/"
        elif moved_x:
            ch = "-"
        else:
            ch = "|"
        cells.append((x, y, ch))
    return cells[:-1]  # drop the endpoint (the city itself)


def render_map(board: Board, route_owner: Dict[int, int], width: int = 96, height: int = 30) -> Text:
    if not board.layout:
        return Text("(board has no layout data)", style="dim")
    pos = _project(board, width, height)
    chars: List[List[str]] = [[" "] * width for _ in range(height)]
    styles: List[List[Optional[str]]] = [[None] * width for _ in range(height)]

    def label_cells(city: str):
        col, row = pos[city]
        code = board.layout[city].code
        start = col - len(code) // 2
        return [(start + i, row, ch) for i, ch in enumerate(code)]

    occupied = {(x, y) for city in pos for x, y, _ in label_cells(city)}

    # Unclaimed first, claimed last so ownership is always visible.
    ordered = sorted(board.routes, key=lambda r: r.id in route_owner)
    for route in ordered:
        owner = route_owner.get(route.id)
        style = SEAT_CLAIM[owner] if owner is not None else f"dim {ROUTE_FG[route.color]}"
        for x, y, ch in _line(pos[route.a], pos[route.b]):
            if (x, y) in occupied or not (0 <= x < width and 0 <= y < height):
                continue
            chars[y][x] = str(owner) if owner is not None else ch
            styles[y][x] = style

    for city in pos:
        for x, y, ch in label_cells(city):
            if 0 <= x < width:
                chars[y][x] = ch
                styles[y][x] = "bold bright_white"

    text = Text()
    for row in range(height):
        for col in range(width):
            text.append(chars[row][col], style=styles[row][col])
        if row < height - 1:
            text.append("\n")
    return text
