"""Write a Markdown checklist of a board's routes and tickets for checking the data
against the physical board and cards.

    .venv/Scripts/python scripts/board_checklist.py usa > docs/usa_map_checklist.md
"""

import sys
from collections import Counter

from ttr.board import load_board


def main(name: str = "usa") -> None:
    board = load_board(name)
    out = [
        f"# {board.name.upper()} map data checklist",
        "",
        f"Generated from `src/ttr/data/{board.name}.json` by `scripts/board_checklist.py`.",
        f"Data status: **{'verified' if board.verified else 'UNVERIFIED'}**.",
        "",
        "Tick each row once it matches the physical board or cards. Fix mistakes in the",
        "JSON, regenerate this file, and set `\"verified\": true` once every row is ticked.",
        "",
        f"## Summary ({len(board.cities)} cities, {len(board.routes)} routes, "
        f"{len(board.tickets)} tickets)",
        "",
    ]
    colors = Counter(r.color.value if r.color else "gray" for r in board.routes)
    spaces = Counter()
    for r in board.routes:
        spaces[r.color.value if r.color else "gray"] += r.length
    out += ["| color | routes | train spaces |", "|---|---|---|"]
    for c in sorted(colors):
        out.append(f"| {c} | {colors[c]} | {spaces[c]} |")
    total_spaces = sum(r.length for r in board.routes)
    doubles = sum(1 for r in board.routes if r.sibling is not None) // 2
    out += ["", f"Total train spaces: {total_spaces}. Double routes: {doubles}.", ""]

    out += ["## Routes (alphabetical by first city)", "", "| ✓ | from | to | length | color | double |",
            "|---|---|---|---|---|---|"]
    rows = []
    for r in board.routes:
        a, b = sorted((r.a, r.b))
        rows.append((a, b, r.length, r.color.value if r.color else "gray", "yes" if r.sibling is not None else ""))
    for a, b, length, color, dbl in sorted(rows):
        out.append(f"| [ ] | {a} | {b} | {length} | {color} | {dbl} |")

    out += ["", "## Destination tickets", "", "| ✓ | from | to | points |", "|---|---|---|---|"]
    for t in sorted(board.tickets, key=lambda t: sorted((t.a, t.b))):
        a, b = sorted((t.a, t.b))
        out.append(f"| [ ] | {a} | {b} | {t.points} |")

    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
