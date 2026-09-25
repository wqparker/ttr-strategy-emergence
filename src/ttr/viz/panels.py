"""Panel widgets for the viewer: the table strip, a seat panel and the event
ticker. They draw a `ttr.viz.perspective.ViewModel` and nothing else, so what is
hidden is decided before the drawing, not here.

Panels are drawn straight at the display scale: every shape is axis-aligned, so
they need no supersampling (unlike the board).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Sequence, Tuple

import pygame

from ttr.cards import TRAIN_COLORS, Color
from ttr.viz import theme
from ttr.viz.perspective import SeatFacts, TicketFact, ViewModel

HAND_ORDER: Tuple[Color, ...] = tuple(TRAIN_COLORS) + (Color.LOCOMOTIVE,)
SANS = "segoeui,arial,helvetica,sans"

_FONTS: Dict[Tuple[int, bool], pygame.font.Font] = {}


def font(size: float, bold: bool = False) -> pygame.font.Font:
    """Cached system font. Fonts made before a `pygame.quit()` are dead, so a
    cache hit is probed and the whole cache dropped if pygame was restarted."""
    key = (max(6, round(size)), bold)
    cached = _FONTS.get(key)
    if cached is not None:
        try:
            cached.get_height()
            return cached
        except pygame.error:
            _FONTS.clear()
    _FONTS[key] = pygame.font.SysFont(SANS, key[0], bold=bold)
    return _FONTS[key]


# --------------------------------------------------------------- primitives


def text(
    s: pygame.Surface,
    body: str,
    pos: Tuple[float, float],
    size: float,
    color: theme.RGB = theme.PANEL_TEXT,
    bold: bool = False,
    right: bool = False,
) -> pygame.Rect:
    img = font(size, bold).render(body, True, color)
    rect = img.get_rect()
    if right:
        rect.topright = (round(pos[0]), round(pos[1]))
    else:
        rect.topleft = (round(pos[0]), round(pos[1]))
    s.blit(img, rect)
    return rect


def card_chip(
    s: pygame.Surface,
    rect: pygame.Rect,
    color: Color,
    label: str = "",
    dim: bool = False,
) -> None:
    """A train-card chip: the card color, a dark edge, and a label inside."""
    fill = theme.CARD[color]
    if dim:
        fill = theme.darker(fill, 0.55)
    radius = max(2, round(min(rect.width, rect.height) * 0.22))
    pygame.draw.rect(s, fill, rect, border_radius=radius)
    if color is Color.LOCOMOTIVE:
        _loco_bands(s, rect, dim)
    pygame.draw.rect(s, theme.darker(fill, 0.45), rect, max(1, round(rect.height / 22)), border_radius=radius)
    if label:
        img = font(rect.height * 0.46, bold=True).render(label, True, theme.chip_text(fill))
        s.blit(img, img.get_rect(center=rect.center))


def _loco_bands(s: pygame.Surface, rect: pygame.Rect, dim: bool) -> None:
    """The locomotive card's rainbow, so it is never mistaken for a gray card."""
    clip = s.get_clip()
    s.set_clip(rect.clip(clip) if clip else rect)
    n = len(theme.LOCO_BANDS)
    w = rect.width / n
    skew = rect.height * 0.35
    for i, band in enumerate(theme.LOCO_BANDS):
        c = theme.darker(band, 0.55) if dim else band
        x = rect.left + i * w - skew
        pygame.draw.polygon(
            s,
            c,
            [(x, rect.bottom), (x + w, rect.bottom), (x + w + skew, rect.top), (x + skew, rect.top)],
        )
    s.set_clip(clip)


def _box(s: pygame.Surface, rect: pygame.Rect, bg: theme.RGB, edge: Optional[theme.RGB] = None) -> None:
    radius = max(2, round(min(rect.width, rect.height) * 0.05))
    pygame.draw.rect(s, bg, rect, border_radius=radius)
    if edge:
        pygame.draw.rect(s, edge, rect, 2, border_radius=radius)


def _count_row(
    s: pygame.Surface,
    pos: Tuple[float, float],
    counts: Counter,
    k: float,
    chip_w: float,
    chip_h: float,
    gap: float,
    per_row: int = 9,
    prefix: str = "",
    dim: bool = False,
) -> float:
    """Chips for every color with a nonzero count. Returns the bottom y."""
    x, y = pos
    i = 0
    for color in HAND_ORDER:
        n = counts.get(color, 0)
        if not n:
            continue
        col, row = i % per_row, i // per_row
        rect = pygame.Rect(
            round(x + col * (chip_w + gap)),
            round(y + row * (chip_h + gap)),
            round(chip_w),
            round(chip_h),
        )
        card_chip(s, rect, color, f"{prefix}{n}", dim=dim)
        i += 1
    rows = max(1, (i + per_row - 1) // per_row)
    return y + rows * (chip_h + gap)


def _text_w(body: str, size: float, bold: bool = False) -> float:
    return font(size, bold).size(body)[0]


def _chips_w(counts: Counter, chip_w: float, gap: float, per_row: int) -> float:
    n = min(per_row, sum(1 for c in HAND_ORDER if counts.get(c, 0)))
    return n * chip_w + max(0, n - 1) * gap if n else 0.0


# -------------------------------------------------------------- table strip


def draw_table(s: pygame.Surface, rect: pygame.Rect, vm: ViewModel, k: float = 1.0,
               left_bound: Optional[float] = None) -> None:
    """Top strip: the 5 face-up cards, the pile counts, the unseen pool in a
    player view, and the current sub-step. The table blocks are centered in the
    strip; the sub-step stays anchored to the right end, and `left_bound` keeps
    the blocks clear of the key legend."""
    _box(s, rect, theme.PANEL_BG)
    pad = 10 * k
    cw, ch = 30 * k, 42 * k
    gap = 14 * k
    top = rect.top + pad + 13 * k

    cards_w = 5 * cw + 4 * 5 * k
    piles = (("deck", vm.table.deck), ("discard", vm.table.discard), ("tickets", vm.table.tickets_left))
    piles_w = 2 * 52 * k + max(_text_w(n.upper(), 9 * k, True) for n, _ in piles)
    piles_w = max(piles_w, 2 * 52 * k + _text_w(str(piles[-1][1]), 20 * k, True))
    total = cards_w + gap + piles_w
    unseen_w = 0.0
    if vm.table.unseen is not None:
        unseen_w = max(
            _text_w("UNSEEN POOL", 9 * k, True),
            _chips_w(vm.table.unseen, 28 * k, 3 * k, 9),
        )
        total += gap + unseen_w

    x = rect.centerx - total / 2
    x = max(left_bound if left_bound is not None else rect.left + pad,
            min(x, rect.right - _substep_w(vm, k) - 12 * k - total))

    text(s, "FACE UP", (x, rect.top + pad), 9 * k, theme.PANEL_LABEL, bold=True)
    for i, color in enumerate(vm.table.market):
        card_chip(s, pygame.Rect(round(x + i * (cw + 5 * k)), round(top), round(cw), round(ch)), color)
    x += cards_w + gap

    for i, (name, n) in enumerate(piles):
        cx = x + i * 52 * k
        text(s, name.upper(), (cx, rect.top + pad), 9 * k, theme.PANEL_LABEL, bold=True)
        text(s, str(n), (cx, top + 4 * k), 20 * k, theme.PANEL_TEXT, bold=True)
    x += piles_w + gap

    if vm.table.unseen is not None:
        text(s, "UNSEEN POOL", (x, rect.top + pad), 9 * k, theme.PANEL_LABEL, bold=True)
        _count_row(s, (x, top + 6 * k), vm.table.unseen, k, 28 * k, 17 * k, 3 * k)

    _substep(s, rect, vm, k)


def _substep_w(vm: ViewModel, k: float) -> float:
    """How much room the right-hand sub-step block needs."""
    t = vm.table
    widths = [_text_w("all-seeing" if vm.all_seeing else f"P{vm.viewer} view (memory L2)", 10 * k, True)]
    if t.game_over:
        widths.append(_text_w("GAME OVER — P0, P1", 15 * k, True))
    else:
        widths.append(_text_w(f"turn {t.turn}  ·  P{t.current_player} to act", 15 * k, True))
        widths.append(_text_w(t.phase.replace("_", " "), 11 * k))
        if t.final_turns_remaining is not None:
            widths.append(_text_w(f"FINAL ROUND · {t.final_turns_remaining} turns left", 11 * k, True))
    return max(widths)


def _substep(s: pygame.Surface, rect: pygame.Rect, vm: ViewModel, k: float) -> None:
    """Right end of the table strip: whose view this is, and the sub-step."""
    t = vm.table
    right = rect.right - 10 * k
    view = "all-seeing" if vm.all_seeing else f"P{vm.viewer} view (memory L{vm.memory_level})"
    text(s, view, (right, rect.top + 8 * k), 10 * k, theme.PANEL_LABEL, bold=True, right=True)
    if t.game_over:
        won = ", ".join(f"P{w}" for w in t.winners)
        text(s, f"GAME OVER — {won or 'nobody'}", (right, rect.top + 24 * k), 15 * k,
             theme.HIGHLIGHT, bold=True, right=True)
        return
    text(s, f"turn {t.turn}  ·  P{t.current_player} to act", (right, rect.top + 24 * k),
         15 * k, theme.PANEL_TEXT, bold=True, right=True)
    text(s, t.phase.replace("_", " "), (right, rect.top + 44 * k), 11 * k, theme.PANEL_DIM, right=True)
    if t.final_turns_remaining is not None:
        text(s, f"FINAL ROUND · {t.final_turns_remaining} turns left", (right, rect.top + 60 * k),
             11 * k, theme.CARD[Color.YELLOW], bold=True, right=True)


# ------------------------------------------------------------ controls


def controls_layout(entries, rect: pygame.Rect, k: float = 1.0, per_col: int = 4):
    """Type size, row height, column width and total width for the key legend,
    sized to fill `rect` (the table strip) down its full height."""
    if not entries:
        return 0.0, 0.0, 0.0, 0.0
    pad = 8 * k
    row_h = (rect.height - 2 * pad) / per_col
    size = max(8 * k, min(13 * k, row_h * 0.66))
    key_w = max(_text_w(keys, size, True) for keys, _ in entries)
    label_w = max(_text_w(name, size) for _, name in entries)
    col_w = key_w + 10 * k + label_w + 20 * k
    cols = -(-len(entries) // per_col)
    return size, row_h, col_w, col_w * cols


def draw_controls(s: pygame.Surface, rect: pygame.Rect, entries, k: float = 1.0,
                  per_col: int = 4) -> None:
    """The key legend down the table strip's left side: (keys, what it does)
    pairs in columns. The app owns the bindings; this only draws them."""
    size, row_h, col_w, _ = controls_layout(entries, rect, k, per_col)
    if not size:
        return
    pad = 8 * k
    key_w = max(_text_w(keys, size, True) for keys, _ in entries)
    for i, (keys, name) in enumerate(entries):
        col, row = i // per_col, i % per_col
        cx = rect.left + pad + col * col_w
        cy = rect.top + pad + row * row_h + (row_h - size * 1.35) / 2
        text(s, keys, (cx, cy), size, theme.PANEL_TEXT, bold=True)
        text(s, name, (cx + key_w + 10 * k, cy), size, theme.PANEL_LABEL)


def draw_buttons(s: pygame.Surface, buttons, k: float = 1.0, active=()) -> None:
    """On-screen controls: (id, label, rect) triples. Ids in `active` are drawn
    lit, for a toggle that is currently on."""
    for name, label, rect in buttons:
        lit = name in active
        pygame.draw.rect(s, theme.PANEL_BG if lit else theme.PANEL_SLOT, rect,
                         border_radius=max(2, round(3 * k)))
        pygame.draw.rect(s, theme.HIGHLIGHT if lit else theme.PANEL_EDGE, rect,
                         max(1, round(k)), border_radius=max(2, round(3 * k)))
        img = font(10 * k, bold=True).render(label, True,
                                             theme.HIGHLIGHT if lit else theme.PANEL_TEXT)
        s.blit(img, img.get_rect(center=rect.center))


# ---------------------------------------------------------------- seat panel


def draw_seat(
    s: pygame.Surface,
    rect: pygame.Rect,
    facts: SeatFacts,
    k: float = 1.0,
    wide: bool = False,
    codes: Optional[Dict[str, str]] = None,
) -> None:
    """One seat's box. `wide` is the full-width bottom panel; otherwise a side
    panel column."""
    accent = theme.PLAYER[facts.seat % len(theme.PLAYER)]
    _box(s, rect, theme.PANEL_SLOT, theme.HIGHLIGHT if facts.to_act else theme.PANEL_EDGE)
    pad = 9 * k
    x, y = rect.left + pad, rect.top + pad

    swatch = pygame.Rect(round(x), round(y + 2 * k), round(11 * k), round(11 * k))
    radius = max(1, round(2 * k))
    pygame.draw.rect(s, accent, swatch, border_radius=radius)
    # A light rim: the black seat's swatch would vanish into the panel otherwise.
    pygame.draw.rect(s, theme.PANEL_DIM, swatch, max(1, round(k)), border_radius=radius)
    name = f"P{facts.seat}" + (f" {facts.name}" if facts.name else "")
    if facts.is_viewer:
        name += " (you)"
    text(s, name, (x + 16 * k, y), 13 * k, theme.PANEL_TEXT, bold=True)
    if facts.to_act:
        text(s, "TO ACT", (rect.right - pad, y + 1 * k), 10 * k, theme.HIGHLIGHT, bold=True, right=True)
    y += 20 * k

    if wide:
        _wide_body(s, rect, facts, y, k, codes)
    else:
        y = _stats(s, facts, (x, y), k, wide=False)
        _narrow_body(s, facts, (x, y), k, codes)


def _stats(s, facts: SeatFacts, pos, k: float, wide: bool) -> float:
    """trains / cards / tickets / points: a row when wide, a 2x2 grid when not.
    All four are public (RULES.md §9 #17), so they show in every view."""
    x, y = pos
    items = (
        ("trains", facts.trains),
        ("cards", facts.hand_size),
        ("tickets", facts.ticket_count),
        ("points", facts.route_points),
    )
    if wide:
        for i, (name, value) in enumerate(items):
            cx = x + i * 62 * k
            text(s, name.upper(), (cx, y), 9 * k, theme.PANEL_LABEL, bold=True)
            text(s, str(value), (cx, y + 11 * k), 19 * k, theme.PANEL_TEXT, bold=True)
        return y + 34 * k
    for i, (name, value) in enumerate(items):
        cx = x + (i % 2) * 84 * k
        cy = y + (i // 2) * 26 * k
        text(s, name.upper(), (cx, cy + 3 * k), 9 * k, theme.PANEL_LABEL, bold=True)
        text(s, str(value), (cx + 72 * k, cy), 15 * k, theme.PANEL_TEXT, bold=True, right=True)
    return y + 56 * k


def _hand_block(s, facts: SeatFacts, pos, k: float, chip: Tuple[float, float], per_row: int) -> float:
    """Exact hand when revealed; memory lower bounds otherwise; at level 0 only
    the hand size, which the stats row already shows."""
    x, y = pos
    cw, ch = chip
    if facts.hand is not None:
        text(s, "HAND", (x, y), 9 * k, theme.PANEL_LABEL, bold=True)
        if not facts.hand_size:
            text(s, "empty", (x, y + 12 * k), 11 * k, theme.PANEL_DIM)
            return y + 26 * k
        return _count_row(s, (x, y + 12 * k), facts.hand, k, cw, ch, 4 * k, per_row=per_row)
    if facts.known is None:
        text(s, "CARDS HIDDEN", (x, y), 9 * k, theme.PANEL_LABEL, bold=True)
        return y + 16 * k
    text(s, "KNOWN CARDS", (x, y), 9 * k, theme.PANEL_LABEL, bold=True)
    bottom = y + 12 * k
    if facts.known:
        # Short chips: the bounds need two more lines under them than a hand does.
        bottom = _count_row(s, (x, bottom), facts.known, k, cw, min(ch, 19 * k), 4 * k,
                            per_row=per_row, prefix="≥", dim=True)
    else:
        text(s, "none seen", (x, bottom), 11 * k, theme.PANEL_DIM)
        bottom += 16 * k
    if facts.unknown:
        text(s, f"+{facts.unknown} unknown", (x, bottom), 11 * k, theme.PANEL_DIM)
        bottom += 16 * k
    return bottom


def _stats_w(facts: SeatFacts, k: float) -> float:
    return 3 * 62 * k + max(_text_w("POINTS", 9 * k, True), _text_w(str(facts.route_points), 19 * k, True))


def _hand_w(facts: SeatFacts, k: float, chip_w: float, per_row: int) -> float:
    if facts.hand is not None:
        if not facts.hand_size:
            return max(_text_w("HAND", 9 * k, True), _text_w("empty", 11 * k))
        return max(_text_w("HAND", 9 * k, True), _chips_w(facts.hand, chip_w, 4 * k, per_row))
    if facts.known is None:
        return _text_w("CARDS HIDDEN", 9 * k, True)
    widths = [_text_w("KNOWN CARDS", 9 * k, True)]
    widths.append(_chips_w(facts.known, chip_w, 4 * k, per_row) if facts.known
                  else _text_w("none seen", 11 * k))
    if facts.unknown:
        widths.append(_text_w(f"+{facts.unknown} unknown", 11 * k))
    return max(widths)


def _tickets_w(facts: SeatFacts, k: float, label: str, per_col: int, col_w: float) -> float:
    label_w = _text_w(label, 9 * k, True)
    if facts.tickets is None:
        return max(label_w, _text_w(f"{facts.ticket_count} held, contents hidden", 11 * k))
    if not facts.tickets:
        return max(label_w, _text_w("none", 11 * k))
    cols = -(-len(facts.tickets) // per_col)
    return max(label_w, cols * col_w - 12 * k)


def _ticket_line(s, t: TicketFact, pos, k: float, codes, size: float, width: float) -> None:
    x, y = pos
    a = codes.get(t.a, t.a) if codes else t.a
    b = codes.get(t.b, t.b) if codes else t.b
    color = theme.DONE if t.done else (theme.PANEL_DIM if t.pending else theme.PANEL_TEXT)
    mark = "✓" if t.done else ("?" if t.pending else "·")
    text(s, mark, (x, y), size, color, bold=t.done)
    text(s, f"{a}–{b}", (x + 11 * k, y), size, color)
    text(s, str(t.points), (x + width, y), size, color, bold=True, right=True)


def _tickets_header(s, facts: SeatFacts, pos, k: float, label: str) -> bool:
    """Draws the caption and the hidden/none cases. True if lines follow."""
    x, y = pos
    text(s, label, (x, y), 9 * k, theme.PANEL_LABEL, bold=True)
    if facts.tickets is None:
        text(s, f"{facts.ticket_count} held, contents hidden", (x, y + 13 * k), 11 * k, theme.PANEL_DIM)
        return False
    if not facts.tickets:
        text(s, "none", (x, y + 13 * k), 11 * k, theme.PANEL_DIM)
        return False
    return True


def _wide_body(s, rect, facts: SeatFacts, y: float, k: float, codes) -> None:
    """Seat 0's panel. The stats row, the hand and the tickets are measured and
    centred together, so the block sits in the middle of the full-width panel."""
    chip_w, per_row, per_col, col_w = 34 * k, 9, 4, 100 * k
    left_w = max(_stats_w(facts, k), _hand_w(facts, k, chip_w, per_row))
    tickets_w = _tickets_w(facts, k, "DESTINATION TICKETS", per_col, col_w)
    gap = 34 * k
    x = max(rect.left + 9 * k, rect.centerx - (left_w + gap + tickets_w) / 2)

    hand_y = _stats(s, facts, (x, y), k, wide=True)
    _hand_block(s, facts, (x, hand_y), k, (chip_w, 46 * k), per_row=per_row)
    tx = x + left_w + gap
    if not _tickets_header(s, facts, (tx, hand_y), k, "DESTINATION TICKETS"):
        return
    for i, t in enumerate(facts.tickets or ()):
        col, row = i // per_col, i % per_col
        _ticket_line(s, t, (tx + col * col_w, hand_y + 14 * k + row * 15 * k), k, codes, 11 * k, 88 * k)


def _narrow_body(s, facts: SeatFacts, pos, k: float, codes) -> None:
    x, y = pos
    y = _hand_block(s, facts, (x, y), k, (30 * k, 19 * k), per_row=5) + 8 * k
    if not _tickets_header(s, facts, (x, y), k, "TICKETS"):
        return
    for i, t in enumerate(facts.tickets or ()):
        _ticket_line(s, t, (x, y + 14 * k + i * 15 * k), k, codes, 11 * k, 84 * k)


# -------------------------------------------------------------------- ticker


def draw_ticker(s: pygame.Surface, rect: pygame.Rect, lines: Sequence[str], k: float = 1.0) -> None:
    """Recent public events, oldest first, along the top of the bottom panel.
    Overlong text is clipped from the left so the newest event stays visible."""
    pygame.draw.rect(s, theme.PANEL_BG, rect)
    if not lines:
        return
    img = font(11 * k).render("   •   ".join(lines), True, theme.TICKER_TEXT)
    inner = rect.width - 16 * k
    pos = (rect.left + 8 * k, rect.centery - img.get_height() / 2)
    if img.get_width() > inner:
        s.blit(img, pos, pygame.Rect(img.get_width() - inner, 0, inner, img.get_height()))
    else:
        s.blit(img, pos)
