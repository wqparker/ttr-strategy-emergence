"""Colors and fonts for the viewer, chosen to resemble the physical board."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ttr.cards import Color

RGB = Tuple[int, int, int]

# Route / card colors ("purple" is the board's pink).
CARD: Dict[Optional[Color], RGB] = {
    None: (178, 182, 186),  # gray route
    Color.PURPLE: (214, 120, 196),
    Color.BLUE: (40, 146, 214),
    Color.ORANGE: (240, 140, 36),
    Color.WHITE: (242, 240, 232),
    Color.GREEN: (126, 190, 58),
    Color.YELLOW: (246, 214, 48),
    Color.BLACK: (46, 50, 56),
    Color.RED: (214, 52, 40),
    Color.LOCOMOTIVE: (160, 160, 160),  # drawn as a rainbow stripe where it matters
}

# Plastic train colors, by seat.
PLAYER: Tuple[RGB, ...] = (
    (200, 30, 30),    # red
    (30, 90, 200),    # blue
    (40, 150, 60),    # green
    (235, 190, 20),   # yellow
    (35, 35, 40),     # black
)

MAP_BG = (224, 228, 226)  # boards without a backdrop
FRAME = (122, 30, 24)  # the board's red border
FRAME_LINE = (196, 150, 70)  # thin gold line inside the frame
WATER = (196, 214, 222)
LAND = (234, 234, 226)
STATE_LINE = (178, 180, 172)
COUNTRY_LINE = (140, 136, 128)
COAST_LINE = (150, 170, 180)
CITY_FILL = (206, 86, 38)
CITY_RING = (70, 30, 20)
CITY_SHINE = (250, 190, 150)
LABEL = (54, 34, 30)
LABEL_HALO = (236, 238, 234)
OUTLINE = (60, 60, 64)
TRAIN_RIM = (16, 16, 18)
TABLE_BG = (238, 234, 222)
PANEL_BG = (34, 36, 42)
PANEL_TEXT = (230, 230, 230)
HIGHLIGHT = (255, 255, 255)


def darker(c: RGB, f: float = 0.6) -> RGB:
    return (int(c[0] * f), int(c[1] * f), int(c[2] * f))


def lighter(c: RGB, f: float = 0.35) -> RGB:
    return tuple(int(v + (255 - v) * f) for v in c)  # type: ignore[return-value]
