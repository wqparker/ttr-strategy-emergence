"""Agent analysis dashboard: one full-screen matplotlib window over one or more
training runs (the JSON `ttr-train-linear`, `ttr-train-dqn` or `ttr-train-ppo` writes).

    ttr-dash runs/linear/q_greedy.json
    ttr-dash runs/linear/q_greedy.json runs/linear/sarsa_greedy.json     # compare runs
    ttr-dash runs/linear/*.json --save runs/linear/dash                   # every page as PNG, no window
    ttr-dash runs/linear/pass2/*.json --group                             # average seeds: X_s0, X_s1 -> X
    ttr-dash --live runs/linear/pass3/q_random_s0.json                    # follow a run training with --live

Pages (keys 1-7 or left/right):

    1 Overview        learning curves (evaluation win share and margin, training
                      margin, TD error, exploration) and the final evaluation next
                      to the random and greedy bots on the same games
    2 Behavior        every evaluation metric over training, bots as reference lines
    3 Training games  the same metrics over the training games themselves (exploring),
                      as rolling means
    4 Action mix      what the agent spends its decisions on, over training and at the end
    5 Weights         final weights as heatmaps, and the largest weights over training
                      (linear runs only)
    6 Evaluations     raw table: every evaluation
    7 Games           raw table: every training game, a page at a time

    o  next evaluation opponent (greedy, random, and any others the runs have)
    r  next run (pages 4-7)
    PgUp / PgDn  page through raw games          s  save this page as PNG
    f  full screen                               q  quit

`--live [SECONDS]` rereads the files every few seconds (default 3) and redraws
the page on screen when one changed; train with `ttr-train-linear --live` so the
file is rewritten while training. Files that don't exist yet are waited for.

Evaluations are greedy play (no exploration) against a fixed set of games per
evaluation; the bots' baselines are measured on the final evaluation's games.
Needs the `[analysis]` extra (matplotlib).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import textwrap
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------ theme
# Validated reference palette (dataviz skill, references/palette.md), light mode.

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BOT_STYLE = {  # reference lines for the scripted bots: gray, told apart by dash and label
    "greedy": {"color": INK_2, "linestyle": (0, (6, 3)), "linewidth": 1.4},
    "random": {"color": INK_3, "linestyle": (0, (1.5, 2.5)), "linewidth": 1.4},
}
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]  # blue <- 0 -> red
OPPONENTS = ("greedy", "random")
# Settings shown on the overview page, by the run's method ("method" in the run file; linear runs predate it).
CONFIG_KEYS = {
    "linear": ("opponent", "games", "alpha", "alpha_end", "lam", "average", "shaping", "epsilon_start",
               "epsilon_end", "reward_mode", "seed"),
    "dqn": ("opponent", "games", "hidden", "n_step", "lr", "lr_end", "batch", "average", "epsilon_end",
            "reward_mode", "seed"),
    "ppo": ("opponent", "games", "games_per_update", "hidden", "lr", "ent_coef", "clip", "gae_lambda", "shaping",
            "ticket_plan", "reward_mode", "seed"),
}

# Metric -> (label, format). Order is display order.
FORMATS: Dict[str, Tuple[str, str]] = {
    "win_share": ("win share", "{:.0%}"),
    "margin": ("margin", "{:+.1f}"),
    "score": ("score", "{:.1f}"),
    "opp_score": ("opponent score", "{:.1f}"),
    "route_points": ("route points", "{:.1f}"),
    "ticket_points": ("ticket points", "{:+.1f}"),
    "tickets_completed": ("tickets completed", "{:.2f}"),
    "tickets_failed": ("tickets failed", "{:.2f}"),
    "tickets_kept": ("tickets kept", "{:.2f}"),
    "longest_bonus": ("longest-path bonus", "{:.0%}"),
    "longest_path": ("longest path", "{:.1f}"),
    "claims": ("claims", "{:.1f}"),
    "mean_claim_length": ("mean claim length", "{:.2f}"),
    "long_claims": ("claims of 5-6", "{:.2f}"),
    "trains_left": ("trains left", "{:.1f}"),
    "locomotives_spent": ("Locomotives spent", "{:.1f}"),
    "draw_color": ("face-up color draws", "{:.1f}"),
    "draw_locomotive": ("face-up Locomotive draws", "{:.1f}"),
    "draw_blind": ("blind draws", "{:.1f}"),
    "ticket_draws": ("ticket draws", "{:.2f}"),
    "final_round_ticket_draws": ("final-round ticket draws", "{:.2f}"),
    "passes": ("passes", "{:.2f}"),
    "triggered_end": ("triggered the end", "{:.0%}"),
    "turns": ("turns (both seats)", "{:.1f}"),
    "won": ("sole wins", "{:.0%}"),
}
BEHAVIOR = ["win_share", "margin", "score", "opp_score", "route_points", "ticket_points",
            "tickets_completed", "tickets_failed", "claims", "mean_claim_length", "long_claims",
            "ticket_draws", "final_round_ticket_draws", "longest_bonus", "draw_blind", "trains_left"]
TRAINING = ["margin", "score", "route_points", "ticket_points", "tickets_completed", "tickets_failed",
            "claims", "mean_claim_length", "ticket_draws", "final_round_ticket_draws", "decisions",
            "mean_abs_td"]
SUMMARY = ["win_share", "margin", "score", "route_points", "ticket_points", "tickets_completed",
           "tickets_failed", "tickets_kept", "claims", "mean_claim_length", "long_claims", "ticket_draws",
           "final_round_ticket_draws", "longest_bonus", "draw_blind", "draw_color", "trains_left"]
ACTIONS = [("claims", "claim"), ("draw_blind", "blind draw"), ("draw_color", "face-up color"),
           ("draw_locomotive", "face-up Locomotive"), ("ticket_draws", "ticket draw"), ("passes", "pass")]
EXTRA_FORMATS = {"decisions": ("decisions per game", "{:.1f}"), "mean_abs_td": ("mean |TD error|", "{:.4f}"),
                 "epsilon": ("epsilon", "{:.3f}"), "game": ("game", "{:.0f}"), "seat": ("seat", "{:.0f}")}


def label(key: str) -> str:
    return (FORMATS.get(key) or EXTRA_FORMATS.get(key) or (key, ""))[0]


def fmt(key: str, value: float) -> str:
    spec = (FORMATS.get(key) or EXTRA_FORMATS.get(key) or ("", "{:.3g}"))[1]
    return spec.format(value)


# ------------------------------------------------------------------- data


class RunFormatError(ValueError):
    pass


@dataclass
class Run:
    name: str
    path: Path
    data: dict
    color: str

    @property
    def config(self) -> dict:
        return self.data.get("config", {})

    @property
    def history(self) -> List[dict]:
        return self.data["history"]

    @property
    def games(self) -> List[dict]:
        return self.data["games"]

    def eval_series(self, opponent: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
        xs = np.array([e["games"] for e in self.history], dtype=float)
        ys = np.array([e["eval"][opponent][key] for e in self.history], dtype=float)
        return xs, ys

    def game_series(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        xs = np.array([r["game"] for r in self.games], dtype=float)
        ys = np.array([r[key] for r in self.games], dtype=float)
        return xs, ys

    def final(self, opponent: str) -> Dict[str, float]:
        """The latest evaluation; empty before the first one (a live run)."""
        return self.history[-1]["eval"][opponent] if self.history else {}

    @property
    def progress(self) -> str:
        p = self.data.get("progress")
        if not p:
            return ""
        return f"{p['games']}/{p['of']}" + ("" if p.get("finished", True) else " training")


def load_run(path: Path, color: str) -> Run:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in ("history", "games", "snapshots", "baselines") if k not in data]
    if missing or (data["history"] and "eval" not in data["history"][0]):
        raise RunFormatError(f"{path} was saved before runs recorded {', '.join(missing) or 'evaluation metrics'}; "
                             f"retrain it with ttr-train-linear")
    return Run(name=Path(path).stem, path=Path(path), data=data, color=color)


def expand(patterns: Sequence) -> List[Path]:
    """Paths from the command line, with wildcards expanded here: PowerShell
    passes `*.json` through unexpanded. A pattern that matches nothing yet is
    dropped (a live view picks its files up once they exist)."""
    out: List[Path] = []
    for pat in patterns:
        text = str(pat)
        if any(ch in text for ch in "*?["):
            out.extend(sorted(Path(m) for m in glob.glob(text)))
        else:
            out.append(Path(text))
    return list(dict.fromkeys(out))


def _mean_tree(items: Sequence):
    """Elementwise mean of equally shaped JSON values (dicts, lists, numbers);
    anything else is taken from the first."""
    first = items[0]
    if isinstance(first, dict):
        return {k: _mean_tree([it[k] for it in items]) for k in first if all(k in it for it in items)}
    if isinstance(first, list):
        n = min(len(it) for it in items)
        return [_mean_tree([it[i] for it in items]) for i in range(n)]
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return float(sum(items) / len(items))  # not np.mean: millions of calls for per-game rows
    return first


def group_runs(runs: Sequence[Run]) -> List[Run]:
    """Merge runs named NAME_s<seed> into one run NAME (n seeds): every recorded
    number is the mean over the seeds, game by game and evaluation by evaluation
    (all seeds evaluate on the same games). Other runs pass through."""
    groups: Dict[str, List[Run]] = {}
    for run in runs:
        groups.setdefault(re.sub(r"_s\d+$", "", run.name), []).append(run)
    out = []
    for i, (name, members) in enumerate(groups.items()):
        if len(members) == 1:
            out.append(Run(members[0].name, members[0].path, members[0].data, SERIES[i % len(SERIES)]))
            continue
        data = {k: _mean_tree([m.data[k] for m in members])
                for k in ("history", "games", "snapshots", "baselines", "weights") if all(k in m.data for m in members)}
        for k in ("method", "features"):
            if k in members[0].data:
                data[k] = members[0].data[k]
        data["config"] = dict(members[0].data.get("config", {}),
                              seed=", ".join(str(m.config.get("seed")) for m in members))
        data["eval_games"] = members[0].data.get("eval_games", "?")
        progress = [m.data.get("progress") for m in members if m.data.get("progress")]
        if progress:
            data["progress"] = {"games": min(p["games"] for p in progress), "of": progress[0]["of"],
                                "finished": all(p.get("finished", True) for p in progress)}
        bests = [m.data.get("best") for m in members if m.data.get("best")]
        if bests:
            data["best"] = {"games": "mean", "margin_vs_greedy": float(np.mean([b["margin_vs_greedy"] for b in bests]))}
        out.append(Run(f"{name} x{len(members)}", members[0].path, data, SERIES[i % len(SERIES)]))
    return out


def start_polling(dash: "Dashboard", seconds: float):
    """Call `dash.poll()` every `seconds` on the figure's timer, and return the
    timer. matplotlib drops a timer callback that returns False or 0, which
    poll() does whenever nothing changed, so the callback returns None."""

    def tick() -> None:
        dash.poll()

    timer = dash.fig.canvas.new_timer(interval=int(seconds * 1000))
    timer.add_callback(tick)
    timer.start()
    return timer


def rolling(ys: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean; the first points average what exists so far."""
    if len(ys) == 0:
        return ys
    c = np.cumsum(np.insert(ys, 0, 0.0))
    idx = np.arange(1, len(ys) + 1)
    lo = np.maximum(0, idx - window)
    return (c[idx] - c[lo]) / (idx - lo)


def window_for(n: int) -> int:
    return max(10, n // 40)


# ------------------------------------------------------------------ drawing


def style_axes(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=4)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def draw_table(ax, header: Sequence[str], rows: Sequence[Sequence[str]], title: str = "",
               col_widths: Optional[Sequence[float]] = None, font: float = 7.5,
               swatches: Optional[Sequence[Optional[str]]] = None, row_height: Optional[float] = None) -> None:
    """A plain table at the top of `ax`: header in secondary ink (wrapped to two
    lines), zebra rows. `row_height` (axes fraction) keeps short tables from
    stretching; None fills the axes. `swatches` underlines header cells in a
    run's color (identity for run columns)."""
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=4)
    if not rows:
        ax.text(0, 1, "no data", color=INK_2, fontsize=font, va="top", transform=ax.transAxes)
        return
    height = 1.0 if row_height is None else min(1.0, (len(rows) + 1.6) * row_height)
    header = [textwrap.fill(h, 11) for h in header]
    table = ax.table(cellText=[list(r) for r in rows], colLabels=list(header), loc="upper left",
                     cellLoc="right", colLoc="right", colWidths=col_widths, bbox=[0, 1 - height, 1, height])
    table.auto_set_font_size(False)
    table.set_fontsize(font)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(SURFACE)
        cell.set_linewidth(0)
        if c == 0:
            cell._loc = "left"
            cell.get_text().set_ha("left")
        if r == 0:
            cell.set_facecolor(SURFACE)
            cell.get_text().set_color(INK_2)
            cell.get_text().set_fontweight("bold")
            if swatches and c < len(swatches) and swatches[c]:
                cell.set_edgecolor(swatches[c])
                cell.visible_edges = "B"
                cell.set_linewidth(2.5)
        else:
            cell.set_facecolor("#f4f3f0" if r % 2 == 0 else SURFACE)
            cell.get_text().set_color(INK)


class Dashboard:
    """Pages drawn into one figure; `show()` opens the window, `save_all()` writes PNGs."""

    PAGE_NAMES = ["Overview", "Behavior", "Training games", "Action mix", "Weights", "Evaluations", "Games"]
    GAMES_PER_PAGE = 32

    def __init__(self, runs: Sequence[Run], figure=None, opponent: str = "greedy",
                 loader: Optional[Callable[[], List[Run]]] = None, sources: Sequence[Path] = ()) -> None:
        """`loader` and `sources` make it live: `poll()` reloads the runs when a
        source file changed. A live dashboard may start with no runs yet."""
        import matplotlib.pyplot as plt

        if not runs and loader is None:
            raise ValueError("no runs to show")
        self.runs = list(runs)
        self.loader = loader
        self.sources = [str(p) for p in sources]  # paths or wildcard patterns
        self._stamps = self._source_stamps()
        self.updated = ""
        self.fig = figure if figure is not None else plt.figure(figsize=(19.2, 10.8), dpi=100)
        self.fig.set_facecolor(SURFACE)
        self.page = 0
        self.opponent = opponent
        self.run_index = 0
        self.games_page = 0
        self.pages: List[Callable[[], None]] = [
            self.page_overview, self.page_behavior, self.page_training, self.page_actions,
            self.page_weights, self.page_evaluations, self.page_games,
        ]

    def opponents(self) -> List[str]:
        """Evaluation opponents every shown run has: greedy and random first, then
        the others in the order the runs recorded them (DQN runs add wary, racer,
        linear)."""
        seen = [set(r.history[0]["eval"]) for r in self.runs if r.history]
        if not seen:
            return list(OPPONENTS)
        common = set.intersection(*seen)
        first = next(r for r in self.runs if r.history).history[0]["eval"]
        return [o for o in OPPONENTS if o in common] + [o for o in first if o in common and o not in OPPONENTS]

    @property
    def run(self) -> Run:
        return self.runs[self.run_index]

    # ------------------------------------------------------------ frame

    def draw(self) -> None:
        self.fig.clear()
        self.fig.set_facecolor(SURFACE)
        if self.runs:
            self.pages[self.page]()
        else:
            self.fig.text(0.5, 0.5, "waiting for " + ", ".join(self.sources)
                          + "\n(start ttr-train-linear or ttr-train-dqn with --live and the same --out)",
                          ha="center", va="center", fontsize=12, color=INK_2)
        names = "   ".join(f"{i + 1} {n}" for i, n in enumerate(self.PAGE_NAMES))
        runs = ", ".join(r.name + (f" [{r.progress}]" if self.loader and r.progress else "") for r in self.runs)
        self.fig.text(0.008, 0.992, f"{self.PAGE_NAMES[self.page]}  ·  runs: {runs}  ·  evaluation opponent: "
                      f"{self.opponent}", fontsize=11, color=INK, va="top", fontweight="bold")
        if self.loader is not None:
            self.fig.text(0.992, 0.992, f"LIVE · updated {self.updated or '—'}", ha="right", va="top",
                          fontsize=9, color=INK_2)
        self.fig.text(0.008, 0.006, f"{names}      ←/→ page   o opponent   r run   PgUp/PgDn games   "
                      f"s save PNG   f full screen   q quit", fontsize=8, color=INK_2, va="bottom")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------ live

    def _source_stamps(self) -> Tuple:
        """(path, mtime, size) of every file the sources name now; a new file
        matching a wildcard changes it too."""
        stamps = []
        for p in expand(self.sources):
            try:
                st = p.stat()
                stamps.append((str(p), st.st_mtime_ns, st.st_size))
            except OSError:
                stamps.append((str(p), None))
        return tuple(stamps)

    def poll(self) -> bool:
        """Reload and redraw if a source file changed. Returns True if it did.
        A file caught mid-replace just waits for the next poll."""
        if self.loader is None:
            return False
        stamps = self._source_stamps()
        if stamps == self._stamps:
            return False
        try:
            runs = self.loader()
        except (ValueError, OSError):  # JSONDecodeError is a ValueError
            return False
        self._stamps = stamps
        self.runs = runs
        self.run_index = min(self.run_index, max(0, len(runs) - 1))
        import time as _time

        self.updated = _time.strftime("%H:%M:%S")
        self.draw()
        return True

    def grid(self, rows: int, cols: int, **kw):
        kw.setdefault("hspace", 0.55)
        kw.setdefault("wspace", 0.28)
        left = kw.pop("left", 0.04)
        return self.fig.add_gridspec(rows, cols, left=left, right=0.99, top=0.9, bottom=0.06, **kw)

    def legend(self, handles, labels, y: float = 0.962) -> None:
        self.fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.008, y), ncol=min(len(labels), 8),
                        frameon=False, fontsize=8, labelcolor=INK_2, handlelength=2.6)

    def bot_lines(self, ax, key: str) -> None:
        """Horizontal reference lines: the bots' final-evaluation value for `key`
        against the current opponent (from the first run that has them)."""
        base = self.runs[0].data.get("baselines", {})
        for bot, style in BOT_STYLE.items():
            value = base.get(bot, {}).get(self.opponent, {}).get(key)
            if value is not None:
                ax.axhline(value, **style, zorder=1)

    def series_handles(self, with_bots: bool = True):
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], color=r.color, linewidth=2) for r in self.runs]
        labels = [r.name for r in self.runs]
        if with_bots:
            for bot, style in BOT_STYLE.items():
                handles.append(Line2D([], [], **style))
                labels.append(f"{bot} bot vs {self.opponent}")
        return handles, labels

    # ------------------------------------------------------------ pages

    def page_overview(self) -> None:
        gs = self.grid(3, 3, width_ratios=[1, 1, 1.15])
        opp = self.opponent
        charts = [
            (gs[0, 0], f"evaluation win share vs {opp}", lambda r: r.eval_series(opp, "win_share"), "win_share", True),
            (gs[0, 1], f"evaluation margin vs {opp}", lambda r: r.eval_series(opp, "margin"), "margin", True),
            (gs[1, 0], "training margin per game (rolling mean)", lambda r: self._rolled(r, "margin"), None, False),
            (gs[1, 1], "mean |TD error| per game (rolling mean)", lambda r: self._rolled(r, "mean_abs_td"), None, False),
            (gs[2, 0], "exploration epsilon", lambda r: r.game_series("epsilon"), None, False),
            (gs[2, 1], "decisions per game (rolling mean)", lambda r: self._rolled(r, "decisions"), None, False),
        ]
        for spec, title, series, key, markers in charts:
            ax = self.fig.add_subplot(spec)
            style_axes(ax, title)
            for run in self.runs:
                xs, ys = series(run)
                ax.plot(xs, ys, color=run.color, linewidth=2, marker="o" if markers else None, markersize=4,
                        markeredgecolor=SURFACE, markeredgewidth=1)
            if key:
                self.bot_lines(ax, key)
            if key == "win_share":
                ax.set_ylim(-0.03, 1.03)
                ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.set_xlabel("training games", fontsize=7.5, color=INK_2)
        self.legend(*self.series_handles())

        # Final evaluation: every run, then the bots, on the same games.
        cols = [(r.name, r.final(opp), r.color) for r in self.runs]
        base = self.runs[0].data.get("baselines", {})
        cols += [(f"{bot} bot", base[bot][opp], None) for bot in ("greedy", "random") if opp in base.get(bot, {})]
        rows = [[label(k)] + [fmt(k, c[1][k]) if k in c[1] else "—" for c in cols] for k in SUMMARY]
        draw_table(self.fig.add_subplot(gs[0:2, 2]), ["metric"] + [c[0] for c in cols], rows,
                   title=f"final evaluation vs {opp} ({self._eval_games()} games each)",
                   swatches=[None] + [c[2] for c in cols], col_widths=[1.5] + [1] * len(cols))

        cfg_rows = [["method"] + [r.config.get("algo", r.data.get("method", "")) for r in self.runs]]
        keys = [k for m in dict.fromkeys(r.data.get("method", "linear") for r in self.runs)
                for k in CONFIG_KEYS.get(m, ())]
        for k in dict.fromkeys(keys):
            cfg_rows.append([k] + [str(r.config.get(k, "")) for r in self.runs])
        cfg_rows.append(["training time"] + [f"{r.history[-1]['seconds']:.0f} s" if r.history else "—"
                                              for r in self.runs])
        cfg_rows.append(["best vs greedy"] + [
            f"{r.data['best']['games']}: {r.data['best']['margin_vs_greedy']:+.1f}"
            if r.data.get("best") else "" for r in self.runs])
        draw_table(self.fig.add_subplot(gs[2, 2]), ["setting"] + [r.name for r in self.runs], cfg_rows,
                   title="run settings", swatches=[None] + [r.color for r in self.runs])

    def _rolled(self, run: Run, key: str):
        xs, ys = run.game_series(key)
        return xs, rolling(ys, window_for(len(ys)))

    def _eval_games(self) -> str:
        return str(self.runs[0].data.get("eval_games", "?"))

    def _multiples(self, keys: Sequence[str], series: Callable[[Run, str], Tuple[np.ndarray, np.ndarray]],
                   bots: bool, markers: bool, raw: Optional[Callable[[Run, str], Tuple[np.ndarray, np.ndarray]]] = None
                   ) -> None:
        cols = 4
        rows = (len(keys) + cols - 1) // cols
        gs = self.grid(rows, cols, hspace=0.62)
        for i, key in enumerate(keys):
            ax = self.fig.add_subplot(gs[i // cols, i % cols])
            style_axes(ax, label(key))
            if raw is not None and len(self.runs) == 1:
                xs, ys = raw(self.runs[0], key)
                ax.plot(xs, ys, color=self.runs[0].color, alpha=0.12, linewidth=0.6)
            for run in self.runs:
                xs, ys = series(run, key)
                ax.plot(xs, ys, color=run.color, linewidth=2, marker="o" if markers else None, markersize=3.5,
                        markeredgecolor=SURFACE, markeredgewidth=0.8)
            if bots:
                self.bot_lines(ax, key)
            fmt_spec = (FORMATS.get(key) or EXTRA_FORMATS.get(key) or ("", ""))[1]
            if "%" in fmt_spec:
                ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        self.legend(*self.series_handles(with_bots=bots))

    def page_behavior(self) -> None:
        self._multiples(BEHAVIOR, lambda r, k: r.eval_series(self.opponent, k), bots=True, markers=True)
        self.fig.text(0.99, 0.992, f"greedy play, {self._eval_games()} games per evaluation", ha="right",
                      va="top", fontsize=8, color=INK_2)

    def page_training(self) -> None:
        self._multiples(TRAINING, lambda r, k: self._rolled(r, k), bots=False, markers=False,
                        raw=lambda r, k: r.game_series(k))
        w = window_for(len(self.runs[0].games))
        self.fig.text(0.99, 0.992, f"training games, exploring; rolling mean over {w} games (faint: every game)",
                      ha="right", va="top", fontsize=8, color=INK_2)

    def page_actions(self) -> None:
        run = self.run
        gs = self.grid(1, 2, width_ratios=[1.6, 1], wspace=0.22)
        ax = self.fig.add_subplot(gs[0, 0])
        style_axes(ax, f"{run.name}: share of turn actions per training game (rolling mean)")
        xs, _ = run.game_series("claims")
        counts = np.array([rolling(run.game_series(k)[1], window_for(len(xs))) for k, _ in ACTIONS])
        shares = counts / np.maximum(counts.sum(axis=0), 1e-9)
        colors = SERIES[:len(ACTIONS)]
        polys = ax.stackplot(xs, shares, colors=colors, edgecolor=SURFACE, linewidth=1)
        ax.set_ylim(0, 1)
        if len(xs) > 1:
            ax.set_xlim(xs[0], xs[-1])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel("training games", fontsize=7.5, color=INK_2)
        self.legend(polys, [n for _, n in ACTIONS])

        # Final per-game counts against the bots, same games.
        ax = self.fig.add_subplot(gs[0, 1])
        style_axes(ax, f"actions per game, final evaluation vs {self.opponent}")
        base = run.data.get("baselines", {})
        groups = [(run.name, run.final(self.opponent), run.color)]
        groups += [(f"{b} bot", base[b][self.opponent], BOT_STYLE[b]["color"]) for b in ("greedy", "random")
                   if self.opponent in base.get(b, {})]
        groups = [g for g in groups if g[1]]  # a live run has no evaluation yet
        if not groups:
            ax.text(0.5, 0.5, "no evaluation yet", ha="center", va="center", color=INK_2, transform=ax.transAxes)
            return
        y = np.arange(len(ACTIONS))
        h = 0.8 / len(groups)
        for i, (name, metrics, color) in enumerate(groups):
            vals = [metrics[k] for k, _ in ACTIONS]
            bars = ax.barh(y + i * h - 0.4 + h / 2, vals, height=h * 0.86, color=color, label=name)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {v:.1f}", va="center",
                        fontsize=7, color=INK_2)
        ax.set_yticks(y, [n for _, n in ACTIONS])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_2)

    def page_weights(self) -> None:
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

        run = self.run
        if "weights" not in run.data:
            self.fig.text(0.5, 0.5, f"{run.name}: no linear weights to show ({run.data.get('method', '?')} run; "
                          f"its networks are in {run.data.get('weights_file', 'the .pt file')})",
                          ha="center", va="center", fontsize=12, color=INK_2)
            return
        feats: Dict[str, List[str]] = run.data["features"]
        weights: Dict[str, List[float]] = run.data["weights"]
        all_blocks = list(feats)
        if "value" in feats:
            # Value + advantage (second pass): one shared state row, then each
            # type's bias and own features.
            types = [t for t in all_blocks if t != "value"]
            state_rows, state_names = ["value (shared)"], feats["value"]
            state = np.array([weights["value"]])
            own = {t: (feats[t], weights[t]) for t in types}
            state_title = f"{run.name}: state value weights (shared by every action)"
        else:
            # First pass: every type carries its own copy of the state features.
            types = all_blocks
            n_state = _shared_prefix([feats[t] for t in types])
            state_rows, state_names = types, feats[types[0]][:n_state]
            state = np.array([weights[t][:n_state] for t in types])
            own = {t: (feats[t][n_state:], weights[t][n_state:]) for t in types}
            state_title = f"{run.name}: state-feature weights (shared names, one row per action type)"
        n_state = len(state_names)
        width = max(len(own[t][0]) for t in types)
        action = np.full((len(types), max(width, 1)), np.nan)
        for i, t in enumerate(types):
            action[i, :len(own[t][1])] = own[t][1]
        lim = float(np.nanmax(np.abs(np.concatenate([state.ravel(), action[~np.isnan(action)]])))) or 1.0
        cmap = LinearSegmentedColormap.from_list("div", DIVERGING).with_extremes(bad=SURFACE)
        norm = TwoSlopeNorm(0.0, -lim, lim)

        gs = self.grid(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.05], hspace=0.35, wspace=0.12, left=0.07)
        ax = self.fig.add_subplot(gs[0, 0])
        style_axes(ax, state_title)
        ax.grid(False)
        ax.imshow(state, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(n_state), state_names, rotation=30, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(state_rows)), state_rows, fontsize=7.5)
        for i in range(len(state_rows)):
            for j in range(n_state):
                ax.text(j, i, f"{state[i, j]:+.2f}", ha="center", va="center", fontsize=6.5, color=INK)

        ax = self.fig.add_subplot(gs[0, 1])
        style_axes(ax, "advantage weights (each type's bias and own features)" if "value" in feats
                   else "action-feature weights (each type's own features)")
        ax.grid(False)
        ax.imshow(action, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks(range(len(types)), types, fontsize=7.5)
        for i, t in enumerate(types):
            for j, name in enumerate(own[t][0]):
                ax.text(j, i, f"{name}\n{action[i, j]:+.3f}", ha="center", va="center", fontsize=6, color=INK)

        # Trajectories of the weights that moved most over training.
        ax = self.fig.add_subplot(gs[1, :])
        snaps = run.data["snapshots"]
        style_axes(ax, "weights that moved most over training (at each evaluation)")
        if snaps:
            xs = np.array([s["games"] for s in snaps], dtype=float)
            series = {}
            for t in all_blocks:
                for j, name in enumerate(feats[t]):
                    series[f"{t}.{name}"] = np.array([s["weights"][t][j] for s in snaps])
            top = sorted(series, key=lambda k: -abs(series[k][-1] - series[k][0]))[:8]
            for color, key in zip(SERIES, top):
                ax.plot(xs, series[key], color=color, linewidth=2, marker="o", markersize=3.5,
                        markeredgecolor=SURFACE, markeredgewidth=0.8, label=key)
            ax.axhline(0, color=INK_3, linewidth=0.8)
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, frameon=False, fontsize=8,
                      labelcolor=INK_2)
            ax.set_xlabel("training games", fontsize=7.5, color=INK_2)

    def page_evaluations(self) -> None:
        run = self.run
        keys = ["win_share", "margin", "score", "route_points", "ticket_points", "tickets_completed",
                "tickets_failed", "claims", "mean_claim_length", "ticket_draws", "final_round_ticket_draws"]
        header = ["games", "epsilon", "seconds", "train win", "train margin", "|TD|"] + \
                 [f"{label(k)}" for k in keys]
        rows = []
        for e in run.history:
            ev = e["eval"][self.opponent]
            t = e["train"]
            rows.append([f"{e['games']}", f"{e['epsilon']:.3f}", f"{e['seconds']:.0f}", f"{t['won']:.0%}",
                         f"{t['margin']:+.1f}", f"{t['mean_abs_td']:.4f}"] + [fmt(k, ev[k]) for k in keys])
        gs = self.grid(1, 1)
        draw_table(self.fig.add_subplot(gs[0, 0]), header, rows, font=8, row_height=0.04,
                   title=f"{run.name}: every evaluation (eval columns: greedy play vs {self.opponent})",
                   col_widths=[0.7] * 6 + [1] * len(keys))

    def page_games(self) -> None:
        run = self.run
        keys = ["game", "epsilon", "seat", "won", "margin", "score", "opp_score", "route_points", "ticket_points",
                "tickets_completed", "tickets_failed", "tickets_kept", "claims", "mean_claim_length",
                "ticket_draws", "final_round_ticket_draws", "draw_blind", "draw_color", "trains_left",
                "decisions", "mean_abs_td"]
        n = len(run.games)
        pages = max(1, (n + self.GAMES_PER_PAGE - 1) // self.GAMES_PER_PAGE)
        self.games_page = min(self.games_page, pages - 1)
        lo = self.games_page * self.GAMES_PER_PAGE
        chunk = run.games[lo:lo + self.GAMES_PER_PAGE]
        rows = [[fmt(k, r[k]) if k in r else "" for k in keys] for r in chunk]
        short = {"final_round_ticket_draws": "final-rd ticket draws", "mean_claim_length": "mean claim len",
                 "tickets_completed": "tix done", "tickets_failed": "tix failed", "tickets_kept": "tix kept",
                 "mean_abs_td": "|TD|", "decisions": "decisions", "draw_color": "color draws",
                 "draw_blind": "blind draws", "opp_score": "opp score", "route_points": "route pts",
                 "ticket_points": "ticket pts", "trains_left": "trains left", "won": "won"}
        header = [short.get(k, label(k)) for k in keys]
        gs = self.grid(1, 1)
        draw_table(self.fig.add_subplot(gs[0, 0]), header, rows, font=7.5, row_height=0.029,
                   title=f"{run.name}: training games {lo + 1}-{lo + len(chunk)} of {n}  "
                         f"(page {self.games_page + 1}/{pages}; PgUp/PgDn, Home/End)")

    # ------------------------------------------------------------ control

    def on_key(self, event) -> None:
        key = event.key
        if key == "right":
            self.page = (self.page + 1) % len(self.pages)
        elif key == "left":
            self.page = (self.page - 1) % len(self.pages)
        elif key and key.isdigit() and 1 <= int(key) <= len(self.pages):
            self.page = int(key) - 1
        elif key == "o":
            opponents = self.opponents()
            at = opponents.index(self.opponent) + 1 if self.opponent in opponents else 0
            self.opponent = opponents[at % len(opponents)]
        elif key == "r":
            self.run_index = (self.run_index + 1) % len(self.runs)
            self.games_page = 0
        elif key == "pagedown":
            self.games_page += 1
        elif key == "pageup":
            self.games_page = max(0, self.games_page - 1)
        elif key == "home":
            self.games_page = 0
        elif key == "end":
            self.games_page = 10 ** 9
        elif key == "s":
            path = Path(f"dash_{self.PAGE_NAMES[self.page].lower().replace(' ', '_')}_{self.opponent}.png")
            self.fig.savefig(path, facecolor=SURFACE)
            print(f"saved {path.resolve()}")
            return
        else:
            return
        self.draw()

    def save_all(self, out_dir: Path) -> List[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, name in enumerate(self.PAGE_NAMES):
            self.page = i
            self.draw()
            path = out_dir / f"{i + 1}_{name.lower().replace(' ', '_')}_{self.opponent}.png"
            self.fig.savefig(path, facecolor=SURFACE)
            paths.append(path)
        return paths


def _shared_prefix(lists: Sequence[Sequence[str]]) -> int:
    n = 0
    for names in zip(*lists):
        if len(set(names)) != 1:
            break
        n += 1
    return n


# ------------------------------------------------------------------ main


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="training run JSON files (ttr-train-linear --out); "
                                                "wildcards like runs/linear/pass3/*.json are expanded here")
    parser.add_argument("--opponent", default="greedy",
                        help="evaluation opponent shown first (greedy, random, or any other the runs evaluated against)")
    parser.add_argument("--page", type=int, default=1, help="page to open on (1-7)")
    parser.add_argument("--save", type=Path, metavar="DIR", help="write every page as PNG to DIR and exit")
    parser.add_argument("--windowed", action="store_true", help="don't start full screen")
    parser.add_argument("--group", action="store_true",
                        help="average runs named NAME_s<seed> into one line per NAME")
    parser.add_argument("--live", type=float, nargs="?", const=3.0, default=0.0, metavar="SECONDS",
                        help="reread the files every SECONDS (default 3) and redraw when they change; "
                             "for runs training with ttr-train-linear --live")
    args = parser.parse_args(argv)
    if args.live and args.save:
        parser.error("--live and --save don't combine")

    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    for key in list(matplotlib.rcParams):
        if key.startswith("keymap.") and key not in ("keymap.fullscreen", "keymap.quit"):
            matplotlib.rcParams[key] = []
    import matplotlib.pyplot as plt

    parsed: Dict[Path, Tuple[Tuple[int, int], Run]] = {}  # path -> ((mtime, size), run): parse only what changed

    def load_all() -> List[Run]:
        paths = expand(args.runs)
        if args.live:
            paths = [p for p in paths if p.exists()]
        elif not paths:
            sys.exit(f"no run files match {' '.join(map(str, args.runs))}")
        loaded = []
        for i, p in enumerate(paths):
            st = p.stat()
            stamp = (st.st_mtime_ns, st.st_size)
            if p not in parsed or parsed[p][0] != stamp:
                parsed[p] = (stamp, load_run(p, SERIES[0]))
            run = parsed[p][1]
            loaded.append(Run(run.name, run.path, run.data, SERIES[i % len(SERIES)]))
        return group_runs(loaded) if args.group else loaded

    try:
        runs = load_all()
    except (RunFormatError, OSError, ValueError) as e:
        if not args.live:
            sys.exit(str(e))
        runs = []  # caught mid-write; the first poll loads it
    if len(runs) > len(SERIES):
        parser.error(f"at most {len(SERIES)} runs at once (--group averages seeds)")
    live = {"loader": load_all, "sources": args.runs} if args.live else {}
    dash = Dashboard(runs, opponent=args.opponent, **live)
    dash.page = max(0, min(len(dash.pages), args.page) - 1)
    if args.save:
        for opp in dash.opponents():
            dash.opponent = opp
            for path in dash.save_all(args.save):
                print(path)
        return
    dash.fig.canvas.manager.set_window_title("Ticket to Ride — agent analysis")
    dash.fig.canvas.mpl_connect("key_press_event", dash.on_key)
    dash.draw()
    if args.live:
        dash._timer = start_polling(dash, args.live)  # keep a reference, or it is garbage-collected
    if not args.windowed:
        try:
            dash.fig.canvas.manager.full_screen_toggle()
        except AttributeError:
            pass
    plt.show()


if __name__ == "__main__":
    main()
