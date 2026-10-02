"""Evaluation dashboard: one matplotlib window over the per-game CSVs that
`scripts/eval_agents.py` writes (MCTS passes, reference agents). `ttr-dash` opens it
when given CSV files instead of training runs.

    ttr-dash runs/mcts/pass2.csv runs/mcts/pass1.csv runs/mcts/pass1_reference.csv --live
    ttr-dash "runs/mcts/*.csv" --control "mcts:reward=score"
    ttr-dash runs/mcts/pass1.csv --save runs/mcts/dash          # every page as PNG, no window

Pages (keys 1-4 or left/right):

    1 Strength   per opponent, every agent's mean margin with a 95% interval, win share, games
    2 Paired     per opponent, every agent minus the control agent on the same games
                 (same batch seed and game number), mean and 95% interval; `c` picks the control
    3 Behavior   for one opponent (`o` cycles), tickets, claims, tempo and score per agent
    4 Progress   games, time a game and per search for every agent and opponent, and the latest
                 progress line of each CSV's OUT.log with an estimated finish

    o  next opponent (page 3)     c  next control agent (page 2)
    s  save this page as PNG      f  full screen      q  quit

A game in several files counts once (the last file given wins). `--live [SECONDS]`
rereads the files when they change. Needs the `[analysis]` extra (matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ttr.learn.dashboard import (GRID, INK, INK_2, INK_3, SERIES, SURFACE, draw_table, expand, start_polling,
                                 style_axes)

TEXT_KEYS = ("agent", "opponent")
BEHAVIOR = [("tickets_kept", "tickets kept", "{:.2f}"), ("tickets_completed", "tickets completed", "{:.2f}"),
            ("tickets_failed", "tickets failed", "{:.2f}"), ("ticket_draws", "ticket draws", "{:.2f}"),
            ("claims", "claims", "{:.1f}"), ("mean_claim_length", "mean claim length", "{:.2f}"),
            ("triggered_end", "ended the game", "{:.0%}"), ("longest_bonus", "longest-path bonus", "{:.0%}"),
            ("score", "score", "{:.0f}"), ("opp_score", "opponent score", "{:.0f}"),
            ("turns", "turns (all seats)", "{:.0f}"), ("win_share", "win share", "{:.0%}")]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
BOTS = ("random", "greedy", "wary", "racer", "collector")  # opponents shown first, in this order
MCTS_KEYS = {"reward": "{}", "opponent": "opp {}", "iterations": "{} it", "prior": "prior {}", "steps": "steps {}",
             "rollout": "roll {}", "c": "c {}", "memory": "memory {}", "known_tickets": "known tickets {}"}


def short(spec: str) -> str:
    """A compact agent name: a run file by its name ("ppo p2b_pool_s3", "linear p7b…s3@best"),
    MCTS by its options ("mcts score · opp racer")."""
    kind, _, arg = spec.partition(":")
    if kind in ("ppo", "dqn", "linear") and arg:
        best = arg.endswith("@best")
        stem = Path(arg.removesuffix("@best")).stem
        if len(stem) > 14:
            stem = re.sub(r"^([^_]+)_.*_(s\d+)$", lambda m: f"{m[1]}…{m[2]}", stem)
        return f"{kind} {stem}" + ("@best" if best else "")
    if kind == "mcts":
        parts = []
        for item in filter(None, arg.split(",")):
            key, _, value = item.partition("=")
            parts.append(MCTS_KEYS[key].format(value) if key in MCTS_KEYS else item)
        return " · ".join(["mcts " + parts[0]] + parts[1:]) if parts else "mcts"
    return spec


def read_csvs(paths: Sequence[Path]) -> List[dict]:
    """Rows of every file, numbers as floats; a game in several files counts once (the last wins)."""
    rows: Dict[tuple, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                row = {k: (v if k in TEXT_KEYS else float(v)) for k, v in r.items() if v != ""}
                rows[row["agent"], row["opponent"], row.get("players", 2), row.get("seed", 0), row["game"]] = row
    return list(rows.values())


def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and the half-width of a normal 95% interval (0 for one value)."""
    m = st.mean(values)
    return m, (1.96 * st.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0)


class EvalData:
    def __init__(self, rows: Sequence[dict]) -> None:
        self.rows = list(rows)
        self.agents: List[str] = list(dict.fromkeys(r["agent"] for r in self.rows))
        seen = list(dict.fromkeys(r["opponent"] for r in self.rows))
        self.opponents: List[str] = sorted(seen, key=lambda o: (BOTS.index(o) if o in BOTS else len(BOTS),
                                                                seen.index(o)))
        self.by_pair: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for r in self.rows:
            self.by_pair[r["agent"], r["opponent"]].append(r)

    def games(self, agent: str, opponent: str) -> List[dict]:
        return self.by_pair.get((agent, opponent), [])

    def paired(self, agent: str, control: str, opponent: str) -> List[float]:
        """Per shared game (same players, batch seed and game number): agent's margin minus the control's."""
        base = {(r.get("players"), r.get("seed"), r["game"]): r["margin"] for r in self.games(control, opponent)}
        return [r["margin"] - base[k] for r in self.games(agent, opponent)
                if (k := (r.get("players"), r.get("seed"), r["game"])) in base]


class EvalDashboard:
    PAGE_NAMES = ["Strength", "Paired", "Behavior", "Progress"]

    def __init__(self, data: EvalData, figure=None, control: Optional[str] = None,
                 sources: Sequence[str] = (), live: bool = False) -> None:
        import matplotlib.pyplot as plt

        self.data = data
        self.sources = [str(s) for s in sources]
        self.live = live
        self.updated = ""
        self._stamps = self._source_stamps()
        self.fig = figure if figure is not None else plt.figure(figsize=(19.2, 10.8), dpi=100)
        self.page = 0
        self.opponent_index = 0
        self.control = control
        self.pages = [self.page_strength, self.page_paired, self.page_behavior, self.page_progress]

    # ------------------------------------------------------------ identity

    def color(self, agent: str) -> str:
        return SERIES[self.data.agents.index(agent) % len(SERIES)]

    def marker(self, agent: str) -> str:
        return MARKERS[(self.data.agents.index(agent) // len(SERIES)) % len(MARKERS)]

    @property
    def opponent(self) -> Optional[str]:
        opps = self.data.opponents
        return opps[self.opponent_index % len(opps)] if opps else None

    def control_agent(self) -> Optional[str]:
        if self.control in self.data.agents:
            return self.control
        return self.data.agents[0] if self.data.agents else None

    # ------------------------------------------------------------ frame

    def draw(self) -> None:
        self.fig.clear()
        self.fig.set_facecolor(SURFACE)
        if self.data.rows:
            self.pages[self.page]()
        else:
            self.fig.text(0.5, 0.5, "waiting for games in " + ", ".join(self.sources), ha="center", va="center",
                          fontsize=12, color=INK_2)
        n = len(self.data.rows)
        self.fig.text(0.008, 0.992, f"{self.PAGE_NAMES[self.page]}  ·  {n} games  ·  "
                      f"{len(self.data.agents)} agents  ·  {len(self.data.opponents)} opponents",
                      fontsize=11, color=INK, va="top", fontweight="bold")
        if self.live:
            self.fig.text(0.992, 0.992, f"LIVE · updated {self.updated or '—'}", ha="right", va="top",
                          fontsize=9, color=INK_2)
        names = "   ".join(f"{i + 1} {p}" for i, p in enumerate(self.PAGE_NAMES))
        self.fig.text(0.008, 0.006, f"{names}      ←/→ page   o opponent   c control   s save PNG   "
                      f"f full screen   q quit", fontsize=8, color=INK_2, va="bottom")
        self.fig.canvas.draw_idle()

    def legend(self) -> None:
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], color=self.color(a), marker=self.marker(a), linestyle="", markersize=7)
                   for a in self.data.agents]
        self.fig.legend(handles, [short(a) for a in self.data.agents], loc="upper left",
                        bbox_to_anchor=(0.008, 0.965), ncol=min(4, len(handles)), frameon=False, fontsize=8.5,
                        labelcolor=INK_2, handletextpad=0.3)

    def opponent_grid(self):
        n = len(self.data.opponents)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        top = 0.86 if len(self.data.agents) <= 4 else 0.82
        gs = self.fig.add_gridspec(rows, cols, left=0.085, right=0.99, top=top, bottom=0.07, hspace=0.42,
                                   wspace=0.5)
        return [self.fig.add_subplot(gs[i // cols, i % cols]) for i in range(n)]

    def interval_panel(self, ax, opponent: str, values: Dict[str, List[float]], title: str, note) -> None:
        """One row per agent with values: mean and 95% interval; its note at the panel's right edge."""
        from matplotlib.transforms import blended_transform_factory

        style_axes(ax, title)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.grid(False, axis="y")
        agents = [a for a in self.data.agents if values.get(a)]
        for y, agent in enumerate(agents):
            m, h = mean_ci(values[agent])
            ax.errorbar([m], [y], xerr=[[h], [h]], color=self.color(agent), marker=self.marker(agent),
                        markersize=6, capsize=3, linewidth=1.6)
            ax.text(0.995, y, note(agent, m, h), transform=blended_transform_factory(ax.transAxes, ax.transData),
                    ha="right", va="center", fontsize=7.5, color=INK_2,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
        ax.axvline(0, color=INK_3, linewidth=1, zorder=1)
        ax.set_yticks(range(len(agents)), [short(a) for a in agents], fontsize=7.5)
        ax.set_ylim(len(agents) - 0.4, -0.6)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + (hi - lo) * 0.75)  # room for the notes
        if not agents:
            ax.text(0.5, 0.5, "no games", transform=ax.transAxes, ha="center", color=INK_2)

    # ------------------------------------------------------------ pages

    def page_strength(self) -> None:
        self.legend()
        for ax, opp in zip(self.opponent_grid(), self.data.opponents):
            values = {a: [r["margin"] for r in self.data.games(a, opp)] for a in self.data.agents}
            wins = {a: st.mean(r["win_share"] for r in self.data.games(a, opp)) for a in self.data.agents
                    if self.data.games(a, opp)}
            self.interval_panel(ax, opp, values, f"margin vs {short(opp)}",
                                lambda a, m, h: f"{m:+.1f} ± {h:.1f}  win {wins[a]:.0%}  n {len(values[a])}")

    def page_paired(self) -> None:
        control = self.control_agent()
        self.legend()
        self.fig.text(0.992, 0.965, f"control: {short(control)}   (c: next)", ha="right", va="top", fontsize=9.5,
                      color=INK)
        for ax, opp in zip(self.opponent_grid(), self.data.opponents):
            values = {a: self.data.paired(a, control, opp) for a in self.data.agents if a != control}
            self.interval_panel(ax, opp, values, f"minus control, same games vs {short(opp)}",
                                lambda a, m, h: f"{m:+.1f} ± {h:.1f}  n {len(values[a])}")

    def page_behavior(self) -> None:
        opp = self.opponent
        self.legend()
        self.fig.text(0.992, 0.965, f"vs {short(opp)}   (o: next opponent)", ha="right", va="top", fontsize=9.5,
                      color=INK)
        agents = [a for a in self.data.agents if self.data.games(a, opp)]
        top = 0.86 if len(self.data.agents) <= 4 else 0.82
        gs = self.fig.add_gridspec(3, 4, left=0.04, right=0.99, top=top, bottom=0.07, hspace=0.5, wspace=0.25)
        for i, (key, title, spec) in enumerate(BEHAVIOR):
            ax = self.fig.add_subplot(gs[i // 4, i % 4])
            style_axes(ax, title)
            ax.grid(False, axis="x")
            for x, agent in enumerate(agents):
                m, h = mean_ci([r[key] for r in self.data.games(agent, opp)])
                ax.bar([x], [m], color=self.color(agent), width=0.7, yerr=[h], ecolor=INK_3, capsize=2)
                ax.annotate(spec.format(m), (x, m), xytext=(0, 2), textcoords="offset points", ha="center",
                            va="bottom", fontsize=7, color=INK_2)
            ax.set_xticks([])
            if key in ("triggered_end", "longest_bonus", "win_share"):
                ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    def page_progress(self) -> None:
        gs = self.fig.add_gridspec(2, 1, left=0.04, right=0.99, top=0.93, bottom=0.07, hspace=0.12,
                                   height_ratios=[3, 1])
        rows = []
        for agent in self.data.agents:
            for opp in self.data.opponents:
                games = self.data.games(agent, opp)
                if not games:
                    continue
                searches = sum(r.get("searches", 0) for r in games)
                per_search = sum(r.get("search_seconds", 0) for r in games) / searches if searches else 0
                m, h = mean_ci([r["margin"] for r in games])
                rows.append([short(agent), short(opp), f"{len(games)}", f"{m:+.1f} ± {h:.1f}",
                             f"{st.mean(r['win_share'] for r in games):.0%}",
                             f"{st.mean(r.get('seconds', 0) for r in games) / 60:.1f}",
                             f"{searches / len(games):.0f}" if searches else "—",
                             f"{per_search:.2f}" if searches else "—"])
        draw_table(self.fig.add_subplot(gs[0]), ["agent", "opponent", "games", "margin (95%)", "win",
                                                 "min a game", "searches a game", "s a search"], rows,
                   title="games so far", col_widths=[0.34, 0.26, 0.06, 0.1, 0.05, 0.06, 0.07, 0.06],
                   row_height=0.045)
        ax = self.fig.add_subplot(gs[1])
        ax.axis("off")
        ax.set_title("latest progress line of each OUT.log", loc="left", fontsize=9.5, color=INK, pad=4)
        ax.text(0, 1, "\n".join(self.progress_lines()) or "no logs", transform=ax.transAxes, va="top",
                fontsize=8.5, color=INK_2, family="monospace")

    def progress_lines(self) -> List[str]:
        out = []
        for path in expand(self.sources):
            log = path.with_suffix(".log")
            if not log.exists():
                continue
            lines = log.read_text(encoding="utf-8").splitlines()
            last = next((x for x in reversed(lines) if " games, " in x or "finished" in x), "")
            done = re.match(r"(\d+)/(\d+) games, (\d+) min", last)
            eta = ""
            if done:
                d, total, minutes = map(int, done.groups())
                if d and d < total:
                    eta = f"  ·  about {minutes / d * (total - d):.0f} min to go in this batch"
            out.append(f"{log}: {last.split(':')[0] if done else last}{eta}")
        return out

    # ------------------------------------------------------------ live and keys

    def _source_stamps(self) -> tuple:
        stamps = []
        for p in expand(self.sources):
            for q in (p, p.with_suffix(".log")):
                try:
                    s = q.stat()
                    stamps.append((str(q), s.st_mtime_ns, s.st_size))
                except OSError:
                    stamps.append((str(q), None))
        return tuple(stamps)

    def poll(self) -> bool:
        stamps = self._source_stamps()
        if stamps == self._stamps:
            return False
        try:
            data = EvalData(read_csvs(expand(self.sources)))
        except (ValueError, OSError, KeyError):  # caught mid-write: the next poll reads it
            return False
        self._stamps = stamps
        self.data = data
        self.updated = time.strftime("%H:%M:%S")
        self.draw()
        return True

    def on_key(self, event) -> None:
        key = event.key
        if key in ("1", "2", "3", "4"):
            self.page = int(key) - 1
        elif key == "right":
            self.page = (self.page + 1) % len(self.pages)
        elif key == "left":
            self.page = (self.page - 1) % len(self.pages)
        elif key == "o":
            self.opponent_index += 1
        elif key == "c" and self.data.agents:
            agents = self.data.agents
            current = self.control_agent()
            self.control = agents[(agents.index(current) + 1) % len(agents)]
        elif key == "s":
            print(self.save_page(Path("runs/eval_dash")))
            return
        elif key == "q":
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            return
        else:
            return
        self.draw()

    def save_page(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.page + 1}_{self.PAGE_NAMES[self.page].lower()}.png"
        self.fig.savefig(path, facecolor=SURFACE)
        return path

    def save_all(self, folder: Path) -> List[Path]:
        paths = []
        for i in range(len(self.pages)):
            self.page = i
            self.draw()
            paths.append(self.save_page(folder))
        return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+", help="per-game CSVs from scripts/eval_agents.py (wildcards expanded here)")
    parser.add_argument("--control", help="agent spec the Paired page subtracts (default: the first agent)")
    parser.add_argument("--page", type=int, default=1, help="page to open on (1-4)")
    parser.add_argument("--save", type=Path, metavar="DIR", help="write every page as PNG to DIR and exit")
    parser.add_argument("--windowed", action="store_true", help="don't start full screen")
    parser.add_argument("--live", type=float, nargs="?", const=10.0, default=0.0, metavar="SECONDS",
                        help="reread the files every SECONDS (default 10) and redraw when they change")
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

    paths = expand(args.csvs)
    if not args.live and not any(p.exists() for p in paths):
        sys.exit(f"no CSV files match {' '.join(args.csvs)}")
    dash = EvalDashboard(EvalData(read_csvs(paths)), control=args.control, sources=args.csvs,
                         live=bool(args.live))
    dash.page = max(0, min(len(dash.pages), args.page) - 1)
    if args.save:
        for path in dash.save_all(args.save):
            print(path)
        return
    dash.fig.canvas.manager.set_window_title("Ticket to Ride — evaluations")
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
