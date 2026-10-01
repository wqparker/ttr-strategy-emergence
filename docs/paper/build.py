"""Build the research paper page (docs/paper/paper.html) from data.json and template.html.

    python docs/paper/compile_data.py      # runs and summaries -> data.json
    python docs/paper/measure.py ...       # instrumented games -> measured.json (merged into data.json)
    python docs/paper/build.py             # -> paper.html

Everything the page draws or quotes as a run result comes from here: chart series,
tables, and the numbers bound into the text (`data-v` spans), so a rebuild after more
training updates them all.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
SET = DATA["settings"]
MEASURED = HERE / "measured.json"  # read directly: measurements can be newer than data.json
MEAS = json.loads(MEASURED.read_text(encoding="utf-8")) if MEASURED.exists() else DATA.get("measured", {})
OPPS = ("greedy", "wary", "racer", "collector")


def sgn(v, d=1):
    if v is None:
        return "–"
    s = f"{abs(v):.{d}f}"
    return ("−" if v < 0 else "+" if v > 0 else "") + s if round(v, d) != 0 else f"{0:.{d}f}"


def pct(v, d=0):
    return "–" if v is None else f"{100 * v:.{d}f}%"


def smooth(values, k):
    """Centered rolling mean over k evaluations (shorter at the ends)."""
    if not values or k <= 1:
        return values
    h = k // 2
    return [round(st.mean(values[max(0, i - h):i + h + 1]), 2) for i in range(len(values))]


def curve(key, label, opp="greedy", k=1):
    s = SET[key]
    c = s["curve"]
    return {"label": label, "x": c["games"], "y": smooth(c[opp], k), "lo": smooth(c.get(opp + "_min"), k),
            "hi": smooth(c.get(opp + "_max"), k), "seeds": len(s["seeds"]), "done": max(s["games_done"]),
            "finished": s["finished"], "smoothed": k}


# ------------------------------------------------------------------ re-scores (fresh games)

def rescore_index():
    g = defaultdict(list)
    for r in DATA["rescores"]:
        g[(r["group"], r["setting"], r["which"], r["opponent"])].append(r)
    out = {}
    for k, rs in g.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        out[k] = {"margin": st.mean(r["margin"] for r in rs), "win": st.mean(r["win"] for r in rs),
                  "seeds": [r["margin"] for r in rs], "players": rs[0]["players"],
                  "tickets": st.mean(r["tickets_completed"] for r in rs)}
    return out


RS = rescore_index()


def rescore_rows(group, labels):
    rows = []
    for setting, label in labels:
        row = {"setting": setting, "label": label}
        for which in ("final", "best"):
            for opp in OPPS:
                v = RS.get((group, setting, which, opp))
                if v:
                    row[f"{which}_{opp}"] = round(v["margin"], 1)
                    row[f"{which}_{opp}_win"] = round(v["win"], 3)
                    if opp == "greedy":
                        row[f"{which}_seeds"] = [round(x) for x in v["seeds"]]
        rows.append(row)
    return rows


LINEAR_ROWS = rescore_rows("linear/pass4", [("p4a_base", "p4a · Q vs random, α 0.05"), ("p4b_avg", "p4b · + averaging"),
                                            ("p4c_slow", "p4c · α 0.02 + averaging"), ("p4d_shape", "p4d · + shaping"),
                                            ("p4e_lam90", "p4e · + λ 0.9"), ("p4f_lam98", "p4f · α 0.2 + λ 0.98")]) \
    + rescore_rows("linear/pass5", [("p5a_q_random", "p5 · Q vs random"), ("p5b_sarsa_random", "p5 · SARSA vs random"),
                                    ("p5c_q_greedy", "p5 · Q vs greedy"), ("p5d_q_mixed", "p5 · Q vs mixed"),
                                    ("p5e_q_self", "p5 · Q vs self")]) \
    + rescore_rows("linear/pass6", [("p6c_q_greedy_decay", "p6 · Q vs greedy, α decay"),
                                    ("p6e_q_self_decay", "p6 · Q vs self, α decay"),
                                    ("p6d_sarsa_greedy_decay", "p6 · SARSA vs greedy, α decay")]) \
    + rescore_rows("linear/pass7", [("p7a_q_lam98_random", "p7 · Q(λ) vs random"),
                                    ("p7b_sarsa_lam98_random", "p7 · SARSA(λ) vs random"),
                                    ("p7c_q_lam98_greedy", "p7 · Q(λ) vs greedy"), ("p7d_q_lam98_self", "p7 · Q(λ) vs self")]) \
    + rescore_rows("linear/pass8", [("p8a_q_lam98_greedy_decay", "p8 · Q(λ) vs greedy, α decay"),
                                    ("p8b_q_lam98_self_decay", "p8 · Q(λ) vs self, α decay")])
DQN_ROWS = rescore_rows("dqn/pass1", [("d1a_n1", "pass 1 · n = 1"), ("d1b_n8", "pass 1 · n = 8"), ("d1c_n32", "pass 1 · n = 32")]) \
    + rescore_rows("dqn/pass2", [("d2b_pool", "pass 2 · pool"), ("d2c_pool_shape", "pass 2 · pool + shaping")]) \
    + rescore_rows("dqn/pass3", [("d3a_pool", "pass 3 · pool, fixed racer"), ("d3b_pool_plan", "pass 3 · + ticket plan")]) \
    + rescore_rows("dqn/random", [("dr_random", "random opponent only")])
PPO_ROWS = rescore_rows("ppo/pass1", [("p1a_pool", "pass 1 · pool + shaping"), ("p1b_pool_plan", "pass 1 · + ticket plan")]) \
    + rescore_rows("ppo/pass2", [("p2a_pool_shape", "pass 2 · shaping, 100k"), ("p2b_pool", "pass 2 · no shaping, 100k")]) \
    + rescore_rows("ppo/pass3", [("p3a_margin", "pass 3 · margin"), ("p3b_score", "pass 3 · score"), ("p3c_win", "pass 3 · win/loss")]) \
    + rescore_rows("ppo/pass4", [("p4a_3p_margin", "pass 4 · 3 players, margin"), ("p4b_3p_score", "pass 4 · 3 players, score"),
                                 ("p4c_4p_margin", "pass 4 · 4 players, margin"), ("p4d_4p_score", "pass 4 · 4 players, score")])
for key, label in (("ppo/pass5/p5a_5p_margin", "pass 5 · 5 players, margin"), ("ppo/pass5/p5b_5p_score", "pass 5 · 5 players, score"),
                   ("ppo/pass5/p5c_4p_self_margin", "pass 5 · 4 players self-play, margin"),
                   ("ppo/pass5/p5d_4p_self_score", "pass 5 · 4 players self-play, score")):
    group, setting = key.rsplit("/", 1)
    if (group, setting, "final", "greedy") in RS:
        PPO_ROWS += rescore_rows(group, [(setting, label)])
    elif key in SET:  # not re-scored yet: training evaluations, last five
        s = SET[key]
        row = {"setting": setting, "label": label, "training_evals": True, "games_done": max(s["games_done"]),
               "finished": s["finished"]}
        for opp in OPPS:
            if opp in s["last5"]:
                row[f"final_{opp}"] = round(s["last5"][opp]["margin"], 1)
                row[f"final_{opp}_win"] = round(s["last5"][opp]["win"], 3)
        row["final_seeds"] = [round(x) for x in s["last5"]["greedy"]["seeds"]]
        PPO_ROWS.append(row)

# ------------------------------------------------------------------ round robin ratings

def method_of(name):
    if name.startswith("linear:"):
        return "Linear"
    if name.startswith("dqn:"):
        return "DQN"
    if name.startswith("ppo:"):
        return "PPO"
    return "Scripted bot"


LABELS = {
    "ppo:p1a_pool_s0": "PPO pass 1 (pool, shaping)", "ppo:p2b_pool_s3": "PPO pass 2 (pool)",
    "ppo:p2a_pool_shape_s1": "PPO pass 2 (pool, shaping)", "ppo:p3a_margin_s0": "PPO pass 3 (margin reward)",
    "ppo:p3b_score_s2": "PPO pass 3 (score reward)", "ppo:p3c_win_s0": "PPO pass 3 (win/loss reward)",
    "linear:p7b_sarsa_lam98_random_s3@best": "Linear SARSA(λ) best checkpoint", "dqn:d3a_pool_s1@best": "DQN pass 3 best checkpoint",
}
RR = DATA["round_robins"][-1]
ELO = [{"name": n, "label": LABELS.get(n, n), "method": method_of(n), "elo": e} for n, e in RR["ratings"]]
RR_HISTORY = [{"file": r["file"], "top": r["ratings"][:4], "n": len(r["ratings"])} for r in DATA["round_robins"]]

# ------------------------------------------------------------------ multiplayer and scripted tables

ARMS = {2: ("ppo/pass3/p3a_margin", "ppo/pass3/p3b_score"), 3: ("ppo/pass4/p4a_3p_margin", "ppo/pass4/p4b_3p_score"),
        4: ("ppo/pass4/p4c_4p_margin", "ppo/pass4/p4d_4p_score"), 5: ("ppo/pass5/p5a_5p_margin", "ppo/pass5/p5b_5p_score")}
TABLE_SIZE = []
for n, (m, s) in ARMS.items():
    for arm, key in (("margin", m), ("score", s)):
        if key in SET:
            l5 = SET[key]["last5"]
            TABLE_SIZE.append({"players": n, "arm": arm, "margin_greedy": l5["greedy"]["margin"], "win_greedy": l5["greedy"]["win"],
                               "margin_racer": l5["racer"]["margin"], "win_racer": l5["racer"]["win"],
                               "route_points": l5["greedy"]["route_points"], "claim_length": l5["greedy"]["mean_claim_length"],
                               "tickets_completed": l5["greedy"]["tickets_completed"], "baseline": 1 / n})

SCRIPTED = MEAS.get("scripted", {}).get("tables", [])
LONE = []
for t in SCRIPTED:
    a, n = t["agents"], t["players"]
    kinds = t["by_kind"]
    if a.count(a[0]) == 1 and len(set(a)) == 2:  # one of a kind among n - 1 of another
        LONE.append({"players": n, "lone": a[0], "others": a[1], "lone_win": kinds[a[0]]["win"], "others_win": kinds[a[1]]["win"],
                     "lone_score": kinds[a[0]]["score"], "others_score": kinds[a[1]]["score"], "baseline": 1 / n})

# ------------------------------------------------------------------ instrumented games

ANAT_ORDER = ["collector", "greedy", "wary", "racer", "linear (Q/SARSA)", "DQN", "PPO margin", "PPO score"]
ANATOMY = []
for r in MEAS.get("anatomy", {}).get("rows", []):
    tag, label = r["key"]
    if tag == f"{label} vs greedy" or (label == "greedy" and tag == "greedy vs greedy"):
        bins = [r["len1"] + r["len2"], r["len3"], r["len4"], r["len5"], r["len6"]]
        ANATOMY.append({"label": label, "bins": bins, "claims": r["claims"], "route_points": r["route_points"],
                        "ticket_points": r["ticket_points"], "tickets_kept": r["tickets_kept"],
                        "tickets_completed": r["tickets_completed"], "own_turns": r["own_turns"], "half_turn": r["half_turn"],
                        "longest_bonus": r["longest_bonus"], "triggered_end": r["triggered_end"], "margin": r["margin"],
                        "win": r["win_share"], "score": r["score"], "opp_score": r["opp_score"],
                        "blocks": r["blocks_per_claim"], "chance": r["chance_length_per_claim"]})
ANATOMY.sort(key=lambda r: ANAT_ORDER.index(r["label"]) if r["label"] in ANAT_ORDER else 99)

ARM_LABEL = {"pass3/p3a_margin": "margin", "pass3/p3b_score": "score", "pass4/p4a_3p_margin": "margin",
             "pass4/p4b_3p_score": "score", "pass4/p4c_4p_margin": "margin", "pass4/p4d_4p_score": "score",
             "pass5/p5a_5p_margin": "margin", "pass5/p5b_5p_score": "score",
             "pass5/p5c_4p_self_margin": "self-play margin", "pass5/p5d_4p_self_score": "self-play score"}
BLOCKING, CROSS, SELFPOOL = [], [], []
for src in ("blocking", "blocking-self"):
    for r in MEAS.get(src, {}).get("rows", []):
        tag, label = r["key"]
        if " cross" in tag:
            CROSS.append({"players": r["players"], "arm": ARM_LABEL.get(label, label), "games": r["games"], "score": r["score"],
                          "win": r["win_share"], "rank": r["rank"], "margin": r["margin"]})
            continue
        if " selfpool " in tag:  # two self-play seats against two pool-trained seats
            SELFPOOL.append({"reward": tag.rsplit(" ", 1)[1], "self": label.startswith("pass5/"), "games": r["games"],
                             "score": r["score"], "win": r["win_share"], "six": r["len6"],
                             "claim_length": r["mean_claim_length"], "longest": r["longest_bonus"]})
            continue
        n, arm, _, bot = tag.split(" ")
        if label == arm:  # the learner's seat
            BLOCKING.append({"players": r["players"], "arm": ARM_LABEL.get(arm, arm), "bot": bot, "games": r["games"],
                             "blocks": r["blocks_per_claim"], "chance": r["chance_length_per_claim"],
                             "uniform": r["chance_uniform_per_claim"], "excess": r["excess_blocks"], "excess_se": r["excess_blocks_se"],
                             "claims": r["claims"], "claim_length": r["mean_claim_length"], "margin": r["margin"], "win": r["win_share"],
                             "six": r["len6"], "opp_tickets": r["opp_tickets_completed"], "opp_trains_left": r["opp_trains_left"],
                             "own_turns": r["own_turns"], "tickets_completed": r["tickets_completed"]})
        elif bot in label:  # a bot beside the learner: the reference rate for corridor overlap
            BLOCKING.append({"players": r["players"], "arm": f"{bot} beside {ARM_LABEL.get(arm, arm)}", "bot": bot, "reference": True,
                             "games": r["games"], "blocks": r["blocks_per_claim"], "chance": r["chance_length_per_claim"],
                             "uniform": r["chance_uniform_per_claim"], "excess": r["excess_blocks"], "excess_se": r["excess_blocks_se"],
                             "claims": r["claims"], "claim_length": r["mean_claim_length"]})

GAP = MEAS.get("ticket-gap", {}).get("rows", [])

# ------------------------------------------------------------------ chart series

CURVES = {
    "linear": [curve("linear/pass5/p5c_q_greedy", "Q, constant α", k=5), curve("linear/pass6/p6c_q_greedy_decay", "Q, α decay", k=5),
               curve("linear/pass7/p7c_q_lam98_greedy", "Q(λ 0.98)", k=5)],
    "dqn": [curve("dqn/pass1/d1a_n1", "n = 1"), curve("dqn/pass1/d1b_n8", "n = 8"), curve("dqn/pass1/d1c_n32", "n = 32")],
    "ppo": [curve("ppo/pass1/p1a_pool", "pass 1, 50k games"), curve("ppo/pass2/p2b_pool", "pass 2, 100k games"),
            curve("ppo/pass1/p1b_pool_plan", "+ ticket plan")],
    "reward": [curve("ppo/pass3/p3a_margin", "margin"), curve("ppo/pass3/p3b_score", "own score"), curve("ppo/pass3/p3c_win", "win/loss")],
    "self_margin": [curve("ppo/pass4/p4c_4p_margin", "pool (pass 4)"), curve("ppo/pass5/p5c_4p_self_margin", "self-play (pass 5)")],
    "self_score": [curve("ppo/pass4/p4d_4p_score", "pool (pass 4)"), curve("ppo/pass5/p5d_4p_self_score", "self-play (pass 5)")],
}


# ------------------------------------------------------------------ numbers bound into the text

def last5(key, opp="greedy", field="margin"):
    return SET[key]["last5"][opp][field]


V = {}
p5c, p5d = SET.get("ppo/pass5/p5c_4p_self_margin"), SET.get("ppo/pass5/p5d_4p_self_score")
if p5c and p5d:
    V["self.games"] = f"{min(max(p5c['games_done']), max(p5d['games_done'])):,}"
    V["self.finished"] = "finished" if p5c["finished"] and p5d["finished"] else "still training"
    V["self.m.greedy"] = sgn(last5("ppo/pass5/p5c_4p_self_margin"))
    V["self.s.greedy"] = sgn(last5("ppo/pass5/p5d_4p_self_score"))
    V["self.m.racer"] = sgn(last5("ppo/pass5/p5c_4p_self_margin", "racer"))
    V["self.s.racer"] = sgn(last5("ppo/pass5/p5d_4p_self_score", "racer"))
    V["self.m.win"] = pct(last5("ppo/pass5/p5c_4p_self_margin", field="win"))
    V["self.m.len"] = f"{last5('ppo/pass5/p5c_4p_self_margin', field='mean_claim_length'):.1f}"
    V["self.s.len"] = f"{last5('ppo/pass5/p5d_4p_self_score', field='mean_claim_length'):.1f}"
    V["self.m.tix"] = f"{last5('ppo/pass5/p5c_4p_self_margin', field='tickets_completed'):.2f}"
    V["self.parity"] = ", ".join(str(g // 1000) + "k" for g in p5c["parity_games"] + p5d["parity_games"] if g)
    V["pool4.parity"] = ", ".join(str(g // 1000) + "k" for g in SET["ppo/pass4/p4c_4p_margin"]["parity_games"]
                                  + SET["ppo/pass4/p4d_4p_score"]["parity_games"] if g)
for key in ("ppo/pass5/p5a_5p_margin", "ppo/pass5/p5b_5p_score"):
    arm = "m" if "margin" in key else "s"
    V[f"p5.{arm}.greedy"] = sgn(last5(key))
    V[f"p5.{arm}.win"] = pct(last5(key, field="win"))
    V[f"p5.{arm}.racer"] = sgn(last5(key, "racer"))
if p5d:  # where the self-play score arm peaked
    c = p5d["curve"]
    k = max(range(len(c["games"])), key=lambda i: c["greedy"][i])
    V["self.s.peak"] = sgn(c["greedy"][k])
    V["self.s.peak_games"] = f"{c['games'][k]:,}"

# tempo and head-to-head numbers from the instrumented games (measure.py blocking)
rows = {tuple(r["key"]): r for src in ("blocking", "blocking-self") for r in MEAS.get(src, {}).get("rows", [])}
tm, ts = rows.get(("2p pass3/p3a_margin vs greedy", "pass3/p3a_margin")), rows.get(("2p pass3/p3b_score vs greedy", "pass3/p3b_score"))
if tm and ts:
    for arm, r in (("m", tm), ("s", ts)):
        V[f"tempo.{arm}.turns"] = f"{2 * r['own_turns']:.0f}"
        V[f"tempo.{arm}.route"] = f"{r['route_points']:.0f}"
        V[f"tempo.{arm}.oppscore"] = f"{r['opp_score']:.0f}"
        V[f"tempo.{arm}.opptrains"] = f"{r['opp_trains_left']:.0f}"
        V[f"tempo.{arm}.opptix"] = f"{r['opp_tickets_completed']:.1f}"
        V[f"tempo.{arm}.six"] = f"{r['len6']:.1f}"
    V["tempo.route_gap"] = f"{tm['route_points'] - ts['route_points']:+.0f}".replace("-", "−")
    V["tempo.route_given"] = f"{ts['route_points'] - tm['route_points']:.0f}"
    V["tempo.opp_taken"] = f"{ts['opp_score'] - tm['opp_score']:.0f}"
for n in (2, 3, 4, 5):
    m_arm, s_arm, _ = {2: ("pass3/p3a_margin", "pass3/p3b_score", 0), 3: ("pass4/p4a_3p_margin", "pass4/p4b_3p_score", 0),
                       4: ("pass4/p4c_4p_margin", "pass4/p4d_4p_score", 0), 5: ("pass5/p5a_5p_margin", "pass5/p5b_5p_score", 0)}[n]
    cm, cs = rows.get((f"{n}p cross", m_arm)), rows.get((f"{n}p cross", s_arm))
    if cm and cs:
        V[f"cross.{n}.edge"] = sgn(cs["score"] - cm["score"])
        V[f"cross.{n}.se"] = f"{cs['margin_se']:.1f}"
        V[f"cross.{n}.swin"] = pct(cs["win_share"])
        V[f"cross.{n}.mwin"] = pct(cm["win_share"])
        V[f"cross.{n}.ssix"] = f"{cs['len6']:.1f}"
        V[f"cross.{n}.msix"] = f"{cm['len6']:.1f}"
        V[f"cross.{n}.shalf"] = f"{cs['half_turn']:.0f}"
        V[f"cross.{n}.mhalf"] = f"{cm['half_turn']:.0f}"
for reward, arm in (("margin", "m"), ("score", "s")):
    a = next((r for r in SELFPOOL if r["reward"] == reward and r["self"]), None)
    b = next((r for r in SELFPOOL if r["reward"] == reward and not r["self"]), None)
    if a and b:  # per seat; each kind holds two of the four seats
        V[f"selfpool.{arm}.edge"] = sgn(a["score"] - b["score"])
        V[f"selfpool.{arm}.swin"] = pct(2 * a["win"])
        V[f"selfpool.{arm}.ssix"] = f"{a['six']:.1f}"
        V[f"selfpool.{arm}.psix"] = f"{b['six']:.1f}"
        V[f"selfpool.{arm}.slongest"] = pct(a["longest"])
        V[f"selfpool.{arm}.plongest"] = pct(b["longest"])
gg = next((r for r in ANATOMY if r["label"] == "greedy"), None)
if gg:
    V["greedy.self.score"] = f"{gg['score']:.0f}"
    V["greedy.self.turns"] = f"{gg['own_turns']:.0f}"
V["generated"] = DATA["generated"].replace("T", " ")
V["runs"] = str(sum(len(s["seeds"]) for s in SET.values()))
V["settings"] = str(len(SET))
V["games"] = f"{sum(sum(s['games_done']) for s in SET.values()) / 1e6:.1f}"
V["rescores"] = str(len(DATA["rescores"]))

PAPER = {
    "vals": V, "curves": CURVES, "elo": ELO, "rr_history": RR_HISTORY, "rr6_pairs": RR["pairs"],
    "linear_rows": LINEAR_ROWS, "dqn_rows": DQN_ROWS, "ppo_rows": PPO_ROWS,
    "table_size": TABLE_SIZE, "lone": LONE, "scripted": SCRIPTED,
    "anatomy": ANATOMY, "blocking": BLOCKING, "cross": CROSS, "gap": GAP,
}


def main() -> None:
    html = (HERE / "template.html").read_text(encoding="utf-8")
    payload = json.dumps(PAPER, separators=(",", ":")).replace("</", "<\\/")
    out = HERE / "paper.html"
    out.write_text(html.replace("__PAPER_DATA__", payload), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e3:.0f} kB); data of {V['generated']}; "
          f"anatomy {len(ANATOMY)}, blocking {len(BLOCKING)}, cross {len(CROSS)}, gap {len(GAP)}, scripted {len(SCRIPTED)}")


if __name__ == "__main__":
    main()
