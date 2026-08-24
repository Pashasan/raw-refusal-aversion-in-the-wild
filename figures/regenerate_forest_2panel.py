"""Two-panel headline forest plot.

Panel A (top):   LMSYS cells (cross-model heterogeneity within a single
                  corpus). 4 cells passing the inclusion criterion.
Panel B (bottom): WildChat cells (9 cells: 2 general-traffic + 7 risky-
                  enriched), with the three dated GPT-4 cells annotated as
                  the within-line drift sequence.

The two-panel layout makes two patterns visible that the single 13-row forest
muddies:
  (i) within-corpus heterogeneity is a clean comparison (each panel = same
      conditional estimand),
  (ii) the GPT-4 drift sequence (0314 -> 1106 -> 0125) is visually grouped.

Inclusion criterion is identical to the headline forest: n_T >= 500,
ESS_T >= 100, AIPW 95% CI width <= 12 pp.

--se-mode clustered (default): AIPW CIs use the default
inference regime (hashed-IP user clusters on WildChat cells, near-duplicate
prompt-group clusters on LMSYS cells) read from
output/comparison/aipw_clustered/default_inference.json.
--se-mode iid draws conversation-level iid intervals instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

_ap = argparse.ArgumentParser()
_ap.add_argument("--treatment-label", dest="treatment_label",
                 choices=["coalesce", "gpt54", "wg"], default="coalesce")
_ap.add_argument("--se-mode", dest="se_mode",
                 choices=["clustered", "iid"], default="clustered")
_args = _ap.parse_args()
_SUF = "" if _args.treatment_label == "coalesce" else f"_{_args.treatment_label}"

DEFAULT_INFERENCE = (ROOT / "output" / "comparison" / "aipw_clustered"
                     / "default_inference.json")
DEF = (json.loads(DEFAULT_INFERENCE.read_text())
       if (_args.se_mode == "clustered" and DEFAULT_INFERENCE.exists()) else {})
if _args.se_mode == "clustered" and not DEF:
    raise SystemExit(f"--se-mode clustered but {DEFAULT_INFERENCE} missing; "
                     f"run collect_default_inference.py first")

MATCH = ROOT / "output" / "comparison" / f"matching{_SUF}" / "matching_results.json"
AIPW = ROOT / "output" / "comparison" / f"aipw{_SUF}" / "aipw_results.json"
print(f"reading {MATCH}\n        {AIPW}")

# Larger, consistent label sizing
matplotlib.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

PANEL_META = [
    ("base", "Llama-13b (base)", "LMSYS"),
    ("chat", "Llama-2-13b-chat", "LMSYS"),
    ("vicuna13b", "Vicuna-13b", "LMSYS"),
    ("vicuna33b", "Vicuna-33b", "LMSYS"),
    ("koala13b", "Koala-13b", "LMSYS"),
    ("gpt35turbo", "GPT-3.5-turbo", "LMSYS"),
    ("gpt4lmsys", "GPT-4", "LMSYS"),
    ("claude1", "Claude-1", "LMSYS"),
    ("claudeinstant1", "Claude-instant-1", "LMSYS"),
    # The three GPT-4o-era cells are WildChat general-traffic samples.
    ("gpt4o_lmsys", "GPT-4o (general)", "WildChat-general"),
    ("gpt4omini_lmsys", "GPT-4o-mini (general)", "WildChat-general"),
    ("gpt41mini_lmsys", "GPT-4.1-mini (general)", "WildChat-general"),
    ("gpt4o_wcrisky", "GPT-4o (risky)", "WildChat-risky"),
    ("gpt4omini_wcrisky", "GPT-4o-mini (risky)", "WildChat-risky"),
    ("gpt41mini_wcrisky", "GPT-4.1-mini (risky)", "WildChat-risky"),
    ("gpt35wc_wcrisky", "GPT-3.5-turbo (risky)", "WildChat-risky"),
    ("gpt4_0314_wcrisky", "GPT-4 (0314, risky)", "WildChat-risky"),
    ("gpt4_1106_wcrisky", "GPT-4-Turbo (1106, risky)", "WildChat-risky"),
    ("gpt4_0125_wcrisky", "GPT-4-Turbo (0125, risky)", "WildChat-risky"),
]

LMSYS_COLOR = "#1F4E79"
WCGENERAL_COLOR = "#2E6E63"
WCRISKY_COLOR = "#7A1F1F"
DRIFT_COLOR = "#C0530C"
SECONDARY_COLOR = "#888888"

match = json.loads(MATCH.read_text())
aipw = json.loads(AIPW.read_text())

N_T_MIN = 500
ESS_T_MIN = 100.0
CI_WIDTH_MAX_PP = 12.0


def passes(tag: str) -> bool:
    a = aipw.get(tag)
    if a is None:
        return False
    ci_width = a["att_ci95_hi"] - a["att_ci95_lo"]
    return (a["n_treated"] >= N_T_MIN and a["ess_treated"] >= ESS_T_MIN
            and ci_width <= CI_WIDTH_MAX_PP)


def pull(tag: str) -> dict:
    a = aipw[tag]
    m = match[tag]
    score_pp = m["score"]["pp_difference"]
    score_se = (m["score"]["ci_upper"] * 100 - score_pp) / 1.96
    emb_pp = m["embedding"]["pp_difference"]
    emb_se = (m["embedding"]["ci_upper"] * 100 - emb_pp) / 1.96
    pp = a["att_pp"]
    if tag in DEF and DEF[tag].get("cluster_mode") in ("ip", "prompt"):
        ci_lo, ci_hi = DEF[tag]["def_lo"], DEF[tag]["def_hi"]
    else:
        ci_lo = pp - 1.96 * a["att_se_pp"]
        ci_hi = pp + 1.96 * a["att_se_pp"]
    return {
        "AIPW":   (pp, a["att_se_pp"]),
        "CI":     (ci_lo, ci_hi),
        "Score":  (score_pp, score_se),
        "Emb":    (emb_pp, emb_se),
        "AIPW-T": (a["att_trimmed_pp"], a["att_trimmed_se_pp"]),
        "p_treated": a["p_treated"],
        "n_treated": a["n_treated"],
    }


lmsys_records = []
wc_records = []
for tag, name, corpus in PANEL_META:
    if tag not in aipw or not passes(tag):
        continue
    d = pull(tag)
    rec = (tag, name, corpus, d)
    (lmsys_records if corpus == "LMSYS" else wc_records).append(rec)

# Sort each panel by AIPW ATT ascending (most negative on top)
lmsys_records.sort(key=lambda r: r[3]["AIPW"][0])
wc_records.sort(key=lambda r: r[3]["AIPW"][0])

DRIFT_TAGS = {"gpt4_0314_wcrisky", "gpt4_1106_wcrisky", "gpt4_0125_wcrisky"}

# ---------------------------------------------------------------------------
# Common x-range so the two panels read on the same axis
# ---------------------------------------------------------------------------
all_pp = []
for recs in (lmsys_records, wc_records):
    for tag, name, corpus, d in recs:
        all_pp.append(d["CI"][0])
        all_pp.append(d["CI"][1])
        for est in ("Score", "Emb", "AIPW-T"):
            ept = d[est][0]
            if not np.isnan(ept):
                all_pp.append(ept)
xlim = (min(all_pp) - 2.0, max(all_pp) + 2.0)


def render_panel(ax, records, color, annotate_drift=False, title=None):
    n = len(records)
    y_positions = list(range(n))[::-1]
    annot_x = xlim[1] + 0.04 * (xlim[1] - xlim[0])

    for y, (tag, name, corpus, d) in zip(y_positions, records):
        is_drift = annotate_drift and tag in DRIFT_TAGS
        if is_drift:
            face = DRIFT_COLOR
        elif corpus == "WildChat-general":
            face = WCGENERAL_COLOR
        else:
            face = color
        edge = "black"
        pp, _se = d["AIPW"]
        ci_lo, ci_hi = d["CI"]
        if not (np.isnan(pp) or np.isnan(ci_lo)):
            ax.errorbar(pp, y, xerr=[[pp - ci_lo], [ci_hi - pp]],
                        fmt="D", markersize=8.5,
                        color=face, mec=edge, mew=0.6,
                        ecolor=face, elinewidth=1.8, capsize=3, zorder=4)
        for est in ("Score", "Emb", "AIPW-T"):
            ept, _ = d[est]
            if np.isnan(ept):
                continue
            ax.plot(ept, y, marker="|", markersize=11, mew=1.5,
                    color=SECONDARY_COLOR, zorder=3)
        ax.text(annot_x, y, f"P(T|risky)={d['p_treated']*100:>5.1f}%   "
                            f"n_T={d['n_treated']:>5,}",
                ha="left", va="center", fontsize=8.5, color="#333333",
                family="monospace")

    ax.axvline(0, color="#444444", lw=1.0, ls="--", zorder=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[1] for r in records])
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    if title:
        ax.set_title(title, fontsize=11.5, pad=4, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, lw=0.4)
    ax.tick_params(axis="y", length=0)


# ---------------------------------------------------------------------------
# Layout: two stacked panels with shared x-axis
# ---------------------------------------------------------------------------
n_lmsys = len(lmsys_records)
n_wc = len(wc_records)
row_h = 0.25  # inches per row (rows packed ~30% tighter; fonts/markers unchanged)
total_h = (n_lmsys + n_wc) * row_h + 2.0
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(8.5, total_h),
    gridspec_kw={"height_ratios": [n_lmsys, n_wc], "hspace": 0.18},
    sharex=True,
)

render_panel(ax_top, lmsys_records, LMSYS_COLOR,
             annotate_drift=False,
             title=f"A. LMSYS cells (n={n_lmsys}) — cross-model heterogeneity within one corpus")
render_panel(ax_bot, wc_records, WCRISKY_COLOR,
             annotate_drift=True,
             title=f"B. WildChat cells (n={n_wc}): general traffic (teal) and "
                   f"risky-enriched — orange diamonds = GPT-4 line drift")

# Draw a thin orange line connecting the three GPT-4 drift cells in panel B
drift_positions = []
y_positions = list(range(n_wc))[::-1]
for y, (tag, name, corpus, d) in zip(y_positions, wc_records):
    if tag in DRIFT_TAGS:
        drift_positions.append((d["AIPW"][0], y))
if len(drift_positions) >= 2:
    drift_positions.sort(key=lambda p: p[1])
    xs = [p[0] for p in drift_positions]
    ys = [p[1] for p in drift_positions]
    ax_bot.plot(xs, ys, color=DRIFT_COLOR, lw=1.5, ls=":", alpha=0.7, zorder=3.5)

ax_bot.set_xlabel("ATT of refusal on user re-engagement (percentage points)")

# Legend below the bottom panel
legend_handles = [
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=LMSYS_COLOR,
               markeredgecolor="black", markersize=8, label="LMSYS cell", lw=0),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=WCGENERAL_COLOR,
               markeredgecolor="black", markersize=8, label="WildChat general", lw=0),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=WCRISKY_COLOR,
               markeredgecolor="black", markersize=8, label="WildChat risky-enriched", lw=0),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=DRIFT_COLOR,
               markeredgecolor="black", markersize=8, label="GPT-4 dated checkpoint", lw=0),
    plt.Line2D([0], [0], marker="|", color=SECONDARY_COLOR, markersize=11, mew=1.5,
               label="Score / Emb / AIPW-T", lw=0),
]
fig.legend(handles=legend_handles, loc="lower center",
           frameon=True, fontsize=9, ncol=5,
           bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, borderpad=0.4)

plt.tight_layout(rect=(0, 0.03, 1, 1))
out = FIG / f"fig_forest_2panel{_SUF}.png"
fig.savefig(out)
print(f"wrote {out}")
print(f"panel A (LMSYS): {n_lmsys} cells; panel B (WildChat-risky): {n_wc} cells")
