"""Single redesigned forest plot for the headline result.

Layout per row (one row per model, ordered by AIPW ATT ascending):
  [model name + corpus tag]    [annotation: P(T) | n_T]    [thin axis with:
       AIPW primary point + 95% CI (dominant visual),
       small tick marks for Score / Emb / AIPW-T (secondary)]

Background row shading distinguishes LMSYS from WildChat. We drop the scatter
plot entirely; this one chart now carries the headline."""
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
                 choices=["coalesce", "gpt54", "wg"], default="coalesce",
                 help="Reads from output/comparison/{matching,aipw}{suffix}/. "
                      "coalesce -> no suffix (legacy gpt54-headline). "
                      "wg -> _wg suffix.")
_ap.add_argument("--se-mode", dest="se_mode",
                 choices=["clustered", "iid"], default="clustered",
                 help="clustered (default): main-panel AIPW CIs "
                      "use the default inference regime (user clusters on "
                      "WildChat cells, prompt-group clusters on LMSYS cells); "
                      "excluded cells stay iid. iid draws conversation-level "
                      "intervals.")
_args = _ap.parse_args()
_LABEL_SUFFIX = "" if _args.treatment_label == "coalesce" else f"_{_args.treatment_label}"
_FIG_SUFFIX = "" if _args.treatment_label == "coalesce" else f"_{_args.treatment_label}"

MATCH = ROOT / "output" / "comparison" / f"matching{_LABEL_SUFFIX}" / "matching_results.json"
AIPW = ROOT / "output" / "comparison" / f"aipw{_LABEL_SUFFIX}" / "aipw_results.json"
print(f"reading {MATCH}\n        {AIPW}")

DEFAULT_INFERENCE = (ROOT / "output" / "comparison" / "aipw_clustered"
                     / "default_inference.json")
DEF = (json.loads(DEFAULT_INFERENCE.read_text())
       if (_args.se_mode == "clustered" and _args.treatment_label == "coalesce"
           and DEFAULT_INFERENCE.exists()) else {})
if _args.se_mode == "clustered" and _args.treatment_label == "coalesce" and not DEF:
    raise SystemExit(f"--se-mode clustered but {DEFAULT_INFERENCE} missing; "
                     f"run collect_default_inference.py first")

matplotlib.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

PANEL_META = [
    # tag                  display name                  corpus
    ("base",               "Llama-13b (base)",           "LMSYS"),
    ("chat",               "Llama-2-13b-chat",           "LMSYS"),
    ("vicuna13b",          "Vicuna-13b",                 "LMSYS"),
    ("vicuna33b",          "Vicuna-33b",                 "LMSYS"),
    ("koala13b",           "Koala-13b",                  "LMSYS"),
    ("gpt35turbo",         "GPT-3.5-turbo",              "LMSYS"),
    ("gpt4lmsys",          "GPT-4",                      "LMSYS"),
    ("claude1",            "Claude-1",                   "LMSYS"),
    ("claudeinstant1",     "Claude-instant-1",           "LMSYS"),
    # The three GPT-4o-era cells are WildChat general-traffic samples.
    ("gpt4o_lmsys",        "GPT-4o",                     "WildChat-general"),
    ("gpt4omini_lmsys",    "GPT-4o-mini",                "WildChat-general"),
    ("gpt41mini_lmsys",    "GPT-4.1-mini",               "WildChat-general"),
    # gpt35wc (regular WildChat) intentionally dropped — gpt35wc_wcrisky is
    # the risky-enriched oversample of the same model on the same corpus and
    # carries the population-of-interest signal. Sampling design is documented
    # in §3 (data) and §4 (estimation).
    ("gpt4o_wcrisky",      "GPT-4o",                     "WildChat-risky"),
    ("gpt4omini_wcrisky",  "GPT-4o-mini",                "WildChat-risky"),
    ("gpt41mini_wcrisky",  "GPT-4.1-mini",               "WildChat-risky"),
    ("gpt35wc_wcrisky",    "GPT-3.5-turbo",              "WildChat-risky"),
    ("gpt4_0314_wcrisky",  "GPT-4 (0314)",               "WildChat-risky"),
    ("gpt4_1106_wcrisky",  "GPT-4-Turbo (1106)",         "WildChat-risky"),
    ("gpt4_0125_wcrisky",  "GPT-4-Turbo (0125)",         "WildChat-risky"),
]

LMSYS_COLOR = "#1F4E79"
WC_COLOR = "#C0530C"
WCGENERAL_COLOR = "#2E6E63"
WCRISKY_COLOR = "#7A1F1F"
SECONDARY_COLOR = "#888888"
CORPUS_COLOR = {"LMSYS": LMSYS_COLOR, "WildChat": WC_COLOR,
                "WildChat-general": WCGENERAL_COLOR,
                "WildChat-risky": WCRISKY_COLOR}

match = json.loads(MATCH.read_text())
aipw = json.loads(AIPW.read_text())

# ---------------------------------------------------------------------------
# Inclusion criterion for the "main-text" 13-cell subset: cells where the
# AIPW nuisance fit and the IPW reweighting are reliable enough that the
# headline ATT is statistically defensible. Cells failing the criterion are
# kept in the appendix for completeness but not in the main forest / table.
#
#   n_T            >= 500   (per-fold m1 trains on >=400 treated)
#   ESS_T          >= 100   (effective treated sample after IPW reweighting)
#   AIPW CI width  <= 12 pp (precision proxy; flags narrow-overlap pathologies
#                            even when n_T is large)
# ---------------------------------------------------------------------------
N_T_MIN = 500
ESS_T_MIN = 100.0
CI_WIDTH_MAX_PP = 12.0


def passes_main_criterion(tag: str) -> bool:
    a = aipw.get(tag)
    if a is None:
        return False
    ci_width = a["att_ci95_hi"] - a["att_ci95_lo"]
    return (a["n_treated"] >= N_T_MIN
            and a["ess_treated"] >= ESS_T_MIN
            and ci_width <= CI_WIDTH_MAX_PP)


def pull(tag: str) -> dict:
    a = aipw[tag]; m = match[tag]
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
        "AIPW":   (pp,                  a["att_se_pp"]),
        "CI":     (ci_lo, ci_hi),
        "Score":  (score_pp, score_se),
        "Emb":    (emb_pp,   emb_se),
        "AIPW-T": (a["att_trimmed_pp"], a["att_trimmed_se_pp"]),
        "p_treated": a["p_treated"],
        "n_treated": a["n_treated"],
        "n": a["n"],
    }


# Order by AIPW ATT (ascending: most negative at top so eye reads strongest
# refusal aversion first)
all_records = []
for tag, name, ds in PANEL_META:
    if tag not in aipw:
        continue
    d = pull(tag)
    all_records.append((tag, name, ds, d))
all_records.sort(key=lambda r: r[3]["AIPW"][0])

main_records = [r for r in all_records if passes_main_criterion(r[0])]


# ---------------------------------------------------------------------------
# Forest helper (used for both the main-text and appendix versions)
# ---------------------------------------------------------------------------
def render_forest(records, out_path: Path, height_per_row: float = 0.36,
                   xlim: tuple = (-60, 40)):
    n = len(records)
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, height_per_row * n + 2.0)))
    y_positions = list(range(n))[::-1]  # top-down
    annot_x = xlim[1] + 0.05 * (xlim[1] - xlim[0])  # just past right edge

    for y, (tag, name, ds, d) in zip(y_positions, records):
        color = CORPUS_COLOR[ds]
        pp, _se = d["AIPW"]
        ci_lo, ci_hi = d["CI"]
        if not (np.isnan(pp) or np.isnan(ci_lo)):
            ax.errorbar(pp, y, xerr=[[pp - ci_lo], [ci_hi - pp]],
                        fmt="D", markersize=7,
                        color=color, mec="black", mew=0.5,
                        ecolor=color, elinewidth=1.6, capsize=3, zorder=4)
        for est in ("Score", "Emb", "AIPW-T"):
            ept, ese = d[est]
            if np.isnan(ept):
                continue
            ax.plot(ept, y, marker="|", markersize=10, mew=1.5,
                    color=SECONDARY_COLOR, zorder=3)
        pT = d["p_treated"]; nT = d["n_treated"]
        ax.text(annot_x, y, f"{ds:<14}{pT*100:>5.1f}%   n_T={nT:>5,}",
                ha="left", va="center", fontsize=6.5, color="#333333",
                family="monospace")

    ax.axvline(0, color="#444444", lw=1.0, ls="--", zorder=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[1] for r in records])
    ax.set_xlabel("ATT of refusal on user re-engagement (percentage points)")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.5, n - 0.5)
    ax.text(annot_x, n - 0.5 + 0.4, "corpus         P(refuse|risky)   n_treated",
            ha="left", va="bottom", fontsize=6.5, color="#333333",
            family="monospace", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.18, lw=0.4)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor=LMSYS_COLOR,
                   markeredgecolor="black", markersize=7, label="LMSYS", lw=0),
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor=WCGENERAL_COLOR,
                   markeredgecolor="black", markersize=7, label="WildChat (general)", lw=0),
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor=WCRISKY_COLOR,
                   markeredgecolor="black", markersize=7, label="WildChat (risky-enriched)", lw=0),
        plt.Line2D([0],[0], marker="|", color=SECONDARY_COLOR, markersize=10, mew=1.5,
                   label="Score / Emb / AIPW-T (point estimates)", lw=0),
    ]
    ax.legend(handles=legend_handles, loc="upper center", frameon=True,
              fontsize=6.5, handletextpad=0.4, borderpad=0.4,
              ncol=3, bbox_to_anchor=(0.5, -0.07))
    plt.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    plt.close(fig)


render_forest(main_records, FIG / f"fig_forest_main{_FIG_SUFFIX}.png",
              xlim=(-40, 20))
render_forest(all_records,  FIG / f"fig_forest{_FIG_SUFFIX}.png",
              xlim=(-60, 40), height_per_row=0.25)


# ---------------------------------------------------------------------------
# Cross-estimator agreement scatter for the main 13-cell subset.
# Each cell contributes 3 points (Score, Emb, AIPW-T) plotted against AIPW.
# Diagonal y=x indicates perfect agreement; spread off-diagonal flags
# estimator-to-estimator disagreement.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), sharex=True, sharey=True)

EST_LABEL = {"Score": "Score-matching ATT", "Emb": "Embedding-matching ATT",
             "AIPW-T": "Crump-trimmed AIPW ATT"}

# Compute a shared axis span from the main records only
aipw_vals = [r[3]["AIPW"][0] for r in main_records]
other_vals = []
for r in main_records:
    for est in ("Score", "Emb", "AIPW-T"):
        v = r[3][est][0]
        if not np.isnan(v):
            other_vals.append(v)
lo = min(min(aipw_vals), min(other_vals)) - 2.0
hi = max(max(aipw_vals), max(other_vals)) + 2.0
span = (lo, hi)

for ax, est in zip(axes, ("Score", "Emb", "AIPW-T")):
    ax.plot(span, span, "--", color="#888888", lw=0.8, zorder=1)
    ax.axhline(0, color="#cccccc", lw=0.5, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.5, zorder=1)
    for tag, name, ds, d in main_records:
        a_pp = d["AIPW"][0]
        e_pp = d[est][0]
        if np.isnan(a_pp) or np.isnan(e_pp):
            continue
        color = CORPUS_COLOR[ds]
        ax.scatter(a_pp, e_pp, s=42, color=color, edgecolor="black",
                   linewidth=0.4, zorder=3)
    ax.set_xlim(*span); ax.set_ylim(*span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(EST_LABEL[est], fontsize=8.5, pad=2)
    ax.set_xlabel("AIPW ATT (pp)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.18, lw=0.4)
axes[0].set_ylabel("Other-estimator ATT (pp)")

legend_handles = [
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=LMSYS_COLOR,
               markeredgecolor="black", markersize=7, label="LMSYS cell", lw=0),
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=WCGENERAL_COLOR,
               markeredgecolor="black", markersize=7, label="WildChat-general cell", lw=0),
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=WCRISKY_COLOR,
               markeredgecolor="black", markersize=7, label="WildChat-risky cell", lw=0),
    plt.Line2D([0],[0], color="#888888", ls="--", lw=0.8,
               label="$y = x$ (perfect agreement)"),
]
fig.legend(handles=legend_handles, loc="lower center", frameon=False,
           fontsize=7.5, ncol=4, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / f"fig_cross_estimator{_FIG_SUFFIX}.png")
print(f"wrote {FIG / f'fig_cross_estimator{_FIG_SUFFIX}.png'}")
plt.close(fig)
print(f"main-text panel: {len(main_records)} cells; appendix panel: {len(all_records)} cells")
print("done")
