"""Regenerate fig_deciles_main.png for the 13-cell main-text panel.

Pooled-quantile decile cutoffs are computed on ALL 19 cells (so D1, ..., D10
mean the same absolute risk band as in fig_deciles_all.png) but only the 13
main-text cells (passing n_T >= 500, ESS_T >= 100, AIPW CI <= 12pp; same
criterion as the headline forest) are plotted here.

Outputs:
  paper/figures/fig_deciles_main.png
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

_ap = argparse.ArgumentParser()
_ap.add_argument("--treatment-label", dest="treatment_label",
                 choices=["coalesce", "gpt54", "wg"], default="coalesce")
_args = _ap.parse_args()
_FIG_SUF = "" if _args.treatment_label == "coalesce" else f"_{_args.treatment_label}"

# Larger, more consistent label sizing per user feedback.
matplotlib.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
C_COLOR = "#1F4E79"   # blue: Cont | no refusal
T_COLOR = "#8B0000"   # dark red: Cont | refusal
REF_COLOR = "#E07A00" # orange line: refusal rate

# Full 19-cell panel (pooled cutoff computation uses all of these)
ALL_PANEL = [
    # ---- LMSYS ----
    ("Llama-2-13b-chat",       "gpt35_message_df_with_users.pkl",
                               "refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("Llama-13b (base)",       "llama13b_message_df_with_users.pkl",
                               "llama13b_refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("Vicuna-13b",             "vicuna13b_message_df_with_users.pkl",
                               "vicuna13b_refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("Vicuna-33b",             "vicuna33b_message_df_with_users.pkl",
                               "vicuna33b_refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("Koala-13b",              "koala13b_message_df_with_users.pkl",
                               "koala13b_refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("GPT-3.5-turbo (LMSYS)",  "gpt35turbo_message_df_with_users.pkl",
                               "gpt35turbo_refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("GPT-4 (LMSYS)",          "gpt4lmsys_message_df_with_users.pkl",
                               "gpt4lmsys_refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("Claude-1",               "claude1_message_df_with_users.pkl",
                               "claude1_refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("Claude-instant-1",       "claudeinstant1_message_df_with_users.pkl",
                               "claudeinstant1_refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("GPT-4o (WC-general)",       "gpt4o_message_df_with_users.pkl",
                                  "gpt4o_refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("GPT-4o-mini (WC-general)",  "gpt4omini_message_df_with_users.pkl",
                                  "gpt4omini_refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("GPT-4.1-mini (WC-general)", "gpt41mini_message_df_with_users.pkl",
                                  "gpt41mini_refusal_data_dynamic_with_labels_gpt54.csv", False),
    ("GPT-3.5-turbo (WC-risky)", "gpt35wc_risky_add_message_df_with_users.pkl",
                                 "gpt35wc_risky_add_refusal_data_dynamic_with_labels_gpt54.csv", True),
    ("GPT-4o (WC-risky)",        "gpt4o_risky_add_message_df_with_users.pkl",
                                 "gpt4o_risky_add_refusal_data_dynamic_with_labels_gpt54.csv", True),
    ("GPT-4o-mini (WC-risky)",   "gpt4omini_risky_add_message_df_with_users.pkl",
                                 "gpt4omini_risky_add_refusal_data_dynamic_with_labels_gpt54.csv", True),
    ("GPT-4.1-mini (WC-risky)",  "gpt41mini_risky_add_message_df_with_users.pkl",
                                 "gpt41mini_risky_add_refusal_data_dynamic_with_labels_gpt54.csv", True),
    ("GPT-4 (0314)",           "gpt4_0314_risky_add_message_df_with_users.pkl",
                               "gpt4_0314_risky_add_refusal_data_dynamic_with_labels_gpt54.csv", True),
    ("GPT-4-Turbo (1106)",     "gpt4_1106_risky_add_message_df_with_users.pkl",
                               "gpt4_1106_risky_add_refusal_data_dynamic_with_labels_gpt54.csv", True),
    ("GPT-4-Turbo (0125)",     "gpt4_0125_risky_add_message_df_with_users.pkl",
                               "gpt4_0125_risky_add_refusal_data_dynamic_with_labels_gpt54.csv", True),
]

# 13 main-text cells (matching headline forest inclusion criterion)
MAIN_NAMES = {
    "Llama-13b (base)", "Vicuna-13b", "Vicuna-33b", "Koala-13b",
    "GPT-4o (WC-general)", "GPT-4o-mini (WC-general)",
    "GPT-3.5-turbo (WC-risky)", "GPT-4o (WC-risky)", "GPT-4o-mini (WC-risky)",
    "GPT-4.1-mini (WC-risky)", "GPT-4 (0314)", "GPT-4-Turbo (1106)",
    "GPT-4-Turbo (0125)",
}


def build_conv(pkl, lab, coalesce_wg=False, treatment_label="coalesce"):
    msg = pickle.load(open(ROOT / pkl, "rb"))
    lb = pd.read_csv(ROOT / lab)
    msg = msg.drop(columns=[c for c in ["refused_answer"] if c in msg.columns])
    if treatment_label == "wg":
        msg = msg.merge(lb[["conversation_id", "message_number", "is_refusal_wg"]],
                        on=["conversation_id", "message_number"], how="left")
        msg["_refusal"] = msg["is_refusal_wg"]
    elif treatment_label == "gpt54":
        msg = msg.merge(lb[["conversation_id", "message_number", "is_refusal_gpt54"]],
                        on=["conversation_id", "message_number"], how="left")
        msg["_refusal"] = msg["is_refusal_gpt54"]
    else:  # coalesce
        if "is_refusal_gpt54" not in lb.columns:
            if not coalesce_wg:
                raise ValueError(f"{lab} missing is_refusal_gpt54")
            lb["is_refusal_gpt54"] = pd.NA
        cols = ["conversation_id", "message_number", "is_refusal_gpt54"]
        if coalesce_wg:
            cols.append("is_refusal_wg")
        msg = msg.merge(lb[cols], on=["conversation_id", "message_number"], how="left")
        if coalesce_wg:
            msg["_refusal"] = msg["is_refusal_gpt54"].combine_first(msg["is_refusal_wg"])
        else:
            msg["_refusal"] = msg["is_refusal_gpt54"]

    fu = msg[(msg["role"] == "user") & (msg["message_number"] == 1)][
        ["conversation_id", "max_concern_score"]
    ]
    r2 = msg[(msg["role"] == "assistant") & (msg["message_number"] == 2)][
        ["conversation_id", "_refusal"]
    ]
    u3 = msg[(msg["role"] == "user") & (msg["message_number"] == 3)][
        ["conversation_id"]
    ].assign(Y=1)
    c = fu.merge(r2, on="conversation_id", how="inner").merge(u3, on="conversation_id", how="left")
    c["Y"] = c["Y"].fillna(0).astype(int)
    c = c.dropna(subset=["_refusal", "max_concern_score"])
    c["T"] = c["_refusal"].astype(int)
    return c


# Pass 1: load all 19 cells, compute pooled-quantile cutoffs over the full pool
print("[pass 1] loading all 19 cells to pool max_concern_score for cutoffs")
per_cell = {}
pooled_scores = []
for name, pkl, lab, coalesce_wg in ALL_PANEL:
    try:
        c = build_conv(pkl, lab, coalesce_wg=coalesce_wg,
                       treatment_label=_args.treatment_label)
    except ValueError as e:
        print(f"  [{name}] skipping — {e}")
        continue
    per_cell[name] = c
    pooled_scores.append(c["max_concern_score"].to_numpy())
    print(f"  {name:30s} n={len(c):>6}")
pooled = np.concatenate(pooled_scores)
inner_cuts = np.quantile(pooled, np.arange(0.1, 1.0, 0.1))
edges = np.concatenate([[-np.inf], inner_cuts, [np.inf]])


def assign_dec(scores):
    return np.digitize(scores, edges[1:-1], right=False) + 1


# Pass 2: plot only the 13 main-text cells. 5x3 grid (15 slots, 13 plots, 2 free
# for the legend) keeps the figure wide rather than tall so it fits the body.
MAIN_PANEL = [t for t in ALL_PANEL if t[0] in MAIN_NAMES]
n = len(MAIN_PANEL)
ncols = 5
nrows = int(np.ceil(n / ncols))  # = 3 for 13 cells

fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 2.5 * nrows))
axes = axes.ravel()

print(f"\n[pass 2] per-cell decile aggregation for {n} main-text cells")
for ax, (name, pkl, lab, coalesce_wg) in zip(axes, MAIN_PANEL):
    c = per_cell[name].copy()
    c["dec"] = assign_dec(c["max_concern_score"].to_numpy())
    deciles = range(1, 11)
    refrate = []; nr_rate = []; ref_rate_cont = []
    for d in deciles:
        sub = c[c["dec"] == d]
        if len(sub) == 0:
            refrate.append(np.nan); nr_rate.append(np.nan); ref_rate_cont.append(np.nan); continue
        refrate.append(sub["T"].mean())
        nr_rate.append(sub.loc[sub["T"] == 0, "Y"].mean() if (1 - sub["T"]).sum() else np.nan)
        ref_rate_cont.append(sub.loc[sub["T"] == 1, "Y"].mean() if sub["T"].sum() else np.nan)
    x = np.arange(1, 11); w = 0.36
    ax.bar(x - w/2, nr_rate, w, color=C_COLOR, alpha=0.85,
           edgecolor="black", linewidth=0.2, label="Cont. | no refusal")
    ax.bar(x + w/2, ref_rate_cont, w, color=T_COLOR, alpha=0.85,
           edgecolor="black", linewidth=0.2, label="Cont. | refusal")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in x])
    ax.set_xlim(0.4, 10.6)
    ax.set_ylim(0, 1.0)
    ax.set_title(name, fontsize=10, pad=3)
    ax.tick_params(axis="y", length=2, pad=1)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.grid(axis="y", alpha=0.2, lw=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, refrate, color=REF_COLOR, marker="o", ms=4, lw=1.4)
    rr_max = np.nanmax(refrate) if not np.all(np.isnan(refrate)) else 0.8
    ax2.set_ylim(0, max(0.8, rr_max * 1.1 if not np.isnan(rr_max) else 0.8))
    ax2.tick_params(axis="y", colors=REF_COLOR)
    ax2.spines["right"].set_color(REF_COLOR)

# Hide leftover slots; use one for the legend
extra = axes[n:]
for ax in extra:
    ax.axis("off")

# Per-row y-label, per-bottom-row x-label
for i in range(nrows):
    axes[i * ncols].set_ylabel("Re-engagement rate")
# Bottom row x-labels (for cells that exist; remaining columns are hidden)
for j in range(ncols):
    idx_bottom = (nrows - 1) * ncols + j
    if idx_bottom < n:
        axes[idx_bottom].set_xlabel("Pooled moderation decile")

handles = [
    plt.Rectangle((0, 0), 1, 1, fc=C_COLOR, alpha=0.85, label="Cont. | no refusal"),
    plt.Rectangle((0, 0), 1, 1, fc=T_COLOR, alpha=0.85, label="Cont. | refusal"),
    plt.Line2D([0],[0], color=REF_COLOR, marker="o", ms=5, lw=1.6,
               label="Refusal rate (right axis)"),
]
if extra.size > 0:
    legend_ax = extra[0]
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=10)
else:
    axes[0].legend(handles=handles, loc="upper right", frameon=True, fontsize=8)

plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.8)
fig.savefig(FIG / f"fig_deciles_main{_FIG_SUF}.png")
print(f"wrote {FIG / f'fig_deciles_main{_FIG_SUF}.png'}")
