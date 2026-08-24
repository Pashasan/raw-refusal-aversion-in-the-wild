"""Generate three additional figures for the paper:
   fig_propensity.png   overlap histograms of propensity per model
   fig_gamma_curves.png p-value vs Gamma curves for sensitivity
   fig_adjusted.png     raw vs score-matched vs AIPW-T continuation rates per model
All three cover the full 19-cell panel.
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

_ap = argparse.ArgumentParser()
_ap.add_argument("--treatment-label", dest="treatment_label",
                 choices=["coalesce", "gpt54", "wg"], default="coalesce")
_args = _ap.parse_args()
_LBL_SUF = "" if _args.treatment_label == "coalesce" else f"_{_args.treatment_label}"
_FIG_SUF = "" if _args.treatment_label == "coalesce" else f"_{_args.treatment_label}"
print(f"treatment_label={_args.treatment_label}  -> reading output/comparison/*{_LBL_SUF}/, "
      f"writing fig_*{_FIG_SUF}.png")

matplotlib.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
T_COLOR = "#8B0000"
C_COLOR = "#1F4E79"

RISK = 0.01
K = 5
CLIP = (0.02, 0.98)

# (tag, display name, dataset, pkl, labels_csv, emb_npz)
# Canonical 19-cell panel (mirrors aipw_with_gpt54.py MODELS).
# fields: (tag, name, ds, pkl, lab, emb, coalesce_wg)
MODELS = [
    # ---- LMSYS ----
    ("chat",            "Llama-2-13b-chat",           "LMSYS",
        "gpt35_message_df_with_users.pkl",
        "refusal_data_dynamic_with_labels_gpt54.csv",
        "first_user_embeddings.npz", False),
    ("base",            "Llama-13b (base)",           "LMSYS",
        "llama13b_message_df_with_users.pkl",
        "llama13b_refusal_data_dynamic_with_labels_gpt54.csv",
        "llama13b_first_user_embeddings.npz", False),
    ("vicuna13b",       "Vicuna-13b",                 "LMSYS",
        "vicuna13b_message_df_with_users.pkl",
        "vicuna13b_refusal_data_dynamic_with_labels_gpt54.csv",
        "vicuna13b_first_user_embeddings.npz", False),
    ("vicuna33b",       "Vicuna-33b",                 "LMSYS",
        "vicuna33b_message_df_with_users.pkl",
        "vicuna33b_refusal_data_dynamic_with_labels_gpt54.csv",
        "vicuna33b_first_user_embeddings.npz", False),
    ("koala13b",        "Koala-13b",                  "LMSYS",
        "koala13b_message_df_with_users.pkl",
        "koala13b_refusal_data_dynamic_with_labels_gpt54.csv",
        "koala13b_first_user_embeddings.npz", False),
    ("gpt35turbo",      "GPT-3.5-turbo (LMSYS)",      "LMSYS",
        "gpt35turbo_message_df_with_users.pkl",
        "gpt35turbo_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt35turbo_first_user_embeddings.npz", False),
    ("gpt4lmsys",       "GPT-4 (LMSYS)",              "LMSYS",
        "gpt4lmsys_message_df_with_users.pkl",
        "gpt4lmsys_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt4lmsys_first_user_embeddings.npz", False),
    ("claude1",         "Claude-1",                   "LMSYS",
        "claude1_message_df_with_users.pkl",
        "claude1_refusal_data_dynamic_with_labels_gpt54.csv",
        "claude1_first_user_embeddings.npz", False),
    ("claudeinstant1",  "Claude-instant-1",           "LMSYS",
        "claudeinstant1_message_df_with_users.pkl",
        "claudeinstant1_refusal_data_dynamic_with_labels_gpt54.csv",
        "claudeinstant1_first_user_embeddings.npz", False),
    # ---- WildChat / general traffic ----
    ("gpt4o_lmsys",     "GPT-4o (WC-general)",        "WildChat",
        "gpt4o_message_df_with_users.pkl",
        "gpt4o_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt4o_first_user_embeddings.npz", False),
    ("gpt4omini_lmsys", "GPT-4o-mini (WC-general)",   "WildChat",
        "gpt4omini_message_df_with_users.pkl",
        "gpt4omini_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt4omini_first_user_embeddings.npz", False),
    ("gpt41mini_lmsys", "GPT-4.1-mini (WC-general)",  "WildChat",
        "gpt41mini_message_df_with_users.pkl",
        "gpt41mini_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt41mini_first_user_embeddings.npz", False),
    # gpt35wc (WildChat general GPT-3.5) is not one of the 19 panel cells.
    # ---- WildChat / risky-enriched (gpt54 + wildguard) ----
    ("gpt4o_wcrisky",        "GPT-4o (WC-risky)",     "WildChat",
        "gpt4o_risky_add_message_df_with_users.pkl",
        "gpt4o_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt4o_risky_add_first_user_embeddings.npz", True),
    ("gpt4omini_wcrisky",    "GPT-4o-mini (WC-risky)","WildChat",
        "gpt4omini_risky_add_message_df_with_users.pkl",
        "gpt4omini_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt4omini_risky_add_first_user_embeddings.npz", True),
    ("gpt41mini_wcrisky",    "GPT-4.1-mini (WC-risky)","WildChat",
        "gpt41mini_risky_add_message_df_with_users.pkl",
        "gpt41mini_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt41mini_risky_add_first_user_embeddings.npz", True),
    ("gpt35wc_wcrisky",      "GPT-3.5-turbo (WC-risky)","WildChat",
        "gpt35wc_risky_add_message_df_with_users.pkl",
        "gpt35wc_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt35wc_risky_add_first_user_embeddings.npz", True),
    # ---- New WildChat-risky cells (wildguard-only) ----
    ("gpt4_0314_wcrisky",    "GPT-4 (0314)",          "WildChat",
        "gpt4_0314_risky_add_message_df_with_users.pkl",
        "gpt4_0314_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt4_0314_risky_add_first_user_embeddings.npz", True),
    ("gpt4_1106_wcrisky",    "GPT-4-Turbo (1106)",    "WildChat",
        "gpt4_1106_risky_add_message_df_with_users.pkl",
        "gpt4_1106_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt4_1106_risky_add_first_user_embeddings.npz", True),
    ("gpt4_0125_wcrisky",    "GPT-4-Turbo (0125)",    "WildChat",
        "gpt4_0125_risky_add_message_df_with_users.pkl",
        "gpt4_0125_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
        "gpt4_0125_risky_add_first_user_embeddings.npz", True),
    # o1mini dropped — single-turn-only convs make ATT undefined.
]


def build_sample(pkl, lab, emb, coalesce_wg=False, treatment_label="coalesce"):
    with open(ROOT / pkl, "rb") as f:
        msg = pickle.load(f)
    lb = pd.read_csv(ROOT / lab)
    msg = msg.drop(columns=[c for c in ["refused_answer"] if c in msg.columns])
    if treatment_label == "wg":
        if "is_refusal_wg" not in lb.columns:
            raise ValueError(f"{lab} missing is_refusal_wg")
        msg = msg.merge(lb[["conversation_id", "message_number", "is_refusal_wg"]],
                        on=["conversation_id", "message_number"], how="left")
        msg["refused_answer"] = msg["is_refusal_wg"].fillna(0).astype(int)
    elif treatment_label == "gpt54":
        if "is_refusal_gpt54" not in lb.columns:
            raise ValueError(f"{lab} missing is_refusal_gpt54")
        msg = msg.merge(lb[["conversation_id", "message_number", "is_refusal_gpt54"]],
                        on=["conversation_id", "message_number"], how="left")
        msg["refused_answer"] = msg["is_refusal_gpt54"].fillna(0).astype(int)
    elif treatment_label == "coalesce":
        if "is_refusal_gpt54" not in lb.columns:
            if not coalesce_wg:
                raise ValueError(f"{lab} missing is_refusal_gpt54 (set coalesce_wg=True for wg-only cells)")
            lb["is_refusal_gpt54"] = pd.NA
        cols = ["conversation_id", "message_number", "is_refusal_gpt54"]
        if coalesce_wg:
            if "is_refusal_wg" not in lb.columns:
                raise ValueError(f"{lab} missing is_refusal_wg (coalesce_wg=True requested)")
            cols.append("is_refusal_wg")
        msg = msg.merge(lb[cols], on=["conversation_id", "message_number"], how="left")
        if coalesce_wg:
            coalesced = msg["is_refusal_gpt54"].combine_first(msg["is_refusal_wg"])
            msg["refused_answer"] = coalesced.fillna(0).astype(int)
        else:
            msg["refused_answer"] = msg["is_refusal_gpt54"].fillna(0).astype(int)
    else:
        raise ValueError(f"unknown treatment_label: {treatment_label!r}")
    first = msg[(msg["role"] == "user") & (msg["message_number"] == 1)][
        ["conversation_id", "max_concern_score"]
    ].rename(columns={"max_concern_score": "max_score"})
    turn2 = msg[(msg["role"] == "assistant") & (msg["message_number"] == 2)][
        ["conversation_id", "refused_answer"]
    ].rename(columns={"refused_answer": "T"})
    turn3 = msg[(msg["role"] == "user") & (msg["message_number"] == 3)][
        ["conversation_id"]
    ].assign(Y=1)
    conv = (first.merge(turn2, on="conversation_id", how="inner")
                 .merge(turn3, on="conversation_id", how="left"))
    conv["Y"] = conv["Y"].fillna(0).astype(int)
    conv = conv.drop_duplicates("conversation_id").reset_index(drop=True)
    z = np.load(ROOT / emb, allow_pickle=True)
    id2i = {c: i for i, c in enumerate(z["conversation_ids"])}
    conv = conv[conv["conversation_id"].isin(id2i)].reset_index(drop=True)
    X_emb = z["embeddings"][[id2i[c] for c in conv["conversation_id"]]]
    risky = conv["max_score"] > RISK
    conv = conv[risky].reset_index(drop=True)
    X_emb = X_emb[risky.values]
    T = conv["T"].values.astype(int)
    Y = conv["Y"].values.astype(int)
    S = conv["max_score"].values.astype(float)
    X = np.concatenate([X_emb, S.reshape(-1, 1)], axis=1)
    return X, T, Y


def make_gbm():
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05,
        l2_regularization=1.0, random_state=0,
    )


def cross_fit_propensity(X, T, Y):
    n = len(T)
    ehat = np.zeros(n); m0 = np.zeros(n); m1 = np.zeros(n)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    for tr, te in skf.split(X, T):
        em = make_gbm().fit(X[tr], T[tr])
        ehat[te] = em.predict_proba(X[te])[:, 1]
        m0m = make_gbm().fit(X[tr][T[tr] == 0], Y[tr][T[tr] == 0])
        m1m = make_gbm().fit(X[tr][T[tr] == 1], Y[tr][T[tr] == 1])
        m0[te] = m0m.predict_proba(X[te])[:, 1]
        m1[te] = m1m.predict_proba(X[te])[:, 1]
    return np.clip(ehat, *CLIP), m0, m1


# ---------------------------------------------------------------------------
# Cross-fit nuisances once for all 19 cells
# ---------------------------------------------------------------------------
print("[propensity] cross-fitting nuisances per model")
prop_data = {}
for tag, name, ds, pkl, lab, emb, coalesce_wg in MODELS:
    try:
        X, T, Y = build_sample(pkl, lab, emb, coalesce_wg=coalesce_wg,
                               treatment_label=_args.treatment_label)
    except ValueError as e:
        print(f"  [{tag}] skipping — {e}")
        continue
    if len(T) == 0 or T.sum() == 0 or (1 - T).sum() == 0:
        print(f"  {name:30s} skipped (empty sample)")
        continue
    ehat, m0, m1 = cross_fit_propensity(X, T, Y)
    prop_data[tag] = {"name": name, "ds": ds,
                      "ehat": ehat, "T": T, "Y": Y, "m0": m0, "m1": m1}
    print(f"  {name:30s} n={len(T):>6}  P(T=1)={T.mean():.3f}")


# ---------------------------------------------------------------------------
# FIGURE 3: propensity overlap histograms (19 cells, 4x4 grid)
# ---------------------------------------------------------------------------
n_models = len(prop_data)
ncols = 4
nrows = int(np.ceil((n_models + 1) / ncols))  # +1 for legend cell
fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, 2.0 * nrows))
axes = axes.ravel()
bins = np.linspace(0, 1, 41)
for ax, (tag, d) in zip(axes[:n_models], prop_data.items()):
    e = d["ehat"]; t = d["T"]
    ax.hist(e[t == 0], bins=bins, color=C_COLOR, alpha=0.55,
            label="No refusal", edgecolor="white", linewidth=0.2)
    ax.hist(e[t == 1], bins=bins, color=T_COLOR, alpha=0.55,
            label="Refusal", edgecolor="white", linewidth=0.2)
    ax.axvspan(0.1, 0.9, ymin=0, ymax=1, color="#fff1c7", alpha=0.35, zorder=0)
    ax.set_title(d["name"], fontsize=7.5, pad=2)
    ax.set_xlim(0, 1)
    ax.tick_params(axis="y", length=2, pad=1, labelsize=6.5)
    ax.tick_params(axis="x", length=2, pad=1, labelsize=6.5)
# Last cell: legend
for ax in axes[n_models:]:
    ax.axis("off")
legend_ax = axes[n_models] if n_models < len(axes) else axes[-1]
handles = [
    plt.Rectangle((0, 0), 1, 1, fc=C_COLOR, alpha=0.55, ec="white", lw=0.2,
                  label="No refusal"),
    plt.Rectangle((0, 0), 1, 1, fc=T_COLOR, alpha=0.55, ec="white", lw=0.2,
                  label="Refusal"),
    plt.Rectangle((0, 0), 1, 1, fc="#fff1c7", alpha=0.9, ec="none",
                  label="Overlap [0.1, 0.9]"),
]
legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=8)
legend_ax.text(
    0.5, 0.12,
    "Nearly-empty overlap bands\nmean the ATT is a local\neffect on the tails.",
    ha="center", va="center", fontsize=7, color="#555555",
    transform=legend_ax.transAxes,
)
# Common axis labels for the edges
for i in range(nrows):
    axes[i * ncols].set_ylabel("Count", fontsize=7.5)
for j in range(ncols):
    idx = (nrows - 1) * ncols + j
    if idx < len(axes):
        axes[idx].set_xlabel(r"Propensity $\hat e(X)$", fontsize=7.5)
plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.6)
fig.savefig(FIG / f"fig_propensity{_FIG_SUF}.png")
print(f"wrote {FIG / f'fig_propensity{_FIG_SUF}.png'}")
plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 4: Rosenbaum Gamma curves (14 lines, grouped by corpus)
# ---------------------------------------------------------------------------
sens = pd.read_csv(ROOT / f"output/comparison/sensitivity{_LBL_SUF}/rosenbaum_gamma.csv")
score_sens = sens[sens["design"] == "score"].copy()

# Use distinct colors for each model; split LMSYS (blues/greens) vs WildChat (reds/oranges)
lmsys_tags = [t for t, n, ds, *_ in MODELS if ds == "LMSYS"]
wc_tags = [t for t, n, ds, *_ in MODELS if ds == "WildChat"]
cmap_lm = plt.cm.Blues(np.linspace(0.35, 0.95, len(lmsys_tags)))
cmap_wc = plt.cm.YlOrRd(np.linspace(0.35, 0.95, len(wc_tags)))
tag_color = {}
for i, t in enumerate(lmsys_tags):
    tag_color[t] = cmap_lm[i]
for i, t in enumerate(wc_tags):
    tag_color[t] = cmap_wc[i]

tag_label = {t: n for t, n, *_ in MODELS}

g_grid = np.linspace(1.0, 6.0, 120)
fig, ax = plt.subplots(figsize=(7.0, 4.2))
# Draw in order of most-robust first so less-robust lines come on top
score_sens["gamma_bound_05"] = pd.to_numeric(score_sens["gamma_bound_05"], errors="coerce")
score_sens_sorted = score_sens.sort_values("gamma_bound_05", ascending=False)
for _, row in score_sens_sorted.iterrows():
    m = row["model"]
    if m not in tag_color:
        continue
    b_plus = int(row["b_plus"]); b_minus = int(row["b_minus"])
    n = b_plus + b_minus
    if n == 0:
        continue
    p_vals = [binom.cdf(b_plus, n, 1.0 / (1.0 + g)) for g in g_grid]
    ax.plot(g_grid, p_vals, color=tag_color[m], lw=1.3, label=tag_label[m])
ax.axhline(0.05, color="#888888", lw=0.6, ls="--")
ax.axhline(0.01, color="#888888", lw=0.6, ls=":")
ax.text(5.95, 0.052, r"$\alpha=0.05$", ha="right", va="bottom",
        fontsize=6.5, color="#666")
ax.text(5.95, 0.012, r"$\alpha=0.01$", ha="right", va="bottom",
        fontsize=6.5, color="#666")
ax.set_xlabel(r"Rosenbaum $\Gamma$ (unobserved-confounding odds ratio)")
ax.set_ylabel("One-sided p-value (score matching)")
ax.set_xlim(1.0, 6.0)
ax.set_ylim(0, 0.6)
ax.legend(loc="center right", frameon=True, fontsize=6.2, ncol=2,
          handletextpad=0.4, borderpad=0.3, columnspacing=0.8)
ax.grid(axis="y", alpha=0.25, lw=0.4)
plt.tight_layout()
fig.savefig(FIG / f"fig_gamma_curves{_FIG_SUF}.png")
print(f"wrote {FIG / f'fig_gamma_curves{_FIG_SUF}.png'}")
plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 5: raw vs matched vs AIPW-T continuation rates, per model (2x7 grid)
# ---------------------------------------------------------------------------
match_results = json.loads(
    (ROOT / f"output/comparison/matching{_LBL_SUF}/matching_results.json").read_text()
)

rows = []
for tag, d in prop_data.items():
    name = d["name"]
    ehat, T, Y, m0, m1 = d["ehat"], d["T"], d["Y"], d["m0"], d["m1"]
    rows.append({"tag": tag, "model": name, "method": "Raw",
                 "group": "No refusal", "rate": Y[T == 0].mean()})
    rows.append({"tag": tag, "model": name, "method": "Raw",
                 "group": "Refusal", "rate": Y[T == 1].mean()})
    if tag in match_results:
        sm = match_results[tag]["score"]
        rows.append({"tag": tag, "model": name, "method": "Matched",
                     "group": "No refusal", "rate": sm["non_refused_rate"]})
        rows.append({"tag": tag, "model": name, "method": "Matched",
                     "group": "Refusal", "rate": sm["refused_rate"]})
    in_ol = (ehat >= 0.1) & (ehat <= 0.9)
    if in_ol.sum() > 20:
        rows.append({"tag": tag, "model": name, "method": "AIPW-T",
                     "group": "No refusal", "rate": m0[in_ol].mean()})
        rows.append({"tag": tag, "model": name, "method": "AIPW-T",
                     "group": "Refusal", "rate": m1[in_ol].mean()})
    else:
        rows.append({"tag": tag, "model": name, "method": "AIPW-T",
                     "group": "No refusal", "rate": np.nan})
        rows.append({"tag": tag, "model": name, "method": "AIPW-T",
                     "group": "Refusal", "rate": np.nan})
adj = pd.DataFrame(rows)

# Inclusion criterion for the "main-text" subset (mirror regenerate_headline_figs.py).
# The diagnostic thresholds are n_T >= 500, ESS_T >= 100, AIPW CI width <= 12 pp.
# Plus an explicit editorial exclusion of the regular gpt35wc cell, which is a
# strict subset of gpt35wc_wcrisky's prompt distribution and is not in the
# canonical headline panel even though it would pass the numeric thresholds.
aipw_full = json.loads(
    (ROOT / f"output/comparison/aipw{_LBL_SUF}/aipw_results.json").read_text()
)
EDITORIAL_EXCLUDE = {"gpt35wc"}


def _passes_main(tag: str) -> bool:
    if tag in EDITORIAL_EXCLUDE:
        return False
    a = aipw_full.get(tag)
    if a is None:
        return False
    ci_w = a["att_ci95_hi"] - a["att_ci95_lo"]
    return (a["n_treated"] >= 500
            and a["ess_treated"] >= 100
            and ci_w <= 12.0)


def _render_adjusted(items, out_path: Path, ncols: int):
    n_models = len(items)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 2.6 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    methods_order = ["Raw", "Matched", "AIPW-T"]
    # Display labels: "Unadj." avoids collision with the method name "RAW".
    methods_display = ["Unadj.", "Matched", "AIPW-T"]
    for ax, (tag, d) in zip(axes[:n_models], items):
        name = d["name"]
        sub = adj[adj["tag"] == tag]
        x = np.arange(len(methods_order)); w = 0.36
        nr = [sub[(sub["method"] == m) & (sub["group"] == "No refusal")]["rate"].values[0]
              for m in methods_order]
        re = [sub[(sub["method"] == m) & (sub["group"] == "Refusal")]["rate"].values[0]
              for m in methods_order]
        ax.bar(x - w/2, nr, w, color=C_COLOR, alpha=0.9,
                edgecolor="black", linewidth=0.3, label="No refusal")
        ax.bar(x + w/2, re, w, color=T_COLOR, alpha=0.9,
                edgecolor="black", linewidth=0.3, label="Refusal")
        ax.set_xticks(x)
        ax.set_xticklabels(methods_display, rotation=0, fontsize=6.5)
        ax.set_title(name, fontsize=7.5, pad=2)
        ax.set_ylim(0, 0.85)
        ax.tick_params(axis="y", length=2, pad=1, labelsize=6.5)
        ax.tick_params(axis="x", length=2, pad=1, labelsize=6.5)
        ax.grid(axis="y", alpha=0.25, lw=0.3)
    for ax in axes[n_models:]:
        ax.axis("off")
    for i in range(nrows):
        axes[i * ncols].set_ylabel("Re-engagement rate", fontsize=7.5)
    axes[0].legend(loc="upper left", frameon=True, fontsize=6.3)
    plt.tight_layout(pad=0.4, w_pad=0.4)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    plt.close(fig)


# Full 19-cell version (appendix)
_render_adjusted(list(prop_data.items()), FIG / f"fig_adjusted{_FIG_SUF}.png", ncols=7)

# Main-text 13-cell subset
main_items = [(t, d) for t, d in prop_data.items() if _passes_main(t)]
_render_adjusted(main_items, FIG / f"fig_adjusted_main{_FIG_SUF}.png", ncols=7)
print(f"main-text adjusted panel: {len(main_items)} cells")

print("done")
