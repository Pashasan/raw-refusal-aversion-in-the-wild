"""Rosenbaum Gamma sensitivity analysis for matching-based ATT estimates.

For each matched-pair analysis (score-caliper and embedding + score),
compute the largest Gamma such that the observed evidence of refusal
reducing continuation remains significant at alpha under the worst-case
unobserved confounder consistent with Gamma.

Rosenbaum (2002), "Observational Studies" 2nd ed, ch. 4; Rosenbaum
(2011) "A New U-Statistic with Superior Design Sensitivity in Matched
Observational Studies" Biometrics. The binary-outcome McNemar setup
we use is in Rosenbaum (2002) 4.3.

Interpretation of Gamma (Rosenbaum 2002 convention):
  Gamma = 1    : no unobserved confounding allowed; equivalent to
                  assuming SOO holds exactly.
  Gamma = 1.5  : unmeasured confounder with 50 percent odds of pushing
                  treatment assignment; our effect survives this.
  Gamma = 2    : unmeasured confounder doubles the treatment odds.
  Gamma = 3-4  : strong evidence; confounder would have to be very
                  large to explain away the effect.

Reports Gamma bounds at alpha in {0.05, 0.01, 0.001} per pair set.

Outputs:
  output/comparison/sensitivity/rosenbaum_gamma.json   nested per-model
  output/comparison/sensitivity/rosenbaum_gamma.csv    flat summary
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from llm_dynamics import config

ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "output" / "comparison"
OUT = OUT_BASE / "sensitivity"  # default; main() overrides based on --treatment-label


def _out_dir(treatment_label: str) -> Path:
    suffix = "" if treatment_label == "coalesce" else f"_{treatment_label}"
    out = OUT_BASE / f"sensitivity{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    return out
OUT.mkdir(parents=True, exist_ok=True)

RISK_THRESHOLD = config.RISK_THRESHOLD                # 0.01
SCORE_CALIPER = config.SCORE_MATCH_CALIPER            # 0.1
EMBED_SIM_THRESHOLD = 0.7
EMBED_SCORE_CAP = config.EMBEDDING_SCORE_CAP          # 0.01
ALPHAS = (0.05, 0.01, 0.001)
MAX_GAMMA = 50.0


# ----------------------------------------------------------------------------
# Core Rosenbaum Gamma computation
# ----------------------------------------------------------------------------
def rosenbaum_gamma_bound(
    b_plus: int, b_minus: int,
    alpha: float = 0.05, max_gamma: float = MAX_GAMMA,
) -> float:
    """Largest Gamma such that the one-sided test (b_plus small relative to
    b_plus + b_minus) remains significant at level alpha under worst-case bias.

    Our directional convention:
      b_plus  = pairs where refused-branch continued and non-refused didn't
                (evidence AGAINST our hypothesis that refusal reduces cont)
      b_minus = pairs where non-refused branch continued and refused didn't
                (evidence FOR our hypothesis)
    We observe b_minus >> b_plus when refusal aversion is real.

    Under H_0 (no treatment effect) and adversarial bias Gamma (confounder
    pushes treated units toward the refused-continued pattern when Gamma > 1,
    AGAINST our observed direction), the discordant indicator in each pair
    has probability p = 1/(1+Gamma), and b_plus ~ Binomial(N, 1/(1+Gamma)).

    Test rejects H_0 in favor of our alternative if b_plus is small enough
    that P(X <= b_plus | X ~ Binom(N, 1/(1+Gamma))) <= alpha.

    As Gamma grows, 1/(1+Gamma) shrinks, the expected b_plus shrinks, and the
    observed b_plus becomes less surprising, so the p-value grows. The
    sensitivity bound is the largest Gamma where the p-value is still <= alpha.
    """
    n = b_plus + b_minus
    if n == 0:
        return float("nan")

    def pval(g: float) -> float:
        p = 1.0 / (1.0 + g)
        return float(binom.cdf(b_plus, n, p))

    # Trivial edge cases
    if pval(1.0) > alpha:
        return 1.0                       # fragile: even at Gamma=1, fails
    if pval(max_gamma) <= alpha:
        return float(max_gamma)          # very robust: saturated at cap

    # Binary search for largest Gamma with pval(Gamma) <= alpha
    lo, hi = 1.0, max_gamma
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if pval(mid) <= alpha:
            lo = mid
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------------
# Helpers to reconstruct matched pairs per model
# ----------------------------------------------------------------------------
def _first_user_continuation(msg_df: pd.DataFrame) -> pd.DataFrame:
    """One row per conversation with (max_score, refused_answer, user_continued)
    where max_score is the first-user-message moderation score, refused_answer
    is the turn-2 assistant refusal label, user_continued is 1(turn 3 exists)."""
    first = msg_df[(msg_df["role"] == "user") &
                   (msg_df["message_number"] == 1)][
        ["conversation_id", "max_concern_score"]].rename(
            columns={"max_concern_score": "max_score"})
    turn2 = msg_df[(msg_df["role"] == "assistant") &
                   (msg_df["message_number"] == 2)][
        ["conversation_id", "refused_answer"]]
    turn3 = msg_df[(msg_df["role"] == "user") &
                   (msg_df["message_number"] == 3)][
        ["conversation_id"]].assign(user_continued=1)
    conv = (first.merge(turn2, on="conversation_id", how="inner")
                 .merge(turn3, on="conversation_id", how="left"))
    conv["user_continued"] = conv["user_continued"].fillna(0).astype(int)
    return conv.drop_duplicates("conversation_id").reset_index(drop=True)


def _score_match(conv: pd.DataFrame, caliper: float = SCORE_CALIPER
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (refused_cont, matched_cont) arrays for score-caliper matching
    on risky first-user prompts. Nearest-neighbour with replacement."""
    risky = conv[conv["max_score"] > RISK_THRESHOLD]
    ref = risky[risky["refused_answer"] == 1].reset_index(drop=True)
    non = risky[risky["refused_answer"] == 0].reset_index(drop=True)
    if len(ref) == 0 or len(non) == 0:
        return np.array([]), np.array([])
    d = np.abs(ref["max_score"].to_numpy()[:, None] - non["max_score"].to_numpy())
    nearest = np.argmin(d, axis=1)
    best = d[np.arange(len(ref)), nearest]
    keep = best <= caliper
    return (ref.loc[keep, "user_continued"].to_numpy(dtype=int),
            non.iloc[nearest[keep]]["user_continued"].to_numpy(dtype=int))


def _embedding_match(conv: pd.DataFrame, embeddings: np.ndarray,
                     conv_ids: list[str],
                     sim_threshold: float = EMBED_SIM_THRESHOLD,
                     score_cap: float = EMBED_SCORE_CAP
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return (refused_cont, matched_cont) arrays for embedding + score
    matching. Aligns embedding matrix rows to the conv dataframe by
    conversation_id lookup, then runs cosine-similarity filter + score cap."""
    id_to_pos = {cid: i for i, cid in enumerate(conv_ids)}
    have = conv["conversation_id"].isin(id_to_pos)
    conv = conv[have].reset_index(drop=True)
    emb = embeddings[[id_to_pos[c] for c in conv["conversation_id"]]]

    risky_mask = conv["max_score"] > RISK_THRESHOLD
    risky = conv[risky_mask].reset_index(drop=True)
    risky_emb = emb[risky_mask.values]
    ref_mask = risky["refused_answer"] == 1

    ref_emb = risky_emb[ref_mask.values]
    non_emb = risky_emb[~ref_mask.values]
    ref = risky[ref_mask].reset_index(drop=True)
    non = risky[~ref_mask].reset_index(drop=True)
    if len(ref) == 0 or len(non) == 0:
        return np.array([]), np.array([])

    # Cosine similarity matrix (embeddings already L2-normalised in clustering step)
    rn = ref_emb / (np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-12)
    cn = non_emb / (np.linalg.norm(non_emb, axis=1, keepdims=True) + 1e-12)
    sims = rn @ cn.T
    nearest = np.argmax(sims, axis=1)
    best_sim = sims[np.arange(len(ref)), nearest]
    pass1 = best_sim >= sim_threshold

    ref_1 = ref[pass1].reset_index(drop=True)
    near_1 = nearest[pass1]
    non_1 = non.iloc[near_1].reset_index(drop=True)

    score_diff = np.abs(
        ref_1["max_score"].to_numpy(dtype=float) -
        non_1["max_score"].to_numpy(dtype=float)
    )
    pass2 = score_diff <= score_cap
    return (ref_1.loc[pass2, "user_continued"].to_numpy(dtype=int),
            non_1.loc[pass2, "user_continued"].to_numpy(dtype=int))


# ----------------------------------------------------------------------------
# Per-model runner
# ----------------------------------------------------------------------------
@dataclass
class SensitivityResult:
    design: str                    # 'score' or 'embedding'
    n_pairs: int
    b_plus: int                    # refused continued and non-refused did not
    b_minus: int                   # non-refused continued and refused did not
    n_both: int                    # both continued
    n_neither: int                 # neither continued
    pp_difference: float           # (refused_cont - non_ref_cont) * 100
    gamma_bound_05: float
    gamma_bound_01: float
    gamma_bound_001: float
    mcnemar_chi2: float
    mcnemar_p: float


def _mcnemar(b_plus: int, b_minus: int) -> tuple[float, float]:
    n = b_plus + b_minus
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b_plus - b_minus) - 1) ** 2 / n
    from scipy.stats import chi2 as chi2_dist
    p = 1 - float(chi2_dist.cdf(chi2, df=1))
    return float(chi2), p


def run_one(tag: str, msg_pkl: str, labels_csv: str, emb_npz: str,
            coalesce_wg: bool = False,
            treatment_label: str = "coalesce") -> dict[str, SensitivityResult]:
    """treatment_label in {coalesce (legacy), gpt54, wg}. See aipw_with_gpt54.build_aipw_sample."""
    msg_p, lab_p, emb_p = ROOT/msg_pkl, ROOT/labels_csv, ROOT/emb_npz
    if not all(p.exists() for p in (msg_p, lab_p, emb_p)):
        return {}

    with open(msg_p, "rb") as f:
        msg = pickle.load(f)
    lab = pd.read_csv(lab_p)
    msg = msg.drop(columns=[c for c in ["refused_answer"] if c in msg.columns])

    if treatment_label == "wg":
        if "is_refusal_wg" not in lab.columns:
            raise ValueError(f"{labels_csv} missing is_refusal_wg")
        msg = msg.merge(lab[["conversation_id","message_number","is_refusal_wg"]],
                        on=["conversation_id","message_number"], how="left")
        msg["refused_answer"] = msg["is_refusal_wg"].fillna(0).astype(int)
    elif treatment_label == "gpt54":
        if "is_refusal_gpt54" not in lab.columns:
            raise ValueError(f"{labels_csv} missing is_refusal_gpt54")
        msg = msg.merge(lab[["conversation_id","message_number","is_refusal_gpt54"]],
                        on=["conversation_id","message_number"], how="left")
        msg["refused_answer"] = msg["is_refusal_gpt54"].fillna(0).astype(int)
    elif treatment_label == "coalesce":
        if "is_refusal_gpt54" not in lab.columns:
            if not coalesce_wg:
                raise ValueError(f"{labels_csv} missing is_refusal_gpt54 (set coalesce_wg=True for wg-only cells)")
            lab["is_refusal_gpt54"] = pd.NA
        cols = ["conversation_id", "message_number", "is_refusal_gpt54"]
        if coalesce_wg:
            if "is_refusal_wg" not in lab.columns:
                raise ValueError(f"{labels_csv} missing is_refusal_wg (coalesce_wg=True requested)")
            cols.append("is_refusal_wg")
        msg = msg.merge(lab[cols], on=["conversation_id","message_number"], how="left")
        if coalesce_wg:
            coalesced = msg["is_refusal_gpt54"].combine_first(msg["is_refusal_wg"])
            msg["refused_answer"] = coalesced.fillna(0).astype(int)
        else:
            msg["refused_answer"] = msg["is_refusal_gpt54"].fillna(0).astype(int)
    else:
        raise ValueError(f"unknown treatment_label: {treatment_label!r}")

    conv = _first_user_continuation(msg)
    z = np.load(emb_p, allow_pickle=True)

    out: dict[str, SensitivityResult] = {}
    # Score-only matching
    ref_c, non_c = _score_match(conv)
    if len(ref_c):
        b_plus  = int(np.sum((ref_c == 1) & (non_c == 0)))
        b_minus = int(np.sum((ref_c == 0) & (non_c == 1)))
        n_both    = int(np.sum((ref_c == 1) & (non_c == 1)))
        n_neither = int(np.sum((ref_c == 0) & (non_c == 0)))
        chi2, p = _mcnemar(b_plus, b_minus)
        out["score"] = SensitivityResult(
            design="score",
            n_pairs=len(ref_c),
            b_plus=b_plus, b_minus=b_minus,
            n_both=n_both, n_neither=n_neither,
            pp_difference=float((ref_c.mean() - non_c.mean()) * 100),
            gamma_bound_05=rosenbaum_gamma_bound(b_plus, b_minus, 0.05),
            gamma_bound_01=rosenbaum_gamma_bound(b_plus, b_minus, 0.01),
            gamma_bound_001=rosenbaum_gamma_bound(b_plus, b_minus, 0.001),
            mcnemar_chi2=chi2, mcnemar_p=p,
        )

    # Embedding + score matching
    ref_c, non_c = _embedding_match(conv, z["embeddings"], list(z["conversation_ids"]))
    if len(ref_c):
        b_plus  = int(np.sum((ref_c == 1) & (non_c == 0)))
        b_minus = int(np.sum((ref_c == 0) & (non_c == 1)))
        n_both    = int(np.sum((ref_c == 1) & (non_c == 1)))
        n_neither = int(np.sum((ref_c == 0) & (non_c == 0)))
        chi2, p = _mcnemar(b_plus, b_minus)
        out["embedding"] = SensitivityResult(
            design="embedding",
            n_pairs=len(ref_c),
            b_plus=b_plus, b_minus=b_minus,
            n_both=n_both, n_neither=n_neither,
            pp_difference=float((ref_c.mean() - non_c.mean()) * 100),
            gamma_bound_05=rosenbaum_gamma_bound(b_plus, b_minus, 0.05),
            gamma_bound_01=rosenbaum_gamma_bound(b_plus, b_minus, 0.01),
            gamma_bound_001=rosenbaum_gamma_bound(b_plus, b_minus, 0.001),
            mcnemar_chi2=chi2, mcnemar_p=p,
        )
    return out


# Canonical 19-cell panel (mirrors aipw_with_gpt54.py MODELS).
# fields: (tag, pkl, labels_csv, embeddings, coalesce_wg)
MODELS = [
    # ---- LMSYS ----
    ("base",            "llama13b_message_df_with_users.pkl",
                        "llama13b_refusal_data_dynamic_with_labels_gpt54.csv",
                        "llama13b_first_user_embeddings.npz", False),
    ("chat",            "gpt35_message_df_with_users.pkl",
                        "refusal_data_dynamic_with_labels_gpt54.csv",
                        "first_user_embeddings.npz", False),
    ("vicuna13b",       "vicuna13b_message_df_with_users.pkl",
                        "vicuna13b_refusal_data_dynamic_with_labels_gpt54.csv",
                        "vicuna13b_first_user_embeddings.npz", False),
    ("vicuna33b",       "vicuna33b_message_df_with_users.pkl",
                        "vicuna33b_refusal_data_dynamic_with_labels_gpt54.csv",
                        "vicuna33b_first_user_embeddings.npz", False),
    ("koala13b",        "koala13b_message_df_with_users.pkl",
                        "koala13b_refusal_data_dynamic_with_labels_gpt54.csv",
                        "koala13b_first_user_embeddings.npz", False),
    ("gpt35turbo",      "gpt35turbo_message_df_with_users.pkl",
                        "gpt35turbo_refusal_data_dynamic_with_labels_gpt54.csv",
                        "gpt35turbo_first_user_embeddings.npz", False),
    ("gpt4lmsys",       "gpt4lmsys_message_df_with_users.pkl",
                        "gpt4lmsys_refusal_data_dynamic_with_labels_gpt54.csv",
                        "gpt4lmsys_first_user_embeddings.npz", False),
    ("claude1",         "claude1_message_df_with_users.pkl",
                        "claude1_refusal_data_dynamic_with_labels_gpt54.csv",
                        "claude1_first_user_embeddings.npz", False),
    ("claudeinstant1",  "claudeinstant1_message_df_with_users.pkl",
                        "claudeinstant1_refusal_data_dynamic_with_labels_gpt54.csv",
                        "claudeinstant1_first_user_embeddings.npz", False),
    ("gpt4o_lmsys",     "gpt4o_message_df_with_users.pkl",
                        "gpt4o_refusal_data_dynamic_with_labels_gpt54.csv",
                        "gpt4o_first_user_embeddings.npz", False),
    ("gpt4omini_lmsys", "gpt4omini_message_df_with_users.pkl",
                        "gpt4omini_refusal_data_dynamic_with_labels_gpt54.csv",
                        "gpt4omini_first_user_embeddings.npz", False),
    ("gpt41mini_lmsys", "gpt41mini_message_df_with_users.pkl",
                        "gpt41mini_refusal_data_dynamic_with_labels_gpt54.csv",
                        "gpt41mini_first_user_embeddings.npz", False),
    # ---- WildChat / regular ----
    ("gpt35wc",         "gpt35wc_message_df_with_users.pkl",
                        "gpt35wc_refusal_data_dynamic_with_labels_gpt54.csv",
                        "gpt35wc_first_user_embeddings.npz", False),
    # ---- WildChat / risky-enriched (gpt54 + wildguard) ----
    ("gpt4o_wcrisky",        "gpt4o_risky_add_message_df_with_users.pkl",
                             "gpt4o_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                             "gpt4o_risky_add_first_user_embeddings.npz", True),
    ("gpt4omini_wcrisky",    "gpt4omini_risky_add_message_df_with_users.pkl",
                             "gpt4omini_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                             "gpt4omini_risky_add_first_user_embeddings.npz", True),
    ("gpt41mini_wcrisky",    "gpt41mini_risky_add_message_df_with_users.pkl",
                             "gpt41mini_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                             "gpt41mini_risky_add_first_user_embeddings.npz", True),
    ("gpt35wc_wcrisky",      "gpt35wc_risky_add_message_df_with_users.pkl",
                             "gpt35wc_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                             "gpt35wc_risky_add_first_user_embeddings.npz", True),
    # ---- New WildChat-risky cells (wildguard-only) ----
    ("gpt4_0314_wcrisky",    "gpt4_0314_risky_add_message_df_with_users.pkl",
                             "gpt4_0314_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                             "gpt4_0314_risky_add_first_user_embeddings.npz", True),
    ("gpt4_1106_wcrisky",    "gpt4_1106_risky_add_message_df_with_users.pkl",
                             "gpt4_1106_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                             "gpt4_1106_risky_add_first_user_embeddings.npz", True),
    ("gpt4_0125_wcrisky",    "gpt4_0125_risky_add_message_df_with_users.pkl",
                             "gpt4_0125_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                             "gpt4_0125_risky_add_first_user_embeddings.npz", True),
    # o1mini dropped: structurally undefined ATT (Y identically 0). See §3.
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None,
                    help="comma-separated subset of MODELS tags. "
                         "If set, results are merged into existing rosenbaum_gamma.json/csv.")
    ap.add_argument("--treatment-label", dest="treatment_label",
                    choices=["coalesce", "gpt54", "wg"], default="coalesce",
                    help="Which judge supplies the treatment label.")
    args = ap.parse_args()
    selected = set(args.tags.split(",")) if args.tags else None
    if selected is not None:
        unknown = selected - {m[0] for m in MODELS}
        if unknown:
            sys.exit(f"unknown tags: {sorted(unknown)}")

    global OUT
    OUT = _out_dir(args.treatment_label)
    print(f"treatment_label={args.treatment_label}  ->  output dir: {OUT}")

    results: dict[str, dict] = {}
    if selected is not None and (OUT / "rosenbaum_gamma.json").exists():
        results = json.loads((OUT / "rosenbaum_gamma.json").read_text())
        print(f"merge mode: loaded {len(results)} existing cells from rosenbaum_gamma.json")

    for tag, pkl, lab, emb, coalesce_wg in MODELS:
        if selected is not None and tag not in selected:
            continue
        print(f"\n{'='*72}\nMODEL: {tag}  (treatment_label={args.treatment_label}, coalesce_wg={coalesce_wg})\n{'='*72}")
        try:
            r = run_one(tag, pkl, lab, emb, coalesce_wg=coalesce_wg,
                        treatment_label=args.treatment_label)
        except ValueError as e:
            print(f"  [{tag}] skipping — {e}"); continue
        if not r:
            print(f"  [{tag}] missing inputs; skipping"); continue
        results[tag] = {d: asdict(v) for d, v in r.items()}
        for design, sr in r.items():
            print(f"  --- {design} matching ---")
            print(f"    n_pairs = {sr.n_pairs:,}   discordant: b_plus={sr.b_plus}  b_minus={sr.b_minus}   "
                  f"concordant: both={sr.n_both} neither={sr.n_neither}")
            print(f"    pp_difference = {sr.pp_difference:+.2f}    "
                  f"McNemar chi2={sr.mcnemar_chi2:.2f}  p={sr.mcnemar_p:.4g}")
            print(f"    Rosenbaum Gamma bound:   "
                  f"alpha=0.05 -> {sr.gamma_bound_05:.3f}    "
                  f"alpha=0.01 -> {sr.gamma_bound_01:.3f}    "
                  f"alpha=0.001 -> {sr.gamma_bound_001:.3f}")

    # Rebuild flat csv from full (possibly merged) results dict
    rows = []
    for tag, designs in results.items():
        for design, sr_dict in designs.items():
            rows.append({"model": tag, **sr_dict})
    (OUT / "rosenbaum_gamma.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(rows).to_csv(OUT / "rosenbaum_gamma.csv", index=False)
    print(f"\nWrote {OUT/'rosenbaum_gamma.json'} ({len(results)} cells) "
          f"and {OUT/'rosenbaum_gamma.csv'}")


if __name__ == "__main__":
    main()
