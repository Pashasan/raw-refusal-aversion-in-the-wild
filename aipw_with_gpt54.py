"""Cross-fit Augmented Inverse Probability Weighting (AIPW) / Double ML
estimates of the ATT and ATE of refusal on first-turn continuation.

Same identifying assumption as the matching design — selection on
observables given (first-user-prompt embedding, moderation score) —
just a different estimator. Cross-fit AIPW with gradient-boosted
nuisance learners (Robins, Rotnitzky & Zhao, JASA 1994; Chernozhukov
et al., Econometrics J 2018; Kennedy, Stat Sci 2024).

Per-model output includes overlap diagnostics (propensity AUC,
fraction in [0.1, 0.9], effective sample size, max IPW weight) and a
Crump-trimmed estimate restricted to the overlap region (Crump, Hotz,
Imbens & Mitnik, Biometrika 2009).

Outputs:
  output/comparison/aipw/aipw_results.json  — full per-model nested results
  output/comparison/aipw/aipw_summary.csv   — flat one-row-per-model summary
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "output" / "comparison"
OUT = OUT_BASE / "aipw"  # default; main() overrides based on --treatment-label
OUT.mkdir(parents=True, exist_ok=True)


def _out_dir(treatment_label: str) -> Path:
    suffix = "" if treatment_label == "coalesce" else f"_{treatment_label}"
    out = OUT_BASE / f"aipw{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    return out

RISK_THRESHOLD = 0.01     # matches config.RISK_THRESHOLD
K_FOLDS = 5
PROP_CLIP = (0.02, 0.98)  # clip extreme propensities for stable IPW weights
TRIM_REGION = (0.1, 0.9)  # Crump et al. 2009 overlap region for trimmed estimates
SEED = 42


# ----------------------------------------------------------------------------
# Single nuisance learner: gradient-boosted classification trees
# ----------------------------------------------------------------------------
def make_gbm() -> Callable:
    """Default nuisance learner: gradient-boosted classification trees.
    Regularization via depth + learning rate keeps propensities off the
    boundary; handles the high-dim mxbai embedding natively."""
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05,
        l2_regularization=1.0, random_state=0,
    )


# ----------------------------------------------------------------------------
# AIPW core
# ----------------------------------------------------------------------------
@dataclass
class AIPWResult:
    # --- sample + treatment-prevalence stats ---
    n: int
    n_treated: int
    n_control: int
    p_treated: float                # P(T=1)
    cont_no_refusal: float          # P(Y=1 | T=0)
    cont_refusal: float             # P(Y=1 | T=1)
    raw_diff_pp: float              # (cont_ref - cont_no_ref) * 100

    # --- AIPW headline estimates (cross-fit, GBM nuisances, full sample) ---
    ate_pp: float
    ate_se_pp: float
    ate_ci95_lo: float
    ate_ci95_hi: float
    att_pp: float
    att_se_pp: float
    att_ci95_lo: float
    att_ci95_hi: float

    # --- decomposition into the two halves of AIPW (ATE-form) ---
    g_comp_pp: float                # outcome regression alone (G-computation)
    ipw_pp: float                   # IPW (Horvitz-Thompson) alone

    # --- propensity / overlap diagnostics ---
    prop_mean: float
    prop_median: float
    prop_min: float
    prop_max: float
    prop_frac_below_01: float       # treatment-rare region — high = bad overlap for ATT
    prop_frac_above_09: float       # treatment-frequent region — high = bad overlap for ATC
    prop_frac_in_overlap: float     # fraction with e in [0.1, 0.9]
    prop_auc: float                 # AUC of propensity vs T; ~0.5 = good overlap, ~1.0 = treatment fully predictable
    ess_treated: float              # effective sample size of treated IPW weights (T/e)
    ess_control: float              # effective sample size of control IPW weights ((1-T)/(1-e))
    max_weight_treated: float       # largest single treated IPW weight
    max_weight_control: float       # largest single control IPW weight

    # --- Crump-trimmed (overlap-region only) AIPW ---
    n_trimmed: int
    n_treated_trimmed: int
    ate_trimmed_pp: float
    ate_trimmed_se_pp: float
    att_trimmed_pp: float
    att_trimmed_se_pp: float

    def as_dict(self) -> dict:
        return asdict(self)


def _psi_arrays(
    T: np.ndarray, Y: np.ndarray, ehat: np.ndarray, m0: np.ndarray, m1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-observation influence-function contributions (ATE, ATT).
    Same formulas as _aipw_from_nuisances; used for cluster-robust SEs."""
    psi_ate = (m1 - m0) + T*(Y - m1)/ehat - (1-T)*(Y - m0)/(1 - ehat)
    p_T = float(T.mean())
    psi_att = T*(Y - m0)/p_T - (1-T)*ehat*(Y - m0)/((1 - ehat)*p_T)
    return psi_ate, psi_att


def _cluster_se(psi: np.ndarray, cluster_ids: np.ndarray) -> tuple[float, int]:
    """Cluster-robust SE of mean(psi): with S_c = sum_{i in c} (psi_i - psi_bar),
    se = sqrt( G/(G-1) * sum_c S_c^2 ) / n."""
    n = len(psi)
    centered = psi - psi.mean()
    sums = pd.Series(centered).groupby(pd.Series(cluster_ids)).sum().to_numpy()
    G = len(sums)
    se = float(np.sqrt(G / (G - 1) * np.sum(sums ** 2)) / n)
    return se, G


def _cross_fit_propensity_auc(
    X: np.ndarray, T: np.ndarray,
    K: int = K_FOLDS, clip: tuple[float, float] = PROP_CLIP, seed: int = SEED,
) -> float:
    """Cross-fit ONLY the propensity model on X and return AUC(ehat, T).
    Used to compare propensity AUC with vs without an added covariate."""
    n = len(T)
    ehat = np.zeros(n)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, T):
        em = make_gbm().fit(X[tr], T[tr])
        ehat[te] = em.predict_proba(X[te])[:, 1]
    ehat = np.clip(ehat, *clip)
    try:
        return float(roc_auc_score(T, ehat))
    except ValueError:
        return float("nan")


def _aipw_from_nuisances(
    T: np.ndarray, Y: np.ndarray, ehat: np.ndarray, m0: np.ndarray, m1: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute (ATE, SE_ATE, ATT, SE_ATT) given clipped nuisance estimates.

    ATE influence function (Robins-Rotnitzky-Zhao, JASA 1994):
        psi_ATE_i = (m1 - m0)(X_i)
                  + T_i (Y_i - m1(X_i)) / e(X_i)
                  - (1-T_i)(Y_i - m0(X_i)) / (1 - e(X_i))

    ATT influence function (Hirano-Imbens-Ridder):
        psi_ATT_i = T_i (Y_i - m0(X_i)) / P(T=1)
                  - (1-T_i) e(X_i) (Y_i - m0(X_i)) / ((1 - e(X_i)) P(T=1))
    """
    n = len(T)
    psi_ate = (m1 - m0) + T*(Y - m1)/ehat - (1-T)*(Y - m0)/(1 - ehat)
    ate = float(psi_ate.mean())
    ate_se = float(psi_ate.std(ddof=1) / np.sqrt(n))
    p_T = float(T.mean())
    psi_att = T*(Y - m0)/p_T - (1-T)*ehat*(Y - m0)/((1 - ehat)*p_T)
    att = float(psi_att.mean())
    att_se = float(psi_att.std(ddof=1) / np.sqrt(n))
    return ate, ate_se, att, att_se


def cross_fit_aipw(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    K: int = K_FOLDS, clip: tuple[float, float] = PROP_CLIP, seed: int = SEED,
    return_extras: bool = False,
) -> AIPWResult | tuple[AIPWResult, dict]:
    """Cross-fit AIPW: fit nuisances on K-1 folds, evaluate on held-out fold.

    return_extras=True (opt-in; default preserves legacy behavior) additionally
    returns {"ehat","m0","m1"} so callers can compute cluster-robust SEs from
    the per-observation influence functions."""
    n = len(T)
    ehat = np.zeros(n); m0 = np.zeros(n); m1 = np.zeros(n)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, T):
        em = make_gbm().fit(X[tr], T[tr])
        ehat[te] = em.predict_proba(X[te])[:, 1]
        m0m = make_gbm().fit(X[tr][T[tr] == 0], Y[tr][T[tr] == 0])
        m1m = make_gbm().fit(X[tr][T[tr] == 1], Y[tr][T[tr] == 1])
        m0[te] = m0m.predict_proba(X[te])[:, 1]
        m1[te] = m1m.predict_proba(X[te])[:, 1]
    ehat = np.clip(ehat, *clip)

    # --- headline estimates on full sample ---
    ate, ate_se, att, att_se = _aipw_from_nuisances(T, Y, ehat, m0, m1)

    # --- diagnostics ---
    raw_diff = float(Y[T == 1].mean() - Y[T == 0].mean())
    g_comp = float((m1 - m0).mean())
    ipw = float(np.mean(T*Y/ehat - (1-T)*Y/(1 - ehat)))

    # IPW weights and ESS
    w_treated = T / ehat
    w_control = (1 - T) / (1 - ehat)
    ess_t = float((w_treated[T == 1].sum()) ** 2 / (w_treated[T == 1] ** 2).sum())
    ess_c = float((w_control[T == 0].sum()) ** 2 / (w_control[T == 0] ** 2).sum())

    # AUC of propensity model (how predictable is T from X)
    try:
        auc = float(roc_auc_score(T, ehat))
    except ValueError:
        auc = float("nan")

    in_overlap = (ehat >= TRIM_REGION[0]) & (ehat <= TRIM_REGION[1])

    # --- Crump-trimmed AIPW: re-estimate on overlap region only ---
    if in_overlap.sum() > 50 and T[in_overlap].sum() > 5 and (1 - T[in_overlap]).sum() > 5:
        ate_t, ate_t_se, att_t, att_t_se = _aipw_from_nuisances(
            T[in_overlap], Y[in_overlap],
            ehat[in_overlap], m0[in_overlap], m1[in_overlap],
        )
    else:
        ate_t = ate_t_se = att_t = att_t_se = float("nan")

    result = AIPWResult(
        n=n, n_treated=int(T.sum()), n_control=int((1 - T).sum()),
        p_treated=float(T.mean()),
        cont_no_refusal=float(Y[T == 0].mean()),
        cont_refusal=float(Y[T == 1].mean()),
        raw_diff_pp=raw_diff * 100,
        ate_pp=ate*100, ate_se_pp=ate_se*100,
        ate_ci95_lo=(ate - 1.96*ate_se)*100, ate_ci95_hi=(ate + 1.96*ate_se)*100,
        att_pp=att*100, att_se_pp=att_se*100,
        att_ci95_lo=(att - 1.96*att_se)*100, att_ci95_hi=(att + 1.96*att_se)*100,
        g_comp_pp=g_comp*100, ipw_pp=ipw*100,
        prop_mean=float(ehat.mean()), prop_median=float(np.median(ehat)),
        prop_min=float(ehat.min()), prop_max=float(ehat.max()),
        prop_frac_below_01=float((ehat < 0.1).mean()),
        prop_frac_above_09=float((ehat > 0.9).mean()),
        prop_frac_in_overlap=float(in_overlap.mean()),
        prop_auc=auc,
        ess_treated=ess_t, ess_control=ess_c,
        max_weight_treated=float(w_treated.max()),
        max_weight_control=float(w_control.max()),
        n_trimmed=int(in_overlap.sum()),
        n_treated_trimmed=int(T[in_overlap].sum()),
        ate_trimmed_pp=ate_t*100, ate_trimmed_se_pp=ate_t_se*100,
        att_trimmed_pp=att_t*100, att_trimmed_se_pp=att_t_se*100,
    )
    if return_extras:
        return result, {"ehat": ehat, "m0": m0, "m1": m1}
    return result


# ----------------------------------------------------------------------------
# Per-model data prep (risky first-user prompts, same sample as matching)
# ----------------------------------------------------------------------------
def build_aipw_sample(msg_pkl: str, labels_csv: str, emb_npz: str,
                      coalesce_wg: bool = False,
                      treatment_label: str = "coalesce",
                      return_ids: bool = False,
                      variant_opts: dict | None = None,
                      ) -> tuple[np.ndarray, ...] | None:
    """Return (X, T, Y) for AIPW on the risky first-user subset of a model.

    return_ids=True (opt-in; default preserves legacy behavior) additionally
    returns the aligned conversation_id array: (X, T, Y, conv_ids).

    variant_opts (opt-in; default None preserves legacy behavior) enables the
    robustness variants. Dict keys:
      meta: DataFrame with conversation_hash, timestamp, hashed_ip,
            temperature, top_p (or None)
      cluster: "ip" | "promptcluster" | None
      add_time / add_temp / restrict_modal_temp: bool
      length_stratum: "short" | "long" | None — restrict TREATED
            conversations to first-assistant-response character lengths
            <= (short) or > (long) the median among treated conversations
            in the analysis sample; controls untouched.
    When set, returns (X, T, Y, info) where info carries the aligned
    cluster_ids array, X_base (X without the time column, for the AUC
    comparison), and bookkeeping counts.

    treatment_label selects which judge supplies T:
      "coalesce" (legacy): per-cell coalesce_wg flag — gpt54 with optional
                  wg fallback for WildChat-risky cells.
      "gpt54":    is_refusal_gpt54 everywhere (NaN -> 0).
      "wg":       is_refusal_wg everywhere (NaN -> 0). Panel-consistent
                  single-judge pipeline."""
    msg_path, lab_path, emb_path = ROOT/msg_pkl, ROOT/labels_csv, ROOT/emb_npz
    if not all(p.exists() for p in (msg_path, lab_path, emb_path)):
        return None

    with open(msg_path, "rb") as f:
        msg = pickle.load(f)
    lab = pd.read_csv(lab_path)
    msg = msg.drop(columns=[c for c in ["refused_answer"] if c in msg.columns])

    if treatment_label == "wg":
        if "is_refusal_wg" not in lab.columns:
            raise ValueError(f"{labels_csv} missing is_refusal_wg")
        msg = msg.merge(lab[["conversation_id", "message_number", "is_refusal_wg"]],
                        on=["conversation_id", "message_number"], how="left")
        msg["refused_answer"] = msg["is_refusal_wg"].fillna(0).astype(int)
    elif treatment_label == "gpt54":
        if "is_refusal_gpt54" not in lab.columns:
            raise ValueError(f"{labels_csv} missing is_refusal_gpt54 "
                             f"(use --treatment-label wg or coalesce)")
        msg = msg.merge(lab[["conversation_id", "message_number", "is_refusal_gpt54"]],
                        on=["conversation_id", "message_number"], how="left")
        msg["refused_answer"] = msg["is_refusal_gpt54"].fillna(0).astype(int)
    elif treatment_label == "coalesce":
        if "is_refusal_gpt54" not in lab.columns:
            if not coalesce_wg:
                raise ValueError(f"{labels_csv} missing is_refusal_gpt54 (set coalesce_wg=True for wg-only cells)")
            lab["is_refusal_gpt54"] = pd.NA
        cols = ["conversation_id", "message_number", "is_refusal_gpt54"]
        if coalesce_wg:
            if "is_refusal_wg" not in lab.columns:
                raise ValueError(f"{labels_csv} missing is_refusal_wg "
                                 f"(needed because coalesce_wg=True)")
            cols.append("is_refusal_wg")
        msg = msg.merge(lab[cols], on=["conversation_id", "message_number"],
                        how="left")
        if coalesce_wg:
            coalesced = msg["is_refusal_gpt54"].combine_first(msg["is_refusal_wg"])
            msg["refused_answer"] = coalesced.fillna(0).astype(int)
        else:
            msg["refused_answer"] = msg["is_refusal_gpt54"].fillna(0).astype(int)
    else:
        raise ValueError(f"unknown treatment_label: {treatment_label!r}")

    fu_cols = ["conversation_id", "max_concern_score"]
    if variant_opts and variant_opts.get("cluster") == "promptcluster":
        if "user_id" not in msg.columns:
            raise ValueError(f"{msg_pkl} has no user_id column (needed for "
                             f"--cluster promptcluster)")
        fu_cols.append("user_id")
    first_user = msg[(msg["role"] == "user") & (msg["message_number"] == 1)][
        fu_cols
    ].rename(columns={"max_concern_score": "max_score"})
    need_len = bool(variant_opts and variant_opts.get("length_stratum"))
    t2_cols = ["conversation_id", "refused_answer"] + (
        ["content"] if need_len else [])
    turn2 = msg[(msg["role"] == "assistant") & (msg["message_number"] == 2)][
        t2_cols
    ].rename(columns={"refused_answer": "T"})
    if need_len:
        turn2["first_resp_len"] = (
            turn2.pop("content").fillna("").astype(str).str.len())
    turn3 = msg[(msg["role"] == "user") & (msg["message_number"] == 3)][
        ["conversation_id"]
    ].assign(Y=1)
    conv = (first_user.merge(turn2, on="conversation_id", how="inner")
                       .merge(turn3, on="conversation_id", how="left"))
    conv["Y"] = conv["Y"].fillna(0).astype(int)
    conv = conv.drop_duplicates("conversation_id").reset_index(drop=True)

    emb = np.load(emb_path, allow_pickle=True)
    id2i = {c: i for i, c in enumerate(emb["conversation_ids"])}
    conv = conv[conv["conversation_id"].isin(id2i)].reset_index(drop=True)
    X_emb = emb["embeddings"][[id2i[c] for c in conv["conversation_id"]]]

    risky = conv["max_score"] > RISK_THRESHOLD
    conv = conv[risky].reset_index(drop=True)
    X_emb = X_emb[risky.values]
    if len(conv) == 0 or conv["T"].sum() == 0 or (1 - conv["T"]).sum() == 0:
        return None

    info: dict = {}
    if variant_opts:
        meta = variant_opts.get("meta")
        if meta is not None:
            # left-join on unique conversation_hash keeps row order and count,
            # so X_emb stays aligned with conv.
            conv = conv.merge(meta, left_on="conversation_id",
                              right_on="conversation_hash", how="left")
            info["n_meta_unmatched"] = int(conv["conversation_hash"].isna().sum())
        if variant_opts.get("restrict_modal_temp"):
            if meta is None:
                raise ValueError("restrict_modal_temp requires meta parquet")
            modal_t = float(conv["temperature"].mode(dropna=True).iloc[0])
            modal_p = float(conv["top_p"].mode(dropna=True).iloc[0])
            keep = ((conv["temperature"] == modal_t)
                    & (conv["top_p"] == modal_p)).to_numpy()
            info["modal_temperature"] = modal_t
            info["modal_top_p"] = modal_p
            info["n_dropped_modal_temp"] = int((~keep).sum())
            conv = conv[keep].reset_index(drop=True)
            X_emb = X_emb[keep]
            if len(conv) == 0 or conv["T"].sum() == 0 or (1 - conv["T"]).sum() == 0:
                return None
        stratum = variant_opts.get("length_stratum")
        if stratum:
            treated = conv["T"].to_numpy(dtype=bool)
            med = float(conv.loc[treated, "first_resp_len"].median())
            if stratum == "short":
                in_stratum = conv["first_resp_len"].to_numpy() <= med
            else:
                in_stratum = conv["first_resp_len"].to_numpy() > med
            keep = ~treated | in_stratum   # controls always kept
            info["length_stratum"] = stratum
            info["length_median_treated"] = med
            info["n_treated_prestratum"] = int(treated.sum())
            info["n_treated_stratum"] = int((treated & keep).sum())
            info["n_dropped_length_stratum"] = int((~keep).sum())
            conv = conv[keep].reset_index(drop=True)
            X_emb = X_emb[keep]
            if len(conv) == 0 or conv["T"].sum() == 0 or (1 - conv["T"]).sum() == 0:
                return None

    T = conv["T"].values.astype(int)
    Y = conv["Y"].values.astype(int)
    S = conv["max_score"].values.astype(float)
    X = np.concatenate([X_emb, S.reshape(-1, 1)], axis=1)

    if variant_opts:
        parts = [X]
        if variant_opts.get("add_temp"):
            t_raw, p_raw = conv["temperature"], conv["top_p"]
            modal_t = float(t_raw.mode(dropna=True).iloc[0])
            modal_p = float(p_raw.mode(dropna=True).iloc[0])
            miss = (t_raw.isna() | p_raw.isna()).to_numpy(dtype=float)
            t_f = t_raw.fillna(modal_t).to_numpy(dtype=float)
            p_f = p_raw.fillna(modal_p).to_numpy(dtype=float)
            parts += [t_f.reshape(-1, 1), p_f.reshape(-1, 1),
                      miss.reshape(-1, 1)]
            info["addtemp_modal_temperature"] = modal_t
            info["addtemp_modal_top_p"] = modal_p
            info["addtemp_n_missing"] = int(miss.sum())
        X_base = np.concatenate(parts, axis=1) if len(parts) > 1 else X
        if variant_opts.get("add_time"):
            ts = pd.to_datetime(conv["timestamp"])
            n_null_ts = int(ts.isna().sum())
            if n_null_ts:
                ts = ts.fillna(ts.min())
            t_days = (ts - ts.min()).dt.total_seconds().to_numpy() / 86400.0
            span = max(float(t_days.max()), 1.0)
            X_full = np.concatenate([X_base, (t_days / span).reshape(-1, 1)],
                                    axis=1)
            info["time_n_null_ts"] = n_null_ts
            info["time_span_days"] = float(t_days.max())
            info["ts_min"] = str(ts.min())
            info["ts_max"] = str(ts.max())
        else:
            X_full = X_base
        cl = variant_opts.get("cluster")
        if cl == "ip":
            ids = conv["hashed_ip"].astype(object).to_numpy().copy()
            null_mask = pd.isna(ids)
            info["cluster_n_null_ids"] = int(null_mask.sum())
            for i in np.flatnonzero(null_mask):   # unmatched -> singleton
                ids[i] = f"__singleton_{i}"
            cluster_ids = ids
        elif cl == "promptcluster":
            ids = conv["user_id"].astype(object).to_numpy().copy()
            null_mask = pd.isna(ids)
            info["cluster_n_null_ids"] = int(null_mask.sum())
            for i in np.flatnonzero(null_mask):   # null id -> singleton
                ids[i] = f"__singleton_{i}"
            cluster_ids = ids
        else:
            cluster_ids = None
        info["cluster_ids"] = cluster_ids
        info["X_base"] = X_base if variant_opts.get("add_time") else None
        return X_full, T, Y, info

    if return_ids:
        return X, T, Y, conv["conversation_id"].to_numpy()
    return X, T, Y


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
# Per-(dataset, model) cells. Tag suffix encodes the prompt source so each row
# is one (dataset, model) bucket — no within-row mixing of LMSYS and WildChat.
# coalesce_wg=True only on WildChat-risky cells where OpenAI's content filter
# rejected ~60% of gpt54 labels and we fill from wildguard.
#
# fields: (tag, pkl, labels_csv, embeddings, coalesce_wg)
MODELS = [
    # ---- LMSYS / regular -------------------------------------------------
    ("base",                  "llama13b_message_df_with_users.pkl",
                              "llama13b_refusal_data_dynamic_with_labels_gpt54.csv",
                              "llama13b_first_user_embeddings.npz",  False),
    ("chat",                  "gpt35_message_df_with_users.pkl",
                              "refusal_data_dynamic_with_labels_gpt54.csv",
                              "first_user_embeddings.npz",            False),
    ("vicuna13b",             "vicuna13b_message_df_with_users.pkl",
                              "vicuna13b_refusal_data_dynamic_with_labels_gpt54.csv",
                              "vicuna13b_first_user_embeddings.npz",  False),
    ("vicuna33b",             "vicuna33b_message_df_with_users.pkl",
                              "vicuna33b_refusal_data_dynamic_with_labels_gpt54.csv",
                              "vicuna33b_first_user_embeddings.npz",  False),
    ("koala13b",              "koala13b_message_df_with_users.pkl",
                              "koala13b_refusal_data_dynamic_with_labels_gpt54.csv",
                              "koala13b_first_user_embeddings.npz",   False),
    ("gpt35turbo",            "gpt35turbo_message_df_with_users.pkl",
                              "gpt35turbo_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt35turbo_first_user_embeddings.npz", False),
    ("gpt4lmsys",             "gpt4lmsys_message_df_with_users.pkl",
                              "gpt4lmsys_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt4lmsys_first_user_embeddings.npz",  False),
    ("claude1",               "claude1_message_df_with_users.pkl",
                              "claude1_refusal_data_dynamic_with_labels_gpt54.csv",
                              "claude1_first_user_embeddings.npz",    False),
    ("claudeinstant1",        "claudeinstant1_message_df_with_users.pkl",
                              "claudeinstant1_refusal_data_dynamic_with_labels_gpt54.csv",
                              "claudeinstant1_first_user_embeddings.npz", False),
    ("gpt4o_lmsys",           "gpt4o_message_df_with_users.pkl",
                              "gpt4o_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt4o_first_user_embeddings.npz",      False),
    ("gpt4omini_lmsys",       "gpt4omini_message_df_with_users.pkl",
                              "gpt4omini_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt4omini_first_user_embeddings.npz",  False),
    ("gpt41mini_lmsys",       "gpt41mini_message_df_with_users.pkl",
                              "gpt41mini_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt41mini_first_user_embeddings.npz",  False),
    # ---- WildChat / regular ---------------------------------------------
    ("gpt35wc",               "gpt35wc_message_df_with_users.pkl",
                              "gpt35wc_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt35wc_first_user_embeddings.npz",    False),
    # ---- WildChat / risky-enriched (coalesce gpt54 + wildguard) ---------
    ("gpt4o_wcrisky",         "gpt4o_risky_add_message_df_with_users.pkl",
                              "gpt4o_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt4o_risky_add_first_user_embeddings.npz", True),
    ("gpt4omini_wcrisky",     "gpt4omini_risky_add_message_df_with_users.pkl",
                              "gpt4omini_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt4omini_risky_add_first_user_embeddings.npz", True),
    ("gpt41mini_wcrisky",     "gpt41mini_risky_add_message_df_with_users.pkl",
                              "gpt41mini_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt41mini_risky_add_first_user_embeddings.npz", True),
    ("gpt35wc_wcrisky",       "gpt35wc_risky_add_message_df_with_users.pkl",
                              "gpt35wc_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt35wc_risky_add_first_user_embeddings.npz", True),
    # ---- WildChat-risky dated GPT-4 cells --------------------------------
    ("gpt4_0314_wcrisky",     "gpt4_0314_risky_add_message_df_with_users.pkl",
                              "gpt4_0314_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt4_0314_risky_add_first_user_embeddings.npz", True),
    ("gpt4_1106_wcrisky",     "gpt4_1106_risky_add_message_df_with_users.pkl",
                              "gpt4_1106_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt4_1106_risky_add_first_user_embeddings.npz", True),
    ("gpt4_0125_wcrisky",     "gpt4_0125_risky_add_message_df_with_users.pkl",
                              "gpt4_0125_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt4_0125_risky_add_first_user_embeddings.npz", True),
    # o1mini dropped from the panel: every WildChat o1-mini conversation has
    # exactly 2 messages, so the re-engagement outcome Y is identically 0 and
    # the ATT is structurally undefined. See paper §3 for discussion.
]


def _variant_name(args) -> str:
    parts = []
    if args.cluster:
        parts.append(f"cluster-{args.cluster}")
    if args.add_time:
        parts.append("addtime")
    if args.add_temp:
        parts.append("addtemp")
    if args.restrict_modal_temp:
        parts.append("modaltemp")
    if args.length_stratum:
        parts.append(f"len{args.length_stratum}")
    return "_".join(parts) if parts else "plain"


def run_variants(args, selected: set[str] | None) -> None:
    """Robustness-variant runner: everything under --outdir, never touches
    output/comparison. One JSON per (variant, cell) under outdir/cells/,
    plus a rebuilt flat summary CSV after every cell."""
    outdir = Path(args.outdir)
    cells_dir = outdir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    variant = _variant_name(args)

    meta = None
    if args.meta_parquet:
        meta = pd.read_parquet(args.meta_parquet,
                               columns=["conversation_hash", "timestamp",
                                        "hashed_ip", "temperature", "top_p"])
        print(f"loaded meta parquet: {len(meta):,} rows")

    variant_opts = {
        "meta": meta,
        "cluster": args.cluster,
        "add_time": args.add_time,
        "add_temp": args.add_temp,
        "restrict_modal_temp": args.restrict_modal_temp,
        "length_stratum": args.length_stratum,
    }
    print(f"VARIANT MODE: {variant}  ->  {outdir}")

    def rebuild_summary() -> None:
        rows = []
        for p in sorted(cells_dir.glob("*.json")):
            rows.append(json.loads(p.read_text()))
        if rows:
            try:
                pd.DataFrame(rows).to_csv(outdir / "aipw_variants_summary.csv",
                                          index=False)
            except PermissionError:
                # another variant process is writing the shared summary right
                # now; per-cell JSONs are authoritative, so skip quietly
                pass

    for tag, pkl, lab, emb, coalesce_wg in MODELS:
        if selected is not None and tag not in selected:
            continue
        print(f"\n{'='*72}\nVARIANT {variant} | MODEL {tag} "
              f"(treatment_label={args.treatment_label})\n{'='*72}")
        try:
            sample = build_aipw_sample(pkl, lab, emb, coalesce_wg=coalesce_wg,
                                       treatment_label=args.treatment_label,
                                       variant_opts=variant_opts)
        except ValueError as e:
            print(f"  [{tag}] skipping — {e}"); continue
        if sample is None:
            print(f"  [{tag}] missing inputs or empty sample; skipping"); continue
        X, T, Y, info = sample

        if args.cluster == "ip" and info.get("n_meta_unmatched", 0) > 0:
            print(f"  WARNING: {info['n_meta_unmatched']} conversations "
                  f"unmatched in meta parquet (singleton clusters)")

        r, extras = cross_fit_aipw(X, T, Y, return_extras=True)
        row = {"variant": variant, "model": tag,
               "cluster_mode": args.cluster or "",
               "add_time": args.add_time, "add_temp": args.add_temp,
               "restrict_modal_temp": args.restrict_modal_temp,
               "length_stratum": args.length_stratum or "",
               **r.as_dict()}
        for k, v in info.items():
            if k not in ("cluster_ids", "X_base"):
                row[k] = v

        cluster_ids = info.get("cluster_ids")
        if cluster_ids is not None:
            psi_ate, psi_att = _psi_arrays(T, Y, extras["ehat"],
                                           extras["m0"], extras["m1"])
            ate_se_cl, G = _cluster_se(psi_ate, cluster_ids)
            att_se_cl, _ = _cluster_se(psi_att, cluster_ids)
            row.update(
                n_clusters=G,
                ate_se_cluster_pp=ate_se_cl * 100,
                ate_ci95_cluster_lo=(r.ate_pp - 1.96 * ate_se_cl * 100),
                ate_ci95_cluster_hi=(r.ate_pp + 1.96 * ate_se_cl * 100),
                att_se_cluster_pp=att_se_cl * 100,
                att_ci95_cluster_lo=(r.att_pp - 1.96 * att_se_cl * 100),
                att_ci95_cluster_hi=(r.att_pp + 1.96 * att_se_cl * 100),
            )
            print(f"  ATT = {r.att_pp:+.2f} pp")
            print(f"    iid SE {r.att_se_pp:.2f}  95% CI "
                  f"[{r.att_ci95_lo:+.2f}, {r.att_ci95_hi:+.2f}]")
            print(f"    {args.cluster}-clustered SE {att_se_cl*100:.2f} "
                  f"(G={G:,})  95% CI "
                  f"[{row['att_ci95_cluster_lo']:+.2f}, "
                  f"{row['att_ci95_cluster_hi']:+.2f}]")
        else:
            print(f"  ATT = {r.att_pp:+.2f} pp   iid SE {r.att_se_pp:.2f}  "
                  f"95% CI [{r.att_ci95_lo:+.2f}, {r.att_ci95_hi:+.2f}]")

        if args.add_time:
            auc_no_time = _cross_fit_propensity_auc(info["X_base"], T)
            row["prop_auc_with_time"] = r.prop_auc
            row["prop_auc_without_time"] = auc_no_time
            print(f"  propensity AUC: with time = {r.prop_auc:.4f}   "
                  f"without time = {auc_no_time:.4f}")
        if args.restrict_modal_temp:
            print(f"  restrict-modal-temp: dropped "
                  f"{info.get('n_dropped_modal_temp', 0)} conversations "
                  f"(modal T={info.get('modal_temperature')}, "
                  f"top_p={info.get('modal_top_p')}); n = {r.n:,}")
        if args.length_stratum:
            print(f"  length-stratum {args.length_stratum}: treated median = "
                  f"{info.get('length_median_treated'):.1f} chars; kept "
                  f"{info.get('n_treated_stratum')} of "
                  f"{info.get('n_treated_prestratum')} treated "
                  f"(dropped {info.get('n_dropped_length_stratum')}); "
                  f"n = {r.n:,}")

        (cells_dir / f"{variant}__{tag}.json").write_text(
            json.dumps(row, indent=2, default=str))
        rebuild_summary()
        print(f"  wrote {cells_dir / (variant + '__' + tag + '.json')}")

    rebuild_summary()
    print(f"\nDone. Summary: {outdir / 'aipw_variants_summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None,
                    help="comma-separated subset of MODELS tags. "
                         "If set, results are merged into existing aipw_results.json "
                         "rather than overwriting from scratch.")
    ap.add_argument("--treatment-label", dest="treatment_label",
                    choices=["coalesce", "gpt54", "wg"], default="coalesce",
                    help="Which judge supplies the treatment label. Default "
                         "'coalesce' = legacy gpt54-headline. 'wg' = panel-wide WildGuard.")
    # --- robustness-variant flags (all opt-in; defaults preserve legacy) ----
    ap.add_argument("--meta-parquet", dest="meta_parquet", default=None,
                    help="WildChat conversation metadata parquet "
                         "(conversation_hash, timestamp, hashed_ip, "
                         "temperature, top_p); merged by conversation_id.")
    ap.add_argument("--cluster", choices=["ip", "promptcluster"], default=None,
                    help="cluster-robust influence-function SEs: 'ip' uses "
                         "hashed_ip from --meta-parquet (WildChat cells only); "
                         "'promptcluster' uses the pickles' user_id column.")
    ap.add_argument("--add-time", dest="add_time", action="store_true",
                    help="append scaled days-since-cell-window-start to X "
                         "(needs --meta-parquet); reports propensity AUC "
                         "with/without the time column.")
    ap.add_argument("--add-temp", dest="add_temp", action="store_true",
                    help="append temperature/top_p (modal-imputed + missing "
                         "indicator) to X (needs --meta-parquet).")
    ap.add_argument("--restrict-modal-temp", dest="restrict_modal_temp",
                    action="store_true",
                    help="drop conversations whose temperature/top_p differ "
                         "from the cell's modal values (needs --meta-parquet).")
    ap.add_argument("--length-stratum", dest="length_stratum",
                    choices=["short", "long"], default=None,
                    help="restrict TREATED conversations to those whose "
                         "first assistant response (turn-2, character count) "
                         "is <= (short) or > (long) the median among treated "
                         "conversations in the analysis sample; controls "
                         "untouched.")
    ap.add_argument("--outdir", default=None,
                    help="REQUIRED with any variant flag; variant outputs go "
                         "here (never output/comparison).")
    args = ap.parse_args()
    selected = set(args.tags.split(",")) if args.tags else None
    if selected is not None:
        unknown = selected - {m[0] for m in MODELS}
        if unknown:
            sys.exit(f"unknown tags: {sorted(unknown)}")

    variant_flags = (args.meta_parquet, args.cluster, args.add_time,
                     args.add_temp, args.restrict_modal_temp,
                     args.length_stratum)
    if any(variant_flags) and not args.outdir:
        sys.exit("--outdir is required whenever a variant flag "
                 "(--meta-parquet/--cluster/--add-time/--add-temp/"
                 "--restrict-modal-temp/--length-stratum) is used")
    if (args.cluster == "ip" or args.add_time or args.add_temp
            or args.restrict_modal_temp) and not args.meta_parquet:
        sys.exit("--meta-parquet is required for --cluster ip, --add-time, "
                 "--add-temp, and --restrict-modal-temp")
    if args.outdir:
        run_variants(args, selected)
        return

    global OUT
    OUT = _out_dir(args.treatment_label)
    print(f"treatment_label={args.treatment_label}  ->  output dir: {OUT}")

    # Merge mode: start from existing JSON; only update the selected cells.
    if selected is not None and (OUT / "aipw_results.json").exists():
        results: dict[str, dict] = json.loads((OUT / "aipw_results.json").read_text())
        print(f"merge mode: loaded {len(results)} existing cells from aipw_results.json")
    else:
        results = {}

    summary_rows = []
    for tag, pkl, lab, emb, coalesce_wg in MODELS:
        if selected is not None and tag not in selected:
            continue
        print(f"\n{'='*72}\nMODEL: {tag}  (treatment_label={args.treatment_label}, coalesce_wg={coalesce_wg})\n{'='*72}")
        try:
            sample = build_aipw_sample(pkl, lab, emb, coalesce_wg=coalesce_wg,
                                       treatment_label=args.treatment_label)
        except ValueError as e:
            print(f"  [{tag}] skipping — {e}"); continue
        if sample is None:
            print(f"  [{tag}] missing inputs; skipping"); continue
        X, T, Y = sample
        r = cross_fit_aipw(X, T, Y)
        results[tag] = r.as_dict()

        print(f"  --- sample stats ---")
        print(f"    n = {r.n:,}   n_treated = {r.n_treated:,}   n_control = {r.n_control:,}")
        print(f"    P(refusal) = {r.p_treated:.3f}")
        print(f"    cont | no-refusal = {r.cont_no_refusal*100:.2f}%")
        print(f"    cont | refusal    = {r.cont_refusal*100:.2f}%")
        print(f"    raw diff (ref - no-ref) = {r.raw_diff_pp:+.2f} pp")
        print(f"  --- AIPW (GBM nuisances, K=5 folds, propensity clipped to {PROP_CLIP}) ---")
        print(f"    ATE = {r.ate_pp:+.2f} pp   SE {r.ate_se_pp:.2f}   95% CI [{r.ate_ci95_lo:+.2f}, {r.ate_ci95_hi:+.2f}]")
        print(f"    ATT = {r.att_pp:+.2f} pp   SE {r.att_se_pp:.2f}   95% CI [{r.att_ci95_lo:+.2f}, {r.att_ci95_hi:+.2f}]")
        print(f"  --- decomposition (ATE = G-comp + IPW correction) ---")
        print(f"    G-computation alone = {r.g_comp_pp:+.2f} pp")
        print(f"    IPW alone           = {r.ipw_pp:+.2f} pp")
        print(f"  --- overlap diagnostics ---")
        print(f"    propensity:  mean={r.prop_mean:.3f}  median={r.prop_median:.3f}  "
              f"min={r.prop_min:.3f}  max={r.prop_max:.3f}")
        print(f"    AUC(propensity, T) = {r.prop_auc:.3f}   "
              f"(0.5 = good overlap; 1.0 = treatment fully predictable)")
        print(f"    P(e<0.1) = {r.prop_frac_below_01:.3f}   "
              f"P(e>0.9) = {r.prop_frac_above_09:.3f}   "
              f"P(0.1<=e<=0.9) = {r.prop_frac_in_overlap:.3f}")
        print(f"    ESS treated = {r.ess_treated:.1f} (of {r.n_treated})   "
              f"ESS control = {r.ess_control:.1f} (of {r.n_control})")
        print(f"    max IPW weight: treated={r.max_weight_treated:.1f}   "
              f"control={r.max_weight_control:.1f}")
        print(f"  --- Crump-trimmed AIPW (e in [0.1, 0.9]) ---")
        print(f"    n_trimmed = {r.n_trimmed:,}   n_treated_trimmed = {r.n_treated_trimmed:,}")
        if not np.isnan(r.att_trimmed_pp):
            print(f"    ATE_trim = {r.ate_trimmed_pp:+.2f} pp (SE {r.ate_trimmed_se_pp:.2f})")
            print(f"    ATT_trim = {r.att_trimmed_pp:+.2f} pp (SE {r.att_trimmed_se_pp:.2f})")
        else:
            print(f"    insufficient observations in overlap region for trimmed estimate")

        summary_rows.append({"model": tag, **r.as_dict()})

    (OUT / "aipw_results.json").write_text(json.dumps(results, indent=2))
    # Rebuild flat summary from the full (possibly merged) results dict so
    # aipw_summary.csv stays in sync with aipw_results.json under --tags.
    summary_df = pd.DataFrame([{"model": tag, **r} for tag, r in results.items()])
    summary_df.to_csv(OUT / "aipw_summary.csv", index=False)
    print(f"\nWrote {OUT/'aipw_results.json'} ({len(results)} cells) "
          f"and {OUT/'aipw_summary.csv'}")


if __name__ == "__main__":
    main()
