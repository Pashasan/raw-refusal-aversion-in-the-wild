"""Replicate the paper's matching analyses (Figures 4 + 5) with gpt-5.4-mini
labels, for both Llama-2-13b-chat and base Llama-2-13b.

For each model:
  - score-based caliper matching on first-message moderation score (Fig 4)
  - embedding-based matching on first-user prompt mxbai embeddings (Fig 5)

Writes JSON + PNGs to output/comparison/matching/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from llm_dynamics import config
from llm_dynamics.matching.score import score_matched_analysis, _first_user_messages
from llm_dynamics.matching.embedding import embedding_matched_analysis

ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "output" / "comparison"
OUT = OUT_BASE / "matching"  # default; main() may override based on --treatment-label
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = config.RISK_THRESHOLD  # 0.01


def _out_dir(treatment_label: str) -> Path:
    suffix = "" if treatment_label == "coalesce" else f"_{treatment_label}"
    out = OUT_BASE / f"matching{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_with_gpt54(pkl: str, labels_csv: str,
                    coalesce_wg: bool = False,
                    treatment_label: str = "coalesce") -> pd.DataFrame:
    """Load message_df and attach a refusal label.

    treatment_label selects which judge supplies `refused_answer`:
      "coalesce" (legacy default): per-cell — uses the cell's `coalesce_wg`
                  flag (False = gpt54 only with NaN->0; True = gpt54 first,
                  wg fallback). Reproduces the original gpt54-headline pipeline.
      "gpt54":    is_refusal_gpt54 everywhere (NaN -> 0). Errors on cells
                  that lack a gpt54 column (the 3 wcrisky-only cells).
      "wg":       is_refusal_wg everywhere (NaN -> 0). Panel-consistent
                  single judge — this is the wg-headline pipeline.
    """
    with open(ROOT / pkl, "rb") as f:
        msg = pickle.load(f)
    labels = pd.read_csv(ROOT / labels_csv)
    msg = msg.drop(columns=[c for c in ["refused_answer"] if c in msg.columns])

    if treatment_label == "wg":
        if "is_refusal_wg" not in labels.columns:
            raise ValueError(f"{labels_csv} missing is_refusal_wg")
        cols = ["conversation_id", "message_number", "is_refusal_wg"]
        merged = msg.merge(labels[cols], on=["conversation_id", "message_number"],
                           how="left")
        merged["refused_answer"] = merged["is_refusal_wg"].fillna(0).astype(int)
        return merged.drop(columns=["is_refusal_wg"])

    if treatment_label == "gpt54":
        if "is_refusal_gpt54" not in labels.columns:
            raise ValueError(f"{labels_csv} missing is_refusal_gpt54 "
                             f"(use --treatment-label wg or coalesce for wg-only cells)")
        cols = ["conversation_id", "message_number", "is_refusal_gpt54"]
        merged = msg.merge(labels[cols], on=["conversation_id", "message_number"],
                           how="left")
        merged["refused_answer"] = merged["is_refusal_gpt54"].fillna(0).astype(int)
        return merged.drop(columns=["is_refusal_gpt54"])

    if treatment_label != "coalesce":
        raise ValueError(f"unknown treatment_label: {treatment_label!r}")
    # legacy per-cell coalesce path (gpt54-headline pipeline)
    if "is_refusal_gpt54" not in labels.columns:
        if not coalesce_wg:
            raise ValueError(f"{labels_csv} missing is_refusal_gpt54 (set coalesce_wg=True for wg-only cells)")
        labels["is_refusal_gpt54"] = pd.NA
    cols = ["conversation_id", "message_number", "is_refusal_gpt54"]
    if coalesce_wg:
        if "is_refusal_wg" not in labels.columns:
            raise ValueError(f"{labels_csv} missing is_refusal_wg "
                             f"(needed because coalesce_wg=True)")
        cols.append("is_refusal_wg")
    merged = msg.merge(labels[cols], on=["conversation_id", "message_number"],
                       how="left")
    if coalesce_wg:
        coalesced = merged["is_refusal_gpt54"].combine_first(merged["is_refusal_wg"])
        merged["refused_answer"] = coalesced.fillna(0).astype(int)
        return merged.drop(columns=["is_refusal_gpt54", "is_refusal_wg"])
    merged["refused_answer"] = merged["is_refusal_gpt54"].fillna(0).astype(int)
    return merged.drop(columns=["is_refusal_gpt54"])


def build_high_concern_files(tag: str, msg_df: pd.DataFrame, full_emb_npz: str,
                             meta_df: pd.DataFrame | None = None,
                             ) -> tuple[Path, Path]:
    """Filter the per-conversation first-user embeddings to risky prompts,
    attach the new gpt-5.4-mini refusal/continuation labels, and write the
    paper's two-file format (npy + csv) to a temp location.

    If ``meta_df`` is given (columns: conversation_id, month, hashed_ip) the
    metadata csv additionally carries those columns, enabling the matching
    variants (--month-block / --exclude-same-ip)."""
    z = np.load(ROOT / full_emb_npz, allow_pickle=True)
    full_emb = z["embeddings"]
    full_ids = z["conversation_ids"]

    conv = _first_user_messages(msg_df)  # already has refused_answer + user_continued
    conv = conv.rename(columns={"max_concern_score": "max_score"})
    high = conv[conv["max_score"] > THRESHOLD].copy()

    id_to_pos = {cid: i for i, cid in enumerate(full_ids)}
    have_emb = high["conversation_id"].isin(id_to_pos)
    dropped = (~have_emb).sum()
    if dropped:
        print(f"  [{tag}] dropping {dropped} risky convs without an embedding")
    high = high[have_emb].reset_index(drop=True)
    rows = [id_to_pos[c] for c in high["conversation_id"]]
    high_emb = full_emb[rows]

    # Add 'content' column (embedding_matched_analysis only needs the columns
    # max_score, refused_answer, user_continued, but the original metadata had
    # content for inspection; we mirror that shape).
    first_user_content = msg_df[(msg_df["role"] == "user") &
                                (msg_df["message_number"] == 1)][
        ["conversation_id", "content"]]
    meta = high.merge(first_user_content, on="conversation_id", how="left")
    meta = meta[["conversation_id", "content", "max_score", "refused_answer",
                 "user_continued"]]
    if meta_df is not None:
        meta = meta.merge(meta_df, on="conversation_id", how="left")
        n_miss = int(meta["hashed_ip"].isna().sum())
        if n_miss:
            print(f"  [{tag}] WARNING: {n_miss}/{len(meta)} risky convs "
                  f"missing WildChat metadata (month/hashed_ip)")

    emb_path = OUT / f"high_concern_embeddings_{tag}_gpt54.npy"
    meta_path = OUT / f"high_concern_metadata_{tag}_gpt54.csv"
    np.save(emb_path, high_emb)
    meta.to_csv(meta_path, index=False)
    print(f"  [{tag}] wrote {emb_path.name} ({high_emb.shape}) and "
          f"{meta_path.name} (n={len(meta)}, refused={int(meta['refused_answer'].sum())})")
    return emb_path, meta_path


def _pre_match_stats(msg_df: pd.DataFrame, threshold: float = THRESHOLD) -> dict:
    """Compute pre-match sample stats on the risky first-user sample — the
    pool the matcher sees before imposing the caliper. Mirrors the sample
    stats block in the AIPW module so the two estimators report on
    comparable denominators."""
    first = msg_df[(msg_df["role"] == "user") &
                   (msg_df["message_number"] == 1)][
        ["conversation_id", "max_concern_score"]].drop_duplicates("conversation_id")
    turn2 = msg_df[(msg_df["role"] == "assistant") &
                   (msg_df["message_number"] == 2)][
        ["conversation_id", "refused_answer"]].drop_duplicates("conversation_id")
    turn3 = msg_df[(msg_df["role"] == "user") &
                   (msg_df["message_number"] == 3)][
        ["conversation_id"]].drop_duplicates().assign(Y=1)
    conv = (first.merge(turn2, on="conversation_id", how="inner")
                 .merge(turn3, on="conversation_id", how="left"))
    conv["Y"] = conv["Y"].fillna(0).astype(int)
    risky = conv[conv["max_concern_score"] > threshold]
    n, n_t = len(risky), int(risky["refused_answer"].sum())
    return {
        "n_risky": n,
        "n_treated_risky": n_t,
        "n_control_risky": int(n - n_t),
        "p_treated_risky": float(n_t / n) if n else float("nan"),
        "cont_no_refusal_risky": float(risky.loc[risky["refused_answer"]==0,"Y"].mean())
            if (n - n_t) else float("nan"),
        "cont_refusal_risky": float(risky.loc[risky["refused_answer"]==1,"Y"].mean())
            if n_t else float("nan"),
    }


def run(tag: str, msg_df: pd.DataFrame, full_emb_npz: str, *,
        meta_df: pd.DataFrame | None = None,
        min_cos: float | None = None,
        month_block: bool = False,
        exclude_same_ip: bool = False,
        emb_only: bool = False,
        run_name: str | None = None) -> dict:
    print(f"\n{'='*72}\nMODEL: {tag}\n{'='*72}")
    suffix = f"_{run_name}" if run_name else ""

    pre = _pre_match_stats(msg_df)
    print("--- sample stats (risky first-user pool, max_score > 0.01) ---")
    print(f"  n = {pre['n_risky']:,}   n_treated = {pre['n_treated_risky']:,}   "
          f"n_control = {pre['n_control_risky']:,}")
    print(f"  P(refusal)         = {pre['p_treated_risky']:.3f}")
    print(f"  cont | no-refusal  = {pre['cont_no_refusal_risky']*100:.2f}%")
    print(f"  cont | refusal     = {pre['cont_refusal_risky']*100:.2f}%")
    raw_diff_pp = (pre["cont_refusal_risky"] - pre["cont_no_refusal_risky"]) * 100
    print(f"  raw diff_pp        = {raw_diff_pp:+.2f} pp")

    score_block = None
    if not emb_only:
        print("\n--- score-based matching (caliper ||d-score|| <= 0.1) ---")
        score_stats, _ = score_matched_analysis(
            msg_df, save_path=OUT / f"score_match_{tag}{suffix}.png",
        )
        match_rate_score = (score_stats.n_pairs / pre["n_treated_risky"]
                             if pre["n_treated_risky"] else float("nan"))
        print(f"  n_pairs = {score_stats.n_pairs:,}   "
              f"match_rate = {match_rate_score*100:.1f}% of treated   "
              f"discarded_treated = {pre['n_treated_risky'] - score_stats.n_pairs:,}")
        print(f"  cont | no-refusal (matched) = {score_stats.non_refused_rate*100:.2f}%")
        print(f"  cont | refusal    (matched) = {score_stats.refused_rate*100:.2f}%")
        print(f"  ATT diff_pp = {score_stats.pp_difference:+.2f}   "
              f"ratio (no-ref / ref) = {score_stats.ratio:.2f}x   "
              f"95% CI [{score_stats.ci_lower*100:+.2f}, {score_stats.ci_upper*100:+.2f}]")
        print(f"  balance: wilcoxon W={score_stats.wilcoxon_W:.1f} p={score_stats.wilcoxon_p:.4g}   "
              f"KS D={score_stats.ks_D:.3f} p={score_stats.ks_p:.4g}")
        print(f"  significance: mcnemar chi2={score_stats.mcnemar_chi2:.2f} p={score_stats.mcnemar_p:.4g}   "
              f"binomial p={score_stats.binomial_p:.4g}")
        score_block = {**asdict(score_stats), "match_rate": match_rate_score,
                       "discarded_treated": pre["n_treated_risky"] - score_stats.n_pairs}

    cos_thr = 0.7 if min_cos is None else min_cos
    print(f"\n--- embedding-based matching (cos >= {cos_thr}, ||d-score|| <= 0.01"
          + (", month-block" if month_block else "")
          + (", exclude-same-ip" if exclude_same_ip else "") + ") ---")
    emb_path, meta_path = build_high_concern_files(tag, msg_df, full_emb_npz,
                                                   meta_df=meta_df)
    want_extras = (meta_df is not None or min_cos is not None
                   or month_block or exclude_same_ip)
    emb_kwargs = {}
    if min_cos is not None:
        emb_kwargs["similarity_threshold"] = min_cos
    if month_block:
        emb_kwargs["month_block"] = True
    if exclude_same_ip:
        emb_kwargs["exclude_same_ip"] = True
    if want_extras:
        emb_kwargs["return_extras"] = True
        emb_kwargs["pair_dump_path"] = OUT / f"pairs_{tag}{suffix}.csv"
    result = embedding_matched_analysis(
        emb_path=emb_path, meta_path=meta_path,
        save_path=OUT / f"emb_match_{tag}{suffix}.png",
        **emb_kwargs,
    )
    extras = None
    if want_extras:
        emb_stats, _, extras = result
    else:
        emb_stats, _ = result

    if emb_stats is None:
        print("  NO matched pairs survive this variant (underpowered).")
        emb_block = {"n_pairs": 0}
    else:
        match_rate_emb = (emb_stats.n_pairs / pre["n_treated_risky"]
                           if pre["n_treated_risky"] else float("nan"))
        print(f"  n_pairs = {emb_stats.n_pairs:,}   "
              f"match_rate = {match_rate_emb*100:.1f}% of treated   "
              f"discarded_treated = {pre['n_treated_risky'] - emb_stats.n_pairs:,}")
        print(f"  mean cosine similarity of matched pairs = {emb_stats.mean_similarity:.3f}")
        print(f"  cont | no-refusal (matched) = {emb_stats.non_refused_rate*100:.2f}%")
        print(f"  cont | refusal    (matched) = {emb_stats.refused_rate*100:.2f}%")
        print(f"  ATT diff_pp = {emb_stats.pp_difference:+.2f}   "
              f"ratio (no-ref / ref) = {emb_stats.ratio:.2f}x   "
              f"95% CI [{emb_stats.ci_lower*100:+.2f}, {emb_stats.ci_upper*100:+.2f}]")
        print(f"  balance: wilcoxon W={emb_stats.wilcoxon_W:.1f} p={emb_stats.wilcoxon_p:.4g}   "
              f"KS D={emb_stats.ks_D:.3f} p={emb_stats.ks_p:.4g}")
        print(f"  significance: mcnemar chi2={emb_stats.mcnemar_chi2:.2f} p={emb_stats.mcnemar_p:.4g}   "
              f"binomial p={emb_stats.binomial_p:.4g}")
        emb_block = {**asdict(emb_stats), "match_rate": match_rate_emb,
                     "discarded_treated": pre["n_treated_risky"] - emb_stats.n_pairs}

    if extras is not None:
        if "baseline_same_ip_share" in extras:
            print(f"  baseline design: {extras['baseline_same_ip_pairs']}/"
                  f"{extras['baseline_n_pairs']} pairs share hashed_ip "
                  f"({extras['baseline_same_ip_share']*100:.1f}%)")
        if "same_ip_share_kept" in extras:
            print(f"  this variant:    {extras['same_ip_pairs_kept']}/"
                  f"{extras['n_pairs']} kept pairs share hashed_ip "
                  f"({extras['same_ip_share_kept']*100:.1f}%)")
        if "pairs_by_threshold" in extras:
            print(f"  pairs by cos threshold: {extras['pairs_by_threshold']}")

    out = {
        "pre_match_sample": pre,
        "raw_diff_pp": raw_diff_pp,
        "score": score_block,
        "embedding": emb_block,
    }
    if extras is not None:
        out["embedding_extras"] = extras
    return out


# Per-(dataset, model) cells. Tag suffix encodes the prompt source so each row
# in the forest is one (dataset, model) bucket — no within-row mixing of
# LMSYS and WildChat. coalesce_wg=True only on WildChat-risky cells where
# OpenAI's content filter rejected ~60% of gpt54 labels and we fill from
# wildguard. See `feedback_judge_choice` and `compare_judges.py`.
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
    # No is_refusal_gpt54 column on these CSVs; coalesce_wg=True falls back
    # to is_refusal_wg via the missing-column shim in load_with_gpt54.
    ("gpt4_0314_wcrisky",     "gpt4_0314_risky_add_message_df_with_users.pkl",
                              "gpt4_0314_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt4_0314_risky_add_first_user_embeddings.npz", True),
    ("gpt4_1106_wcrisky",     "gpt4_1106_risky_add_message_df_with_users.pkl",
                              "gpt4_1106_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt4_1106_risky_add_first_user_embeddings.npz", True),
    ("gpt4_0125_wcrisky",     "gpt4_0125_risky_add_message_df_with_users.pkl",
                              "gpt4_0125_risky_add_refusal_data_dynamic_with_labels_gpt54.csv",
                              "gpt4_0125_risky_add_first_user_embeddings.npz", True),
    # o1mini dropped: WildChat o1-mini conversations are all single-turn
    # (2 messages), so re-engagement is structurally absent. See paper §3.
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None,
                    help="comma-separated subset of MODELS tags. "
                         "If set, results are merged into existing matching_results.json "
                         "rather than overwriting from scratch.")
    ap.add_argument("--treatment-label", dest="treatment_label",
                    choices=["coalesce", "gpt54", "wg"], default="coalesce",
                    help="Which judge supplies refused_answer. Default 'coalesce' "
                         "= legacy gpt54-headline pipeline (per-cell coalesce_wg). "
                         "'wg' = WildGuard panel-headline pipeline.")
    # ---- matching-design variant flags (all opt-in; defaults preserve the
    # ---- headline pipeline exactly). Any variant flag requires --outdir so
    # ---- nothing under output/comparison/ is ever rewritten.
    ap.add_argument("--outdir", default=None,
                    help="Override the output directory (required for variant runs).")
    ap.add_argument("--meta-parquet", dest="meta_parquet", default=None,
                    help="WildChat conversation-metadata parquet (conversation_hash, "
                         "timestamp, hashed_ip); attaches calendar month + hashed_ip "
                         "to the high-concern metadata by conversation_id.")
    ap.add_argument("--month-block", dest="month_block", action="store_true",
                    help="Restrict embedding matches to the same calendar month "
                         "(requires --meta-parquet).")
    ap.add_argument("--exclude-same-ip", dest="exclude_same_ip", action="store_true",
                    help="Forbid embedding pairs sharing hashed_ip "
                         "(requires --meta-parquet).")
    ap.add_argument("--min-cos", dest="min_cos", type=float, default=None,
                    help="Override the embedding-similarity threshold (default 0.7).")
    ap.add_argument("--emb-only", dest="emb_only", action="store_true",
                    help="Skip the score-based matching design (variant runs).")
    ap.add_argument("--run-name", dest="run_name", default=None,
                    help="Suffix for results json / pngs / pair dumps so multiple "
                         "variants can share one --outdir.")
    args = ap.parse_args()
    selected = set(args.tags.split(",")) if args.tags else None
    if selected is not None:
        unknown = selected - {m[0] for m in MODELS}
        if unknown:
            sys.exit(f"unknown tags: {sorted(unknown)}")

    variant = (args.meta_parquet is not None or args.month_block
               or args.exclude_same_ip or args.min_cos is not None
               or args.emb_only or args.run_name is not None)
    if variant and not args.outdir:
        sys.exit("--outdir is required for any variant run "
                 "(--meta-parquet/--month-block/--exclude-same-ip/--min-cos/"
                 "--emb-only/--run-name)")
    if (args.month_block or args.exclude_same_ip) and not args.meta_parquet:
        sys.exit("--month-block/--exclude-same-ip require --meta-parquet")

    global OUT
    if args.outdir:
        OUT = Path(args.outdir)
        OUT.mkdir(parents=True, exist_ok=True)
        print(f"outdir override  ->  output dir: {OUT}")
    else:
        OUT = _out_dir(args.treatment_label)
        print(f"treatment_label={args.treatment_label}  ->  output dir: {OUT}")

    meta_df = None
    if args.meta_parquet:
        mp = pd.read_parquet(args.meta_parquet,
                             columns=["conversation_hash", "timestamp", "hashed_ip"])
        mp = mp.rename(columns={"conversation_hash": "conversation_id"})
        mp["month"] = pd.to_datetime(mp["timestamp"]).dt.strftime("%Y-%m")
        meta_df = mp[["conversation_id", "month", "hashed_ip"]]
        print(f"metadata parquet: {len(meta_df):,} conversations "
              f"({args.meta_parquet})")

    results_name = (f"matching_results_{args.run_name}.json" if args.run_name
                    else "matching_results.json")
    if selected is not None and (OUT / results_name).exists():
        out: dict[str, dict] = json.loads((OUT / results_name).read_text())
        print(f"merge mode: loaded {len(out)} existing cells from {results_name}")
    else:
        out = {}

    for tag, pkl, lab, emb, coalesce_wg in MODELS:
        if selected is not None and tag not in selected:
            continue
        paths = [ROOT/pkl, ROOT/lab, ROOT/emb]
        if not all(p.exists() for p in paths):
            missing = [p.name for p in paths if not p.exists()]
            print(f"[{tag}] skipping — missing: {missing}"); continue
        try:
            msg = load_with_gpt54(pkl, lab, coalesce_wg=coalesce_wg,
                                  treatment_label=args.treatment_label)
        except ValueError as e:
            print(f"[{tag}] skipping — {e}"); continue
        out[tag] = run(tag, msg, emb, meta_df=meta_df, min_cos=args.min_cos,
                       month_block=args.month_block,
                       exclude_same_ip=args.exclude_same_ip,
                       emb_only=args.emb_only, run_name=args.run_name)
    (OUT / results_name).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT/results_name}  ({len(out)} models)")


if __name__ == "__main__":
    main()
