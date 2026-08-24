"""Figure 5: embedding-based matching on risky prompts with score constraint."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from ..paper_style import set_paper_style, DARK_BLUE, LIGHT_BLUE, ORANGE
from ._stats import MatchStatsResult, compute_matching_statistics


def _load_high_concern(emb_path: Path, meta_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    if len(embeddings) != len(meta):
        raise ValueError(
            f"Embedding/metadata size mismatch: {len(embeddings)} vs {len(meta)}"
        )
    return embeddings, meta


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between all rows of ``a`` and ``b``."""
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    return an @ bn.T


def _select_pairs(
    sims: np.ndarray,
    ref_meta: pd.DataFrame,
    non_meta: pd.DataFrame,
    similarity_threshold: float,
    score_cap: float,
) -> dict:
    """Greedy nearest-neighbour + the paper's two-stage filter.

    Exactly reproduces the original cell-44 logic: (1) best cosine >=
    ``similarity_threshold``; (2) ``|Δ max_score| <= score_cap``. Applying
    stage 2 to all rows and AND-ing with stage 1 selects the identical
    pair set (same membership, same order) as the original sequential
    filtering, while also exposing the per-row intermediates so callers
    can report counts at alternative thresholds.
    """
    nearest = np.argmax(sims, axis=1)
    best_sim = sims[np.arange(sims.shape[0]), nearest]
    score_diff = np.abs(
        ref_meta["max_score"].to_numpy(dtype=float)
        - non_meta["max_score"].to_numpy(dtype=float)[nearest]
    )
    keep = (best_sim >= similarity_threshold) & (score_diff <= score_cap)
    return {
        "ref_kept": ref_meta[keep].reset_index(drop=True),
        "matched": non_meta.iloc[nearest[keep]].reset_index(drop=True),
        "sims_kept": best_sim[keep],
        "nearest": nearest,
        "best_sim": best_sim,
        "score_diff": score_diff,
        "keep": keep,
    }


def _degenerate_stats(
    refused_scores: np.ndarray,
    non_refused_scores: np.ndarray,
    refused_cont: np.ndarray,
    non_refused_cont: np.ndarray,
    cosine_similarities: np.ndarray,
) -> MatchStatsResult:
    """Fallback for perfectly balanced pairs (every paired score difference
    exactly zero), where scipy's wilcoxon raises. Mirrors
    ``compute_matching_statistics`` but reports the wilcoxon balance test as
    NaN — balance is exact by construction. Only reachable under very tight
    similarity thresholds (near-verbatim resampled prompts)."""
    from scipy import stats as sps

    n_pairs = len(refused_scores)
    refused_rate = float(np.mean(refused_cont))
    matched_rate = float(np.mean(non_refused_cont))
    diff = (refused_rate - matched_rate) * 100.0
    ratio = matched_rate / refused_rate if refused_rate > 0 else float("inf")
    ks_stat, ks_p = sps.ks_2samp(refused_scores, non_refused_scores)
    only_ref = np.sum((refused_cont == 1) & (non_refused_cont == 0))
    only_non = np.sum((refused_cont == 0) & (non_refused_cont == 1))
    if only_ref + only_non > 0:
        mcnemar_stat = (abs(only_ref - only_non) - 1) ** 2 / (only_ref + only_non)
        mcnemar_p = 1 - sps.chi2.cdf(mcnemar_stat, df=1)
    else:
        mcnemar_stat, mcnemar_p = 0.0, 1.0
    binom = sps.binomtest(int(np.sum(refused_cont)), n_pairs, matched_rate,
                          alternative="two-sided").pvalue
    se = np.sqrt(
        (refused_rate * (1 - refused_rate) + matched_rate * (1 - matched_rate)) / n_pairs
    )
    return MatchStatsResult(
        n_pairs=n_pairs,
        refused_rate=refused_rate,
        non_refused_rate=matched_rate,
        pp_difference=diff,
        ratio=ratio,
        ci_lower=(refused_rate - matched_rate) - 1.96 * se,
        ci_upper=(refused_rate - matched_rate) + 1.96 * se,
        wilcoxon_W=float("nan"),
        wilcoxon_p=float("nan"),
        ks_D=float(ks_stat),
        ks_p=float(ks_p),
        mcnemar_chi2=float(mcnemar_stat),
        mcnemar_p=float(mcnemar_p),
        binomial_p=float(binom),
        mean_similarity=float(np.mean(cosine_similarities)),
    )


def embedding_matched_analysis(
    emb_path: Path | None = None,
    meta_path: Path | None = None,
    *,
    similarity_threshold: float = 0.7,
    score_cap: float = config.EMBEDDING_SCORE_CAP,
    save_path: str | Path | None = None,
    month_block: bool = False,
    exclude_same_ip: bool = False,
    return_extras: bool = False,
    pair_dump_path: str | Path | None = None,
):
    """Two-stage matching as in the notebook: (1) drop pairs with cosine
    similarity below ``similarity_threshold``; (2) require
    ``|Δmoderation_score| <= score_cap``. The two filters are applied in order,
    matching the cell-44 implementation exactly so the resulting sample size
    lines up with the paper.

    Variant options (all opt-in; defaults reproduce the paper design exactly):

    month_block
        Only allow matches within the same calendar month (requires a
        ``month`` column in the metadata csv). Disallowed candidates are
        masked to -inf in the cosine matrix *before* the argmax.
    exclude_same_ip
        Forbid pairs sharing ``hashed_ip`` (requires a ``hashed_ip`` column).
        Masked before the argmax, like month_block.
    return_extras
        Return ``(stats, fig, extras)`` instead of ``(stats, fig)``. The
        extras dict includes pool sizes, pair counts at alternative
        similarity thresholds, and — when ip/month metadata is attached —
        the same-IP / same-month share of both the *baseline* design's
        pairs (cos >= 0.7, unmasked) and the current run's kept pairs.
        When no pairs survive, returns ``(None, None, extras)`` instead of
        raising, so callers can report counts for underpowered variants.
    pair_dump_path
        If set, write the matched pairs (ids, scores, outcomes, and any
        attached metadata columns) to this csv.
    """
    emb_path = Path(emb_path) if emb_path else config.HIGH_CONCERN_EMBEDDINGS_NPY
    meta_path = Path(meta_path) if meta_path else config.HIGH_CONCERN_METADATA_CSV
    embeddings, meta = _load_high_concern(emb_path, meta_path)

    mask_refused = meta["refused_answer"].to_numpy() == 1
    ref_emb = embeddings[mask_refused]
    non_emb = embeddings[~mask_refused]
    ref_meta = meta[mask_refused].reset_index(drop=True)
    non_meta = meta[~mask_refused].reset_index(drop=True)

    sims = _cosine_matrix(ref_emb, non_emb)

    have_ip = "hashed_ip" in meta.columns
    have_month = "month" in meta.columns
    extras: dict = {
        "n_refused": int(len(ref_meta)),
        "n_non_refused": int(len(non_meta)),
        "similarity_threshold": float(similarity_threshold),
        "score_cap": float(score_cap),
        "month_block": bool(month_block),
        "exclude_same_ip": bool(exclude_same_ip),
    }

    # Baseline-design diagnostics (computed on the *unmasked* matrix at the
    # paper's threshold, before any variant masking below).
    if return_extras and (have_ip or have_month):
        base = _select_pairs(sims, ref_meta, non_meta, 0.7,
                             config.EMBEDDING_SCORE_CAP)
        n_base = int(base["keep"].sum())
        extras["baseline_n_pairs"] = n_base
        if have_ip and n_base:
            same_ip = (base["ref_kept"]["hashed_ip"].to_numpy()
                       == base["matched"]["hashed_ip"].to_numpy())
            extras["baseline_same_ip_pairs"] = int(same_ip.sum())
            extras["baseline_same_ip_share"] = float(same_ip.mean())
        if have_month and n_base:
            same_m = (base["ref_kept"]["month"].to_numpy()
                      == base["matched"]["month"].to_numpy())
            extras["baseline_same_month_pairs"] = int(same_m.sum())
            extras["baseline_same_month_share"] = float(same_m.mean())

    # Variant masking: applied to the cosine matrix before the argmax so a
    # refused prompt is re-matched to its best *allowed* control (not simply
    # dropped when its unrestricted nearest neighbour is disallowed).
    if month_block:
        if not have_month:
            raise ValueError("month_block requires a 'month' column in the metadata csv")
        if meta["month"].isna().any():
            raise ValueError("month_block: metadata has null months")
        codes = pd.factorize(meta["month"].astype(str))[0]
        sims[codes[mask_refused][:, None] != codes[~mask_refused][None, :]] = -np.inf
    if exclude_same_ip:
        if not have_ip:
            raise ValueError("exclude_same_ip requires a 'hashed_ip' column in the metadata csv")
        if meta["hashed_ip"].isna().any():
            raise ValueError("exclude_same_ip: metadata has null hashed_ip")
        codes = pd.factorize(meta["hashed_ip"].astype(str))[0]
        sims[codes[mask_refused][:, None] == codes[~mask_refused][None, :]] = -np.inf

    sel = _select_pairs(sims, ref_meta, non_meta, similarity_threshold, score_cap)
    ref_kept, matched, sims_kept = sel["ref_kept"], sel["matched"], sel["sims_kept"]

    if return_extras:
        extras["n_ref_no_candidate"] = int(np.isneginf(sel["best_sim"]).sum())
        extras["n_pairs"] = int(len(ref_kept))
        extras["pairs_by_threshold"] = {
            str(t): int(((sel["best_sim"] >= t) & (sel["score_diff"] <= score_cap)).sum())
            for t in (0.7, 0.9, 0.95, 0.99, 0.995, 0.999)
        }
        if have_ip and len(ref_kept):
            same_ip = (ref_kept["hashed_ip"].to_numpy()
                       == matched["hashed_ip"].to_numpy())
            extras["same_ip_pairs_kept"] = int(same_ip.sum())
            extras["same_ip_share_kept"] = float(same_ip.mean())
        if have_month and len(ref_kept):
            same_m = (ref_kept["month"].to_numpy() == matched["month"].to_numpy())
            extras["same_month_pairs_kept"] = int(same_m.sum())
            extras["same_month_share_kept"] = float(same_m.mean())

    if pair_dump_path is not None and len(ref_kept):
        cols = ["conversation_id", "max_score", "refused_answer", "user_continued"]
        cols += [c for c in ("month", "hashed_ip") if c in meta.columns]
        dump = pd.concat(
            [ref_kept[cols].add_prefix("ref_").reset_index(drop=True),
             matched[cols].add_prefix("ctl_").reset_index(drop=True)],
            axis=1,
        )
        dump["cosine"] = np.asarray(sims_kept, dtype=float)
        dump.to_csv(pair_dump_path, index=False)

    if len(ref_kept) == 0:
        if return_extras:
            return None, None, extras
        raise RuntimeError(
            "No matched pairs satisfy the score cap; consider loosening it."
        )

    r_scores = ref_kept["max_score"].to_numpy(dtype=float)
    m_scores = matched["max_score"].to_numpy(dtype=float)
    stats_kwargs = dict(
        refused_scores=r_scores,
        non_refused_scores=m_scores,
        refused_cont=ref_kept["user_continued"].to_numpy(dtype=int),
        non_refused_cont=matched["user_continued"].to_numpy(dtype=int),
    )
    if np.all(r_scores == m_scores):
        # scipy's wilcoxon raises when every paired difference is zero.
        stats = _degenerate_stats(cosine_similarities=sims_kept, **stats_kwargs)
    else:
        stats = compute_matching_statistics(cosine_similarities=sims_kept,
                                            **stats_kwargs)
    fig = _plot(
        ref_kept["max_score"].to_numpy(),
        matched["max_score"].to_numpy(),
        sims_kept,
        stats,
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if return_extras:
        return stats, fig, extras
    return stats, fig


def _plot(
    refused_scores: np.ndarray,
    matched_scores: np.ndarray,
    sims: np.ndarray,
    stats: MatchStatsResult,
) -> plt.Figure:
    set_paper_style("double")
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.5))
    n = stats.n_pairs
    se_m = np.sqrt(stats.non_refused_rate * (1 - stats.non_refused_rate) / n)
    se_r = np.sqrt(stats.refused_rate * (1 - stats.refused_rate) / n)
    rates = [stats.non_refused_rate, stats.refused_rate]
    errs = [se_m, se_r]

    bars = axes[0].bar(
        [0, 1], rates, yerr=errs, width=0.6,
        color=[LIGHT_BLUE, DARK_BLUE], alpha=0.85,
        capsize=3, edgecolor="black", linewidth=0.8,
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["No refusal", "Refusal"])
    axes[0].set_ylabel("Continuation rate")
    axes[0].set_title("Continuation rates")
    axes[0].set_ylim(0, max(0.05, max(rates) * 1.5))
    for b, r, e in zip(bars, rates, errs):
        axes[0].text(
            b.get_x() + b.get_width() / 2, r + e + 0.02,
            f"{r:.3f}", ha="center", va="bottom", fontsize=7,
        )

    lo = min(refused_scores.min(), matched_scores.min())
    hi = max(refused_scores.max(), matched_scores.max())
    if hi <= lo:  # degenerate range (e.g. very few, identical-score pairs)
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, 20)
    axes[1].hist(matched_scores, bins=bins, alpha=0.6, color=ORANGE,
                 label="No refusal", edgecolor="black", linewidth=0.5)
    axes[1].hist(refused_scores, bins=bins, alpha=0.6, color=DARK_BLUE,
                 label="Refusal", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Moderation score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Score distribution")
    axes[1].legend(loc="upper right")

    axes[2].hist(sims, bins=20, alpha=0.7, color=DARK_BLUE,
                 edgecolor=DARK_BLUE, linewidth=1)
    mean_s = float(sims.mean())
    axes[2].axvline(mean_s, color="black", linestyle="--", linewidth=1.2,
                    label=f"Mean: {mean_s:.3f}")
    axes[2].set_xlabel("Cosine similarity")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Embedding similarity")
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    return fig
