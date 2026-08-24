"""Shared statistical tests for the two matching designs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class MatchStatsResult:
    n_pairs: int
    refused_rate: float
    non_refused_rate: float
    pp_difference: float
    ratio: float
    ci_lower: float
    ci_upper: float
    wilcoxon_W: float
    wilcoxon_p: float
    ks_D: float
    ks_p: float
    mcnemar_chi2: float
    mcnemar_p: float
    binomial_p: float
    mean_similarity: float | None = None


def compute_matching_statistics(
    refused_scores: np.ndarray,
    non_refused_scores: np.ndarray,
    refused_cont: np.ndarray,
    non_refused_cont: np.ndarray,
    *,
    cosine_similarities: np.ndarray | None = None,
) -> MatchStatsResult:
    """Run the paper's suite of balance + continuation tests on a matched set.

    Parameters
    ----------
    refused_scores, non_refused_scores
        Moderation scores of the refused and matched-non-refused prompts
        (aligned, so position ``i`` in each is a pair).
    refused_cont, non_refused_cont
        Continuation indicators (0/1) for the same pairs.
    cosine_similarities
        Optional cosine similarity between each pair's embeddings.
    """
    n_pairs = len(refused_scores)
    refused_rate = float(np.mean(refused_cont))
    matched_rate = float(np.mean(non_refused_cont))
    diff = (refused_rate - matched_rate) * 100.0
    ratio = matched_rate / refused_rate if refused_rate > 0 else float("inf")

    # Balance tests.
    w_stat, w_p = stats.wilcoxon(refused_scores, non_refused_scores)
    ks_stat, ks_p = stats.ks_2samp(refused_scores, non_refused_scores)

    # Continuation tests.
    both = np.sum((refused_cont == 1) & (non_refused_cont == 1))
    only_ref = np.sum((refused_cont == 1) & (non_refused_cont == 0))
    only_non = np.sum((refused_cont == 0) & (non_refused_cont == 1))
    neither = np.sum((refused_cont == 0) & (non_refused_cont == 0))
    if only_ref + only_non > 0:
        mcnemar_stat = (abs(only_ref - only_non) - 1) ** 2 / (only_ref + only_non)
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
    else:
        mcnemar_stat, mcnemar_p = 0.0, 1.0

    binom = stats.binomtest(int(np.sum(refused_cont)), n_pairs, matched_rate,
                             alternative="two-sided").pvalue

    se = np.sqrt(
        (refused_rate * (1 - refused_rate) + matched_rate * (1 - matched_rate)) / n_pairs
    )
    ci = ((refused_rate - matched_rate) - 1.96 * se,
          (refused_rate - matched_rate) + 1.96 * se)

    return MatchStatsResult(
        n_pairs=n_pairs,
        refused_rate=refused_rate,
        non_refused_rate=matched_rate,
        pp_difference=diff,
        ratio=ratio,
        ci_lower=ci[0],
        ci_upper=ci[1],
        wilcoxon_W=float(w_stat),
        wilcoxon_p=float(w_p),
        ks_D=float(ks_stat),
        ks_p=float(ks_p),
        mcnemar_chi2=float(mcnemar_stat),
        mcnemar_p=float(mcnemar_p),
        binomial_p=float(binom),
        mean_similarity=float(np.mean(cosine_similarities)) if cosine_similarities is not None else None,
    )
