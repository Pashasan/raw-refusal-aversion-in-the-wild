"""Figure 4: score-based caliper matching on risky prompts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from ..paper_style import set_paper_style, DARK_BLUE, LIGHT_BLUE, ORANGE
from ._stats import MatchStatsResult, compute_matching_statistics


def _first_user_messages(msg_df: pd.DataFrame) -> pd.DataFrame:
    first = msg_df[(msg_df["role"] == "user") & (msg_df["message_number"] == 1)][
        ["conversation_id", "max_concern_score"]
    ].copy()
    asst = msg_df[(msg_df["role"] == "assistant") & (msg_df["message_number"] == 2)][
        ["conversation_id", "refused_answer"]
    ]
    third = msg_df[(msg_df["role"] == "user") & (msg_df["message_number"] == 3)][
        ["conversation_id"]
    ].assign(user_continued=1)

    conv = first.merge(asst, on="conversation_id", how="inner")
    conv = conv.merge(third, on="conversation_id", how="left")
    conv["user_continued"] = conv["user_continued"].fillna(0).astype(int)
    return conv


def _greedy_caliper_match(
    refused: pd.DataFrame,
    non_refused: pd.DataFrame,
    caliper: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """For each refused prompt, find the closest non-refused within ``caliper``.

    Matching is with replacement (a non-refused prompt can be reused) to mirror
    the notebook's implementation.
    """
    ref_scores = refused["max_score"].to_numpy(dtype=float)[:, None]
    non_scores = non_refused["max_score"].to_numpy(dtype=float)
    distances = np.abs(ref_scores - non_scores)
    nearest = np.argmin(distances, axis=1)
    best_dist = distances[np.arange(len(refused)), nearest]
    keep = best_dist <= caliper
    refused_kept = refused.iloc[keep].reset_index(drop=True)
    matched = non_refused.iloc[nearest[keep]].reset_index(drop=True)
    return refused_kept, matched, best_dist[keep]


def score_matched_analysis(
    msg_df: pd.DataFrame,
    *,
    threshold: float = config.RISK_THRESHOLD,
    caliper: float = config.SCORE_MATCH_CALIPER,
    save_path: str | Path | None = None,
) -> tuple[MatchStatsResult, plt.Figure]:
    """Run the paragraph-1 matching of Section 4.2 and produce Figure 4."""
    conv = _first_user_messages(msg_df)
    conv = conv.rename(columns={"max_concern_score": "max_score"})
    # Add max_score if not already present (notebook used 'max_score' derived from score columns).
    high_concern = conv[conv["max_score"] > threshold].copy()
    refused = high_concern[high_concern["refused_answer"] == 1]
    non_refused = high_concern[high_concern["refused_answer"] == 0]
    if len(refused) == 0 or len(non_refused) == 0:
        raise RuntimeError("Insufficient risky refused/non-refused prompts for matching.")

    refused_kept, matched, match_dist = _greedy_caliper_match(refused, non_refused, caliper)
    stats = compute_matching_statistics(
        refused_scores=refused_kept["max_score"].to_numpy(dtype=float),
        non_refused_scores=matched["max_score"].to_numpy(dtype=float),
        refused_cont=refused_kept["user_continued"].to_numpy(dtype=int),
        non_refused_cont=matched["user_continued"].to_numpy(dtype=int),
    )

    fig = _plot(
        refused_kept["max_score"].to_numpy(),
        matched["max_score"].to_numpy(),
        refused_kept["user_continued"].to_numpy(),
        matched["user_continued"].to_numpy(),
        match_dist,
        stats,
        caliper,
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return stats, fig


def _plot(
    refused_scores: np.ndarray,
    matched_scores: np.ndarray,
    refused_cont: np.ndarray,
    matched_cont: np.ndarray,
    match_distances: np.ndarray,
    stats: MatchStatsResult,
    caliper: float,
) -> plt.Figure:
    set_paper_style("double")
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.5))
    n = len(refused_scores)
    se_matched = np.sqrt(stats.non_refused_rate * (1 - stats.non_refused_rate) / n)
    se_refused = np.sqrt(stats.refused_rate * (1 - stats.refused_rate) / n)
    rates = [stats.non_refused_rate, stats.refused_rate]
    errors = [se_matched, se_refused]

    bars = axes[0].bar(
        [0, 1], rates, yerr=errors, width=0.6,
        color=[LIGHT_BLUE, DARK_BLUE], alpha=0.85,
        capsize=3, edgecolor="black", linewidth=0.8,
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["No refusal", "Refusal"])
    axes[0].set_ylabel("Continuation rate")
    axes[0].set_title("Continuation rates")
    top = max(rates)
    axes[0].set_ylim(0, max(0.05, top * 1.25))
    for bar, rate, err in zip(bars, rates, errors):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, rate + err + 0.02,
            f"{rate:.3f}", ha="center", va="bottom", fontsize=7,
        )

    bins = np.linspace(
        min(refused_scores.min(), matched_scores.min()),
        max(refused_scores.max(), matched_scores.max()),
        20,
    )
    axes[1].hist(matched_scores, bins=bins, alpha=0.6, color=ORANGE,
                 label="No refusal", edgecolor="black", linewidth=0.5)
    axes[1].hist(refused_scores, bins=bins, alpha=0.6, color=DARK_BLUE,
                 label="Refusal", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Moderation score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Score distribution")
    axes[1].legend(loc="upper right")

    axes[2].hist(match_distances, bins=20, alpha=0.7, color=DARK_BLUE,
                 edgecolor=DARK_BLUE, linewidth=1)
    mean_d = float(match_distances.mean())
    axes[2].axvline(mean_d, color="black", linestyle="--", linewidth=1.2,
                    label=f"Mean: {mean_d:.3f}")
    axes[2].set_xlabel("Absolute score difference")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title(f"Matching quality (caliper {caliper})")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    return fig
