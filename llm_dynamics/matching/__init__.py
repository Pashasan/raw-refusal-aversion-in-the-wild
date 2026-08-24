"""Matching-based causal analyses (Section 4.2 of the paper)."""

from .score import score_matched_analysis
from .embedding import embedding_matched_analysis

__all__ = ["score_matched_analysis", "embedding_matched_analysis"]
