"""
scoring.py — Weighted quality scoring engine.

Aggregates per-metric sub-scores into a final score out of 100 and
generates human-readable interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config as cfg
from src.analysis import FullAnalysis, MetricResult


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ScoreReport:
    """Complete scoring output."""
    sub_scores:         dict[str, float]   # metric_key → sub_score (0–100)
    weighted_scores:    dict[str, float]   # metric_key → weighted contribution
    total_score:        float              # 0–100
    band:               str               # e.g. "Good"
    reductions:         list[str]         # Why the score was reduced
    improvements:       list[str]         # What enhancements helped / would help


def _band(score: float) -> str:
    """Return the quality band label for *score*."""
    for threshold, label in cfg.SCORE_BANDS:
        if score >= threshold:
            return label
    return "Poor"


def compute_score(analysis: FullAnalysis) -> ScoreReport:
    """
    Compute a transparent weighted score from a :class:`FullAnalysis`.

    The weights are defined in :data:`src.config.SCORE_WEIGHTS` and
    sum to 100.  Each metric contributes:

        contribution_i = (sub_score_i / 100) × weight_i

    The total is the sum of all contributions.
    """
    weights = cfg.SCORE_WEIGHTS
    metrics = analysis.as_dict()

    sub_scores:      dict[str, float] = {}
    weighted_scores: dict[str, float] = {}
    reductions:      list[str] = []
    improvements:    list[str] = []

    for key, weight in weights.items():
        metric: MetricResult | None = metrics.get(key)
        if metric is None:
            # Composition might be missing if not computed
            sub_scores[key] = 50.0
            weighted_scores[key] = weight * 0.5
            continue

        sub = float(np.clip(metric.score, 0, 100))
        sub_scores[key] = round(sub, 1)
        contribution = (sub / 100.0) * weight
        weighted_scores[key] = round(contribution, 2)

        if sub < 70:
            lost = round(weight * (1 - sub / 100), 2)
            reductions.append(
                f"**{metric.name}** ({metric.label}): lost {lost:.1f} pts — {metric.explanation}"
            )
        if metric.suggestion and metric.suggestion.lower() not in ("no change needed.", "no exposure correction needed.", "no contrast enhancement needed.", "no sharpening required.", "no white balance correction needed.", "no saturation change needed.", "no dynamic range intervention required."):
            if sub < 80:
                improvements.append(f"• {metric.name}: {metric.suggestion}")

    total = float(np.clip(sum(weighted_scores.values()), 0, 100))

    return ScoreReport(
        sub_scores=sub_scores,
        weighted_scores=weighted_scores,
        total_score=round(total, 1),
        band=_band(total),
        reductions=reductions,
        improvements=improvements,
    )
