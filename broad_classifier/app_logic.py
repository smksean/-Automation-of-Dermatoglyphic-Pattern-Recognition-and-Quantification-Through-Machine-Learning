"""Presentation rules for broad-pattern predictions."""

from __future__ import annotations

from dataclasses import dataclass

from .inference import PredictionResult


LOW_PROBABILITY_THRESHOLD = 0.60
LOW_AGREEMENT_THRESHOLD = 0.80
SMALL_MARGIN_THRESHOLD = 0.15


@dataclass(frozen=True)
class PredictionAssessment:
    needs_review: bool
    reasons: tuple[str, ...]


def assess_prediction(result: PredictionResult) -> PredictionAssessment:
    """Identify ensemble outputs that need cautious manual interpretation."""
    reasons: list[str] = []
    if result.predicted_probability < LOW_PROBABILITY_THRESHOLD:
        reasons.append("The leading model probability is below 60%.")
    if result.agreement < LOW_AGREEMENT_THRESHOLD:
        reasons.append("Fewer than four of the five fold models agree.")
    if result.top_two_margin < SMALL_MARGIN_THRESHOLD:
        reasons.append("The two leading classes are separated by less than 15 points.")
    return PredictionAssessment(needs_review=bool(reasons), reasons=tuple(reasons))
