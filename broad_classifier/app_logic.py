"""Presentation rules for broad-pattern predictions."""

from __future__ import annotations

from dataclasses import dataclass

from .inference import PredictionResult


LOW_PROBABILITY_THRESHOLD = 0.60
LOW_AGREEMENT_THRESHOLD = 0.80
SMALL_MARGIN_THRESHOLD = 0.15
LOW_SUBTYPE_CONFIDENCE_THRESHOLD = 0.70
RARE_WHORL_SUBTYPES = frozenset(
    {"central_pocket_loop_whorl", "double_loop_whorl"}
)


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


def assess_subtype_prediction(
    predicted_subtype: str,
    confidence: float,
    broad_prediction_needs_review: bool = False,
) -> PredictionAssessment:
    """Identify conditional subtype outputs that require expert review."""
    reasons: list[str] = []
    if broad_prediction_needs_review:
        reasons.append("The upstream broad-pattern prediction requires review.")
    if confidence < LOW_SUBTYPE_CONFIDENCE_THRESHOLD:
        reasons.append("The leading subtype score is below 70%.")
    if predicted_subtype in RARE_WHORL_SUBTYPES:
        reasons.append(
            "This minority whorl subtype had limited training examples and weak "
            "cross-validation performance."
        )
    return PredictionAssessment(needs_review=bool(reasons), reasons=tuple(reasons))
