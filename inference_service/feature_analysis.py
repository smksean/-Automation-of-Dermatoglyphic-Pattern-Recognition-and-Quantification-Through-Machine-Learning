"""Conservative image-derived feature summaries for uploaded fingerprints.

The measurements in this module are descriptive quality and ridge-flow proxies.
They deliberately do not manufacture core, delta, minutia, or TFRC values where
the project has no validated uploaded-image detector.
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from broad_classifier.inference import crop_foreground


ANALYSIS_MAX_SIDE = 800
ORIENTATION_BLOCK = 32


def _largest_foreground_mask(image: np.ndarray) -> np.ndarray:
    dark_pixels = np.where(image < 245, 255, 0).astype(np.uint8)
    kernel_size = max(9, int(round(min(image.shape) * 0.04)) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    connected = cv2.morphologyEx(dark_pixels, cv2.MORPH_CLOSE, kernel)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        connected,
    )
    mask = np.zeros_like(image, dtype=np.uint8)
    if component_count <= 1:
        return mask
    largest_label = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    mask[labels == largest_label] = 255
    return mask


def _orientation_field(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[dict[str, float]], float]:
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    blocks: list[dict[str, float]] = []
    coherence_values: list[float] = []

    for top in range(0, image.shape[0] - ORIENTATION_BLOCK + 1, ORIENTATION_BLOCK):
        for left in range(0, image.shape[1] - ORIENTATION_BLOCK + 1, ORIENTATION_BLOCK):
            block_mask = (
                mask[
                    top : top + ORIENTATION_BLOCK,
                    left : left + ORIENTATION_BLOCK,
                ]
                > 0
            )
            if float(block_mask.mean()) < 0.55:
                continue
            block_x = gradient_x[
                top : top + ORIENTATION_BLOCK,
                left : left + ORIENTATION_BLOCK,
            ][block_mask]
            block_y = gradient_y[
                top : top + ORIENTATION_BLOCK,
                left : left + ORIENTATION_BLOCK,
            ][block_mask]
            xx = float(np.sum(block_x * block_x))
            yy = float(np.sum(block_y * block_y))
            xy = float(np.sum(block_x * block_y))
            energy = xx + yy
            if energy < 1.0:
                continue
            coherence = float(np.hypot(xx - yy, 2.0 * xy) / (energy + 1e-6))
            angle = float(0.5 * np.arctan2(2.0 * xy, xx - yy) + np.pi / 2.0)
            blocks.append(
                {
                    "x": float((left + ORIENTATION_BLOCK / 2) / image.shape[1]),
                    "y": float((top + ORIENTATION_BLOCK / 2) / image.shape[0]),
                    "angleDegrees": float(np.degrees(angle) % 180.0),
                    "coherence": round(coherence, 4),
                }
            )
            coherence_values.append(coherence)

    mean_coherence = float(np.mean(coherence_values)) if coherence_values else 0.0
    return blocks, mean_coherence


def _quality_score(
    coverage: float,
    contrast: float,
    sharpness: float,
    coherence: float,
    source_shape: tuple[int, int],
) -> tuple[int, str, list[str]]:
    coverage_score = float(np.clip(coverage / 0.58, 0.0, 1.0))
    contrast_score = float(np.clip((contrast - 12.0) / 60.0, 0.0, 1.0))
    sharpness_score = float(np.clip(sharpness / 4000.0, 0.0, 1.0))
    coherence_score = float(np.clip((coherence - 0.18) / 0.42, 0.0, 1.0))
    score = int(
        round(
            100.0
            * (
                0.20 * coverage_score
                + 0.25 * contrast_score
                + 0.25 * sharpness_score
                + 0.30 * coherence_score
            )
        )
    )
    reasons: list[str] = []
    if min(source_shape) < 600:
        reasons.append(
            "The source is below 600 pixels on its shorter side; fine-ridge feature analysis is limited."
        )
        score = min(score, 64)
    if coverage < 0.40:
        reasons.append("The detected fingerprint foreground has limited image coverage.")
    if contrast < 28.0:
        reasons.append("Ridge-to-valley contrast is low.")
    if sharpness < 500.0:
        reasons.append("The impression appears blurred or lacks fine ridge detail.")
    if coherence < 0.30:
        reasons.append("Local ridge flow is weak or inconsistent.")

    if score >= 75 and not reasons:
        grade = "good"
    elif score >= 48:
        grade = "review"
    else:
        grade = "insufficient"
    return score, grade, reasons


def _render_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    orientation_blocks: list[dict[str, float]],
) -> str | None:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (64, 139, 121), 2)
    length = max(7, ORIENTATION_BLOCK // 3)
    for block in orientation_blocks:
        if block["coherence"] < 0.32:
            continue
        center_x = int(round(block["x"] * image.shape[1]))
        center_y = int(round(block["y"] * image.shape[0]))
        angle = np.radians(block["angleDegrees"])
        offset_x = int(round(np.cos(angle) * length))
        offset_y = int(round(np.sin(angle) * length))
        cv2.line(
            canvas,
            (center_x - offset_x, center_y - offset_y),
            (center_x + offset_x, center_y + offset_y),
            (146, 100, 29),
            1,
            cv2.LINE_AA,
        )
    encoded, buffer = cv2.imencode(
        ".jpg",
        canvas,
        [int(cv2.IMWRITE_JPEG_QUALITY), 78],
    )
    if not encoded:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def analyze_fingerprint_features(
    decoded_image: np.ndarray,
    *,
    include_overlay: bool,
) -> dict[str, Any]:
    """Return deterministic, non-identifying image and ridge-flow summaries."""
    source_height, source_width = decoded_image.shape
    cropped = crop_foreground(decoded_image)
    scale = min(1.0, ANALYSIS_MAX_SIDE / max(cropped.shape))
    if scale < 1.0:
        working = cv2.resize(
            cropped,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = cropped.copy()

    mask = _largest_foreground_mask(working)
    foreground = mask > 0
    if not np.any(foreground):
        return {
            "quality": {
                "score": 0,
                "grade": "insufficient",
                "needsReview": True,
                "reasons": ["A usable fingerprint foreground could not be isolated."],
            },
            "measurements": {
                "sourceWidth": source_width,
                "sourceHeight": source_height,
                "foregroundCoverage": 0.0,
                "ridgeContrast": 0.0,
                "sharpness": 0.0,
                "orientationCoherence": 0.0,
                "ridgeDensityProxy": 0.0,
            },
            "orientationBlocks": [],
            "overlayDataUrl": None,
            **_unsupported_endpoint_payloads(),
        }

    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(working)
    foreground_values = working[foreground]
    coverage = float(foreground.mean())
    contrast = float(np.std(foreground_values))
    laplacian = cv2.Laplacian(working, cv2.CV_64F)
    sharpness = float(np.var(laplacian[foreground]))
    orientation_blocks, coherence = _orientation_field(working, mask)
    ridge_binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    ridge_density = float(np.mean(ridge_binary[foreground] > 0))
    score, grade, reasons = _quality_score(
        coverage,
        contrast,
        sharpness,
        coherence,
        (source_height, source_width),
    )

    return {
        "quality": {
            "score": score,
            "grade": grade,
            "needsReview": grade != "good",
            "reasons": reasons,
        },
        "measurements": {
            "sourceWidth": source_width,
            "sourceHeight": source_height,
            "foregroundCoverage": round(coverage, 4),
            "ridgeContrast": round(contrast, 2),
            "sharpness": round(sharpness, 2),
            "orientationCoherence": round(coherence, 4),
            "ridgeDensityProxy": round(ridge_density, 4),
        },
        "orientationBlocks": orientation_blocks,
        "overlayDataUrl": (
            _render_overlay(working, mask, orientation_blocks)
            if include_overlay
            else None
        ),
        **_unsupported_endpoint_payloads(),
    }


def _unsupported_endpoint_payloads() -> dict[str, Any]:
    return {
        "minutiae": {
            "status": "validation_required",
            "ridgeEndings": None,
            "bifurcations": None,
            "reason": (
                "The cohort contains examiner-marked EFS minutiae, but no uploaded-image "
                "detector has passed comparison against those annotations."
            ),
        },
        "landmarks": {
            "status": "validation_required",
            "cores": None,
            "deltas": None,
            "reason": "Automated core and delta locations are withheld until landmark validation is completed.",
        },
        "ridgeCount": {
            "status": "not_reported",
            "value": None,
            "reason": (
                "TFRC field 9.322 is absent from the restored records, so an image-derived "
                "core-to-delta count is not reported as a validated endpoint."
            ),
        },
        "scope": (
            "Descriptive image-quality and ridge-flow analysis; not identity matching, "
            "forensic minutiae certification, or validated TFRC."
        ),
    }
