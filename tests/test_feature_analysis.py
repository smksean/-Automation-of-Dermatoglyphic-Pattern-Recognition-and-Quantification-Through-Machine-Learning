from __future__ import annotations

import unittest

import cv2
import numpy as np

from inference_service.feature_analysis import analyze_fingerprint_features


class FingerprintFeatureAnalysisTests(unittest.TestCase):
    def test_ridge_flow_summary_is_finite_and_bounded(self) -> None:
        image = np.full((800, 720), 255, dtype=np.uint8)
        center = (360, 470)
        for radius in range(55, 310, 13):
            cv2.ellipse(image, center, (radius, int(radius * 0.75)), 0, 195, 345, 45, 4)

        result = analyze_fingerprint_features(image, include_overlay=True)

        self.assertIn(result["quality"]["grade"], {"good", "review", "insufficient"})
        self.assertGreaterEqual(result["quality"]["score"], 0)
        self.assertLessEqual(result["quality"]["score"], 100)
        for field in (
            "foregroundCoverage",
            "orientationCoherence",
            "ridgeDensityProxy",
        ):
            self.assertTrue(np.isfinite(result["measurements"][field]))
            self.assertGreaterEqual(result["measurements"][field], 0.0)
            self.assertLessEqual(result["measurements"][field], 1.0)
        self.assertTrue(result["overlayDataUrl"].startswith("data:image/jpeg;base64,"))

    def test_small_source_is_capped_for_review(self) -> None:
        image = np.tile(np.arange(320, dtype=np.uint8), (320, 1))
        result = analyze_fingerprint_features(image, include_overlay=False)
        self.assertNotEqual(result["quality"]["grade"], "good")
        self.assertLessEqual(result["quality"]["score"], 64)
        self.assertTrue(any("below 600 pixels" in reason for reason in result["quality"]["reasons"]))
        self.assertIsNone(result["overlayDataUrl"])

    def test_unvalidated_endpoints_are_never_fabricated(self) -> None:
        image = np.full((700, 700), 255, dtype=np.uint8)
        result = analyze_fingerprint_features(image, include_overlay=False)
        self.assertIsNone(result["minutiae"]["ridgeEndings"])
        self.assertIsNone(result["minutiae"]["bifurcations"])
        self.assertIsNone(result["landmarks"]["cores"])
        self.assertIsNone(result["landmarks"]["deltas"])
        self.assertIsNone(result["ridgeCount"]["value"])


if __name__ == "__main__":
    unittest.main()
