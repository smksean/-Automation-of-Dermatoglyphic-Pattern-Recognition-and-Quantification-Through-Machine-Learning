from __future__ import annotations

from io import BytesIO
import unittest

import cv2
import numpy as np
from PIL import Image

from broad_classifier.inference import (
    CLASS_NAMES,
    IMAGE_SIZE,
    InputImageError,
    decode_image,
    image_to_tensor,
    preprocess_image,
)
from broad_classifier.app_logic import assess_prediction
from broad_classifier.inference import PredictionResult


def encoded_image(array: np.ndarray, image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format=image_format)
    return buffer.getvalue()


class BroadClassifierInferenceTests(unittest.TestCase):
    def test_fixed_class_order(self) -> None:
        self.assertEqual(
            CLASS_NAMES,
            ("arch", "left_slant_loop", "right_slant_loop", "whorl"),
        )

    def test_decode_and_preprocess_are_deterministic(self) -> None:
        image = np.full((240, 180), 255, dtype=np.uint8)
        cv2.ellipse(image, (90, 120), (55, 90), 0, 0, 360, 60, 3)
        image_bytes = encoded_image(image)
        decoded = decode_image(image_bytes)
        first = preprocess_image(decoded)
        second = preprocess_image(decoded)
        self.assertEqual(first.shape, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(first.dtype, np.uint8)
        np.testing.assert_array_equal(first, second)

    def test_image_tensor_shape_and_range(self) -> None:
        image = np.full((IMAGE_SIZE, IMAGE_SIZE), 127, dtype=np.uint8)
        tensor = image_to_tensor(image)
        self.assertEqual(tuple(tensor.shape), (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertTrue(np.isfinite(tensor.numpy()).all())

    def test_rejects_empty_or_corrupt_file(self) -> None:
        with self.assertRaises(InputImageError):
            decode_image(b"")
        with self.assertRaises(InputImageError):
            decode_image(b"not an image")

    def test_rejects_tiny_image(self) -> None:
        tiny = np.full((32, 32), 255, dtype=np.uint8)
        with self.assertRaises(InputImageError):
            decode_image(encoded_image(tiny))

    def test_rejects_unsupported_image_format(self) -> None:
        image = np.full((100, 100), 255, dtype=np.uint8)
        with self.assertRaises(InputImageError):
            decode_image(encoded_image(image, "BMP"))

    def test_assessment_flags_uncertain_prediction(self) -> None:
        result = PredictionResult(
            predicted_class="arch",
            predicted_probability=0.48,
            class_probabilities={
                "arch": 0.48,
                "left_slant_loop": 0.42,
                "right_slant_loop": 0.06,
                "whorl": 0.04,
            },
            fold_predictions=(
                "arch",
                "arch",
                "left_slant_loop",
                "left_slant_loop",
                "arch",
            ),
            agreement=0.60,
            top_two_margin=0.06,
        )
        assessment = assess_prediction(result)
        self.assertTrue(assessment.needs_review)
        self.assertEqual(len(assessment.reasons), 3)

    def test_assessment_accepts_internally_consistent_prediction(self) -> None:
        result = PredictionResult(
            predicted_class="whorl",
            predicted_probability=0.91,
            class_probabilities={
                "arch": 0.02,
                "left_slant_loop": 0.03,
                "right_slant_loop": 0.04,
                "whorl": 0.91,
            },
            fold_predictions=("whorl",) * 5,
            agreement=1.0,
            top_two_margin=0.87,
        )
        self.assertFalse(assess_prediction(result).needs_review)


if __name__ == "__main__":
    unittest.main()
