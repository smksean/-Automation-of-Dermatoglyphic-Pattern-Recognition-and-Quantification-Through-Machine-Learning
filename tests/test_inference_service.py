from __future__ import annotations

from io import BytesIO
import os
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from inference_service.app import app


def png_bytes() -> bytes:
    image = np.full((128, 96), 255, dtype=np.uint8)
    image[20:108, 28:68] = 90
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


class InferenceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_does_not_require_authentication(self) -> None:
        with patch("inference_service.app.subtype_available", return_value=True):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_predict_requires_configured_bearer_token(self) -> None:
        with patch.dict(os.environ, {"API_TOKEN": "research-secret"}):
            response = self.client.post(
                "/predict",
                files={"image": ("print.png", png_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 401)

    def test_predict_returns_web_application_contract(self) -> None:
        expected = {
            "predictedClass": "whorl",
            "score": 0.91,
            "probabilities": {
                "arch": 0.02,
                "left_slant_loop": 0.03,
                "right_slant_loop": 0.04,
                "whorl": 0.91,
            },
            "foldPredictions": ["whorl"] * 5,
            "agreement": 1.0,
            "topTwoMargin": 0.87,
            "needsReview": False,
            "reviewReasons": [],
            "subtype": None,
            "processingSeconds": 1.25,
        }
        with patch.dict(os.environ, {"API_TOKEN": "research-secret"}), patch(
            "inference_service.app.runtime.predict", return_value=expected
        ):
            response = self.client.post(
                "/predict",
                headers={"Authorization": "Bearer research-secret"},
                files={"image": ("print.png", png_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_predict_rejects_unsupported_content_type(self) -> None:
        response = self.client.post(
            "/predict",
            files={"image": ("print.bmp", b"not-a-bitmap", "image/bmp")},
        )
        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
