from __future__ import annotations

from io import BytesIO
import base64
from hashlib import sha256
from pathlib import Path
import os
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from inference_service.app import (
    SUBTYPE_ARTIFACT_SHA256,
    _materialize_subtype_secrets,
    app,
)


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

    def test_encoded_subtype_secrets_are_verified_and_materialized(self) -> None:
        artifact_directory = (
            Path(__file__).resolve().parents[1] / "inference_service" / "artifacts"
        )
        with TemporaryDirectory() as source_name, TemporaryDirectory() as target_name:
            source = Path(source_name)
            target = Path(target_name)
            for task, expected_digest in SUBTYPE_ARTIFACT_SHA256.items():
                artifact = artifact_directory / f"{task}_subtype_classifier.pkl"
                self.assertEqual(
                    expected_digest,
                    sha256(artifact.read_bytes()).hexdigest(),
                )
                encoded = base64.b64encode(artifact.read_bytes()).decode("ascii")
                (source / f"{task}_subtype_classifier.b64").write_text(encoded, encoding="ascii")

            result = _materialize_subtype_secrets(source, target)
            self.assertEqual(result, target)
            for task in SUBTYPE_ARTIFACT_SHA256:
                self.assertEqual(
                    (artifact_directory / f"{task}_subtype_classifier.pkl").read_bytes(),
                    (target / f"{task}_subtype_classifier.pkl").read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
