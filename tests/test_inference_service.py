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
import torch
from fastapi.testclient import TestClient
from PIL import Image

from inference_service.app import (
    FINGER_IDS,
    SequentialBroadEnsemble,
    SUBTYPE_ARTIFACT_SHA256,
    _materialize_subtype_secrets,
    app,
    jobs,
    jobs_lock,
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
        with jobs_lock:
            jobs.clear()

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

    def test_background_job_returns_completed_prediction(self) -> None:
        expected = {
            "predictedClass": "arch",
            "score": 0.9,
            "subtype": None,
        }
        with patch("inference_service.app.runtime.predict", return_value=expected):
            submitted = self.client.post(
                "/jobs",
                files={"image": ("print.png", png_bytes(), "image/png")},
            )
        self.assertEqual(submitted.status_code, 202)
        job_id = submitted.json()["jobId"]

        completed = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.json()["status"], "complete")
        self.assertEqual(completed.json()["result"], expected)
        self.assertEqual(completed.headers["cache-control"], "no-store")

    def test_batch_job_collects_ten_images_and_returns_one_result(self) -> None:
        expected = {
            "fingerResults": [],
            "counts": {"arch": 1, "loop": 7, "whorl": 2},
            "pii": 11,
            "reviewCount": 0,
            "featureReviewCount": 0,
            "processingSeconds": 12.5,
            "scope": "test",
        }
        created = self.client.post("/batch-jobs")
        self.assertEqual(created.status_code, 201)
        job_id = created.json()["jobId"]

        for finger_id in FINGER_IDS:
            uploaded = self.client.post(
                f"/batch-jobs/{job_id}/images/{finger_id}",
                files={"image": (f"{finger_id}.png", png_bytes(), "image/png")},
            )
            self.assertEqual(uploaded.status_code, 200)
        self.assertEqual(uploaded.json()["uploadedCount"], 10)

        with patch("inference_service.app.runtime.predict_batch", return_value=expected) as predict_batch:
            started = self.client.post(f"/batch-jobs/{job_id}/start")
        self.assertEqual(started.status_code, 202)
        predict_batch.assert_called_once()

        completed = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.json()["status"], "complete")
        self.assertEqual(completed.json()["result"]["pii"], 11)
        self.assertNotIn("images", completed.json())

    def test_batch_job_cannot_start_with_missing_fingers(self) -> None:
        created = self.client.post("/batch-jobs")
        job_id = created.json()["jobId"]
        response = self.client.post(f"/batch-jobs/{job_id}/start")
        self.assertEqual(response.status_code, 400)

    def test_batch_ensemble_loads_each_checkpoint_once(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.call_count = 0

            def eval(self):
                return self

            def load_state_dict(self, _state, strict=True):
                self.strict = strict

            def __call__(self, batch):
                self.call_count += 1
                logits = torch.zeros((batch.shape[0], 4), dtype=torch.float32)
                logits[:, 0] = 4.0
                return logits

        model = FakeModel()
        ensemble = SequentialBroadEnsemble(Path("unused"))
        images = [np.full((320, 320), 127, dtype=np.uint8) for _ in range(10)]
        with patch("inference_service.app.build_model", return_value=model), patch(
            "inference_service.app.torch.load",
            return_value={},
        ) as checkpoint_load:
            results, _ = ensemble.predict_preprocessed_batch(images)

        self.assertEqual(len(results), 10)
        self.assertTrue(all(result.predicted_class == "arch" for result in results))
        self.assertEqual(checkpoint_load.call_count, 5)
        self.assertEqual(model.call_count, 25)

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
