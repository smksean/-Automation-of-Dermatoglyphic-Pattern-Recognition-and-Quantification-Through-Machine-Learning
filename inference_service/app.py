"""Private HTTP inference service for the research web application."""

from __future__ import annotations

import base64
import hmac
import gc
from hashlib import sha256
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from broad_classifier.app_logic import assess_prediction, assess_subtype_prediction
from broad_classifier.inference import (
    CLASS_NAMES,
    EXPECTED_CHECKPOINTS,
    InputImageError,
    PredictionResult,
    build_model,
    decode_image,
    image_to_tensor,
    preprocess_image,
)
from broad_classifier.model_assets import ensure_checkpoints
from broad_classifier.subtype_inference import predict_subtype, subtype_available


ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
ALLOWED_CONTENT_TYPES = frozenset({"image/png", "image/jpeg", "image/tiff"})
DEFAULT_LOCAL_MODELS = ROOT / "models" / "efficientnet_320_cv"
DEFAULT_LOCAL_SUBTYPES = (
    ROOT
    / "data"
    / "processed"
    / "subtype_review_completed_2026-09-06"
    / "selected_subtype_model_artifacts"
)
PACKAGED_SUBTYPES = ROOT / "inference_service" / "artifacts"
RENDER_SECRET_SUBTYPES = Path("/etc/secrets")
MATERIALIZED_SUBTYPES = Path("/tmp/dermatoglyphic-subtypes")
SUBTYPE_ARTIFACT_SHA256 = {
    "arch": "3cd758560fe43364a1732a00fe9fe02a31153e8a0c80771babcbee2710b6d817",
    "whorl": "56397dbb25309f9edfa40fbce010336d2a1a244aaa6fae62358893c35114dc74",
}


def _model_directory() -> Path:
    configured = os.environ.get("BROAD_CLASSIFIER_MODEL_DIR")
    if configured:
        return Path(configured)
    if DEFAULT_LOCAL_MODELS.is_dir():
        return DEFAULT_LOCAL_MODELS
    return Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/dermatoglyphic-models"))


def _subtype_directory() -> Path:
    configured = os.environ.get("SUBTYPE_CLASSIFIER_MODEL_DIR")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [DEFAULT_LOCAL_SUBTYPES, RENDER_SECRET_SUBTYPES, PACKAGED_SUBTYPES]
    )
    for candidate in candidates:
        if all(
            (candidate / f"{task}_subtype_classifier.pkl").is_file()
            for task in SUBTYPE_ARTIFACT_SHA256
        ):
            return candidate
        materialized = _materialize_subtype_secrets(candidate, MATERIALIZED_SUBTYPES)
        if materialized is not None:
            return materialized
    return PACKAGED_SUBTYPES


def _materialize_subtype_secrets(
    source_directory: Path,
    target_directory: Path,
) -> Optional[Path]:
    encoded_paths = {
        task: source_directory / f"{task}_subtype_classifier.b64"
        for task in SUBTYPE_ARTIFACT_SHA256
    }
    if not all(path.is_file() for path in encoded_paths.values()):
        return None

    target_directory.mkdir(parents=True, exist_ok=True)
    for task, encoded_path in encoded_paths.items():
        try:
            artifact_bytes = base64.b64decode(
                encoded_path.read_text(encoding="ascii"),
                validate=True,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Invalid encoded {task} subtype artifact.") from exc
        if sha256(artifact_bytes).hexdigest() != SUBTYPE_ARTIFACT_SHA256[task]:
            raise RuntimeError(f"Integrity check failed for {task} subtype artifact.")
        artifact_path = target_directory / f"{task}_subtype_classifier.pkl"
        artifact_path.write_bytes(artifact_bytes)
        artifact_path.chmod(0o600)
    return target_directory


class SequentialBroadEnsemble:
    """Memory-conscious five-checkpoint inference for small CPU containers."""

    def __init__(self, model_directory: Path) -> None:
        self.checkpoint_paths = tuple(
            model_directory / filename for filename in EXPECTED_CHECKPOINTS
        )

    def predict_bytes(self, image_bytes: bytes) -> tuple[PredictionResult, np.ndarray]:
        decoded = decode_image(image_bytes)
        preprocessed = preprocess_image(decoded)
        tensor = image_to_tensor(preprocessed)
        probability_rows: list[np.ndarray] = []
        model = build_model()
        model.eval()

        with torch.inference_mode():
            for checkpoint_path in self.checkpoint_paths:
                state_dictionary = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(state_dictionary, strict=True)
                probabilities = torch.softmax(model(tensor), dim=1)[0]
                probability_rows.append(probabilities.cpu().numpy())
                del state_dictionary

        per_fold = np.stack(probability_rows)
        mean_probabilities = per_fold.mean(axis=0)
        predicted_index = int(mean_probabilities.argmax())
        fold_indices = per_fold.argmax(axis=1)
        sorted_probabilities = np.sort(mean_probabilities)
        result = PredictionResult(
            predicted_class=CLASS_NAMES[predicted_index],
            predicted_probability=float(mean_probabilities[predicted_index]),
            class_probabilities={
                name: float(mean_probabilities[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            fold_predictions=tuple(CLASS_NAMES[int(index)] for index in fold_indices),
            agreement=float(np.mean(fold_indices == predicted_index)),
            top_two_margin=float(sorted_probabilities[-1] - sorted_probabilities[-2]),
        )
        del model, tensor
        gc.collect()
        return result, preprocessed


class ModelRuntime:
    """Thread-safe lazy owner of the CPU model ensemble."""

    def __init__(self) -> None:
        self._ensemble: SequentialBroadEnsemble | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._ensemble is not None

    def ensemble(self) -> SequentialBroadEnsemble:
        if self._ensemble is None:
            with self._lock:
                if self._ensemble is None:
                    directory = ensure_checkpoints(_model_directory())
                    self._ensemble = SequentialBroadEnsemble(directory)
        return self._ensemble

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        started = time.perf_counter()
        result, preprocessed = self.ensemble().predict_bytes(image_bytes)
        assessment = assess_prediction(result)

        subtype_payload: dict[str, Any] | None = None
        subtype_directory = _subtype_directory()
        if result.predicted_class in {"arch", "whorl"} and subtype_available(
            subtype_directory,
            broad_class=result.predicted_class,
        ):
            subtype_result = predict_subtype(
                result.predicted_class,
                preprocessed,
                artifact_dir=subtype_directory,
            )
            subtype_assessment = assess_subtype_prediction(
                str(subtype_result["predicted_subtype"]),
                float(subtype_result["confidence"]),
                broad_prediction_needs_review=assessment.needs_review,
            )
            subtype_payload = {
                "predictedSubtype": subtype_result["predicted_subtype"],
                "score": subtype_result["confidence"],
                "probabilities": subtype_result["probabilities"],
                "needsReview": subtype_assessment.needs_review,
                "reviewReasons": list(subtype_assessment.reasons),
                "scope": subtype_result["research_scope"],
            }

        return {
            "predictedClass": result.predicted_class,
            "score": result.predicted_probability,
            "probabilities": result.class_probabilities,
            "foldPredictions": list(result.fold_predictions),
            "agreement": result.agreement,
            "topTwoMargin": result.top_two_margin,
            "needsReview": assessment.needs_review,
            "reviewReasons": list(assessment.reasons),
            "subtype": subtype_payload,
            "processingSeconds": time.perf_counter() - started,
        }


runtime = ModelRuntime()
app = FastAPI(
    title="Dermatoglyphic Research Inference API",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def require_api_token(authorization: Optional[str] = Header(default=None)) -> None:
    configured_token = os.environ.get("API_TOKEN")
    if not configured_token:
        return
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not hmac.compare_digest(supplied, configured_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.get("/health")
def health() -> dict[str, Any]:
    subtype_directory = _subtype_directory()
    return {
        "status": "ok",
        "broadModelLoaded": runtime.loaded,
        "archSubtypeAvailable": subtype_available(subtype_directory, "arch"),
        "whorlSubtypeAvailable": subtype_available(subtype_directory, "whorl"),
    }


@app.post("/predict", dependencies=[Depends(require_api_token)])
def predict(image: UploadFile = File(...)) -> JSONResponse:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image format. Use PNG, JPEG, or TIFF.",
        )

    image_bytes = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The deployment upload limit is 4 MB per image.",
        )

    try:
        payload = runtime.predict(image_bytes)
    except InputImageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model service could not complete this request.",
        ) from exc

    return JSONResponse(payload, headers={"Cache-Control": "no-store"})
