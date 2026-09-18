"""Private HTTP inference service for the research web application."""

from __future__ import annotations

import base64
import hmac
import gc
import logging
import secrets
from contextlib import asynccontextmanager
from hashlib import sha256
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)
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
from inference_service.feature_analysis import analyze_fingerprint_features


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
JOB_TTL_SECONDS = 10 * 60
MAX_ACTIVE_JOBS = 2
MAX_BATCH_UPLOAD_BYTES = 20 * 1024 * 1024
MODEL_BATCH_SIZE = max(1, int(os.environ.get("MODEL_BATCH_SIZE", "2")))
FINGER_IDS = (
    "right-thumb",
    "right-index",
    "right-middle",
    "right-ring",
    "right-little",
    "left-thumb",
    "left-index",
    "left-middle",
    "left-ring",
    "left-little",
)
logger = logging.getLogger(__name__)


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
        return self.predict_preprocessed_batch((preprocessed,))[0][0], preprocessed

    def predict_preprocessed_batch(
        self,
        images: Sequence[np.ndarray],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[PredictionResult], float]:
        """Run every checkpoint once across a memory-conscious image batch."""
        if not images:
            raise ValueError("At least one preprocessed image is required.")
        started = time.perf_counter()
        tensor = torch.cat([image_to_tensor(image) for image in images], dim=0)
        probability_rows: list[np.ndarray] = []
        model = build_model()
        model.eval()

        with torch.inference_mode():
            for checkpoint_index, checkpoint_path in enumerate(self.checkpoint_paths, start=1):
                state_dictionary = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(state_dictionary, strict=True)
                checkpoint_probabilities: list[np.ndarray] = []
                for start in range(0, len(images), MODEL_BATCH_SIZE):
                    batch = tensor[start : start + MODEL_BATCH_SIZE]
                    probabilities = torch.softmax(model(batch), dim=1)
                    checkpoint_probabilities.append(probabilities.cpu().numpy())
                probability_rows.append(np.concatenate(checkpoint_probabilities, axis=0))
                del state_dictionary
                if progress_callback is not None:
                    progress_callback(checkpoint_index, len(self.checkpoint_paths))

        per_fold = np.stack(probability_rows, axis=0)
        results: list[PredictionResult] = []
        for image_index in range(len(images)):
            image_probabilities = per_fold[:, image_index, :]
            mean_probabilities = image_probabilities.mean(axis=0)
            predicted_index = int(mean_probabilities.argmax())
            fold_indices = image_probabilities.argmax(axis=1)
            sorted_probabilities = np.sort(mean_probabilities)
            results.append(
                PredictionResult(
                    predicted_class=CLASS_NAMES[predicted_index],
                    predicted_probability=float(mean_probabilities[predicted_index]),
                    class_probabilities={
                        name: float(mean_probabilities[index])
                        for index, name in enumerate(CLASS_NAMES)
                    },
                    fold_predictions=tuple(
                        CLASS_NAMES[int(index)] for index in fold_indices
                    ),
                    agreement=float(np.mean(fold_indices == predicted_index)),
                    top_two_margin=float(
                        sorted_probabilities[-1] - sorted_probabilities[-2]
                    ),
                )
            )
        del model, tensor
        gc.collect()
        return results, time.perf_counter() - started


class ModelRuntime:
    """Thread-safe lazy owner of the CPU model ensemble."""

    def __init__(self) -> None:
        self._ensemble: SequentialBroadEnsemble | None = None
        self._lock = threading.Lock()
        self._prediction_lock = threading.Lock()

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
        with self._prediction_lock:
            return self._predict_locked(image_bytes)

    def _predict_locked(self, image_bytes: bytes) -> dict[str, Any]:
        started = time.perf_counter()
        decoded = decode_image(image_bytes)
        preprocessed = preprocess_image(decoded)
        result = self.ensemble().predict_preprocessed_batch((preprocessed,))[0][0]
        assessment = assess_prediction(result)
        feature_analysis = analyze_fingerprint_features(
            decoded,
            include_overlay=True,
        )

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
            "featureAnalysis": feature_analysis,
            "processingSeconds": time.perf_counter() - started,
        }

    def predict_batch(
        self,
        image_items: Sequence[tuple[str, bytes]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        with self._prediction_lock:
            return self._predict_batch_locked(image_items, progress_callback)

    def _predict_batch_locked(
        self,
        image_items: Sequence[tuple[str, bytes]],
        progress_callback: Callable[[int, int], None] | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        preprocessed_images: list[np.ndarray] = []
        feature_analyses: list[dict[str, Any]] = []
        for _, image_bytes in image_items:
            decoded = decode_image(image_bytes)
            preprocessed_images.append(preprocess_image(decoded))
            feature_analyses.append(
                analyze_fingerprint_features(decoded, include_overlay=False)
            )

        broad_results, _ = self.ensemble().predict_preprocessed_batch(
            preprocessed_images,
            progress_callback=progress_callback,
        )
        counts = {"arch": 0, "loop": 0, "whorl": 0}
        pii = 0
        finger_results: list[dict[str, Any]] = []
        review_count = 0
        feature_review_count = 0
        for (finger_id, _), result, features in zip(
            image_items,
            broad_results,
            feature_analyses,
        ):
            assessment = assess_prediction(result)
            family = (
                "loop"
                if result.predicted_class in {"left_slant_loop", "right_slant_loop"}
                else result.predicted_class
            )
            contribution = {"arch": 0, "loop": 1, "whorl": 2}[family]
            counts[family] += 1
            pii += contribution
            review_count += int(assessment.needs_review)
            feature_review_count += int(features["quality"]["needsReview"])
            finger_results.append(
                {
                    "fingerId": finger_id,
                    "prediction": {
                        "predictedClass": result.predicted_class,
                        "score": result.predicted_probability,
                        "probabilities": result.class_probabilities,
                        "foldPredictions": list(result.fold_predictions),
                        "agreement": result.agreement,
                        "topTwoMargin": result.top_two_margin,
                        "needsReview": assessment.needs_review,
                        "reviewReasons": list(assessment.reasons),
                        "subtype": None,
                        "featureAnalysis": features,
                        "processingSeconds": 0.0,
                    },
                }
            )

        return {
            "fingerResults": finger_results,
            "counts": counts,
            "pii": pii,
            "reviewCount": review_count,
            "featureReviewCount": feature_review_count,
            "processingSeconds": time.perf_counter() - started,
            "scope": (
                "Broad-pattern batch inference and model-derived PII. Conditional subtype "
                "inference is intentionally omitted from the ten-finger batch."
            ),
        }


runtime = ModelRuntime()
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Prepare network-backed assets before Render marks the service ready."""
    ensure_checkpoints(_model_directory())
    _subtype_directory()
    yield


app = FastAPI(
    title="Dermatoglyphic Research Inference API",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


def require_api_token(authorization: Optional[str] = Header(default=None)) -> None:
    configured_token = os.environ.get("API_TOKEN")
    if not configured_token:
        return
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not hmac.compare_digest(supplied, configured_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def _read_validated_upload(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image format. Use PNG, JPEG, or TIFF.",
        )
    image_bytes = image.file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The deployment upload limit is 4 MB per image.",
        )
    return image_bytes


def _prune_finished_jobs(now: float) -> None:
    expired = [
        job_id
        for job_id, job in jobs.items()
        if job["status"] in {"collecting", "complete", "failed"}
        and now - job["updatedAt"] > JOB_TTL_SECONDS
    ]
    for job_id in expired:
        del jobs[job_id]


def _run_prediction_job(job_id: str, image_bytes: bytes) -> None:
    with jobs_lock:
        jobs[job_id].update(
            status="running",
            progress=0.1,
            message="Preparing fingerprint analysis",
            updatedAt=time.monotonic(),
        )
    try:
        result = runtime.predict(image_bytes)
    except Exception:
        logger.exception("Background inference job %s failed", job_id)
        with jobs_lock:
            jobs[job_id].update(
                status="failed",
                error="The inference service could not complete this analysis.",
                progress=1.0,
                updatedAt=time.monotonic(),
            )
    else:
        with jobs_lock:
            jobs[job_id].update(
                status="complete",
                result=result,
                progress=1.0,
                message="Analysis complete",
                updatedAt=time.monotonic(),
            )


def _run_batch_prediction_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs[job_id]
        stored_images = job.pop("images")
        image_items = [(finger_id, stored_images[finger_id]) for finger_id in FINGER_IDS]
        job.update(
            status="running",
            progress=0.08,
            message="Preprocessing ten fingerprints",
            updatedAt=time.monotonic(),
        )

    def update_progress(checkpoint: int, total: int) -> None:
        with jobs_lock:
            jobs[job_id].update(
                progress=round(0.15 + 0.8 * checkpoint / total, 3),
                message=f"Running ensemble checkpoint {checkpoint} of {total}",
                updatedAt=time.monotonic(),
            )

    try:
        result = runtime.predict_batch(image_items, update_progress)
    except Exception:
        logger.exception("Background batch inference job %s failed", job_id)
        with jobs_lock:
            jobs[job_id].update(
                status="failed",
                error="The inference service could not complete the ten-finger analysis.",
                progress=1.0,
                updatedAt=time.monotonic(),
            )
    else:
        with jobs_lock:
            jobs[job_id].update(
                status="complete",
                result=result,
                progress=1.0,
                message="Ten-finger analysis complete",
                updatedAt=time.monotonic(),
            )


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
    image_bytes = _read_validated_upload(image)

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


@app.post("/jobs", dependencies=[Depends(require_api_token)], status_code=202)
def create_job(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
) -> JSONResponse:
    image_bytes = _read_validated_upload(image)
    now = time.monotonic()
    with jobs_lock:
        _prune_finished_jobs(now)
        active_jobs = sum(
            job["status"] in {"collecting", "queued", "running"}
            for job in jobs.values()
        )
        if active_jobs >= MAX_ACTIVE_JOBS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="The research inference service is busy. Try again shortly.",
            )
        job_id = secrets.token_urlsafe(18)
        jobs[job_id] = {
            "kind": "single",
            "status": "queued",
            "progress": 0.0,
            "message": "Queued for analysis",
            "updatedAt": now,
        }
    background_tasks.add_task(_run_prediction_job, job_id, image_bytes)
    return JSONResponse(
        {"jobId": job_id, "status": "queued"},
        status_code=status.HTTP_202_ACCEPTED,
        headers={"Cache-Control": "no-store"},
    )


@app.post(
    "/batch-jobs",
    dependencies=[Depends(require_api_token)],
    status_code=201,
)
def create_batch_job() -> JSONResponse:
    now = time.monotonic()
    with jobs_lock:
        _prune_finished_jobs(now)
        active_jobs = sum(
            job["status"] in {"collecting", "queued", "running"}
            for job in jobs.values()
        )
        if active_jobs >= MAX_ACTIVE_JOBS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="The research inference service is busy. Try again shortly.",
            )
        job_id = secrets.token_urlsafe(18)
        jobs[job_id] = {
            "kind": "batch",
            "status": "collecting",
            "uploadedCount": 0,
            "progress": 0.0,
            "message": "Waiting for ten labeled fingerprints",
            "images": {},
            "updatedAt": now,
        }
    return JSONResponse(
        {"jobId": job_id, "status": "collecting", "uploadedCount": 0},
        status_code=status.HTTP_201_CREATED,
        headers={"Cache-Control": "no-store"},
    )


@app.post(
    "/batch-jobs/{job_id}/images/{finger_id}",
    dependencies=[Depends(require_api_token)],
)
def upload_batch_image(
    job_id: str,
    finger_id: str,
    image: UploadFile = File(...),
) -> JSONResponse:
    if finger_id not in FINGER_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown finger position.",
        )
    image_bytes = _read_validated_upload(image)
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None or job.get("kind") != "batch":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ten-finger analysis job not found.",
            )
        if job["status"] != "collecting":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This ten-finger job is no longer accepting images.",
            )
        existing_size = len(job["images"].get(finger_id, b""))
        proposed_total = (
            sum(len(payload) for payload in job["images"].values())
            - existing_size
            + len(image_bytes)
        )
        if proposed_total > MAX_BATCH_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="The ten-finger batch exceeds the 20 MB in-memory safety limit.",
            )
        job["images"][finger_id] = image_bytes
        uploaded_count = len(job["images"])
        job.update(
            uploadedCount=uploaded_count,
            progress=round(uploaded_count / (len(FINGER_IDS) * 10), 3),
            message=f"Uploaded {uploaded_count} of {len(FINGER_IDS)} fingerprints",
            updatedAt=time.monotonic(),
        )
    return JSONResponse(
        {"jobId": job_id, "status": "collecting", "uploadedCount": uploaded_count},
        headers={"Cache-Control": "no-store"},
    )


@app.post(
    "/batch-jobs/{job_id}/start",
    dependencies=[Depends(require_api_token)],
    status_code=202,
)
def start_batch_job(job_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None or job.get("kind") != "batch":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ten-finger analysis job not found.",
            )
        if job["status"] != "collecting":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This ten-finger job has already started.",
            )
        missing = [finger_id for finger_id in FINGER_IDS if finger_id not in job["images"]]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload all ten finger positions before starting analysis.",
            )
        job.update(
            status="queued",
            progress=0.1,
            message="Ten-finger batch queued",
            updatedAt=time.monotonic(),
        )
    background_tasks.add_task(_run_batch_prediction_job, job_id)
    return JSONResponse(
        {"jobId": job_id, "status": "queued"},
        status_code=status.HTTP_202_ACCEPTED,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_token)])
def get_job(job_id: str) -> JSONResponse:
    with jobs_lock:
        _prune_finished_jobs(time.monotonic())
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis job not found.",
            )
        payload = {
            key: value
            for key, value in job.items()
            if key not in {"updatedAt", "images"}
        }
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})
