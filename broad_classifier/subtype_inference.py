"""Subtype prototype inference helpers.

These helpers are optional and depend on private local artifacts produced from
the completed expert subtype review. They are not part of the public broad
classifier deployment payload.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "subtype_review_completed_2026-09-06"
    / "selected_subtype_model_artifacts"
)

TASK_BY_BROAD_CLASS = {
    "arch": "arch",
    "whorl": "whorl",
}

DISPLAY_NAMES = {
    "plain_arch": "Plain arch",
    "tented_arch": "Tented arch",
    "plain_whorl": "Plain whorl",
    "central_pocket_loop_whorl": "Central pocket loop whorl",
    "double_loop_whorl": "Double loop whorl",
}


class SubtypeInferenceUnavailable(RuntimeError):
    """Raised when optional subtype artifacts are not available."""


@lru_cache(maxsize=1)
def _feature_extractor() -> tuple[torch.nn.Module, Any]:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms()


@lru_cache(maxsize=4)
def _load_classifier(task: str, artifact_dir: str) -> Any:
    path = Path(artifact_dir) / f"{task}_subtype_classifier.pkl"
    if not path.is_file():
        raise SubtypeInferenceUnavailable(f"Subtype classifier not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def subtype_available(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    broad_class: str | None = None,
) -> bool:
    """Return whether the requested local subtype artifact is available."""
    if broad_class is not None:
        task = TASK_BY_BROAD_CLASS.get(broad_class)
        return bool(
            task and (artifact_dir / f"{task}_subtype_classifier.pkl").is_file()
        )
    return all(
        (artifact_dir / f"{task}_subtype_classifier.pkl").is_file()
        for task in ("arch", "whorl")
    )


def _to_rgb_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def extract_resnet18_embedding(image: Image.Image | np.ndarray) -> np.ndarray:
    """Create the frozen ImageNet ResNet-18 feature vector used in training."""
    model, preprocess = _feature_extractor()
    pil_image = _to_rgb_image(image)
    tensor = preprocess(pil_image).unsqueeze(0)
    with torch.inference_mode():
        embedding = model(tensor).cpu().numpy()
    return embedding.astype(np.float32)


def predict_subtype(
    broad_class: str,
    image: Image.Image | np.ndarray,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    """Predict an arch or whorl subtype from a prepared fingerprint image."""
    task = TASK_BY_BROAD_CLASS.get(broad_class)
    if task is None:
        raise ValueError("Subtype prediction is available only for arch and whorl.")

    classifier = _load_classifier(task, str(artifact_dir))
    embedding = extract_resnet18_embedding(image)
    labels = [str(label) for label in classifier.classes_]
    probabilities = classifier.predict_proba(embedding)[0]
    best_index = int(np.argmax(probabilities))
    predicted_label = labels[best_index]

    return {
        "task": task,
        "predicted_subtype": predicted_label,
        "display_subtype": DISPLAY_NAMES.get(predicted_label, predicted_label),
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(labels, probabilities)
        },
        "research_scope": "expert-reviewed subtype prototype with subject-grouped development validation",
    }
