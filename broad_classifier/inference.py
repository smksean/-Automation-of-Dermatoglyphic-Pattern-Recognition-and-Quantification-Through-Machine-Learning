"""Deterministic preprocessing and ensemble inference for broad patterns."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Final, Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch import Tensor, nn
from torchvision.models import efficientnet_b0


CLASS_NAMES: Final[tuple[str, ...]] = (
    "arch",
    "left_slant_loop",
    "right_slant_loop",
    "whorl",
)
DISPLAY_NAMES: Final[dict[str, str]] = {
    "arch": "Arch",
    "left_slant_loop": "Left-slant loop",
    "right_slant_loop": "Right-slant loop",
    "whorl": "Whorl",
}

IMAGE_SIZE: Final[int] = 320
MAX_UPLOAD_BYTES: Final[int] = 20 * 1024 * 1024
MAX_IMAGE_PIXELS: Final[int] = 40_000_000
MIN_IMAGE_SIDE: Final[int] = 64
ALLOWED_FORMATS: Final[frozenset[str]] = frozenset({"PNG", "JPEG", "TIFF"})
IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)
EXPECTED_CHECKPOINTS: Final[tuple[str, ...]] = tuple(
    f"efficientnet_b0_320_fold_{fold}.pt" for fold in range(1, 6)
)


class InputImageError(ValueError):
    """Raised when an uploaded image is unsafe or unsuitable for inference."""


@dataclass(frozen=True)
class PredictionResult:
    """Serializable output from the five-checkpoint ensemble."""

    predicted_class: str
    predicted_probability: float
    class_probabilities: dict[str, float]
    fold_predictions: tuple[str, ...]
    agreement: float
    top_two_margin: float

    @property
    def display_label(self) -> str:
        return DISPLAY_NAMES[self.predicted_class]


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Validate and decode an uploaded PNG, JPEG, or TIFF as grayscale."""
    if not image_bytes:
        raise InputImageError("The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise InputImageError("The uploaded file exceeds the 20 MB limit.")

    try:
        with Image.open(BytesIO(image_bytes)) as probe:
            image_format = (probe.format or "").upper()
            width, height = probe.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InputImageError("The uploaded file is not a readable image.") from exc

    if image_format not in ALLOWED_FORMATS:
        allowed = ", ".join(sorted(ALLOWED_FORMATS))
        raise InputImageError(f"Unsupported image format. Use {allowed}.")
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        raise InputImageError(
            f"The image must be at least {MIN_IMAGE_SIDE} pixels on each side."
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise InputImageError("The decoded image exceeds the 40-megapixel limit.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise InputImageError("The image could not be decoded as grayscale.")
    return image


def crop_foreground(image: np.ndarray, margin_fraction: float = 0.06) -> np.ndarray:
    """Crop white margins using the rule used to construct the training data."""
    foreground = image < 245
    row_density = foreground.mean(axis=1)
    column_density = foreground.mean(axis=0)
    rows = np.flatnonzero(row_density > 0.01)
    columns = np.flatnonzero(column_density > 0.01)
    if len(rows) == 0 or len(columns) == 0:
        return image

    top, bottom = int(rows[0]), int(rows[-1] + 1)
    left, right = int(columns[0]), int(columns[-1] + 1)
    margin = int(round(max(bottom - top, right - left) * margin_fraction))
    top = max(0, top - margin)
    bottom = min(image.shape[0], bottom + margin)
    left = max(0, left - margin)
    right = min(image.shape[1], right + margin)
    return image[top:bottom, left:right]


def resize_with_white_padding(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Preserve aspect ratio and center an image on a white square."""
    height, width = image.shape
    scale = min(size / height, size / width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((size, size), 255, dtype=np.uint8)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Apply the frozen crop, CLAHE, and resize pipeline."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected a two-dimensional uint8 grayscale image.")
    cropped = crop_foreground(image)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(cropped)
    return resize_with_white_padding(enhanced)


def image_to_tensor(image: np.ndarray) -> Tensor:
    """Convert a preprocessed grayscale image to an ImageNet-normalized tensor."""
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE) or image.dtype != np.uint8:
        raise ValueError(f"Expected a {IMAGE_SIZE} x {IMAGE_SIZE} uint8 image.")
    tensor = torch.from_numpy(image.copy()).to(torch.float32).div_(255.0)
    tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    standard_deviation = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return tensor.sub_(mean).div_(standard_deviation).unsqueeze(0)


def build_model() -> nn.Module:
    """Construct the exact four-output EfficientNet-B0 architecture."""
    model = efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(input_features, 4))
    return model


class BroadPatternEnsemble:
    """Five-fold EfficientNet ensemble used by the broad-classification app."""

    def __init__(self, models: Sequence[nn.Module], device: torch.device) -> None:
        if len(models) != 5:
            raise ValueError("The deployment ensemble requires exactly five models.")
        self.models = tuple(models)
        self.device = device

    @classmethod
    def from_directory(
        cls,
        model_directory: Path,
        device_name: str = "cpu",
    ) -> "BroadPatternEnsemble":
        model_directory = model_directory.resolve()
        expected_paths = [model_directory / name for name in EXPECTED_CHECKPOINTS]
        missing = [path.name for path in expected_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing deployment checkpoints: " + ", ".join(missing)
            )

        device = torch.device(device_name)
        models: list[nn.Module] = []
        for checkpoint_path in expected_paths:
            state_dictionary = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
            model = build_model()
            model.load_state_dict(state_dictionary, strict=True)
            model.to(device)
            model.eval()
            models.append(model)
        return cls(models, device)

    def predict_preprocessed(self, image: np.ndarray) -> PredictionResult:
        """Predict from an already-preprocessed 320 x 320 grayscale image."""
        tensor = image_to_tensor(image).to(self.device)
        probability_rows: list[np.ndarray] = []
        with torch.inference_mode():
            for model in self.models:
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
                probability_rows.append(probabilities.cpu().numpy())

        per_fold = np.stack(probability_rows)
        mean_probabilities = per_fold.mean(axis=0)
        predicted_index = int(mean_probabilities.argmax())
        fold_indices = per_fold.argmax(axis=1)
        sorted_probabilities = np.sort(mean_probabilities)
        return PredictionResult(
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

    def predict_bytes(self, image_bytes: bytes) -> tuple[PredictionResult, np.ndarray]:
        """Validate, preprocess, and predict an uploaded image entirely in memory."""
        decoded = decode_image(image_bytes)
        preprocessed = preprocess_image(decoded)
        return self.predict_preprocessed(preprocessed), preprocessed


def checkpoint_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a checkpoint without loading it."""
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
