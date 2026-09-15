"""Broad fingerprint-pattern classification inference package."""

try:
    from .inference import (
        CLASS_NAMES,
        DISPLAY_NAMES,
        BroadPatternEnsemble,
        InputImageError,
        PredictionResult,
        decode_image,
        preprocess_image,
    )
except ModuleNotFoundError as exc:
    if exc.name != "cv2":
        raise
    CLASS_NAMES = ()
    DISPLAY_NAMES = {}
    BroadPatternEnsemble = None
    InputImageError = RuntimeError
    PredictionResult = None
    decode_image = None
    preprocess_image = None

__all__ = [
    "CLASS_NAMES",
    "DISPLAY_NAMES",
    "BroadPatternEnsemble",
    "InputImageError",
    "PredictionResult",
    "decode_image",
    "preprocess_image",
]
