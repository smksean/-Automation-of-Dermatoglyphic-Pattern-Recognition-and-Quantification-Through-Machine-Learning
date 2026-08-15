"""Load the five-model ensemble and run a private non-holdout smoke test."""

from __future__ import annotations

import csv
import io
from pathlib import Path
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = ROOT / "models" / "efficientnet_320_cv"
PACKAGE_ZIP = ROOT / "data" / "processed" / "roll_efficientnet_320_package.zip"


def private_cv_sample() -> tuple[Path, str]:
    """Return a development image without reading locked-holdout metadata."""
    with zipfile.ZipFile(PACKAGE_ZIP) as archive:
        raw_metadata = archive.read("roll_320_clahe_cv_metadata.csv").decode("utf-8")
    row = next(csv.DictReader(io.StringIO(raw_metadata)))
    return ROOT / Path(row["png_path"]), row["broad_class"]


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from broad_classifier.inference import (
        EXPECTED_CHECKPOINTS,
        BroadPatternEnsemble,
        checkpoint_sha256,
    )

    for checkpoint_name in EXPECTED_CHECKPOINTS:
        path = MODEL_DIRECTORY / checkpoint_name
        print(f"{checkpoint_name}: {checkpoint_sha256(path)}")

    print("Loading five checkpoints on CPU...")
    ensemble = BroadPatternEnsemble.from_directory(MODEL_DIRECTORY)
    sample_path, expected_class = private_cv_sample()
    image_bytes = sample_path.read_bytes()
    first, preprocessed = ensemble.predict_bytes(image_bytes)
    second, repeated_preprocessed = ensemble.predict_bytes(image_bytes)

    if not np.array_equal(preprocessed, repeated_preprocessed):
        raise RuntimeError("Preprocessing was not deterministic.")
    if first != second:
        raise RuntimeError("Repeated model inference was not deterministic.")

    print(f"Private smoke-test expected class: {expected_class}")
    print(f"Private smoke-test predicted class: {first.predicted_class}")
    print(f"Mean top probability: {first.predicted_probability:.4f}")
    print(f"Fold agreement: {first.agreement:.0%}")
    print(f"Top-two margin: {first.top_two_margin:.4f}")
    print("Deterministic repeat: yes")
    print("Locked holdout opened: no")
    print("Broad-classifier model verification passed.")


if __name__ == "__main__":
    main()
