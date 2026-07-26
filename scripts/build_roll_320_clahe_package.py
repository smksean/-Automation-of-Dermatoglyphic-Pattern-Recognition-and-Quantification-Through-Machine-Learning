"""Build the high-resolution roll-fingerprint package used by Colab notebook 07."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "processed" / "roll_broad_model_split.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "efficientnet_320_package"
FIGURE_DIR = ROOT / "data" / "processed" / "figures"
ZIP_PATH = ROOT / "data" / "processed" / "roll_efficientnet_320_package.zip"
IMAGE_SIZE = 320
SEED = 31415


def crop_foreground(image: np.ndarray, margin_fraction: float = 0.06) -> np.ndarray:
    """Crop mostly white margins using robust row/column foreground density."""
    foreground = image < 245
    row_density = foreground.mean(axis=1)
    col_density = foreground.mean(axis=0)
    rows = np.flatnonzero(row_density > 0.01)
    cols = np.flatnonzero(col_density > 0.01)
    if len(rows) == 0 or len(cols) == 0:
        return image
    top, bottom = int(rows[0]), int(rows[-1] + 1)
    left, right = int(cols[0]), int(cols[-1] + 1)
    margin = int(round(max(bottom - top, right - left) * margin_fraction))
    top, bottom = max(0, top - margin), min(image.shape[0], bottom + margin)
    left, right = max(0, left - margin), min(image.shape[1], right + margin)
    return image[top:bottom, left:right]


def resize_with_white_padding(image: np.ndarray, size: int) -> np.ndarray:
    """Preserve aspect ratio and center the print on a white square."""
    height, width = image.shape
    scale = min(size / height, size / width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def preprocess(image_path: Path) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read {image_path}")
    original_shape = image.shape
    cropped = crop_foreground(image)
    crop_shape = cropped.shape
    # CLAHE is applied before downsampling so local ridge contrast is retained.
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(cropped)
    return resize_with_white_padding(enhanced, IMAGE_SIZE), original_shape, crop_shape


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    dtype = {"subject_id": "string", "finger_position": "string", "resolution": "string"}
    metadata = pd.read_csv(SOURCE_CSV, dtype=dtype)
    label_names = sorted(metadata["broad_class"].unique())
    label_to_id = {label: index for index, label in enumerate(label_names)}
    metadata["label_id"] = metadata["broad_class"].map(label_to_id).astype("int64")

    arrays: list[np.ndarray] = []
    original_heights: list[int] = []
    original_widths: list[int] = []
    crop_heights: list[int] = []
    crop_widths: list[int] = []
    for relative_path in tqdm(metadata["png_path"], desc="Crop, enhance, resize"):
        array, original_shape, crop_shape = preprocess(ROOT / Path(relative_path))
        arrays.append(array)
        original_heights.append(original_shape[0])
        original_widths.append(original_shape[1])
        crop_heights.append(crop_shape[0])
        crop_widths.append(crop_shape[1])

    images = np.stack(arrays)
    metadata["original_height"] = original_heights
    metadata["original_width"] = original_widths
    metadata["crop_height"] = crop_heights
    metadata["crop_width"] = crop_widths

    # Reconstruct both previously evaluated subject sets, then lock a never-evaluated holdout.
    old_test_subjects = set(metadata.loc[metadata["split"].eq("test"), "subject_id"])
    old_pool = sorted(set(metadata["subject_id"]) - old_test_subjects)
    old_rng = np.random.default_rng(42)
    old_rng.shuffle(old_pool)
    resnet_test_subjects = set(old_pool[:30])
    previously_evaluated = old_test_subjects | resnet_test_subjects
    clean_subjects = sorted(set(metadata["subject_id"]) - previously_evaluated)
    rng = np.random.default_rng(SEED)
    rng.shuffle(clean_subjects)
    locked_holdout_subjects = set(clean_subjects[:30])
    cv_subjects = set(clean_subjects[30:])

    metadata["experiment_role"] = np.select(
        [
            metadata["subject_id"].isin(cv_subjects),
            metadata["subject_id"].isin(locked_holdout_subjects),
            metadata["subject_id"].isin(previously_evaluated),
        ],
        ["cross_validation", "locked_holdout", "previously_evaluated"],
        default="UNASSIGNED",
    )
    if (metadata["experiment_role"] == "UNASSIGNED").any():
        raise RuntimeError("Some subjects were not assigned an experiment role.")

    cv_mask = metadata["experiment_role"].eq("cross_validation").to_numpy()
    holdout_mask = metadata["experiment_role"].eq("locked_holdout").to_numpy()
    cv_npz_path = OUTPUT_DIR / "roll_320_clahe_cv.npz"
    holdout_npz_path = OUTPUT_DIR / "roll_320_clahe_locked_holdout.npz"
    cv_metadata_path = OUTPUT_DIR / "roll_320_clahe_cv_metadata.csv"
    holdout_metadata_path = OUTPUT_DIR / "roll_320_clahe_locked_holdout_metadata.csv"
    mapping_path = OUTPUT_DIR / "label_mapping.json"
    readme_path = OUTPUT_DIR / "README.txt"
    np.savez_compressed(
        cv_npz_path,
        images=images[cv_mask],
        labels=metadata.loc[cv_mask, "label_id"].to_numpy(np.int64),
        label_names=np.asarray(label_names),
    )
    np.savez_compressed(
        holdout_npz_path,
        images=images[holdout_mask],
        labels=metadata.loc[holdout_mask, "label_id"].to_numpy(np.int64),
        label_names=np.asarray(label_names),
    )
    metadata.loc[cv_mask].to_csv(cv_metadata_path, index=False)
    metadata.loc[holdout_mask].to_csv(holdout_metadata_path, index=False)
    mapping_path.write_text(json.dumps(label_to_id, indent=2), encoding="utf-8")
    readme_path.write_text(
        "320x320 foreground-cropped, CLAHE-enhanced roll fingerprint package.\n"
        "Use only for the agreed private research project. Do not share publicly.\n"
        "Notebook 07 loads only the cross-validation NPZ. The separate locked-holdout\n"
        "NPZ must remain unopened until the full modelling design is frozen.\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (
            cv_npz_path,
            cv_metadata_path,
            holdout_npz_path,
            holdout_metadata_path,
            mapping_path,
            readme_path,
        ):
            archive.write(path, arcname=path.name)

    print("Images:", images.shape, images.dtype)
    print("Subjects by role:", metadata.groupby("experiment_role")["subject_id"].nunique().to_dict())
    print("Rows by role:", metadata["experiment_role"].value_counts().to_dict())
    print("ZIP:", ZIP_PATH, f"{ZIP_PATH.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
