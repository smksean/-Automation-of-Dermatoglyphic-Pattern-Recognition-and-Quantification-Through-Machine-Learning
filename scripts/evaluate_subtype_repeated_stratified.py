"""Repeated stratified evaluation for the non-holdout subtype prototype."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "processed" / "subtype_review_completed_2026-09-06"
RESULTS_DIR = ROOT / "results" / "subtype_repeated_stratified"
PRIVATE_DIR = EXPORT_DIR / "repeated_stratified_evaluation"
SEED = 244664
N_SPLITS = 5
N_REPEATS = 10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def class_counts(labels: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(labels).items()))


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = ImageOps.autocontrast(image)
    return np.asarray(image, dtype=np.float32) / 255.0


def block_means(image: np.ndarray, blocks: int) -> np.ndarray:
    height, width = image.shape
    trimmed = image[: height - height % blocks, : width - width % blocks]
    block_h = trimmed.shape[0] // blocks
    block_w = trimmed.shape[1] // blocks
    return trimmed.reshape(blocks, block_h, blocks, block_w).mean(axis=(1, 3)).ravel()


def gradient_histogram(image: np.ndarray, cells: int = 8, bins: int = 9) -> np.ndarray:
    gy, gx = np.gradient(image)
    magnitude = np.hypot(gx, gy)
    orientation = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    height, width = image.shape
    mag = magnitude[: height - height % cells, : width - width % cells]
    ori = orientation[: height - height % cells, : width - width % cells]
    cell_h = mag.shape[0] // cells
    cell_w = mag.shape[1] // cells
    features = []
    for row in range(cells):
        for column in range(cells):
            mag_cell = mag[row * cell_h : (row + 1) * cell_h, column * cell_w : (column + 1) * cell_w].ravel()
            ori_cell = ori[row * cell_h : (row + 1) * cell_h, column * cell_w : (column + 1) * cell_w].ravel()
            hist, _ = np.histogram(ori_cell, bins=bins, range=(0.0, 180.0), weights=mag_cell)
            norm = np.linalg.norm(hist)
            features.append(hist / norm if norm else hist)
    return np.concatenate(features).astype(np.float32)


def classical_features(rows: list[dict[str, str]]) -> np.ndarray:
    feature_rows = []
    for row in rows:
        image = load_image(EXPORT_DIR / row["model_image_path"])
        small = np.asarray(
            Image.fromarray((image * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        center = image[32:288, 32:288]
        feature_rows.append(
            np.concatenate(
                [
                    small.ravel(),
                    block_means(image, 16),
                    block_means(center, 8),
                    gradient_histogram(image),
                ]
            ).astype(np.float32)
        )
    return np.vstack(feature_rows)


def load_transfer_embeddings(name: str, rows: list[dict[str, str]]) -> np.ndarray:
    cache_path = EXPORT_DIR / "preliminary_subtype_transfer_baseline" / f"{name}_resnet18_embeddings.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Run scripts/train_preliminary_subtype_transfer_baselines.py first: {cache_path}"
        )
    embeddings = np.load(cache_path)["embeddings"]
    if len(embeddings) != len(rows):
        raise ValueError(f"Embedding row count mismatch for {name}")
    return embeddings


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def evaluate_repeated(
    experiment: str,
    rows: list[dict[str, str]],
    features: np.ndarray,
    labels: list[str],
    model: Any,
    classes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )
    fold_rows = []
    report_rows = []
    prediction_rows = []
    labels_array = np.asarray(labels)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, labels_array), 1):
        model.fit(features[train_idx], labels_array[train_idx])
        pred = model.predict(features[test_idx])
        truth = labels_array[test_idx]
        fold_rows.append(
            {
                "experiment": experiment,
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "accuracy": accuracy_score(truth, pred),
                "balanced_accuracy": balanced_accuracy_score(truth, pred),
                "macro_f1": f1_score(truth, pred, average="macro"),
                "weighted_f1": f1_score(truth, pred, average="weighted"),
                "train_class_counts": class_counts(labels_array[train_idx].tolist()),
                "test_class_counts": class_counts(truth.tolist()),
            }
        )
        report = classification_report(
            truth, pred, labels=classes, output_dict=True, zero_division=0
        )
        for label in classes:
            report_rows.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "label": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1_score": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
            )
        for row_index, prediction in zip(test_idx, pred):
            source = rows[int(row_index)]
            prediction_rows.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "review_id": source["review_id"],
                    "record_key": source["record_key"],
                    "true_subtype": source["confirmed_subtype"],
                    "predicted_subtype": str(prediction),
                    "correct": str(prediction) == source["confirmed_subtype"],
                    "confidence": source["confidence"],
                    "model_image_path": source["model_image_path"],
                }
            )
    return fold_rows, report_rows, prediction_rows


def aggregate_folds(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    experiments = sorted({row["experiment"] for row in fold_rows})
    metrics = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    rows = []
    for experiment in experiments:
        matches = [row for row in fold_rows if row["experiment"] == experiment]
        output = {
            "experiment": experiment,
            "folds": len(matches),
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
        }
        for metric in metrics:
            summary = summarize([float(row[metric]) for row in matches])
            for key, value in summary.items():
                output[f"{metric}_{key}"] = value
        rows.append(output)
    return rows


def aggregate_reports(report_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["experiment"], row["label"]) for row in report_rows})
    output = []
    for experiment, label in keys:
        matches = [row for row in report_rows if row["experiment"] == experiment and row["label"] == label]
        row = {"experiment": experiment, "label": label, "folds": len(matches)}
        for metric in ("precision", "recall", "f1_score"):
            summary = summarize([float(item[metric]) for item in matches])
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_std"] = summary["std"]
        row["support_total"] = sum(float(item["support"]) for item in matches)
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    private_dir = PRIVATE_DIR.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    arch_rows = read_csv(EXPORT_DIR / "arch_subtype_modeling_dataset.csv")
    whorl_rows = read_csv(EXPORT_DIR / "whorl_subtype_modeling_dataset.csv")
    whorl_rows = [
        row for row in whorl_rows if row["confirmed_subtype"] != "accidental_whorl"
    ]

    experiments = [
        (
            "arch_classical_linear_svc_balanced",
            arch_rows,
            classical_features(arch_rows),
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("classifier", LinearSVC(class_weight="balanced", dual="auto", max_iter=20000, random_state=SEED)),
                ]
            ),
            ["plain_arch", "tented_arch"],
        ),
        (
            "whorl_resnet18_logistic_balanced",
            whorl_rows,
            load_transfer_embeddings("whorl_first_pass", whorl_rows),
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("classifier", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)),
                ]
            ),
            ["plain_whorl", "central_pocket_loop_whorl", "double_loop_whorl"],
        ),
    ]

    all_folds = []
    all_reports = []
    all_predictions = []
    for experiment, rows, features, model, classes in experiments:
        labels = [row["confirmed_subtype"] for row in rows]
        fold_rows, report_rows, prediction_rows = evaluate_repeated(
            experiment, rows, features, labels, model, classes
        )
        all_folds.extend(fold_rows)
        all_reports.extend(report_rows)
        all_predictions.extend(prediction_rows)

    fold_columns = [
        "experiment",
        "fold",
        "train_rows",
        "test_rows",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "train_class_counts",
        "test_class_counts",
    ]
    write_csv(results_dir / "repeated_stratified_fold_metrics.csv", all_folds, fold_columns)
    aggregate_rows = aggregate_folds(all_folds)
    write_csv(results_dir / "repeated_stratified_summary.csv", aggregate_rows, list(aggregate_rows[0].keys()))
    class_rows = aggregate_reports(all_reports)
    write_csv(results_dir / "repeated_stratified_per_class_summary.csv", class_rows, list(class_rows[0].keys()))
    write_csv(
        private_dir / "repeated_stratified_test_predictions.csv",
        all_predictions,
        [
            "experiment",
            "fold",
            "review_id",
            "record_key",
            "true_subtype",
            "predicted_subtype",
            "correct",
            "confidence",
            "model_image_path",
        ],
    )
    write_json(
        results_dir / "repeated_stratified_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "evaluation_scope": "non-holdout expert-reviewed subtype prototype; stratified, not subject-grouped",
            "source_provenance": "review package generated from non-locked-holdout records; locked-holdout generic records excluded before Supabase upload",
            "private_predictions": (
                private_dir / "repeated_stratified_test_predictions.csv"
            ).relative_to(ROOT).as_posix(),
        },
    )

    print("Repeated stratified subtype evaluation complete.")
    for row in aggregate_rows:
        print(
            f"{row['experiment']}: macro_f1={row['macro_f1_mean']:.3f} "
            f"+/- {row['macro_f1_std']:.3f}; balanced_accuracy={row['balanced_accuracy_mean']:.3f} "
            f"+/- {row['balanced_accuracy_std']:.3f}"
        )


if __name__ == "__main__":
    main()
