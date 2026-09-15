"""Train preliminary subtype-classification baselines.

This is a first-pass research baseline from the completed Supabase subtype
review export. It intentionally keeps row-level outputs under ``data/`` and
writes only aggregate, non-biometric metrics under ``results/``.
"""

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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = (
    ROOT / "data" / "processed" / "subtype_review_completed_2026-09-06"
)
DEFAULT_RESULTS_DIR = ROOT / "results" / "subtype_preliminary_baseline"
PRIVATE_OUTPUT_SUBDIR = "preliminary_subtype_baseline"
SEED = 244664

EXPERIMENTS = {
    "arch_binary": {
        "table": "arch_subtype_modeling_dataset.csv",
        "classes": ("plain_arch", "tented_arch"),
        "test_size": 0.25,
    },
    "whorl_first_pass": {
        "table": "whorl_subtype_modeling_dataset.csv",
        "classes": (
            "plain_whorl",
            "central_pocket_loop_whorl",
            "double_loop_whorl",
        ),
        "test_size": 0.25,
    },
}


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = ImageOps.autocontrast(image)
    return np.asarray(image, dtype=np.float32) / 255.0


def block_means(image: np.ndarray, blocks: int = 16) -> np.ndarray:
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
    trimmed_mag = magnitude[: height - height % cells, : width - width % cells]
    trimmed_ori = orientation[: height - height % cells, : width - width % cells]
    cell_h = trimmed_mag.shape[0] // cells
    cell_w = trimmed_mag.shape[1] // cells
    features: list[np.ndarray] = []
    for row in range(cells):
        for column in range(cells):
            mag_cell = trimmed_mag[
                row * cell_h : (row + 1) * cell_h,
                column * cell_w : (column + 1) * cell_w,
            ].ravel()
            ori_cell = trimmed_ori[
                row * cell_h : (row + 1) * cell_h,
                column * cell_w : (column + 1) * cell_w,
            ].ravel()
            hist, _ = np.histogram(
                ori_cell, bins=bins, range=(0.0, 180.0), weights=mag_cell
            )
            norm = np.linalg.norm(hist)
            features.append(hist / norm if norm else hist)
    return np.concatenate(features).astype(np.float32)


def extract_features(image_path: Path) -> np.ndarray:
    image = load_image(image_path)
    small = np.asarray(
        Image.fromarray((image * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    center = image[32:288, 32:288]
    features = [
        small.ravel(),
        block_means(image, blocks=16),
        block_means(center, blocks=8),
        gradient_histogram(image, cells=8, bins=9),
    ]
    return np.concatenate(features).astype(np.float32)


def class_counts(labels: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(labels).items()))


def report_rows(report: dict[str, Any], experiment: str, model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, values in report.items():
        if not isinstance(values, dict):
            continue
        row = {
            "experiment": experiment,
            "model": model_name,
            "label": label,
            "precision": values.get("precision", ""),
            "recall": values.get("recall", ""),
            "f1_score": values.get("f1-score", ""),
            "support": values.get("support", ""),
        }
        rows.append(row)
    return rows


def split_rows(rows: list[dict[str, str]], test_size: float) -> tuple[list[int], list[int]]:
    labels = [row["confirmed_subtype"] for row in rows]
    indices = list(range(len(rows)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=SEED,
        stratify=labels,
    )
    return sorted(train_idx), sorted(test_idx)


def evaluate_model(
    model: Any,
    model_name: str,
    experiment: str,
    classes: tuple[str, ...],
    x_train: np.ndarray,
    y_train: list[str],
    x_test: np.ndarray,
    y_test: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    model.fit(x_train, y_train)
    predictions = list(model.predict(x_test))
    metrics = {
        "experiment": experiment,
        "model": model_name,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
        "train_class_counts": class_counts(y_train),
        "test_class_counts": class_counts(y_test),
    }
    report = classification_report(
        y_test,
        predictions,
        labels=list(classes),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=list(classes)).tolist()
    return metrics, report_rows(report, experiment, model_name), matrix


def run_experiment(
    name: str,
    config: dict[str, Any],
    export_dir: Path,
    results_dir: Path,
    private_dir: Path,
) -> dict[str, Any]:
    rows = [
        row
        for row in read_csv(export_dir / config["table"])
        if row["confirmed_subtype"] in config["classes"]
    ]
    if not rows:
        raise ValueError(f"No rows found for {name}")

    train_idx, test_idx = split_rows(rows, float(config["test_size"]))
    feature_rows = [
        extract_features(export_dir / row["model_image_path"])
        for row in rows
    ]
    features = np.vstack(feature_rows)
    labels = [row["confirmed_subtype"] for row in rows]
    x_train = features[train_idx]
    x_test = features[test_idx]
    y_train = [labels[index] for index in train_idx]
    y_test = [labels[index] for index in test_idx]

    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "linear_svc_balanced": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=SEED,
                        dual="auto",
                        max_iter=20000,
                    ),
                ),
            ]
        ),
    }

    aggregate_rows: list[dict[str, Any]] = []
    report_table_rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, Any] = {
        "labels": list(config["classes"]),
        "matrices": {},
    }
    prediction_rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        metrics, per_class_rows, matrix = evaluate_model(
            model,
            model_name,
            name,
            tuple(config["classes"]),
            x_train,
            y_train,
            x_test,
            y_test,
        )
        aggregate_rows.append(metrics)
        report_table_rows.extend(per_class_rows)
        confusion_payload["matrices"][model_name] = matrix

        predictions = list(model.predict(x_test))
        for row_index, prediction in zip(test_idx, predictions):
            source = rows[row_index]
            prediction_rows.append(
                {
                    "experiment": name,
                    "model": model_name,
                    "review_id": source["review_id"],
                    "record_key": source["record_key"],
                    "true_subtype": source["confirmed_subtype"],
                    "predicted_subtype": prediction,
                    "correct": prediction == source["confirmed_subtype"],
                    "confidence": source["confidence"],
                    "model_image_path": source["model_image_path"],
                }
            )

    write_csv(
        results_dir / f"{name}_metrics.csv",
        aggregate_rows,
        [
            "experiment",
            "model",
            "train_rows",
            "test_rows",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "train_class_counts",
            "test_class_counts",
        ],
    )
    write_csv(
        results_dir / f"{name}_classification_report.csv",
        report_table_rows,
        ["experiment", "model", "label", "precision", "recall", "f1_score", "support"],
    )
    write_json(results_dir / f"{name}_confusion_matrices.json", confusion_payload)
    write_csv(
        private_dir / f"{name}_test_predictions.csv",
        prediction_rows,
        [
            "experiment",
            "model",
            "review_id",
            "record_key",
            "true_subtype",
            "predicted_subtype",
            "correct",
            "confidence",
            "model_image_path",
        ],
    )
    return {
        "experiment": name,
        "rows": len(rows),
        "classes": list(config["classes"]),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "class_counts": class_counts(labels),
        "metrics": aggregate_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    results_dir = args.results_dir.resolve()
    private_dir = export_dir / PRIVATE_OUTPUT_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name, config in EXPERIMENTS.items():
        print(f"Running {name}...")
        summaries.append(run_experiment(name, config, export_dir, results_dir, private_dir))

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source_export_dir": export_dir.relative_to(ROOT).as_posix(),
        "results_dir": results_dir.relative_to(ROOT).as_posix(),
        "private_prediction_dir": private_dir.relative_to(ROOT).as_posix(),
        "evaluation_protocol": (
            "non-holdout expert-reviewed subtype prototype; stratified train/test "
            "split by accepted subtype label; not grouped by subject because "
            "subject_id is not present in the Supabase export"
        ),
        "feature_method": (
            "64x64 autocontrast grayscale pixels, block-mean intensity summaries, "
            "and 8x8 cell gradient-orientation histograms from 320x320 cropped inputs"
        ),
        "experiments": summaries,
        "research_limitations": [
            "Use repeated stratified evaluation for the most stable current prototype metrics.",
            "If private subject metadata is later restored, rerun grouped evaluation.",
            "Keep row-level predictions and images under data/ because they are biometric research records.",
            "The whorl task is heavily imbalanced toward plain_whorl.",
        ],
    }
    write_json(results_dir / "subtype_preliminary_baseline_summary.json", summary)

    combined_metrics: list[dict[str, Any]] = []
    for experiment in summaries:
        combined_metrics.extend(experiment["metrics"])
    write_csv(
        results_dir / "subtype_preliminary_baseline_metrics.csv",
        combined_metrics,
        [
            "experiment",
            "model",
            "train_rows",
            "test_rows",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "train_class_counts",
            "test_class_counts",
        ],
    )

    print("Preliminary subtype baselines complete.")
    for row in combined_metrics:
        print(
            f"{row['experiment']} / {row['model']}: "
            f"accuracy={row['accuracy']:.3f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.3f}, "
            f"macro_f1={row['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
