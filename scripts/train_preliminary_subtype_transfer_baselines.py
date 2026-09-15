"""Train preliminary subtype baselines using frozen ResNet-18 embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
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
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = (
    ROOT / "data" / "processed" / "subtype_review_completed_2026-09-06"
)
DEFAULT_RESULTS_DIR = ROOT / "results" / "subtype_preliminary_transfer_baseline"
PRIVATE_OUTPUT_SUBDIR = "preliminary_subtype_transfer_baseline"
SEED = 244664
BATCH_SIZE = 16

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


def class_counts(labels: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(labels).items()))


def load_transfer_model() -> tuple[torch.nn.Module, Any]:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms()


def extract_embeddings(
    rows: list[dict[str, str]], export_dir: Path, cache_path: Path
) -> np.ndarray:
    if cache_path.is_file():
        return np.load(cache_path)["embeddings"]

    model, preprocess = load_transfer_model()
    embeddings: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            image = Image.open(export_dir / row["model_image_path"]).convert("RGB")
            batch.append(preprocess(image))
            if len(batch) == BATCH_SIZE or index == len(rows):
                tensor = torch.stack(batch)
                features = model(tensor).cpu().numpy().astype(np.float32)
                embeddings.append(features)
                batch.clear()
                print(f"Embedded images: {index}/{len(rows)}")

    all_embeddings = np.vstack(embeddings)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=all_embeddings)
    return all_embeddings


def report_rows(report: dict[str, Any], experiment: str, model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, values in report.items():
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "experiment": experiment,
                "model": model_name,
                "label": label,
                "precision": values.get("precision", ""),
                "recall": values.get("recall", ""),
                "f1_score": values.get("f1-score", ""),
                "support": values.get("support", ""),
            }
        )
    return rows


def evaluate(
    model: Any,
    model_name: str,
    experiment: str,
    classes: tuple[str, ...],
    x_train: np.ndarray,
    y_train: list[str],
    x_test: np.ndarray,
    y_test: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]], list[str]]:
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
    return metrics, report_rows(report, experiment, model_name), matrix, predictions


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
    labels = [row["confirmed_subtype"] for row in rows]
    train_idx, test_idx = train_test_split(
        list(range(len(rows))),
        test_size=float(config["test_size"]),
        random_state=SEED,
        stratify=labels,
    )
    train_idx = sorted(train_idx)
    test_idx = sorted(test_idx)

    cache_path = private_dir / f"{name}_resnet18_embeddings.npz"
    embeddings = extract_embeddings(rows, export_dir, cache_path)
    x_train = embeddings[train_idx]
    x_test = embeddings[test_idx]
    y_train = [labels[index] for index in train_idx]
    y_test = [labels[index] for index in test_idx]

    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "resnet18_linear_svc_balanced": Pipeline(
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
        "resnet18_logistic_balanced": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=SEED,
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
        metrics, per_class_rows, matrix, predictions = evaluate(
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

    combined_metrics: list[dict[str, Any]] = []
    for experiment in summaries:
        combined_metrics.extend(experiment["metrics"])

    write_csv(
        results_dir / "subtype_preliminary_transfer_metrics.csv",
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
    write_json(
        results_dir / "subtype_preliminary_transfer_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "source_export_dir": export_dir.relative_to(ROOT).as_posix(),
            "results_dir": results_dir.relative_to(ROOT).as_posix(),
            "private_prediction_dir": private_dir.relative_to(ROOT).as_posix(),
            "feature_method": "Frozen ImageNet ResNet-18 embeddings from 320x320 fingerprint-only crops.",
            "evaluation_protocol": (
                "non-holdout expert-reviewed subtype prototype; stratified train/test "
                "split by accepted subtype label; not grouped by subject because "
                "subject_id is not present in the Supabase export"
            ),
            "experiments": summaries,
            "research_limitations": [
                "Use repeated stratified evaluation for the most stable current prototype metrics.",
                "If private subject metadata is later restored, rerun grouped evaluation.",
                "The whorl task remains strongly imbalanced toward plain_whorl.",
                "Frozen ImageNet features are a baseline, not a tuned fingerprint-specific model.",
            ],
        },
    )

    print("Preliminary transfer subtype baselines complete.")
    for row in combined_metrics:
        print(
            f"{row['experiment']} / {row['model']}: "
            f"accuracy={row['accuracy']:.3f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.3f}, "
            f"macro_f1={row['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
