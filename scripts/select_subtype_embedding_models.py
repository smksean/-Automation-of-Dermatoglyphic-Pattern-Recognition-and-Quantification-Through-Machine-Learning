"""Select subtype classifiers using cached ResNet-18 embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "processed" / "subtype_review_completed_2026-09-06"
RESULTS_DIR = ROOT / "results" / "subtype_embedding_model_selection"
PRIVATE_DIR = EXPORT_DIR / "subtype_embedding_model_selection"
SEED = 244664
N_SPLITS = 5
N_REPEATS = 5

TASKS = {
    "arch": {
        "table": "arch_subtype_modeling_dataset.csv",
        "embedding_cache": "preliminary_subtype_transfer_baseline/arch_binary_resnet18_embeddings.npz",
        "classes": ["plain_arch", "tented_arch"],
    },
    "whorl": {
        "table": "whorl_subtype_modeling_dataset.csv",
        "embedding_cache": "preliminary_subtype_transfer_baseline/whorl_first_pass_resnet18_embeddings.npz",
        "classes": [
            "plain_whorl",
            "central_pocket_loop_whorl",
            "double_loop_whorl",
        ],
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_task(task_name: str, config: dict[str, Any]) -> tuple[list[dict[str, str]], np.ndarray, list[str]]:
    rows = [
        row
        for row in read_csv(EXPORT_DIR / config["table"])
        if row["confirmed_subtype"] in config["classes"]
    ]
    embeddings = np.load(EXPORT_DIR / config["embedding_cache"])["embeddings"]
    if len(embeddings) != len(rows):
        raise ValueError(f"Embedding count mismatch for {task_name}")
    labels = [row["confirmed_subtype"] for row in rows]
    return rows, embeddings, labels


def candidate_models(task_name: str) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_C0.1": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=SEED)),
            ]
        ),
        "logistic_C1": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED)),
            ]
        ),
        "linear_svc_C0.1": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LinearSVC(C=0.1, class_weight="balanced", dual="auto", max_iter=20000, random_state=SEED)),
            ]
        ),
        "linear_svc_C1": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=20000, random_state=SEED)),
            ]
        ),
        "rbf_svc_C1": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=SEED)),
            ]
        ),
        "rbf_svc_C10": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(C=10.0, gamma="scale", class_weight="balanced", random_state=SEED)),
            ]
        ),
    }
    if task_name == "arch":
        candidates["mlp_small"] = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, max_iter=1000, random_state=SEED)),
            ]
        )
    return candidates


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def class_counts(labels: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(labels).items()))


def evaluate_candidate(
    task_name: str,
    model_name: str,
    model: Any,
    rows: list[dict[str, str]],
    embeddings: np.ndarray,
    labels: list[str],
    classes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )
    labels_array = np.asarray(labels)
    fold_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(embeddings, labels_array), 1):
        model.fit(embeddings[train_idx], labels_array[train_idx])
        pred = model.predict(embeddings[test_idx])
        truth = labels_array[test_idx]
        fold_rows.append(
            {
                "task": task_name,
                "model": model_name,
                "fold": fold,
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
                    "task": task_name,
                    "model": model_name,
                    "fold": fold,
                    "label": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1_score": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
            )
        for row_index, prediction in zip(test_idx, pred):
            row = rows[int(row_index)]
            prediction_rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "fold": fold,
                    "review_id": row["review_id"],
                    "record_key": row["record_key"],
                    "true_subtype": row["confirmed_subtype"],
                    "predicted_subtype": str(prediction),
                    "correct": str(prediction) == row["confirmed_subtype"],
                    "model_image_path": row["model_image_path"],
                }
            )
    return fold_rows, report_rows, prediction_rows


def aggregate_folds(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["task"], row["model"]) for row in fold_rows})
    output = []
    for task, model in keys:
        rows = [row for row in fold_rows if row["task"] == task and row["model"] == model]
        record = {"task": task, "model": model, "folds": len(rows)}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
            stats = summarize([float(row[metric]) for row in rows])
            for key, value in stats.items():
                record[f"{metric}_{key}"] = value
        output.append(record)
    return output


def aggregate_reports(report_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["task"], row["model"], row["label"]) for row in report_rows})
    output = []
    for task, model, label in keys:
        rows = [
            row
            for row in report_rows
            if row["task"] == task and row["model"] == model and row["label"] == label
        ]
        record = {"task": task, "model": model, "label": label, "folds": len(rows)}
        for metric in ("precision", "recall", "f1_score"):
            stats = summarize([float(row[metric]) for row in rows])
            record[f"{metric}_mean"] = stats["mean"]
            record[f"{metric}_std"] = stats["std"]
        record["support_total"] = sum(float(row["support"]) for row in rows)
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), help="Run one task only.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    selected = {args.task: TASKS[args.task]} if args.task else TASKS

    all_folds: list[dict[str, Any]] = []
    all_reports: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for task_name, config in selected.items():
        rows, embeddings, labels = load_task(task_name, config)
        for model_name, model in candidate_models(task_name).items():
            print(f"Evaluating {task_name} / {model_name}")
            fold_rows, report_rows, prediction_rows = evaluate_candidate(
                task_name,
                model_name,
                model,
                rows,
                embeddings,
                labels,
                list(config["classes"]),
            )
            all_folds.extend(fold_rows)
            all_reports.extend(report_rows)
            all_predictions.extend(prediction_rows)

    write_csv(
        RESULTS_DIR / "embedding_model_selection_fold_metrics.csv",
        all_folds,
        [
            "task",
            "model",
            "fold",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "train_class_counts",
            "test_class_counts",
        ],
    )
    summary_rows = aggregate_folds(all_folds)
    write_csv(
        RESULTS_DIR / "embedding_model_selection_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )
    per_class_rows = aggregate_reports(all_reports)
    write_csv(
        RESULTS_DIR / "embedding_model_selection_per_class.csv",
        per_class_rows,
        list(per_class_rows[0].keys()),
    )
    write_csv(
        PRIVATE_DIR / "embedding_model_selection_predictions.csv",
        all_predictions,
        [
            "task",
            "model",
            "fold",
            "review_id",
            "record_key",
            "true_subtype",
            "predicted_subtype",
            "correct",
            "model_image_path",
        ],
    )
    write_json(
        RESULTS_DIR / "embedding_model_selection_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "scope": "non-holdout expert-reviewed subtype prototype; repeated stratified evaluation",
        },
    )
    print("Model selection complete.")
    for row in sorted(summary_rows, key=lambda item: (item["task"], -item["macro_f1_mean"])):
        print(
            f"{row['task']} / {row['model']}: "
            f"macro_f1={row['macro_f1_mean']:.3f} +/- {row['macro_f1_std']:.3f}; "
            f"balanced_accuracy={row['balanced_accuracy_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
