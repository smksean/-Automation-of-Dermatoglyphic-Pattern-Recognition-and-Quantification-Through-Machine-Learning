"""Select subtype classifiers with repeated subject-grouped validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "processed" / "subtype_review_completed_2026-09-06"
RESTORE_DIR = ROOT / "data" / "processed" / "sd302_2026_restoration"
RESULTS_DIR = ROOT / "results" / "subtype_grouped_model_selection"
PRIVATE_DIR = EXPORT_DIR / "grouped_model_selection"
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
        "classes": ["plain_whorl", "central_pocket_loop_whorl", "double_loop_whorl"],
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


def candidate_models(task_name: str) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_C0.1": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=SEED)),
        ]),
        "logistic_C1": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED)),
        ]),
        "linear_svc_C0.1": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearSVC(C=0.1, class_weight="balanced", dual="auto", max_iter=20000, random_state=SEED)),
        ]),
        "linear_svc_C1": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=20000, random_state=SEED)),
        ]),
        "rbf_svc_C1": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=SEED)),
        ]),
        "rbf_svc_C10": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=10.0, gamma="scale", class_weight="balanced", random_state=SEED)),
        ]),
    }
    if task_name == "arch":
        candidates["mlp_small"] = Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, max_iter=1000, random_state=SEED)),
        ])
    return candidates


def repeated_group_splits(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
) -> Iterator[tuple[int, int, int, np.ndarray, np.ndarray]]:
    required_labels = set(labels.tolist())
    accepted_repeats = 0
    candidate_seed = seed
    attempts = 0
    while accepted_repeats < n_repeats:
        attempts += 1
        if attempts > 1000:
            raise RuntimeError("Could not construct enough class-complete grouped repeats")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=candidate_seed,
        )
        candidate_folds = list(splitter.split(features, labels, groups))
        class_complete = all(
            set(labels[train_idx].tolist()) == required_labels
            and set(labels[test_idx].tolist()) == required_labels
            for train_idx, test_idx in candidate_folds
        )
        if not class_complete:
            candidate_seed += 1
            continue

        accepted_repeats += 1
        for split, (train_idx, test_idx) in enumerate(candidate_folds, 1):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            if train_groups & test_groups:
                raise AssertionError("Subject leakage detected in grouped split")
            yield accepted_repeats, split, candidate_seed, train_idx, test_idx
        candidate_seed += 1


def load_task(task_name: str, config: dict[str, Any]) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        row for row in read_csv(EXPORT_DIR / config["table"])
        if row["confirmed_subtype"] in config["classes"]
    ]
    embeddings = np.load(EXPORT_DIR / config["embedding_cache"])["embeddings"]
    if len(embeddings) != len(rows):
        raise ValueError(f"Embedding count mismatch for {task_name}: {len(embeddings)} != {len(rows)}")

    linkage = {row["record_key"]: row for row in read_csv(RESTORE_DIR / "subtype_subject_linkage.csv")}
    missing = [row["record_key"] for row in rows if row["record_key"] not in linkage]
    if missing:
        raise ValueError(f"Missing subject linkage for {len(missing)} {task_name} rows")
    labels = np.asarray([row["confirmed_subtype"] for row in rows])
    groups = np.asarray([linkage[row["record_key"]]["subject_id"] for row in rows])
    return rows, embeddings, labels, groups


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def evaluate_candidate(
    task_name: str,
    model_name: str,
    model: Any,
    rows: list[dict[str, str]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for repeat, split, split_seed, train_idx, test_idx in repeated_group_splits(embeddings, labels, groups):
        estimator = clone(model)
        estimator.fit(embeddings[train_idx], labels[train_idx])
        prediction = estimator.predict(embeddings[test_idx])
        truth = labels[test_idx]
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        fold_rows.append({
            "task": task_name,
            "model": model_name,
            "repeat": repeat,
            "split": split,
            "split_seed": split_seed,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_subjects": len(train_groups),
            "test_subjects": len(test_groups),
            "subject_overlap": len(train_groups & test_groups),
            "accuracy": accuracy_score(truth, prediction),
            "balanced_accuracy": balanced_accuracy_score(truth, prediction),
            "macro_f1": f1_score(truth, prediction, labels=classes, average="macro", zero_division=0),
            "weighted_f1": f1_score(truth, prediction, labels=classes, average="weighted", zero_division=0),
            "test_class_counts": dict(sorted(Counter(truth.tolist()).items())),
        })
        report = classification_report(truth, prediction, labels=classes, output_dict=True, zero_division=0)
        for label in classes:
            report_rows.append({
                "task": task_name,
                "model": model_name,
                "repeat": repeat,
                "split": split,
                "split_seed": split_seed,
                "label": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1_score": report[label]["f1-score"],
                "support": report[label]["support"],
            })
        for row_index, predicted in zip(test_idx, prediction):
            source = rows[int(row_index)]
            prediction_rows.append({
                "task": task_name,
                "model": model_name,
                "repeat": repeat,
                "split": split,
                "split_seed": split_seed,
                "review_id": source["review_id"],
                "record_key": source["record_key"],
                "subject_id": str(groups[int(row_index)]),
                "true_subtype": source["confirmed_subtype"],
                "predicted_subtype": str(predicted),
                "correct": str(predicted) == source["confirmed_subtype"],
            })
    return fold_rows, report_rows, prediction_rows


def aggregate_folds(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for task, model in sorted({(row["task"], row["model"]) for row in fold_rows}):
        matches = [row for row in fold_rows if row["task"] == task and row["model"] == model]
        record: dict[str, Any] = {"task": task, "model": model, "folds": len(matches)}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
            for statistic, value in summarize([float(row[metric]) for row in matches]).items():
                record[f"{metric}_{statistic}"] = value
        output.append(record)
    return output


def aggregate_reports(report_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["task"], row["model"], row["label"]) for row in report_rows})
    for task, model, label in keys:
        matches = [
            row for row in report_rows
            if row["task"] == task and row["model"] == model and row["label"] == label
        ]
        record: dict[str, Any] = {"task": task, "model": model, "label": label, "folds": len(matches)}
        for metric in ("precision", "recall", "f1_score"):
            statistics = summarize([float(row[metric]) for row in matches])
            record[f"{metric}_mean"] = statistics["mean"]
            record[f"{metric}_std"] = statistics["std"]
        record["support_total"] = sum(float(row["support"]) for row in matches)
        output.append(record)
    return output


def subject_class_counts(labels: np.ndarray, groups: np.ndarray) -> dict[str, int]:
    return {
        label: len(set(groups[labels == label]))
        for label in sorted(set(labels.tolist()))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS))
    args = parser.parse_args()
    selected = {args.task: TASKS[args.task]} if args.task else TASKS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

    all_folds: list[dict[str, Any]] = []
    all_reports: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    cohorts: dict[str, Any] = {}
    for task_name, config in selected.items():
        rows, embeddings, labels, groups = load_task(task_name, config)
        cohorts[task_name] = {
            "images": len(rows),
            "subjects": len(set(groups)),
            "image_class_counts": dict(sorted(Counter(labels.tolist()).items())),
            "subject_class_counts": subject_class_counts(labels, groups),
        }
        for model_name, model in candidate_models(task_name).items():
            print(f"Evaluating {task_name} / {model_name}")
            folds, reports, predictions = evaluate_candidate(
                task_name, model_name, model, rows, embeddings, labels, groups, list(config["classes"])
            )
            all_folds.extend(folds)
            all_reports.extend(reports)
            all_predictions.extend(predictions)

    if any(int(row["subject_overlap"]) for row in all_folds):
        raise AssertionError("At least one evaluation fold contains subject overlap")

    fold_columns = list(all_folds[0])
    write_csv(RESULTS_DIR / "grouped_fold_metrics.csv", all_folds, fold_columns)
    summary_rows = aggregate_folds(all_folds)
    write_csv(RESULTS_DIR / "grouped_model_summary.csv", summary_rows, list(summary_rows[0]))
    class_rows = aggregate_reports(all_reports)
    write_csv(RESULTS_DIR / "grouped_per_class_summary.csv", class_rows, list(class_rows[0]))
    write_csv(RESULTS_DIR / "grouped_cohort_summary.csv", [
        {"task": task, **values} for task, values in cohorts.items()
    ], ["task", "images", "subjects", "image_class_counts", "subject_class_counts"])
    write_csv(PRIVATE_DIR / "grouped_predictions.csv", all_predictions, list(all_predictions[0]))
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "splitter": "StratifiedGroupKFold with deterministic class-complete repeat selection",
        "evaluation_scope": "non-holdout expert-reviewed subtype development; subject-grouped",
        "selection_metric": "mean macro F1 across 25 grouped folds",
        "cohorts": cohorts,
        "maximum_subject_overlap": max(int(row["subject_overlap"]) for row in all_folds),
    }
    (RESULTS_DIR / "grouped_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("Grouped subtype model selection complete.")
    for row in sorted(summary_rows, key=lambda item: (item["task"], -item["macro_f1_mean"])):
        print(
            f"{row['task']} / {row['model']}: macro_f1={row['macro_f1_mean']:.3f} "
            f"+/- {row['macro_f1_std']:.3f}; balanced_accuracy={row['balanced_accuracy_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
