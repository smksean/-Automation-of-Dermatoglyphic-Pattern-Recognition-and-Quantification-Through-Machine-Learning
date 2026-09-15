"""Train selected subtype prototype artifacts for local inference."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "processed" / "subtype_review_completed_2026-09-06"
ARTIFACT_DIR = EXPORT_DIR / "selected_subtype_model_artifacts"
SEED = 244664

TASKS = {
    "arch": {
        "table": "arch_subtype_modeling_dataset.csv",
        "embedding_cache": "preliminary_subtype_transfer_baseline/arch_binary_resnet18_embeddings.npz",
        "classes": ["plain_arch", "tented_arch"],
        "model": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        ),
        "reported_macro_f1": 0.526706,
        "reported_balanced_accuracy": 0.567154,
    },
    "whorl": {
        "table": "whorl_subtype_modeling_dataset.csv",
        "embedding_cache": "preliminary_subtype_transfer_baseline/whorl_first_pass_resnet18_embeddings.npz",
        "classes": [
            "plain_whorl",
            "central_pocket_loop_whorl",
            "double_loop_whorl",
        ],
        "model": LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        ),
        "reported_macro_f1": 0.446040,
        "reported_balanced_accuracy": 0.468596,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["confirmed_subtype"] for row in rows).items()))


def load_rows_and_embeddings(config: dict[str, Any]) -> tuple[list[dict[str, str]], np.ndarray]:
    rows = [
        row
        for row in read_csv(EXPORT_DIR / config["table"])
        if row["confirmed_subtype"] in config["classes"]
    ]
    embeddings = np.load(EXPORT_DIR / config["embedding_cache"])["embeddings"]
    if len(embeddings) != len(rows):
        raise ValueError("Embedding count does not match table row count.")
    return rows, embeddings


def train_task(task: str, config: dict[str, Any]) -> dict[str, Any]:
    rows, embeddings = load_rows_and_embeddings(config)
    labels = [row["confirmed_subtype"] for row in rows]
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", config["model"]),
        ]
    )
    model.fit(embeddings, labels)

    artifact_path = ARTIFACT_DIR / f"{task}_subtype_classifier.pkl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as handle:
        pickle.dump(model, handle)

    return {
        "task": task,
        "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
        "classes": list(config["classes"]),
        "training_rows": len(rows),
        "class_counts": class_counts(rows),
        "reported_subject_grouped_macro_f1": config["reported_macro_f1"],
        "reported_subject_grouped_balanced_accuracy": config[
            "reported_balanced_accuracy"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), help="Train one task only.")
    args = parser.parse_args()

    selected = {args.task: TASKS[args.task]} if args.task else TASKS
    summaries = [train_task(task, config) for task, config in selected.items()]
    write_json(
        ARTIFACT_DIR / "selected_subtype_artifacts_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "feature_extractor": "torchvision ResNet-18 ImageNet frozen penultimate embeddings",
            "scope": "non-holdout expert-reviewed subtype prototype; subject-grouped validation",
            "tasks": summaries,
            "usage_note": (
                "Run arch artifact only for broad-class arch predictions and whorl "
                "artifact only for broad-class whorl predictions."
            ),
        },
    )
    for summary in summaries:
        print(
            f"{summary['task']}: trained {summary['training_rows']} rows -> "
            f"{summary['artifact_path']}"
        )


if __name__ == "__main__":
    main()
