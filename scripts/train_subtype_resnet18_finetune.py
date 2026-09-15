"""Fine-tune lightweight ResNet-18 subtype prototypes on reviewed labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "processed" / "subtype_review_completed_2026-09-06"
RESULTS_DIR = ROOT / "results" / "subtype_resnet18_finetune"
PRIVATE_DIR = EXPORT_DIR / "subtype_resnet18_finetune"
SEED = 244664
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = {
    "arch_resnet18_finetune": {
        "table": "arch_subtype_modeling_dataset.csv",
        "classes": ["plain_arch", "tented_arch"],
        "epochs_head": 8,
        "epochs_unfreeze": 4,
        "batch_size": 8,
        "learning_rate_head": 1e-3,
        "learning_rate_unfreeze": 2e-4,
        "test_size": 0.25,
        "validation_size": 0.25,
    },
    "whorl_resnet18_finetune": {
        "table": "whorl_subtype_modeling_dataset.csv",
        "classes": [
            "plain_whorl",
            "central_pocket_loop_whorl",
            "double_loop_whorl",
        ],
        "epochs_head": 6,
        "epochs_unfreeze": 3,
        "batch_size": 12,
        "learning_rate_head": 1e-3,
        "learning_rate_unfreeze": 2e-4,
        "test_size": 0.20,
        "validation_size": 0.20,
    },
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


class FingerprintSubtypeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        label_to_index: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.rows = rows
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        image = Image.open(EXPORT_DIR / row["model_image_path"]).convert("RGB")
        label = self.label_to_index[row["confirmed_subtype"]]
        return self.transform(image), torch.tensor(label, dtype=torch.long), index


def stratified_three_way_split(
    rows: list[dict[str, str]], test_size: float, validation_size: float
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    labels = [row["confirmed_subtype"] for row in rows]
    train_val, test = train_test_split(
        rows,
        test_size=test_size,
        random_state=SEED,
        stratify=labels,
    )
    train_val_labels = [row["confirmed_subtype"] for row in train_val]
    relative_validation = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation,
        random_state=SEED,
        stratify=train_val_labels,
    )
    return train, validation, test


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["confirmed_subtype"] for row in rows).items()))


def make_sampler(rows: list[dict[str, str]], label_to_index: dict[str, int]) -> WeightedRandomSampler:
    counts = Counter(row["confirmed_subtype"] for row in rows)
    weights = [1.0 / counts[row["confirmed_subtype"]] for row in rows]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    mean, std = ResNet18_Weights.DEFAULT.transforms().mean, ResNet18_Weights.DEFAULT.transforms().std
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(6),
            transforms.RandomAffine(0, translate=(0.025, 0.025), scale=(0.96, 1.04)),
            transforms.ColorJitter(brightness=0.08, contrast=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def make_model(class_count: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model.to(DEVICE)


def unfreeze_final_block(model: nn.Module) -> None:
    for parameter in model.layer4.parameters():
        parameter.requires_grad_(True)
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)


def class_weight_tensor(rows: list[dict[str, str]], classes: list[str]) -> torch.Tensor:
    counts = Counter(row["confirmed_subtype"] for row in rows)
    weights = [len(rows) / (len(classes) * counts[label]) for label in classes]
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    truth: list[int] = []
    predicted: list[int] = []
    row_indices: list[int] = []
    for images, labels, indices in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += float(loss.item()) * len(labels)
        truth.extend(labels.detach().cpu().tolist())
        predicted.extend(logits.argmax(dim=1).detach().cpu().tolist())
        row_indices.extend(indices.tolist())
    return loss_sum / len(loader.dataset), truth, predicted, row_indices


def metric_dict(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, list[int], list[int], list[int], dict[str, float]]:
    with torch.inference_mode():
        loss, truth, predicted, indices = run_epoch(model, loader, criterion)
    return loss, truth, predicted, indices, metric_dict(truth, predicted)


def train_experiment(name: str, config: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in read_csv(EXPORT_DIR / config["table"])
        if row["confirmed_subtype"] in config["classes"]
    ]
    train_rows, validation_rows, test_rows = stratified_three_way_split(
        rows, float(config["test_size"]), float(config["validation_size"])
    )
    classes = list(config["classes"])
    label_to_index = {label: index for index, label in enumerate(classes)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    train_transform, eval_transform = make_transforms()

    train_ds = FingerprintSubtypeDataset(train_rows, label_to_index, train_transform)
    validation_ds = FingerprintSubtypeDataset(validation_rows, label_to_index, eval_transform)
    test_ds = FingerprintSubtypeDataset(test_rows, label_to_index, eval_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        sampler=make_sampler(train_rows, label_to_index),
    )
    validation_loader = DataLoader(validation_ds, batch_size=int(config["batch_size"]), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(config["batch_size"]), shuffle=False)

    model = make_model(len(classes))
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(train_rows, classes))
    history: list[dict[str, Any]] = []
    best_state = None
    best_macro_f1 = -1.0

    stages = [
        ("head", int(config["epochs_head"]), float(config["learning_rate_head"])),
        ("unfreeze_layer4", int(config["epochs_unfreeze"]), float(config["learning_rate_unfreeze"])),
    ]
    for stage, epochs, learning_rate in stages:
        if stage == "unfreeze_layer4":
            unfreeze_final_block(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=1e-4,
        )
        for epoch in range(1, epochs + 1):
            train_loss, train_truth, train_pred, _ = run_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_truth, val_pred, _, val_metrics = evaluate_model(
                model, validation_loader, criterion
            )
            train_metrics = metric_dict(train_truth, train_pred)
            record = {
                "experiment": name,
                "stage": stage,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in val_metrics.items()},
            }
            history.append(record)
            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(
                f"{name} {stage} epoch {epoch}: "
                f"val_macro_f1={val_metrics['macro_f1']:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_truth, test_pred, test_indices, test_metrics = evaluate_model(
        model, test_loader, criterion
    )
    report = classification_report(
        test_truth,
        test_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        test_truth, test_pred, labels=list(range(len(classes)))
    ).tolist()

    prediction_rows = []
    for row_index, truth_index, pred_index in zip(test_indices, test_truth, test_pred):
        row = test_rows[row_index]
        prediction_rows.append(
            {
                "experiment": name,
                "review_id": row["review_id"],
                "record_key": row["record_key"],
                "true_subtype": index_to_label[truth_index],
                "predicted_subtype": index_to_label[pred_index],
                "correct": truth_index == pred_index,
                "confidence": row["confidence"],
                "model_image_path": row["model_image_path"],
            }
        )

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "best_validation_macro_f1": best_macro_f1,
            "config": config,
        },
        PRIVATE_DIR / f"{name}.pt",
    )
    write_csv(
        PRIVATE_DIR / f"{name}_test_predictions.csv",
        prediction_rows,
        [
            "experiment",
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
        "name": name,
        "classes": classes,
        "row_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "class_counts": {
            "train": class_counts(train_rows),
            "validation": class_counts(validation_rows),
            "test": class_counts(test_rows),
        },
        "history": history,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "best_validation_macro_f1": best_macro_f1,
    }


def flatten_history(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        rows.extend(result["history"])
    return rows


def metrics_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        row = {
            "experiment": result["name"],
            "train_rows": result["row_counts"]["train"],
            "validation_rows": result["row_counts"]["validation"],
            "test_rows": result["row_counts"]["test"],
            "best_validation_macro_f1": result["best_validation_macro_f1"],
            "test_loss": result["test_loss"],
            **{f"test_{key}": value for key, value in result["test_metrics"].items()},
            "train_class_counts": result["class_counts"]["train"],
            "validation_class_counts": result["class_counts"]["validation"],
            "test_class_counts": result["class_counts"]["test"],
        }
        rows.append(row)
    return rows


def report_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for label, values in result["classification_report"].items():
            if isinstance(values, dict):
                rows.append(
                    {
                        "experiment": result["name"],
                        "label": label,
                        "precision": values.get("precision", ""),
                        "recall": values.get("recall", ""),
                        "f1_score": values.get("f1-score", ""),
                        "support": values.get("support", ""),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(EXPERIMENTS), help="Run one experiment only.")
    args = parser.parse_args()

    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = {args.only: EXPERIMENTS[args.only]} if args.only else EXPERIMENTS
    results = []
    for name, config in selected.items():
        print(f"Running {name} on {DEVICE}...")
        results.append(train_experiment(name, config))

    history = flatten_history(results)
    write_csv(RESULTS_DIR / "resnet18_finetune_training_history.csv", history, list(history[0].keys()))
    metrics = metrics_rows(results)
    write_csv(RESULTS_DIR / "resnet18_finetune_metrics.csv", metrics, list(metrics[0].keys()))
    reports = report_rows(results)
    write_csv(
        RESULTS_DIR / "resnet18_finetune_classification_report.csv",
        reports,
        ["experiment", "label", "precision", "recall", "f1_score", "support"],
    )
    write_json(
        RESULTS_DIR / "resnet18_finetune_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(DEVICE),
            "seed": SEED,
            "evaluation_scope": "non-holdout expert-reviewed subtype prototype; stratified split",
            "results": [
                {
                    "experiment": result["name"],
                    "classes": result["classes"],
                    "row_counts": result["row_counts"],
                    "class_counts": result["class_counts"],
                    "best_validation_macro_f1": result["best_validation_macro_f1"],
                    "test_metrics": result["test_metrics"],
                    "confusion_matrix": result["confusion_matrix"],
                }
                for result in results
            ],
        },
    )
    print("Fine-tuning complete.")
    for row in metrics:
        print(
            f"{row['experiment']}: test_macro_f1={row['test_macro_f1']:.3f}, "
            f"test_balanced_accuracy={row['test_balanced_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
