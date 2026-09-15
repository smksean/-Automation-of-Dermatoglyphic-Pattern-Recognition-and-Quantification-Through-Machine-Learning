"""Join completed subtype labels back to private source metadata.

This script prepares the research-grade subtype table needed for grouped,
subject-disjoint evaluation. It requires the private EfficientNet metadata file
that is intentionally not stored in Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = (
    ROOT / "data" / "processed" / "subtype_review_completed_2026-09-06"
)
DEFAULT_METADATA_PATH = (
    ROOT / "data" / "processed" / "efficientnet_320_package" / "roll_320_clahe_metadata.csv"
)
DEFAULT_OUTPUT_PATH = DEFAULT_EXPORT_DIR / "subtype_modeling_with_private_metadata.csv"


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
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_record_key(row: dict[str, str]) -> str:
    source = "|".join(
        str(row.get(column, ""))
        for column in ("subject_id", "finger_position", "collection_type", "png_path")
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    metadata_path = args.metadata_path.resolve()
    output_path = args.output_path.resolve()

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Private EfficientNet metadata is required before subject-disjoint "
            f"subtype evaluation can be run: {metadata_path}"
        )

    labels = read_csv(export_dir / "subtype_modeling_dataset.csv")
    metadata = read_csv(metadata_path)
    metadata_by_key = {}
    duplicate_keys = []
    for row in metadata:
        record_key = make_record_key(row)
        if record_key in metadata_by_key:
            duplicate_keys.append(record_key)
        metadata_by_key[record_key] = row

    joined = []
    missing_keys = []
    metadata_fields = [
        "subject_id",
        "finger_position",
        "collection_type",
        "experiment_role",
        "png_path",
        "primary_label",
        "all_pattern_labels",
        "broad_class",
        "subtype",
        "num_pattern_labels",
    ]
    for row in labels:
        metadata_row = metadata_by_key.get(row["record_key"])
        if metadata_row is None:
            missing_keys.append(row["record_key"])
            continue
        joined_row = dict(row)
        for field in metadata_fields:
            joined_row[f"source_{field}"] = metadata_row.get(field, "")
        joined.append(joined_row)

    if duplicate_keys:
        raise ValueError(f"Duplicate metadata record keys found: {duplicate_keys[:10]}")
    if missing_keys:
        raise ValueError(f"Subtype rows missing from metadata: {missing_keys[:10]}")

    columns = list(joined[0].keys()) if joined else []
    write_csv(output_path, joined, columns)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_path": str(metadata_path),
        "output_path": str(output_path),
        "label_rows": len(labels),
        "metadata_rows": len(metadata),
        "joined_rows": len(joined),
        "unique_subjects": len({row["source_subject_id"] for row in joined}),
        "subtype_counts": count_by(joined, "confirmed_subtype"),
        "source_experiment_role_counts": count_by(joined, "source_experiment_role"),
        "main_type_counts": count_by(joined, "current_main_type"),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)

    print(f"Joined rows: {len(joined)}")
    print(f"Unique subjects: {summary['unique_subjects']}")
    print("Subtype counts:")
    for key, value in summary["subtype_counts"].items():
        print(f"  {key}: {value}")
    print("Source experiment roles:")
    for key, value in summary["source_experiment_role_counts"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
