"""Export completed expert subtype-review data from Supabase.

The output is private biometric research data. It is written under ``data/``,
which is intentionally ignored by Git in this repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    ROOT / "data" / "processed" / "subtype_review_completed_2026-09-06"
)
DEFAULT_BUCKET = "fingerprint-review"
DEFAULT_EXPORT_PATH = "exports/subtype_labeling_latest.csv"
USER_AGENT = "codex-local-supabase-research-export/1.0"

ITEM_COLUMNS = (
    "review_id",
    "record_key",
    "image_path",
    "current_main_type",
    "source_primary_code",
    "recorded_pattern_codes",
    "alternative_type_warning",
    "permitted_subtypes",
    "created_at",
)

ANNOTATION_COLUMNS = (
    "review_id",
    "confirmed_subtype",
    "confidence",
    "review_action",
    "main_type_issue",
    "review_notes",
    "reviewer_id",
    "first_reviewed_at",
    "reviewed_at",
    "revision",
)

EXPORT_COLUMNS = (
    *ITEM_COLUMNS,
    *[column for column in ANNOTATION_COLUMNS if column != "review_id"],
    "local_image_path",
    "image_sha256",
)


def request_bytes(url: str, key: str) -> bytes:
    request = Request(
        url,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "User-Agent": USER_AGENT,
        },
    )
    with urlopen(request, timeout=60) as response:
        return response.read()


def get_json(base_url: str, key: str, path: str) -> list[dict[str, Any]]:
    return json.loads(request_bytes(base_url.rstrip("/") + path, key).decode("utf-8"))


def download_storage_object(
    base_url: str, key: str, bucket: str, object_path: str
) -> bytes:
    quoted_bucket = quote(bucket, safe="")
    quoted_path = quote(object_path, safe="/")
    url = f"{base_url.rstrip()}/storage/v1/object/{quoted_bucket}/{quoted_path}"
    return request_bytes(url, key)


def complete(annotation: dict[str, Any] | None) -> bool:
    if not annotation:
        return False
    return all(
        str(annotation.get(field, "") or "").strip()
        for field in (
            "confirmed_subtype",
            "confidence",
            "review_action",
            "reviewer_id",
        )
    )


def validate(item: dict[str, Any], annotation: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    allowed = item.get("permitted_subtypes") or []
    subtype = annotation.get("confirmed_subtype")
    confidence = annotation.get("confidence")
    action = annotation.get("review_action")
    main_type_issue = annotation.get("main_type_issue")
    notes = str(annotation.get("review_notes") or "").strip()

    if subtype not in allowed:
        errors.append("subtype not permitted")
    if confidence not in {"high", "medium", "low"}:
        errors.append("bad confidence")
    if action not in {"accept", "adjudicate", "exclude"}:
        errors.append("bad action")
    if main_type_issue and main_type_issue not in {"incorrect", "uncertain"}:
        errors.append("bad main-type issue")
    if subtype == "unclear" and action == "accept":
        errors.append("unclear accepted")
    if main_type_issue:
        if subtype != "unclear":
            errors.append("main issue without unclear")
        if action != "adjudicate":
            errors.append("main issue not adjudicate")
        if not notes:
            errors.append("main issue lacks notes")
    if action in {"adjudicate", "exclude"} and not notes:
        errors.append("action lacks notes")
    return errors


def merged_rows(
    items: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    annotations_by_id = {
        str(annotation["review_id"]): annotation for annotation in annotations
    }
    rows: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda row: str(row["review_id"])):
        annotation = annotations_by_id.get(str(item["review_id"]), {})
        row = {column: item.get(column, "") for column in ITEM_COLUMNS}
        row.update(
            {
                column: annotation.get(column, "")
                for column in ANNOTATION_COLUMNS
                if column != "review_id"
            }
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            normalized = dict(row)
            permitted = normalized.get("permitted_subtypes")
            if isinstance(permitted, list):
                normalized["permitted_subtypes"] = "|".join(str(value) for value in permitted)
            writer.writerow({column: normalized.get(column, "") for column in columns})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_images(
    base_url: str,
    key: str,
    bucket: str,
    output_dir: Path,
    rows: list[dict[str, Any]],
    skip_images: bool,
) -> None:
    for index, row in enumerate(rows, start=1):
        image_path = str(row["image_path"])
        main_type = str(row["current_main_type"])
        filename = Path(image_path).name
        local_path = output_dir / "review_images" / main_type / filename
        row["local_image_path"] = local_path.relative_to(output_dir).as_posix()

        if local_path.exists():
            row["image_sha256"] = hashlib.sha256(local_path.read_bytes()).hexdigest()
            continue
        if skip_images:
            raise FileNotFoundError(
                f"Missing local image while --skip-images is active: {local_path}"
            )

        image_bytes = download_storage_object(base_url, key, bucket, image_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(image_bytes)
        row["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        if index % 50 == 0 or index == len(rows):
            print(f"Downloaded review images: {index}/{len(rows)}")


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bucket", default=os.environ.get("SUPABASE_BUCKET", DEFAULT_BUCKET))
    parser.add_argument(
        "--export-path",
        default=os.environ.get("SUPABASE_EXPORT_PATH", DEFAULT_EXPORT_PATH),
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Write tables only and reuse any already downloaded images.",
    )
    args = parser.parse_args()

    base_url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not base_url or not key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the current shell."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    item_select = quote(",".join(ITEM_COLUMNS), safe=",")
    items = get_json(
        base_url,
        key,
        f"/rest/v1/review_items?select={item_select}&order=review_id.asc",
    )
    annotations = get_json(
        base_url,
        key,
        "/rest/v1/annotations?select=*&order=review_id.asc",
    )
    events = get_json(
        base_url,
        key,
        "/rest/v1/annotation_events?select=review_id,revision,created_at",
    )
    shared_csv = download_storage_object(base_url, key, args.bucket, args.export_path)

    annotations_by_id = {
        str(annotation["review_id"]): annotation for annotation in annotations
    }
    items_by_id = {str(item["review_id"]): item for item in items}
    completed_ids = {
        review_id
        for review_id, annotation in annotations_by_id.items()
        if complete(annotation)
    }
    pending_ids = sorted(set(items_by_id) - completed_ids)
    orphan_ids = sorted(set(annotations_by_id) - set(items_by_id))
    invalid = [
        {"review_id": review_id, "errors": errors}
        for review_id, annotation in sorted(annotations_by_id.items())
        if review_id in items_by_id
        for errors in [validate(items_by_id[review_id], annotation)]
        if errors
    ]

    rows = merged_rows(items, annotations)
    export_images(base_url, key, args.bucket, output_dir, rows, args.skip_images)

    accepted_rows = [
        row
        for row in rows
        if complete(annotations_by_id.get(str(row["review_id"])))
        and row.get("review_action") == "accept"
        and row.get("confirmed_subtype") != "unclear"
    ]
    adjudication_rows = [
        row
        for row in rows
        if row.get("review_action") in {"adjudicate", "exclude"}
        or row.get("confirmed_subtype") == "unclear"
    ]

    raw_dir = output_dir / "raw_supabase"
    write_json(raw_dir / "review_items.json", items)
    write_json(raw_dir / "annotations.json", annotations)
    write_json(raw_dir / "annotation_events.json", events)
    (raw_dir / "subtype_labeling_latest_from_bucket.csv").write_bytes(shared_csv)

    write_csv(output_dir / "subtype_labeling_latest.csv", rows, EXPORT_COLUMNS)
    write_csv(output_dir / "accepted_subtype_labels.csv", accepted_rows, EXPORT_COLUMNS)
    write_csv(
        output_dir / "adjudication_or_unclear_subtype_labels.csv",
        adjudication_rows,
        EXPORT_COLUMNS,
    )

    summary = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_project_url": base_url,
        "bucket": args.bucket,
        "review_items": len(items),
        "saved_annotations": len(annotations),
        "completed_annotations": len(completed_ids),
        "pending_items": len(pending_ids),
        "invalid_annotations": len(invalid),
        "orphan_annotations": len(orphan_ids),
        "audit_events": len(events),
        "accepted_modeling_rows": len(accepted_rows),
        "adjudication_or_unclear_rows": len(adjudication_rows),
        "main_type_counts": count_by(rows, "current_main_type"),
        "review_action_counts": count_by(rows, "review_action"),
        "confidence_counts": count_by(rows, "confidence"),
        "subtype_counts_all_completed": count_by(rows, "confirmed_subtype"),
        "subtype_counts_accepted_only": count_by(accepted_rows, "confirmed_subtype"),
        "pending_review_ids": pending_ids,
        "orphan_review_ids": orphan_ids,
        "invalid_annotations_detail": invalid,
    }
    write_json(output_dir / "subtype_review_summary.json", summary)

    readme = f"""# Completed Subtype Review Export

This directory contains a private local export of the completed expert
subtype-review data from Supabase.

Exported at UTC: `{summary['exported_at_utc']}`

Source project URL: `{base_url}`
Storage bucket: `{args.bucket}`

## Files

- `subtype_labeling_latest.csv`: merged review item and annotation table.
- `accepted_subtype_labels.csv`: accepted, specific subtype labels for model development.
- `adjudication_or_unclear_subtype_labels.csv`: rows marked unclear, adjudicate, or exclude.
- `subtype_review_summary.json`: machine-readable audit summary.
- `raw_supabase/`: raw exported Supabase tables and bucket CSV.
- `review_images/`: downloaded private review PNGs, grouped by current main type.

## Modeling Rule

Use `accepted_subtype_labels.csv` for the first subtype-modeling pass. Keep
`adjudication_or_unclear_subtype_labels.csv` out of clean supervised training
until a named adjudicator resolves those cases.

Do not commit this directory. It contains row-level biometric research data.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Output directory: {output_dir}")
    print(f"Review items: {len(items)}")
    print(f"Completed annotations: {len(completed_ids)}")
    print(f"Accepted modeling rows: {len(accepted_rows)}")
    print(f"Adjudication/unclear rows: {len(adjudication_rows)}")
    print(f"Pending items: {len(pending_ids)}")
    print(f"Invalid annotations: {len(invalid)}")
    print("Subtype counts, accepted only:")
    for key, value in summary["subtype_counts_accepted_only"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
