"""Audit the private Supabase subtype-review upload against the local package."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = (
    ROOT
    / "data"
    / "processed"
    / "unlabeled_subtype_review_package_2026-08-04"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL", "").strip()
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    bucket_name = os.environ.get("SUPABASE_BUCKET", "fingerprint-review").strip()
    if not url or not service_key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the current shell."
        )

    sys.path.insert(0, str(ROOT))
    from annotation_app.app_logic import build_export_csv
    from scripts.upload_subtype_review_to_supabase import verify_package
    from supabase import create_client

    package_dir = args.package_dir.resolve()
    local_rows = verify_package(package_dir)
    local_by_id = {row["review_id"]: row for row in local_rows}
    client = create_client(url, service_key)

    bucket = client.storage.get_bucket(bucket_name)
    if bucket.public:
        raise RuntimeError(f"Storage bucket {bucket_name!r} is unexpectedly public.")

    item_columns = (
        "review_id,record_key,image_path,current_main_type,source_primary_code,"
        "recorded_pattern_codes,alternative_type_warning,permitted_subtypes"
    )
    remote_items = (
        client.table("review_items")
        .select(item_columns)
        .order("review_id")
        .execute()
        .data
        or []
    )
    remote_by_id = {str(row["review_id"]): row for row in remote_items}
    if set(remote_by_id) != set(local_by_id):
        missing = sorted(set(local_by_id) - set(remote_by_id))
        extra = sorted(set(remote_by_id) - set(local_by_id))
        raise RuntimeError(
            f"Review ID mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )

    type_counts = Counter(str(row["current_main_type"]) for row in remote_items)
    expected_counts = Counter(row["current_main_type"] for row in local_rows)
    if type_counts != expected_counts:
        raise RuntimeError(
            f"Main-type counts differ: remote={type_counts}, local={expected_counts}"
        )

    package_prefix = package_dir.name
    for number, local_row in enumerate(local_rows, start=1):
        review_id = local_row["review_id"]
        remote_row = remote_by_id[review_id]
        local_image = package_dir / local_row["image_file"]
        expected_path = (
            Path(package_prefix)
            / "images"
            / local_row["current_main_type"]
            / Path(local_row["image_file"]).name
        ).as_posix()
        if remote_row["image_path"] != expected_path:
            raise RuntimeError(f"Unexpected image path for {review_id}")
        remote_image = client.storage.from_(bucket_name).download(expected_path)
        if hashlib.sha256(remote_image).digest() != hashlib.sha256(
            local_image.read_bytes()
        ).digest():
            raise RuntimeError(f"Remote image hash mismatch for {review_id}")
        if number % 50 == 0 or number == len(local_rows):
            print(f"Verified remote image hashes: {number}/{len(local_rows)}")

    annotations = client.table("annotations").select("*").execute().data or []
    annotations_by_id = {
        str(annotation["review_id"]): annotation for annotation in annotations
    }
    expected_csv = build_export_csv(remote_items, annotations_by_id)
    remote_csv = client.storage.from_(bucket_name).download(
        "exports/subtype_labeling_latest.csv"
    )
    if remote_csv != expected_csv:
        raise RuntimeError("The shared CSV does not match current database state.")

    events = (
        client.table("annotation_events")
        .select("review_id,revision")
        .execute()
        .data
        or []
    )
    revisions = ", ".join(
        f"{row['review_id']}=r{row['revision']}" for row in annotations
    ) or "none"
    print(f"Private bucket: yes ({bucket_name})")
    print(
        f"Review items: {len(remote_items)} "
        f"({type_counts['arch']} arch, {type_counts['whorl']} whorl)"
    )
    print(f"Saved annotations: {len(annotations)} ({revisions})")
    print(f"Audit events: {len(events)}")
    print(f"Shared CSV rows: {len(remote_items)}; matches database: yes")
    print("Supabase upload audit passed.")


if __name__ == "__main__":
    main()
