"""Securely upload the local subtype-review package to a private Supabase project."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = (
    ROOT
    / "data"
    / "processed"
    / "unlabeled_subtype_review_package_2026-08-04"
)


def verify_package(package_dir: Path) -> list[dict[str, str]]:
    template_path = package_dir / "subtype_labeling_template.csv"
    checksum_path = package_dir / "CHECKSUMS_SHA256.txt"
    if not template_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(
            "The package must contain subtype_labeling_template.csv and "
            "CHECKSUMS_SHA256.txt."
        )

    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split("  ", 1)
        path = package_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {relative_path}")

    with template_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["review_id"] for row in rows}) != len(rows):
        raise ValueError("The review template is empty or has duplicate review IDs.")
    for row in rows:
        image_path = package_dir / row["image_file"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    return rows


def chunks(values: list[dict[str, object]], size: int = 100):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the package and configuration without uploading anything.",
    )
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    rows = verify_package(package_dir)

    url = os.environ.get("SUPABASE_URL", "").strip()
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    bucket_name = os.environ.get("SUPABASE_BUCKET", "fingerprint-review").strip()
    if not url or not service_key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the current shell."
        )

    print(f"Validated {len(rows)} review records in {package_dir}")
    print(f"Target bucket: {bucket_name} (must already exist and be private)")
    if args.dry_run:
        print("Dry run complete; nothing was uploaded.")
        return

    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError(
            "Install the app dependencies first: pip install -r requirements.txt"
        ) from exc

    client = create_client(url, service_key)
    package_prefix = package_dir.name
    database_rows: list[dict[str, object]] = []

    for number, row in enumerate(rows, start=1):
        local_image = package_dir / row["image_file"]
        relative_image = Path(row["image_file"])
        storage_path = (
            Path(package_prefix)
            / "images"
            / row["current_main_type"]
            / relative_image.name
        ).as_posix()
        client.storage.from_(bucket_name).upload(
            path=storage_path,
            file=local_image.read_bytes(),
            file_options={
                "content-type": "image/png",
                "cache-control": "private, no-store",
                "upsert": "true",
            },
        )
        database_rows.append(
            {
                "review_id": row["review_id"],
                "record_key": row["record_key"],
                "image_path": storage_path,
                "current_main_type": row["current_main_type"],
                "source_primary_code": row["source_primary_code"],
                "recorded_pattern_codes": row["recorded_pattern_codes"],
                "alternative_type_warning": row["alternative_type_warning"] == "yes",
                "permitted_subtypes": row["permitted_subtypes"].split("|"),
            }
        )
        if number % 25 == 0 or number == len(rows):
            print(f"Uploaded images: {number}/{len(rows)}")

    for batch in chunks(database_rows):
        client.table("review_items").upsert(
            batch, on_conflict="review_id"
        ).execute()

    sys.path.insert(0, str(ROOT))
    from annotation_app.app_logic import build_export_csv

    initial_csv = build_export_csv(database_rows, {})
    client.storage.from_(bucket_name).upload(
        path="exports/subtype_labeling_latest.csv",
        file=initial_csv,
        file_options={
            "content-type": "text/csv; charset=utf-8",
            "cache-control": "private, no-store",
            "upsert": "true",
        },
    )

    remote_count = len(
        client.table("review_items").select("review_id").execute().data or []
    )
    if remote_count != len(rows):
        raise RuntimeError(
            f"Upload finished but Supabase has {remote_count} items; expected {len(rows)}."
        )
    print(f"Upload complete: {remote_count} review items and the initial shared CSV.")


if __name__ == "__main__":
    main()
