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


def select_review_rows(
    rows: list[dict[str, str]], pilot_per_type: int | None = None
) -> list[dict[str, str]]:
    """Return all rows or a stable, balanced arch/whorl pilot selection."""
    if pilot_per_type is None:
        return rows
    if pilot_per_type < 1:
        raise ValueError("--pilot-per-type must be at least 1.")

    selected: list[dict[str, str]] = []
    for main_type in ("arch", "whorl"):
        matches = sorted(
            (row for row in rows if row["current_main_type"] == main_type),
            key=lambda row: row["review_id"],
        )
        if len(matches) < pilot_per_type:
            raise ValueError(
                f"Requested {pilot_per_type} {main_type} pilot images, "
                f"but only {len(matches)} are available."
            )
        selected.extend(matches[:pilot_per_type])
    return sorted(selected, key=lambda row: row["review_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the package and selection without uploading anything.",
    )
    parser.add_argument(
        "--pilot-per-type",
        type=int,
        metavar="N",
        help="Upload only N arch and N whorl images for a balanced pilot.",
    )
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    all_rows = verify_package(package_dir)
    rows = select_review_rows(all_rows, args.pilot_per_type)

    type_counts = {
        main_type: sum(row["current_main_type"] == main_type for row in rows)
        for main_type in ("arch", "whorl")
    }
    print(f"Validated all {len(all_rows)} review records in {package_dir}")
    print(
        f"Selected {len(rows)} records: "
        f"{type_counts['arch']} arch and {type_counts['whorl']} whorl"
    )
    if args.dry_run:
        print("Dry run complete; no credentials were read and nothing was uploaded.")
        return

    url = os.environ.get("SUPABASE_URL", "").strip()
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    bucket_name = os.environ.get("SUPABASE_BUCKET", "fingerprint-review").strip()
    if not url or not service_key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the current shell."
        )

    print(f"Target bucket: {bucket_name} (created privately by the database migration)")

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

    remote_rows = client.table("review_items").select("review_id").execute().data or []
    remote_ids = {str(row["review_id"]) for row in remote_rows}
    expected_ids = {row["review_id"] for row in rows}
    missing_ids = sorted(expected_ids - remote_ids)
    if missing_ids:
        raise RuntimeError(
            "Upload finished but Supabase is missing selected review IDs: "
            + ", ".join(missing_ids)
        )
    print(
        f"Upload complete: verified {len(expected_ids)} selected items; "
        f"Supabase now contains {len(remote_ids)} review items and the shared CSV."
    )


if __name__ == "__main__":
    main()
