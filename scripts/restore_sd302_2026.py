"""Verify and index the restricted 2026 NIST SD302 research archives.

The script reads SD302g IRR records directly from the ZIP archive. It does not
extract biometric images unless ``--extract-images`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo


FS = "\x1c"
GS = "\x1d"
RS = "\x1e"
US = "\x1f"

EXPECTED_SHA256 = {
    "participants.csv": "19c2d672ce4ad8c1ff307ca08849005ebfcdad736b5282fea6ce3776ec7ed2b2",
    "ERRATA_SD302.txt": "ac34a920d2fb0aa337efaa41ab013c191cc95a55bae12cc2746f6f5f88823358",
    "sd302a.zip": "d78c1cd6089625eb856bc135fa615def9a389cb057b0cef66f43ae34328dc582",
    "sd302b.zip": "0e90c09665bd7537594f9d11d726a4ef2d89b311d2ed37fb8c284a9903070fcd",
    "sd302g.zip": "8477ef79621bba71a126d4db20cbf6072c83b6dae2beb91e018ae2d79988eeea",
}

IRR_NAME_PATTERN = re.compile(
    r"^(?P<subject_id>\d{8})_"
    r"(?P<device>[A-Z])_"
    r"(?P<resolution>\d+)_"
    r"(?P<capture_type>roll|slap|plain)_"
    r"(?P<finger_position>\d{2})\.irr$"
)

BROAD_CLASS_MAP = {
    "AU": "arch",
    "LS": "left_slant_loop",
    "RS": "right_slant_loop",
    "WU": "whorl",
    "UC": "unclassifiable",
}

SUBTYPE_MAP = {
    "AU+PA": "plain_arch",
    "AU+TA": "tented_arch",
    "WU+PW": "plain_whorl",
    "WU+CP": "central_pocket_loop_whorl",
    "WU+DL": "double_loop_whorl",
    "WU+AW": "accidental_whorl",
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_field(text: str, tag: str) -> str | None:
    match = re.search(rf"{re.escape(tag)}:([^{GS}{FS}]*)", text)
    return match.group(1) if match else None


def split_repeated_field(value: str | None) -> list[list[str]]:
    if not value:
        return []
    return [
        [part.strip() for part in entry.split(US) if part.strip()]
        for entry in value.split(RS)
        if entry
    ]


def label_from_parts(parts: list[str]) -> str:
    return "+".join(parts)


def broad_class(label: str | None) -> str | None:
    if not label:
        return None
    return BROAD_CLASS_MAP.get(label.split("+")[0], "unknown")


def subtype_name(label: str | None) -> str | None:
    if not label:
        return None
    return SUBTYPE_MAP.get(label)


def collection_from_member(member: str) -> str:
    parts = {part.lower() for part in PurePosixPath(member).parts}
    if "baseline" in parts:
        return "baseline"
    if "challengers" in parts:
        return "challengers"
    return "unknown"


def expected_png_member(metadata: dict[str, str], collection: str) -> tuple[str, str] | None:
    subject = metadata["subject_id"]
    device = metadata["device"]
    resolution = metadata["resolution"]
    capture = metadata["capture_type"]
    finger = metadata["finger_position"]

    if collection == "challengers":
        return (
            "sd302a.zip",
            f"images/challengers/{device}/roll/png/{subject}_{device}_roll_{finger}.png",
        )
    if collection == "baseline":
        folder = "slap-segmented" if capture == "slap" and int(finger) <= 10 else capture
        return (
            "sd302b.zip",
            f"images/baseline/{device}/{resolution}/{folder}/png/"
            f"{subject}_{device}_{resolution}_{capture}_{finger}.png",
        )
    return None


def legacy_png_path(archive_name: str, member: str) -> str:
    """Recreate the Windows-relative path used to hash subtype review records."""
    return f"{archive_name.removesuffix('.zip')}/{member}".replace("/", "\\")


def review_record_key(
    subject_id: str,
    finger_position: str,
    collection: str,
    png_path: str,
) -> str:
    source = "|".join((subject_id, finger_position, collection, png_path))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def safe_extract_member(archive: ZipFile, info: ZipInfo, destination: Path) -> Path:
    relative = PurePosixPath(info.filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {info.filename}")
    output = destination.joinpath(*relative.parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, output.open("wb") as target:
        while chunk := source.read(8 * 1024 * 1024):
            target.write(chunk)
    return output


def verify_sources(archive_dir: Path) -> list[dict[str, object]]:
    records = []
    for name, expected in EXPECTED_SHA256.items():
        path = archive_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required source is missing: {path}")
        actual = sha256(path)
        records.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": actual,
                "expected_sha256": expected,
                "verified": actual == expected,
            }
        )
    failed = [record["name"] for record in records if not record["verified"]]
    if failed:
        raise RuntimeError(f"Checksum verification failed: {', '.join(failed)}")
    return records


def count_participants(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def parse_archives(
    archive_dir: Path,
    output_dir: Path,
    extract_images: bool,
    extract_irr: bool,
) -> dict[str, object]:
    image_rows: list[dict[str, object]] = []
    pattern_rows: list[dict[str, object]] = []
    field_counts: Counter[str] = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "efs_feature_points.csv"
    feature_handle = feature_path.open("w", encoding="utf-8", newline="")
    feature_fields = [
        "record_key",
        "subject_id",
        "finger_position",
        "collection_type",
        "irr_member",
        "feature_family",
        "feature_index",
        "x",
        "y",
        "direction",
        "minutia_type",
        "attributes_json",
    ]
    feature_writer = csv.DictWriter(feature_handle, fieldnames=feature_fields)
    feature_writer.writeheader()
    feature_point_count = 0

    image_archives = {
        "sd302a.zip": ZipFile(archive_dir / "sd302a.zip"),
        "sd302b.zip": ZipFile(archive_dir / "sd302b.zip"),
    }
    image_members = {name: set(archive.namelist()) for name, archive in image_archives.items()}
    g_archive = ZipFile(archive_dir / "sd302g.zip")

    try:
        irr_infos = sorted(
            (info for info in g_archive.infolist() if info.filename.lower().endswith(".irr")),
            key=lambda info: info.filename,
        )
        for info in irr_infos:
            filename_metadata = IRR_NAME_PATTERN.match(PurePosixPath(info.filename).name)
            if not filename_metadata:
                continue
            metadata = filename_metadata.groupdict()
            collection = collection_from_member(info.filename)
            text = g_archive.read(info).decode("latin-1", errors="ignore")

            labels = [label_from_parts(parts) for parts in split_repeated_field(get_field(text, "9.307"))]
            cores = split_repeated_field(get_field(text, "9.320"))
            deltas = split_repeated_field(get_field(text, "9.321"))
            minutiae = split_repeated_field(get_field(text, "9.331"))
            minutia_types = Counter(parts[3] for parts in minutiae if len(parts) > 3)

            for tag in ("9.307", "9.320", "9.321", "9.331"):
                if get_field(text, tag) is not None:
                    field_counts[tag] += 1

            expected_image = expected_png_member(metadata, collection)
            image_archive_name = expected_image[0] if expected_image else None
            image_member = expected_image[1] if expected_image else None
            image_found = bool(
                image_archive_name
                and image_member
                and image_member in image_members[image_archive_name]
            )
            image_bytes = 0
            historical_png_path = None
            record_key = None
            if image_found and image_archive_name and image_member:
                image_info = image_archives[image_archive_name].getinfo(image_member)
                image_bytes = image_info.file_size
                historical_png_path = legacy_png_path(image_archive_name, image_member)
                record_key = review_record_key(
                    metadata["subject_id"],
                    metadata["finger_position"],
                    collection,
                    historical_png_path,
                )
                if extract_images:
                    safe_extract_member(
                        image_archives[image_archive_name],
                        image_info,
                        archive_dir.parent / "linked_images" / image_archive_name.removesuffix(".zip"),
                    )

            if extract_irr:
                safe_extract_member(g_archive, info, archive_dir.parent / "sd302g")

            for family, features in (("core", cores), ("delta", deltas), ("minutia", minutiae)):
                for feature_index, parts in enumerate(features, start=1):
                    feature_writer.writerow(
                        {
                            "record_key": record_key,
                            "subject_id": metadata["subject_id"],
                            "finger_position": metadata["finger_position"],
                            "collection_type": collection,
                            "irr_member": info.filename,
                            "feature_family": family,
                            "feature_index": feature_index,
                            "x": parts[0] if len(parts) > 0 else None,
                            "y": parts[1] if len(parts) > 1 else None,
                            "direction": parts[2] if len(parts) > 2 else None,
                            "minutia_type": parts[3] if family == "minutia" and len(parts) > 3 else None,
                            "attributes_json": json.dumps(parts, separators=(",", ":")),
                        }
                    )
                    feature_point_count += 1

            primary_label = labels[0] if labels else None
            row: dict[str, object] = {
                **metadata,
                "collection_type": collection,
                "irr_member": info.filename,
                "all_pattern_labels": "|".join(labels),
                "num_pattern_labels": len(labels),
                "primary_label": primary_label,
                "broad_class": broad_class(primary_label),
                "subtype": subtype_name(primary_label),
                "core_count": len(cores),
                "delta_count": len(deltas),
                "minutiae_count": len(minutiae),
                "ridge_ending_count": minutia_types.get("E", 0),
                "bifurcation_count": minutia_types.get("B", 0),
                "unknown_minutia_count": sum(
                    count for kind, count in minutia_types.items() if kind not in {"E", "B"}
                ),
                "image_archive": image_archive_name,
                "image_member": image_member,
                "legacy_png_path": historical_png_path,
                "record_key": record_key,
                "image_found": image_found,
                "image_bytes": image_bytes,
            }
            image_rows.append(row)

            for index, label in enumerate(labels, start=1):
                pattern_rows.append(
                    {
                        **metadata,
                        "collection_type": collection,
                        "irr_member": info.filename,
                        "entry_index": index,
                        "pattern_label": label,
                        "broad_class": broad_class(label),
                        "subtype": subtype_name(label),
                    }
                )
    finally:
        feature_handle.close()
        g_archive.close()
        for archive in image_archives.values():
            archive.close()

    write_csv(output_dir / "irr_feature_inventory.csv", image_rows)
    write_csv(output_dir / "irr_pattern_entries.csv", pattern_rows)

    subtype_linkage = restore_subtype_linkage(output_dir, image_rows)

    return {
        "irr_records": len(image_rows),
        "subjects": len({str(row["subject_id"]) for row in image_rows}),
        "pattern_entries": len(pattern_rows),
        "images_linked": sum(bool(row["image_found"]) for row in image_rows),
        "linked_image_bytes": sum(int(row["image_bytes"]) for row in image_rows),
        "records_with_cores": field_counts["9.320"],
        "records_with_deltas": field_counts["9.321"],
        "records_with_minutiae": field_counts["9.331"],
        "total_cores": sum(int(row["core_count"]) for row in image_rows),
        "total_deltas": sum(int(row["delta_count"]) for row in image_rows),
        "total_minutiae": sum(int(row["minutiae_count"]) for row in image_rows),
        "total_ridge_endings": sum(int(row["ridge_ending_count"]) for row in image_rows),
        "total_bifurcations": sum(int(row["bifurcation_count"]) for row in image_rows),
        "feature_point_rows": feature_point_count,
        "broad_class_counts": dict(Counter(str(row["broad_class"]) for row in image_rows)),
        "subtype_linkage": subtype_linkage,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows generated for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def restore_subtype_linkage(
    output_dir: Path,
    image_rows: list[dict[str, object]],
) -> dict[str, int | str]:
    project_root = Path(__file__).resolve().parents[1]
    subtype_path = (
        project_root
        / "data"
        / "processed"
        / "subtype_review_completed_2026-09-06"
        / "subtype_modeling_dataset.csv"
    )
    if not subtype_path.exists():
        return {"status": "subtype dataset not present", "rows": 0, "matched": 0, "unmatched": 0}

    inventory_by_key = {
        str(row["record_key"]): row
        for row in image_rows
        if row.get("record_key")
    }
    with subtype_path.open("r", encoding="utf-8-sig", newline="") as handle:
        subtype_rows = list(csv.DictReader(handle))

    linked_rows = []
    unmatched = []
    for subtype in subtype_rows:
        source = inventory_by_key.get(subtype["record_key"])
        if source is None:
            unmatched.append(subtype["review_id"])
            continue
        linked_rows.append(
            {
                "review_id": subtype["review_id"],
                "record_key": subtype["record_key"],
                "subject_id": source["subject_id"],
                "finger_position": source["finger_position"],
                "collection_type": source["collection_type"],
                "device": source["device"],
                "resolution": source["resolution"],
                "confirmed_subtype": subtype["confirmed_subtype"],
                "confidence": subtype["confidence"],
                "review_action": subtype["review_action"],
                "source_primary_code": subtype["source_primary_code"],
                "recorded_pattern_codes": subtype["recorded_pattern_codes"],
                "core_count": source["core_count"],
                "delta_count": source["delta_count"],
                "minutiae_count": source["minutiae_count"],
                "ridge_ending_count": source["ridge_ending_count"],
                "bifurcation_count": source["bifurcation_count"],
                "image_archive": source["image_archive"],
                "image_member": source["image_member"],
                "irr_member": source["irr_member"],
            }
        )

    if linked_rows:
        write_csv(output_dir / "subtype_subject_linkage.csv", linked_rows)
    return {
        "status": "complete" if not unmatched else "partial",
        "rows": len(subtype_rows),
        "matched": len(linked_rows),
        "unmatched": len(unmatched),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("data/raw/nist_sd302_2026/archives"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/sd302_2026_restoration"),
    )
    parser.add_argument("--extract-images", action="store_true")
    parser.add_argument("--extract-irr", action="store_true")
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    output_dir = args.output_dir.resolve()
    sources = verify_sources(archive_dir)
    summary = parse_archives(
        archive_dir,
        output_dir,
        extract_images=args.extract_images,
        extract_irr=args.extract_irr,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NIST Special Database 302",
        "release_context": "SD302g refreshed 2026-03-11",
        "participant_rows": count_participants(archive_dir / "participants.csv"),
        "sources": sources,
        "summary": summary,
        "extracted_images": args.extract_images,
        "extracted_irr": args.extract_irr,
    }
    manifest_path = output_dir / "restoration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nPrivate outputs: {output_dir}")


if __name__ == "__main__":
    main()
