"""Prepare a clean local subtype-modeling dataset from the Supabase export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = (
    ROOT / "data" / "processed" / "subtype_review_completed_2026-09-06"
)
FINGERPRINT_CROP_BOX = (50, 48, 370, 368)


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


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    accepted_path = export_dir / "accepted_subtype_labels.csv"
    if not accepted_path.is_file():
        raise FileNotFoundError(accepted_path)

    accepted_rows = read_csv(accepted_path)
    model_image_root = export_dir / "model_input_images_320"
    model_rows: list[dict[str, Any]] = []
    rejected_dimensions: list[dict[str, str]] = []

    for row in accepted_rows:
        review_id = row["review_id"]
        subtype = row["confirmed_subtype"]
        review_image_path = export_dir / row["local_image_path"]
        if not review_image_path.is_file():
            raise FileNotFoundError(review_image_path)

        with Image.open(review_image_path) as image:
            grayscale = image.convert("L")
            if grayscale.size != (420, 448):
                rejected_dimensions.append(
                    {
                        "review_id": review_id,
                        "path": row["local_image_path"],
                        "size": f"{grayscale.size[0]}x{grayscale.size[1]}",
                    }
                )
                continue
            cropped = grayscale.crop(FINGERPRINT_CROP_BOX)
            output_path = model_image_root / subtype / f"{review_id}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(output_path, format="PNG", optimize=True)

        image_bytes = output_path.read_bytes()
        model_row = dict(row)
        model_row["model_image_path"] = output_path.relative_to(export_dir).as_posix()
        model_row["model_image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        model_row["model_image_width"] = 320
        model_row["model_image_height"] = 320
        model_row["recommended_first_pass"] = (
            "yes" if subtype != "accidental_whorl" else "no"
        )
        model_row["modeling_note"] = (
            "too few examples for standalone first-pass class"
            if subtype == "accidental_whorl"
            else ""
        )
        model_rows.append(model_row)

    if rejected_dimensions:
        raise ValueError(
            "Some review images did not match the expected 420x448 review "
            f"format: {rejected_dimensions[:5]}"
        )

    columns = list(model_rows[0].keys()) if model_rows else []
    write_csv(export_dir / "subtype_modeling_dataset.csv", model_rows, columns)

    recommended_rows = [
        row for row in model_rows if row["recommended_first_pass"] == "yes"
    ]
    write_csv(
        export_dir / "subtype_modeling_first_pass_recommended.csv",
        recommended_rows,
        columns,
    )

    arch_rows = [
        row for row in model_rows if row["confirmed_subtype"] in {"plain_arch", "tented_arch"}
    ]
    whorl_rows = [
        row
        for row in model_rows
        if row["confirmed_subtype"]
        in {"plain_whorl", "central_pocket_loop_whorl", "double_loop_whorl"}
    ]
    write_csv(export_dir / "arch_subtype_modeling_dataset.csv", arch_rows, columns)
    write_csv(export_dir / "whorl_subtype_modeling_dataset.csv", whorl_rows, columns)

    summary = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_export_dir": str(export_dir),
        "accepted_rows": len(accepted_rows),
        "model_rows": len(model_rows),
        "recommended_first_pass_rows": len(recommended_rows),
        "arch_binary_rows": len(arch_rows),
        "whorl_multiclass_first_pass_rows": len(whorl_rows),
        "crop_box_left_top_right_bottom": FINGERPRINT_CROP_BOX,
        "subtype_counts_all_model_rows": count_by(model_rows, "confirmed_subtype"),
        "subtype_counts_recommended_first_pass": count_by(
            recommended_rows, "confirmed_subtype"
        ),
        "confidence_counts": count_by(model_rows, "confidence"),
        "main_type_counts": count_by(model_rows, "current_main_type"),
        "excluded_from_first_pass": {
            "accidental_whorl": sum(
                row["confirmed_subtype"] == "accidental_whorl" for row in model_rows
            )
        },
        "notes": [
            "Images are cropped from the private 420x448 review PNGs to remove the text banner.",
            "Use subject-disjoint splitting only after original subject metadata is restored or joined.",
            "Rows marked adjudicate, exclude, or unclear are not included in this clean modeling dataset.",
        ],
    }
    write_json(export_dir / "subtype_modeling_summary.json", summary)

    print(f"Model rows: {len(model_rows)}")
    print(f"Recommended first-pass rows: {len(recommended_rows)}")
    print(f"Arch binary rows: {len(arch_rows)}")
    print(f"Whorl multiclass first-pass rows: {len(whorl_rows)}")
    print("Subtype counts:")
    for key, value in summary["subtype_counts_all_model_rows"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
