"""Build the private client package for fingerprint subtype relabelling."""

from __future__ import annotations

from pathlib import Path
import hashlib
import shutil
import textwrap
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "processed" / "efficientnet_320_package"
ARRAY_PATH = SOURCE_DIR / "roll_320_clahe_dataset.npz"
METADATA_PATH = SOURCE_DIR / "roll_320_clahe_metadata.csv"
INSTRUCTIONS_PATH = ROOT / "relabeling" / "CLIENT_RELABELING_INSTRUCTIONS.md"
OUTPUT_DIR = ROOT / "data" / "processed" / "client_subtype_relabeling_package"
IMAGE_DIR = OUTPUT_DIR / "images"
SHEET_DIR = OUTPUT_DIR / "contact_sheets"
ZIP_PATH = ROOT / "data" / "processed" / "client_subtype_relabeling_package.zip"

PATTERNS = [
    "plain_arch",
    "tented_arch",
    "left_slant_loop",
    "right_slant_loop",
    "plain_whorl",
    "central_pocket_loop_whorl",
    "double_loop_whorl",
    "accidental_whorl",
    "unclear",
]


def build_contact_sheet(items: list[tuple[str, Image.Image]], output_path: Path) -> None:
    columns, rows = 4, 3
    tile_width, tile_height = 340, 370
    sheet = Image.new("L", (columns * tile_width, rows * tile_height), 255)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for position, (review_id, image) in enumerate(items):
        row, column = divmod(position, columns)
        x, y = column * tile_width, row * tile_height
        sheet.paste(image, (x + 10, y + 28))
        draw.text((x + 10, y + 8), review_id, fill=0, font=font)
    sheet.save(output_path, optimize=True)


def main() -> None:
    for directory in (OUTPUT_DIR, IMAGE_DIR, SHEET_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(
        METADATA_PATH,
        dtype={"subject_id": "string", "finger_position": "string"},
    )
    with np.load(ARRAY_PATH, allow_pickle=False) as prepared:
        images = prepared["images"]

    needs_review = (
        metadata["experiment_role"].ne("locked_holdout")
        & (
            metadata["broad_class"].isin(["arch", "whorl"])
            | metadata["num_pattern_labels"].gt(1)
        )
    )
    selected = metadata.loc[needs_review].copy()
    selected["source_array_index"] = selected.index
    selected = selected.reset_index(drop=True)
    selected.insert(0, "review_id", [f"R{i:04d}" for i in range(1, len(selected) + 1)])
    selected["image_file"] = [
        f"images/{review_id}_subject-{subject}_finger-{finger}.png"
        for review_id, subject, finger in zip(
            selected["review_id"], selected["subject_id"], selected["finger_position"]
        )
    ]

    for _, row in selected.iterrows():
        image = Image.fromarray(images[int(row["source_array_index"])])
        image.save(OUTPUT_DIR / row["image_file"], optimize=True)

    for start in range(0, len(selected), 12):
        page = selected.iloc[start : start + 12]
        items = [
            (row["review_id"], Image.open(OUTPUT_DIR / row["image_file"]).convert("L"))
            for _, row in page.iterrows()
        ]
        page_number = start // 12 + 1
        build_contact_sheet(items, SHEET_DIR / f"contact_sheet_{page_number:03d}.png")
        for _, image in items:
            image.close()

    template_columns = [
        "review_id",
        "image_file",
        "subject_id",
        "finger_position",
        "collection_type",
        "existing_primary_label",
        "existing_broad_class",
        "existing_subtype",
        "existing_all_pattern_labels",
        "existing_num_pattern_labels",
        "confirmed_primary_pattern",
        "confirmed_secondary_pattern",
        "confidence",
        "image_quality",
        "review_action",
        "review_notes",
        "reviewer_id",
        "review_date",
    ]
    template = pd.DataFrame(
        {
            "review_id": selected["review_id"],
            "image_file": selected["image_file"],
            "subject_id": selected["subject_id"],
            "finger_position": selected["finger_position"],
            "collection_type": selected["collection_type"],
            "existing_primary_label": selected["primary_label"],
            "existing_broad_class": selected["broad_class"],
            "existing_subtype": selected["subtype"],
            "existing_all_pattern_labels": selected["all_pattern_labels"],
            "existing_num_pattern_labels": selected["num_pattern_labels"],
            "confirmed_primary_pattern": "",
            "confirmed_secondary_pattern": "",
            "confidence": "",
            "image_quality": "",
            "review_action": "",
            "review_notes": "",
            "reviewer_id": "",
            "review_date": "",
        }
    )[template_columns]
    template.to_csv(OUTPUT_DIR / "fingerprint_relabeling_template.csv", index=False)

    guide = f"""\
PERMITTED PRIMARY/SECONDARY PATTERNS
{", ".join(PATTERNS)}

CONFIDENCE: high | medium | low
IMAGE QUALITY: good | usable | poor
REVIEW ACTION: accept | adjudicate | exclude

SHORT VISUAL GUIDE
- plain_arch: ridges flow from one side to the other with a gentle central rise.
- tented_arch: pronounced upthrust, sharp angle, or arch-like central formation.
- left_slant_loop/right_slant_loop: one recurving loop; apply the project's
  official left/right opening convention.
- plain_whorl: clear circular, spiral, or complete recurving whorl formation.
- central_pocket_loop_whorl: loop structure containing a distinct central whorl.
- double_loop_whorl: two separate loop formations forming an S-like pattern.
- accidental_whorl: combination that does not fit the other whorl definitions.
- unclear: the most specific subtype cannot be determined reliably.

Use the client's official fingerprint classification standard when it differs
from this condensed guide.
"""
    (OUTPUT_DIR / "label_guide.txt").write_text(
        textwrap.dedent(guide), encoding="utf-8"
    )
    shutil.copy2(INSTRUCTIONS_PATH, OUTPUT_DIR / "CLIENT_RELABELING_INSTRUCTIONS.md")

    manifest = (
        f"Review images: {len(selected)}\n"
        f"Subjects represented: {selected['subject_id'].nunique()}\n"
        f"Contact sheets: {(len(selected) + 11) // 12}\n"
        f"Locked-holdout images included: 0\n"
        f"Cross-validation-role images: {(selected['experiment_role'] == 'cross_validation').sum()}\n"
        f"Previously-evaluated-role images: {(selected['experiment_role'] == 'previously_evaluated').sum()}\n"
        f"Multi-pattern images: {(selected['num_pattern_labels'] > 1).sum()}\n"
    )
    (OUTPUT_DIR / "package_manifest.txt").write_text(manifest, encoding="utf-8")

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUT_DIR))

    digest = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
    print(manifest, end="")
    print(f"ZIP: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024**2:.2f} MB)")
    print("SHA256:", digest)


if __name__ == "__main__":
    main()
