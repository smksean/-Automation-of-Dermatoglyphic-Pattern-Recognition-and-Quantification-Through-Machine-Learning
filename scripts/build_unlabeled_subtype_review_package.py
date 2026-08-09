"""Build a private review package for generic arch and whorl subtypes."""

from __future__ import annotations

import hashlib
import shutil
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "processed" / "efficientnet_320_package"
ARRAY_PATH = SOURCE_DIR / "roll_320_clahe_dataset.npz"
METADATA_PATH = SOURCE_DIR / "roll_320_clahe_metadata.csv"
INSTRUCTIONS_PATH = ROOT / "relabeling" / "UNLABELED_SUBTYPE_REVIEW_INSTRUCTIONS.md"

PACKAGE_NAME = "unlabeled_subtype_review_package_2026-08-04"
OUTPUT_DIR = ROOT / "data" / "processed" / PACKAGE_NAME
ZIP_PATH = ROOT / "data" / "processed" / f"{PACKAGE_NAME}.zip"
ZIP_DIGEST_PATH = ROOT / "data" / "processed" / f"{PACKAGE_NAME}.zip.sha256"

BROAD_MAP = {
    "AU": "arch",
    "AU+PA": "arch",
    "AU+TA": "arch",
    "LS": "left_slant_loop",
    "RS": "right_slant_loop",
    "WU": "whorl",
    "WU+PW": "whorl",
    "WU+CP": "whorl",
    "WU+DL": "whorl",
    "WU+AW": "whorl",
}

SUBTYPE_MAP = {
    "AU": "arch",
    "AU+PA": "plain_arch",
    "AU+TA": "tented_arch",
    "LS": "left_slant_loop",
    "RS": "right_slant_loop",
    "WU": "whorl",
    "WU+PW": "plain_whorl",
    "WU+CP": "central_pocket_loop_whorl",
    "WU+DL": "double_loop_whorl",
    "WU+AW": "accidental_whorl",
}

ALLOWED_SUBTYPES = {
    "arch": ["plain_arch", "tented_arch", "unclear"],
    "whorl": [
        "plain_whorl",
        "central_pocket_loop_whorl",
        "double_loop_whorl",
        "accidental_whorl",
        "unclear",
    ],
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "arialbd.ttf")
        if bold
        else ("DejaVuSans.ttf", "arial.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def centered_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    width: int,
    fill: int = 0,
) -> None:
    draw.text(((width - text_width(draw, text, font)) // 2, y), text, font=font, fill=fill)


def make_review_image(
    image: np.ndarray,
    review_id: str,
    main_type: str,
    alternative_warning: bool,
) -> Image.Image:
    width, height = 420, 448
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(22, bold=True)
    body_font = load_font(17)
    warning_font = load_font(14, bold=True)

    centered_text(draw, 10, f"{review_id}  |  MAIN TYPE: {main_type.upper()}", header_font, width)
    fingerprint = Image.fromarray(image).convert("L")
    canvas.paste(fingerprint, ((width - fingerprint.width) // 2, 48))
    draw.rectangle((49, 47, 370, 368), outline=190, width=1)
    centered_text(draw, 378, "NEEDS SUBTYPE LABEL", header_font, width)

    if alternative_warning:
        centered_text(
            draw,
            414,
            "WARNING: ALTERNATIVE MAIN-TYPE CODE RECORDED",
            warning_font,
            width,
        )
    else:
        centered_text(draw, 415, "Use the permitted subtype list in the CSV", body_font, width)

    draw.rectangle((0, 0, width - 1, height - 1), outline=180, width=1)
    return canvas


def build_contact_sheet(items: list[tuple[str, Image.Image]], output_path: Path) -> None:
    columns, rows = 4, 3
    tile_width, tile_height = 420, 448
    sheet = Image.new("L", (columns * tile_width, rows * tile_height), 255)
    for position, (_, image) in enumerate(items):
        row, column = divmod(position, columns)
        sheet.paste(image, (column * tile_width, row * tile_height))
    sheet.save(output_path, format="PNG", optimize=True)


def record_key(row: pd.Series) -> str:
    source = "|".join(
        str(row[column])
        for column in ("subject_id", "finger_position", "collection_type", "png_path")
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def mapped_broad_classes(pattern_codes: str) -> set[str]:
    return {
        BROAD_MAP.get(code, "unknown")
        for code in str(pattern_codes).split("|")
        if code
    }


def validate_metadata(metadata: pd.DataFrame) -> dict[str, int]:
    expected_broad = metadata["primary_label"].map(BROAD_MAP)
    expected_subtype = metadata["primary_label"].map(SUBTYPE_MAP)
    missing_pngs = sum(
        not (ROOT / Path(str(path).replace("\\", "/"))).is_file()
        for path in metadata["png_path"]
    )
    duplicate_rows = int(
        metadata.duplicated(
            ["subject_id", "finger_position", "collection_type", "png_path"]
        ).sum()
    )
    audit = {
        "rows": len(metadata),
        "subjects": metadata["subject_id"].nunique(),
        "unknown_primary_codes": int(expected_broad.isna().sum()),
        "broad_mapping_mismatches": int((expected_broad != metadata["broad_class"]).sum()),
        "subtype_mapping_mismatches": int((expected_subtype != metadata["subtype"]).sum()),
        "missing_source_pngs": missing_pngs,
        "duplicate_metadata_rows": duplicate_rows,
        "single_entry_records": int(metadata["num_pattern_labels"].eq(1).sum()),
        "multiple_entry_records": int(metadata["num_pattern_labels"].gt(1).sum()),
    }
    required_zero = (
        "unknown_primary_codes",
        "broad_mapping_mismatches",
        "subtype_mapping_mismatches",
        "missing_source_pngs",
        "duplicate_metadata_rows",
    )
    failures = {key: audit[key] for key in required_zero if audit[key] != 0}
    if failures:
        raise ValueError(f"Metadata audit failed: {failures}")
    return audit


def write_label_guide(path: Path) -> None:
    guide = """\
ARCH SUBTYPES
- plain_arch: ridges flow from one side to the other with a gentle central rise.
- tented_arch: a pronounced upthrust, sharp angle, or arch-like central formation.
- unclear: the exact arch subtype cannot be determined reliably.

WHORL SUBTYPES
- plain_whorl: one or more ridges form or tend to form a complete circuit; the
  standard imaginary line between the two deltas touches or crosses an inner
  recurving ridge.
- central_pocket_loop_whorl: a central whorl lies within a loop-like formation;
  the standard imaginary line between the two deltas does not touch or cross an
  inner recurving ridge.
- double_loop_whorl: two separate loop formations with distinct shoulders,
  cores, and deltas, often forming an S-like pattern.
- accidental_whorl: a combination of pattern types or a whorl that does not fit
  the other defined whorl subtypes.
- unclear: the exact whorl subtype cannot be determined reliably.

CONFIDENCE: high | medium | low
IMAGE QUALITY: good | usable | poor
REVIEW ACTION: accept | adjudicate | exclude
MAIN TYPE ISSUE: blank | incorrect | uncertain

When the main type appears wrong or uncertain, use confirmed_subtype=unclear,
set review_action=adjudicate, and explain the issue in review_notes.

Reference:
https://tsapps.nist.gov/trainingtool/FundamentalAFISSearching/PatternClassification.html
"""
    path.write_text(textwrap.dedent(guide), encoding="utf-8")


def main() -> None:
    if OUTPUT_DIR.exists() or ZIP_PATH.exists() or ZIP_DIGEST_PATH.exists():
        raise FileExistsError(
            f"Package output already exists. Move or remove it before rebuilding: {OUTPUT_DIR}"
        )

    metadata = pd.read_csv(
        METADATA_PATH,
        dtype={"subject_id": "string", "finger_position": "string"},
    )
    audit = validate_metadata(metadata)

    metadata = metadata.copy()
    metadata["source_array_index"] = metadata.index
    metadata["mapped_broad_classes"] = metadata["all_pattern_labels"].map(
        mapped_broad_classes
    )
    metadata["alternative_type_warning"] = metadata["mapped_broad_classes"].map(
        lambda values: "yes" if len(values) > 1 else "no"
    )

    generic_mask = (
        ((metadata["broad_class"] == "arch") & (metadata["subtype"] == "arch"))
        | ((metadata["broad_class"] == "whorl") & (metadata["subtype"] == "whorl"))
    )
    all_generic = metadata.loc[generic_mask].copy()
    locked_generic = all_generic.loc[
        all_generic["experiment_role"] == "locked_holdout"
    ].copy()
    selected = all_generic.loc[
        all_generic["experiment_role"] != "locked_holdout"
    ].copy()
    selected = selected.sort_values(
        ["broad_class", "subject_id", "finger_position", "collection_type"]
    ).reset_index(drop=True)

    expected_selected = len(all_generic) - len(locked_generic)
    if len(selected) != expected_selected:
        raise AssertionError("Selected image count does not match the subtype-gap audit.")

    OUTPUT_DIR.mkdir(parents=True)
    for broad_class in ALLOWED_SUBTYPES:
        (OUTPUT_DIR / "images" / broad_class).mkdir(parents=True)
        (OUTPUT_DIR / "contact_sheets" / broad_class).mkdir(parents=True)

    counters = {"arch": 0, "whorl": 0}
    review_ids = []
    image_files = []
    record_keys = []
    allowed_values = []
    for _, row in selected.iterrows():
        broad_class = row["broad_class"]
        counters[broad_class] += 1
        prefix = "A" if broad_class == "arch" else "W"
        review_id = f"{prefix}{counters[broad_class]:04d}"
        review_ids.append(review_id)
        image_files.append(
            f"images/{broad_class}/{review_id}_{broad_class}_needs_subtype.png"
        )
        record_keys.append(record_key(row))
        allowed_values.append("|".join(ALLOWED_SUBTYPES[broad_class]))

    selected.insert(0, "review_id", review_ids)
    selected.insert(1, "record_key", record_keys)
    selected["image_file"] = image_files
    selected["permitted_subtypes"] = allowed_values

    with np.load(ARRAY_PATH, allow_pickle=False) as prepared:
        images = prepared["images"]
        if len(images) != len(metadata):
            raise ValueError(
                f"Image array has {len(images)} rows but metadata has {len(metadata)} rows."
            )
        for _, row in selected.iterrows():
            review_image = make_review_image(
                images[int(row["source_array_index"])],
                row["review_id"],
                row["broad_class"],
                row["alternative_type_warning"] == "yes",
            )
            review_image.save(
                OUTPUT_DIR / row["image_file"], format="PNG", optimize=True
            )

    for broad_class in ALLOWED_SUBTYPES:
        class_rows = selected.loc[selected["broad_class"] == broad_class]
        for start in range(0, len(class_rows), 12):
            page = class_rows.iloc[start : start + 12]
            items = []
            for _, row in page.iterrows():
                path = OUTPUT_DIR / row["image_file"]
                with Image.open(path) as image:
                    items.append((row["review_id"], image.convert("L").copy()))
            page_number = start // 12 + 1
            build_contact_sheet(
                items,
                OUTPUT_DIR
                / "contact_sheets"
                / broad_class
                / f"{broad_class}_contact_sheet_{page_number:03d}.png",
            )
            for _, image in items:
                image.close()

    template = pd.DataFrame(
        {
            "review_id": selected["review_id"],
            "record_key": selected["record_key"],
            "image_file": selected["image_file"],
            "current_main_type": selected["broad_class"],
            "source_primary_code": selected["primary_label"],
            "recorded_pattern_codes": selected["all_pattern_labels"],
            "alternative_type_warning": selected["alternative_type_warning"],
            "permitted_subtypes": selected["permitted_subtypes"],
            "confirmed_subtype": "",
            "confidence": "",
            "image_quality": "",
            "review_action": "",
            "main_type_issue": "",
            "review_notes": "",
            "reviewer_id": "",
            "review_date": "",
        }
    )
    template.to_csv(OUTPUT_DIR / "subtype_labeling_template.csv", index=False)

    shutil.copy2(INSTRUCTIONS_PATH, OUTPUT_DIR / "START_HERE_INSTRUCTIONS.md")
    write_label_guide(OUTPUT_DIR / "label_guide.txt")

    selected_warning_count = int(selected["alternative_type_warning"].eq("yes").sum())
    manifest = (
        f"Package purpose: generic arch and whorl subtype review\n"
        f"Source modelling images: {len(metadata)}\n"
        f"All generic arch/whorl subtype gaps: {len(all_generic)}\n"
        f"Review images included: {len(selected)}\n"
        f"  arch: {(selected['broad_class'] == 'arch').sum()}\n"
        f"  whorl: {(selected['broad_class'] == 'whorl').sum()}\n"
        f"Subjects represented: {selected['subject_id'].nunique()}\n"
        f"Alternative-main-type warnings: {selected_warning_count}\n"
        f"Locked-holdout images included: 0\n"
        f"Locked-holdout generic images excluded: {len(locked_generic)}\n"
        f"  excluded arch: {(locked_generic['broad_class'] == 'arch').sum()}\n"
        f"  excluded whorl: {(locked_generic['broad_class'] == 'whorl').sum()}\n"
        f"Contact sheets: {sum((len(selected[selected['broad_class'] == c]) + 11) // 12 for c in ALLOWED_SUBTYPES)}\n"
    )
    (OUTPUT_DIR / "package_manifest.txt").write_text(manifest, encoding="utf-8")

    cross_broad_all = int(
        metadata["mapped_broad_classes"].map(lambda values: len(values) > 1).sum()
    )
    audit_text = (
        "BROAD-LABEL AUDIT\n"
        "=================\n"
        "The current main labels were extracted from the first examiner-supplied\n"
        "pattern entry in EBTS field 9.307; they were not model predictions.\n\n"
        "Verified code mapping:\n"
        "  AU, AU+PA, AU+TA -> arch\n"
        "  LS -> left_slant_loop\n"
        "  RS -> right_slant_loop\n"
        "  WU, WU+PW, WU+CP, WU+DL, WU+AW -> whorl\n\n"
        f"Metadata rows audited: {audit['rows']}\n"
        f"Subjects: {audit['subjects']}\n"
        f"Unknown primary codes: {audit['unknown_primary_codes']}\n"
        f"Broad mapping mismatches: {audit['broad_mapping_mismatches']}\n"
        f"Subtype mapping mismatches: {audit['subtype_mapping_mismatches']}\n"
        f"Missing linked source PNGs: {audit['missing_source_pngs']}\n"
        f"Duplicate metadata rows: {audit['duplicate_metadata_rows']}\n"
        f"Single examiner-entry records: {audit['single_entry_records']}\n"
        f"Multiple examiner-entry records: {audit['multiple_entry_records']}\n"
        f"Records with alternatives spanning broad types: {cross_broad_all}\n"
        f"Such warnings in this review package: {selected_warning_count}\n\n"
        "Conclusion: the extraction, file linkage, and code-to-main-type mapping\n"
        "are internally consistent. This audit does not independently confirm the\n"
        "visual pattern of every fingerprint. Warning records require conservative\n"
        "expert review and adjudication when the displayed main type is doubtful.\n"
    )
    (OUTPUT_DIR / "broad_label_audit.txt").write_text(audit_text, encoding="utf-8")

    files_to_hash = sorted(
        path for path in OUTPUT_DIR.rglob("*") if path.is_file()
    )
    checksum_lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(OUTPUT_DIR).as_posix()}"
        for path in files_to_hash
    ]
    (OUTPUT_DIR / "CHECKSUMS_SHA256.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUT_DIR))

    zip_digest = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
    ZIP_DIGEST_PATH.write_text(
        f"{zip_digest}  {ZIP_PATH.name}\n", encoding="utf-8"
    )

    print(manifest, end="")
    print(f"ZIP: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024**2:.2f} MB)")
    print(f"SHA256: {zip_digest}")


if __name__ == "__main__":
    main()
