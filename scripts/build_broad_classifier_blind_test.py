"""Build a neutral-filename test pack for the broad-classification web app."""

from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path
import random
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "processed" / "roll_broad_model_split.csv"
OUTPUT_DIRECTORY = ROOT / "data" / "broad_classifier_blind_test"
IMAGE_DIRECTORY = OUTPUT_DIRECTORY / "images"
ANSWER_DIRECTORY = OUTPUT_DIRECTORY / "answer_key"
CLASS_NAMES = (
    "arch",
    "left_slant_loop",
    "right_slant_loop",
    "whorl",
)
SAMPLES_PER_CLASS = 5
SEED = 20260816


def read_rows() -> list[dict[str, str]]:
    with SOURCE_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        raise RuntimeError("The fixed subject-disjoint test split is empty.")
    return test_rows


def select_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select five images per class from different subjects when possible."""
    rng = random.Random(SEED)
    selected: list[dict[str, str]] = []
    used_subjects: set[str] = set()

    for class_name in CLASS_NAMES:
        candidates = sorted(
            (row for row in rows if row["broad_class"] == class_name),
            key=lambda row: (
                row["subject_id"],
                row["finger_position"],
                row["collection_type"],
            ),
        )
        rng.shuffle(candidates)
        class_selection = []
        for row in candidates:
            if row["subject_id"] in used_subjects:
                continue
            class_selection.append(row)
            used_subjects.add(row["subject_id"])
            if len(class_selection) == SAMPLES_PER_CLASS:
                break
        if len(class_selection) != SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Could not select {SAMPLES_PER_CLASS} unique-subject {class_name} images."
            )
        selected.extend(class_selection)

    rng.shuffle(selected)
    return selected


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(
            "The blind test folder already exists. Move it before rebuilding: "
            f"{OUTPUT_DIRECTORY}"
        )

    selected = select_rows(read_rows())
    IMAGE_DIRECTORY.mkdir(parents=True)
    ANSWER_DIRECTORY.mkdir(parents=True)

    answer_rows: list[dict[str, str]] = []
    result_rows: list[dict[str, str]] = []
    checksum_lines: list[str] = []
    for number, source_row in enumerate(selected, start=1):
        sample_id = f"sample_{number:03d}"
        file_name = f"{sample_id}.png"
        source_path = ROOT / Path(source_row["png_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = IMAGE_DIRECTORY / file_name
        shutil.copyfile(source_path, destination)

        answer_rows.append(
            {
                "sample_id": sample_id,
                "file_name": file_name,
                "expected_broad_class": source_row["broad_class"],
            }
        )
        result_rows.append(
            {
                "sample_id": sample_id,
                "file_name": file_name,
                "model_prediction": "",
                "model_probability": "",
                "fold_agreement": "",
                "top_two_margin": "",
                "review_notes": "",
            }
        )
        checksum_lines.append(f"{sha256(destination.read_bytes()).hexdigest()}  {file_name}")

    write_csv(
        ANSWER_DIRECTORY / "DO_NOT_OPEN_UNTIL_FINISHED.csv",
        ["sample_id", "file_name", "expected_broad_class"],
        answer_rows,
    )
    write_csv(
        OUTPUT_DIRECTORY / "record_your_results.csv",
        [
            "sample_id",
            "file_name",
            "model_prediction",
            "model_probability",
            "fold_agreement",
            "top_two_margin",
            "review_notes",
        ],
        result_rows,
    )
    (OUTPUT_DIRECTORY / "CHECKSUMS_SHA256.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIRECTORY / "START_HERE.txt").write_text(
        "BROAD CLASSIFIER BLIND TEST\n"
        "===========================\n\n"
        "1. Open the images folder.\n"
        "2. Upload each neutral-named PNG to the broad classifier.\n"
        "3. Record the displayed result in record_your_results.csv.\n"
        "4. Do not open the answer_key folder until all predictions are recorded.\n"
        "5. Compare the predictions with DO_NOT_OPEN_UNTIL_FINISHED.csv.\n\n"
        "The pack contains 20 images: five from each broad class. Images come\n"
        "from the earlier fixed subject-disjoint test set and were excluded from\n"
        "the EfficientNet cross-validation training cohort. The final locked\n"
        "holdout is not included. Image pixels are copied from the raw source\n"
        "files; no labels, captions, or overlays are added. Filenames do not\n"
        "reveal the expected class.\n",
        encoding="utf-8",
    )

    print(f"Blind test images: {len(selected)}")
    print(f"Distinct subjects: {len({row['subject_id'] for row in selected})}")
    for class_name in CLASS_NAMES:
        print(
            f"{class_name}: "
            f"{sum(row['broad_class'] == class_name for row in selected)}"
        )
    print("Locked holdout images: 0")
    print(f"Output: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
