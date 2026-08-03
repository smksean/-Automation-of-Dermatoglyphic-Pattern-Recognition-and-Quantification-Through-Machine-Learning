"""Create a small, labeled README gallery from the local training split.

The source SD 302 images and metadata remain in the ignored local data folders.
Published files contain no subject identifier in their filename or visible label.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "roll_broad_model_split.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "training_samples"

CLASS_LABELS = {
    "arch": "Arch",
    "left_slant_loop": "Left-slant loop",
    "right_slant_loop": "Right-slant loop",
    "whorl": "Whorl",
}

SAMPLES_PER_CLASS = 2
TILE_SIZE = (360, 400)
IMAGE_BOX = (320, 320)


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def select_samples() -> dict[str, list[dict[str, str]]]:
    with SPLIT_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected: dict[str, list[dict[str, str]]] = {}
    for class_name in CLASS_LABELS:
        candidates = []
        seen_subjects = set()

        for row in rows:
            if row["split"] != "train" or row["broad_class"] != class_name:
                continue
            if row["subject_id"] in seen_subjects:
                continue
            seen_subjects.add(row["subject_id"])
            candidates.append(row)

        if len(candidates) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"Expected at least {SAMPLES_PER_CLASS} training subjects for {class_name}; "
                f"found {len(candidates)}."
            )

        rng = random.Random(f"training-gallery-{class_name}-42")
        selected[class_name] = rng.sample(candidates, SAMPLES_PER_CLASS)

    return selected


def make_tile(source_path: Path, title: str) -> Image.Image:
    tile = Image.new("L", TILE_SIZE, color="white")
    draw = ImageDraw.Draw(tile)
    title_font = load_font(24)

    with Image.open(source_path) as source:
        image = ImageOps.contain(source.convert("L"), IMAGE_BOX)

    image_x = (TILE_SIZE[0] - image.width) // 2
    image_y = 58 + (IMAGE_BOX[1] - image.height) // 2
    tile.paste(image, (image_x, image_y))

    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_box[2] - title_box[0]
    draw.text(((TILE_SIZE[0] - title_width) // 2, 16), title, fill="black", font=title_font)
    draw.rectangle((0, 0, TILE_SIZE[0] - 1, TILE_SIZE[1] - 1), outline=210, width=1)
    return tile


def main() -> None:
    selected = select_samples()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tiles: list[Image.Image] = []
    for class_name, rows in selected.items():
        display_label = CLASS_LABELS[class_name]
        for sample_number, row in enumerate(rows, start=1):
            source_path = PROJECT_ROOT / Path(row["png_path"].replace("\\", "/"))
            if not source_path.is_file():
                raise FileNotFoundError(source_path)

            title = f"{display_label} — sample {sample_number}"
            tile = make_tile(source_path, title)
            output_path = OUTPUT_DIR / f"{class_name}_{sample_number:02d}.png"
            tile.save(output_path, format="PNG", optimize=True)
            tiles.append(tile)

    gallery = Image.new(
        "L",
        (TILE_SIZE[0] * SAMPLES_PER_CLASS, TILE_SIZE[1] * len(CLASS_LABELS)),
        color="white",
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, SAMPLES_PER_CLASS)
        gallery.paste(tile, (column * TILE_SIZE[0], row * TILE_SIZE[1]))

    gallery.save(OUTPUT_DIR / "training_samples_gallery.png", format="PNG", optimize=True)
    print(f"Created {len(tiles)} labeled samples and one gallery in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
