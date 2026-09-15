"""Build subject-safe aggregate quantification tables from restored SD302 EFS data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESTORE_DIR = ROOT / "data" / "processed" / "sd302_2026_restoration"
PRIVATE_DIR = RESTORE_DIR / "quantification_reference"
RESULTS_DIR = ROOT / "results" / "quantification_reference"
FINGER_POSITIONS = {f"{position:02d}" for position in range(1, 11)}
CLASSIFIABLE_PATTERNS = {"arch", "left_slant_loop", "right_slant_loop", "whorl"}
LOOP_PATTERNS = {"left_slant_loop", "right_slant_loop"}


def pattern_intensity_index(patterns: list[str]) -> int:
    """Return the ten-finger PII: loop count plus twice the whorl count."""
    if len(patterns) != 10 or any(pattern not in CLASSIFIABLE_PATTERNS for pattern in patterns):
        raise ValueError("PII requires exactly ten classifiable finger patterns")
    return sum(pattern in LOOP_PATTERNS for pattern in patterns) + 2 * patterns.count("whorl")


def select_canonical_impressions(inventory: pd.DataFrame) -> pd.DataFrame:
    """Select the uniform 1000-ppi baseline V rolled-impression series."""
    canonical = inventory.loc[
        (inventory["collection_type"] == "baseline")
        & (inventory["device"] == "V")
        & (inventory["resolution"] == 1000)
        & (inventory["capture_type"] == "roll")
    ].copy()
    duplicates = canonical.duplicated(["subject_id", "finger_position"], keep=False)
    if duplicates.any():
        raise ValueError("Canonical source contains duplicate subject/finger records")
    return canonical.sort_values(["subject_id", "finger_position"]).reset_index(drop=True)


def describe(values: pd.Series, measure: str) -> dict[str, object]:
    values = values.dropna().astype(float)
    return {
        "measure": measure,
        "n": int(values.size),
        "mean": round(float(values.mean()), 3),
        "std": round(float(values.std(ddof=1)), 3),
        "min": round(float(values.min()), 3),
        "q1": round(float(values.quantile(0.25)), 3),
        "median": round(float(values.median()), 3),
        "q3": round(float(values.quantile(0.75)), 3),
        "max": round(float(values.max()), 3),
    }


def build_subject_table(canonical: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_id, group in canonical.groupby("subject_id", sort=True):
        positions = set(group["finger_position"])
        patterns = group["broad_class"].tolist()
        complete_images = positions == FINGER_POSITIONS
        pii_eligible = complete_images and all(pattern in CLASSIFIABLE_PATTERNS for pattern in patterns)
        rows.append(
            {
                "subject_id": subject_id,
                "finger_count": int(group["finger_position"].nunique()),
                "classifiable_finger_count": int(group["broad_class"].isin(CLASSIFIABLE_PATTERNS).sum()),
                "complete_ten_finger_images": complete_images,
                "pattern_intensity_eligible": pii_eligible,
                "pattern_intensity_index": pattern_intensity_index(patterns) if pii_eligible else np.nan,
                "arch_count": int((group["broad_class"] == "arch").sum()),
                "loop_count": int(group["broad_class"].isin(LOOP_PATTERNS).sum()),
                "whorl_count": int((group["broad_class"] == "whorl").sum()),
                "unclassifiable_count": int((group["broad_class"] == "unclassifiable").sum()),
                "total_minutiae": int(group["minutiae_count"].sum()) if complete_images else np.nan,
                "total_ridge_endings": int(group["ridge_ending_count"].sum()) if complete_images else np.nan,
                "total_bifurcations": int(group["bifurcation_count"].sum()) if complete_images else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_repeat_agreement(inventory: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    reference = canonical[["subject_id", "finger_position", "broad_class", "minutiae_count"]].rename(
        columns={"broad_class": "reference_pattern", "minutiae_count": "reference_minutiae"}
    )
    alternatives = inventory.loc[inventory["device"] != "V"].merge(
        reference, on=["subject_id", "finger_position"], how="inner"
    )
    alternatives["absolute_minutiae_difference"] = (
        alternatives["minutiae_count"] - alternatives["reference_minutiae"]
    ).abs()
    rows: list[dict[str, object]] = []
    for device, group in [("all", alternatives), *alternatives.groupby("device", sort=True)]:
        rows.append(
            {
                "comparison_device": device,
                "paired_impressions": int(len(group)),
                "broad_pattern_agreement_pct": round(
                    float((group["broad_class"] == group["reference_pattern"]).mean() * 100), 2
                ),
                "minutiae_count_mae": round(float(group["absolute_minutiae_difference"].mean()), 3),
                "minutiae_count_median_absolute_difference": round(
                    float(group["absolute_minutiae_difference"].median()), 3
                ),
                "minutiae_count_spearman": round(
                    float(group["minutiae_count"].corr(group["reference_minutiae"], method="spearman")), 3
                ),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(inventory_path: Path, private_dir: Path, results_dir: Path) -> dict[str, object]:
    inventory = pd.read_csv(
        inventory_path,
        dtype={"subject_id": str, "finger_position": str},
    )
    canonical = select_canonical_impressions(inventory)
    subjects = build_subject_table(canonical)
    complete_subjects = subjects.loc[subjects["complete_ten_finger_images"]]
    pii_subjects = subjects.loc[subjects["pattern_intensity_eligible"]]

    private_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(private_dir / "canonical_finger_quantification.csv", index=False)
    subjects.to_csv(private_dir / "subject_quantification.csv", index=False)

    coverage = pd.DataFrame(
        [
            {"measure": "restored subjects", "value": inventory["subject_id"].nunique()},
            {"measure": "canonical baseline V impressions", "value": len(canonical)},
            {"measure": "subjects with ten canonical images", "value": len(complete_subjects)},
            {"measure": "subjects eligible for pattern intensity", "value": len(pii_subjects)},
            {"measure": "canonical unclassifiable impressions", "value": int((canonical["broad_class"] == "unclassifiable").sum())},
        ]
    )
    coverage.to_csv(results_dir / "coverage_summary.csv", index=False)

    pattern_distribution = (
        canonical.groupby("broad_class", dropna=False)
        .size()
        .rename("images")
        .reset_index()
    )
    pattern_distribution["percent"] = (100 * pattern_distribution["images"] / len(canonical)).round(2)
    pattern_distribution.to_csv(results_dir / "canonical_pattern_distribution.csv", index=False)

    pd.DataFrame([describe(pii_subjects["pattern_intensity_index"], "pattern intensity index")]).to_csv(
        results_dir / "pattern_intensity_summary.csv", index=False
    )
    pii_distribution = (
        pii_subjects["pattern_intensity_index"]
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("pattern_intensity_index")
        .rename("subjects")
        .reset_index()
    )
    pii_distribution.to_csv(results_dir / "pattern_intensity_distribution.csv", index=False)
    pd.DataFrame(
        [
            describe(complete_subjects["total_minutiae"], "ten-finger examiner minutiae total"),
            describe(complete_subjects["total_ridge_endings"], "ten-finger ridge-ending total"),
            describe(complete_subjects["total_bifurcations"], "ten-finger bifurcation total"),
        ]
    ).to_csv(results_dir / "ten_finger_minutiae_summary.csv", index=False)

    minutiae_by_pattern = canonical.groupby("broad_class")["minutiae_count"].agg(
        images="size", mean="mean", std="std", median="median", minimum="min", maximum="max"
    ).reset_index()
    for column in ["mean", "std", "median"]:
        minutiae_by_pattern[column] = minutiae_by_pattern[column].round(3)
    minutiae_by_pattern.to_csv(results_dir / "minutiae_by_pattern.csv", index=False)

    repeat_agreement = build_repeat_agreement(inventory, canonical)
    repeat_agreement.to_csv(results_dir / "repeat_impression_agreement.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_source": "SD302b baseline device V, 1000 ppi, rolled impressions",
        "pattern_intensity_formula": "loop_count + 2 * whorl_count across ten classifiable fingers",
        "subject_level_outputs": str(private_dir.relative_to(ROOT)).replace("\\", "/"),
        "public_output_scope": "aggregate values only",
        "ridge_count_status": "not a primary reported endpoint in this phase; SD302g has no validated expert field 9.322 ridge-count annotations",
        "canonical_impressions": int(len(canonical)),
        "complete_ten_finger_subjects": int(len(complete_subjects)),
        "pattern_intensity_subjects": int(len(pii_subjects)),
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=RESTORE_DIR / "irr_feature_inventory.csv")
    parser.add_argument("--private-dir", type=Path, default=PRIVATE_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    print(json.dumps(write_outputs(args.inventory, args.private_dir, args.results_dir), indent=2))


if __name__ == "__main__":
    main()
