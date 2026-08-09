"""Validation and CSV-export logic shared by the Streamlit application."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping
from typing import Any


ALLOWED_SUBTYPES = {
    "arch": ("plain_arch", "tented_arch", "unclear"),
    "whorl": (
        "plain_whorl",
        "central_pocket_loop_whorl",
        "double_loop_whorl",
        "accidental_whorl",
        "unclear",
    ),
}

CONFIDENCE_VALUES = ("high", "medium", "low")
ACTION_VALUES = ("accept", "adjudicate", "exclude")
MAIN_TYPE_ISSUE_VALUES = ("", "incorrect", "uncertain")

REVIEW_FIELDS = (
    "confirmed_subtype",
    "confidence",
    "review_action",
    "main_type_issue",
    "review_notes",
    "reviewer_id",
    "reviewed_at",
    "revision",
)


def validate_annotation(
    item: Mapping[str, Any], annotation: Mapping[str, Any]
) -> list[str]:
    """Return all validation errors for a proposed subtype annotation."""
    errors: list[str] = []
    main_type = str(item.get("current_main_type", ""))
    allowed = tuple(item.get("permitted_subtypes") or ALLOWED_SUBTYPES.get(main_type, ()))
    subtype = str(annotation.get("confirmed_subtype", ""))
    confidence = str(annotation.get("confidence", ""))
    action = str(annotation.get("review_action", ""))
    main_type_issue = str(annotation.get("main_type_issue", ""))
    notes = str(annotation.get("review_notes", "")).strip()
    reviewer_id = str(annotation.get("reviewer_id", "")).strip()

    if subtype not in allowed:
        errors.append(f"Select a permitted {main_type} subtype.")
    if confidence not in CONFIDENCE_VALUES:
        errors.append("Select confidence: high, medium, or low.")
    if action not in ACTION_VALUES:
        errors.append("Select a review action: accept, adjudicate, or exclude.")
    if main_type_issue not in MAIN_TYPE_ISSUE_VALUES:
        errors.append("Main-type issue must be blank, incorrect, or uncertain.")
    if not reviewer_id:
        errors.append("A reviewer ID is required.")

    if subtype == "unclear" and action == "accept":
        errors.append("An unclear subtype must be adjudicated or excluded.")
    if main_type_issue:
        if subtype != "unclear":
            errors.append("Use subtype 'unclear' when the main type is disputed.")
        if action != "adjudicate":
            errors.append("A main-type issue must be sent for adjudication.")
        if not notes:
            errors.append("Explain the main-type issue in the review notes.")
    if action in {"adjudicate", "exclude"} and not notes:
        errors.append("Add review notes for adjudicated or excluded images.")

    return errors


def annotation_is_complete(annotation: Mapping[str, Any] | None) -> bool:
    """Return whether a stored annotation has the required completed fields."""
    if not annotation:
        return False
    return all(
        str(annotation.get(field, "")).strip()
        for field in (
            "confirmed_subtype",
            "confidence",
            "review_action",
            "reviewer_id",
        )
    )


def progress_counts(
    items: Iterable[Mapping[str, Any]],
    annotations_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Calculate total, completed, pending, warning, and action counts."""
    item_list = list(items)
    completed = sum(
        annotation_is_complete(annotations_by_id.get(str(item["review_id"])))
        for item in item_list
    )
    return {
        "total": len(item_list),
        "completed": completed,
        "pending": len(item_list) - completed,
        "warnings": sum(bool(item.get("alternative_type_warning")) for item in item_list),
        "adjudicate": sum(
            annotation.get("review_action") == "adjudicate"
            for annotation in annotations_by_id.values()
        ),
        "excluded": sum(
            annotation.get("review_action") == "exclude"
            for annotation in annotations_by_id.values()
        ),
    }


def build_export_csv(
    items: Iterable[Mapping[str, Any]],
    annotations_by_id: Mapping[str, Mapping[str, Any]],
) -> bytes:
    """Build the current reviewer CSV in stable review-ID order."""
    fieldnames = (
        "review_id",
        "record_key",
        "image_path",
        "current_main_type",
        "source_primary_code",
        "recorded_pattern_codes",
        "alternative_type_warning",
        "permitted_subtypes",
        *REVIEW_FIELDS,
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    for item in sorted(items, key=lambda row: str(row["review_id"])):
        annotation = annotations_by_id.get(str(item["review_id"]), {})
        permitted = item.get("permitted_subtypes", ())
        if isinstance(permitted, (list, tuple)):
            permitted = "|".join(str(value) for value in permitted)
        row = {
            "review_id": item.get("review_id", ""),
            "record_key": item.get("record_key", ""),
            "image_path": item.get("image_path", ""),
            "current_main_type": item.get("current_main_type", ""),
            "source_primary_code": item.get("source_primary_code", ""),
            "recorded_pattern_codes": item.get("recorded_pattern_codes", ""),
            "alternative_type_warning": "yes"
            if item.get("alternative_type_warning")
            else "no",
            "permitted_subtypes": permitted,
        }
        row.update({field: annotation.get(field, "") for field in REVIEW_FIELDS})
        writer.writerow(row)

    return buffer.getvalue().encode("utf-8-sig")


def filtered_items(
    items: Iterable[Mapping[str, Any]],
    annotations_by_id: Mapping[str, Mapping[str, Any]],
    main_type: str = "all",
    status: str = "pending",
    warnings_only: bool = False,
) -> list[Mapping[str, Any]]:
    """Filter review items without changing their stable ordering."""
    result = []
    for item in sorted(items, key=lambda row: str(row["review_id"])):
        review_id = str(item["review_id"])
        complete = annotation_is_complete(annotations_by_id.get(review_id))
        if main_type != "all" and item.get("current_main_type") != main_type:
            continue
        if status == "pending" and complete:
            continue
        if status == "completed" and not complete:
            continue
        if warnings_only and not item.get("alternative_type_warning"):
            continue
        result.append(item)
    return result
