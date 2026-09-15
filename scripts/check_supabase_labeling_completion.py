"""Check completion status for the Supabase subtype-review labels."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


def _get_json(base_url: str, key: str, path: str) -> list[dict[str, Any]]:
    request = Request(
        base_url.rstrip("/") + path,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "User-Agent": "codex-local-supabase-audit/1.0",
        },
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _complete(annotation: dict[str, Any] | None) -> bool:
    if not annotation:
        return False
    return all(
        str(annotation.get(field, "") or "").strip()
        for field in (
            "confirmed_subtype",
            "confidence",
            "review_action",
            "reviewer_id",
        )
    )


def _validate(item: dict[str, Any], annotation: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    allowed = item.get("permitted_subtypes") or []
    subtype = annotation.get("confirmed_subtype")
    confidence = annotation.get("confidence")
    action = annotation.get("review_action")
    main_type_issue = annotation.get("main_type_issue")
    notes = str(annotation.get("review_notes") or "").strip()

    if subtype not in allowed:
        errors.append("subtype not permitted")
    if confidence not in {"high", "medium", "low"}:
        errors.append("bad confidence")
    if action not in {"accept", "adjudicate", "exclude"}:
        errors.append("bad action")
    if main_type_issue and main_type_issue not in {"incorrect", "uncertain"}:
        errors.append("bad main-type issue")
    if subtype == "unclear" and action == "accept":
        errors.append("unclear accepted")
    if main_type_issue:
        if subtype != "unclear":
            errors.append("main issue without unclear")
        if action != "adjudicate":
            errors.append("main issue not adjudicate")
        if not notes:
            errors.append("main issue lacks notes")
    if action in {"adjudicate", "exclude"} and not notes:
        errors.append("action lacks notes")
    return errors


def _print_counts(label: str, rows: list[dict[str, Any]], field: str) -> None:
    counts = Counter(str(row.get(field, "")) for row in rows)
    print(f"{label}:")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


def main() -> None:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the current shell."
        )

    item_select = quote(
        "review_id,record_key,image_path,current_main_type,source_primary_code,"
        "recorded_pattern_codes,alternative_type_warning,permitted_subtypes",
        safe=",",
    )
    items = _get_json(
        url, key, f"/rest/v1/review_items?select={item_select}&order=review_id.asc"
    )
    annotations = _get_json(
        url, key, "/rest/v1/annotations?select=*&order=review_id.asc"
    )
    events = _get_json(
        url, key, "/rest/v1/annotation_events?select=review_id,revision"
    )

    items_by_id = {str(item["review_id"]): item for item in items}
    annotations_by_id = {
        str(annotation["review_id"]): annotation for annotation in annotations
    }
    completed_ids = {
        review_id
        for review_id, annotation in annotations_by_id.items()
        if _complete(annotation)
    }
    pending_ids = sorted(set(items_by_id) - completed_ids)
    orphan_ids = sorted(set(annotations_by_id) - set(items_by_id))
    invalid = [
        (review_id, errors)
        for review_id, annotation in sorted(annotations_by_id.items())
        if review_id in items_by_id
        for errors in [_validate(items_by_id[review_id], annotation)]
        if errors
    ]

    print(f"Review items: {len(items)}")
    print(f"Saved annotations: {len(annotations)}")
    print(f"Completed annotations: {len(completed_ids)}")
    print(f"Pending items: {len(pending_ids)}")
    print(f"Invalid saved annotations: {len(invalid)}")
    print(f"Orphan annotations: {len(orphan_ids)}")
    print(f"Audit events: {len(events)}")
    _print_counts("Main types", items, "current_main_type")
    _print_counts("Actions", annotations, "review_action")
    _print_counts("Confidence", annotations, "confidence")
    _print_counts("Subtypes", annotations, "confirmed_subtype")

    if pending_ids:
        print("First pending IDs: " + ", ".join(pending_ids[:20]))
    if invalid:
        preview = "; ".join(
            f"{review_id}: {' | '.join(errors)}" for review_id, errors in invalid[:10]
        )
        print(f"Invalid annotation preview: {preview}")
    if orphan_ids:
        print("Orphan annotation IDs: " + ", ".join(orphan_ids[:20]))

    complete = (
        bool(items)
        and not pending_ids
        and not invalid
        and not orphan_ids
        and len(completed_ids) == len(items)
    )
    print(f"Labeling complete: {'yes' if complete else 'no'}")


if __name__ == "__main__":
    main()
