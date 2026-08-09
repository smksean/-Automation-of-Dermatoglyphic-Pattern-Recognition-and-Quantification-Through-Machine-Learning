from __future__ import annotations

import csv
import io
import unittest

from annotation_app.app_logic import (
    build_export_csv,
    filtered_items,
    progress_counts,
    validate_annotation,
)


ITEMS = [
    {
        "review_id": "A0001",
        "record_key": "arch-key",
        "image_path": "review/arch/A0001.png",
        "current_main_type": "arch",
        "source_primary_code": "AU",
        "recorded_pattern_codes": "AU",
        "alternative_type_warning": False,
        "permitted_subtypes": ["plain_arch", "tented_arch", "unclear"],
    },
    {
        "review_id": "W0001",
        "record_key": "whorl-key",
        "image_path": "review/whorl/W0001.png",
        "current_main_type": "whorl",
        "source_primary_code": "WU",
        "recorded_pattern_codes": "WU|LS",
        "alternative_type_warning": True,
        "permitted_subtypes": [
            "plain_whorl",
            "central_pocket_loop_whorl",
            "double_loop_whorl",
            "accidental_whorl",
            "unclear",
        ],
    },
]


def valid_arch_annotation() -> dict[str, str]:
    return {
        "confirmed_subtype": "plain_arch",
        "confidence": "high",
        "review_action": "accept",
        "main_type_issue": "",
        "review_notes": "",
        "reviewer_id": "professor-1",
    }


class AnnotationLogicTests(unittest.TestCase):
    def test_valid_annotation(self) -> None:
        self.assertEqual(validate_annotation(ITEMS[0], valid_arch_annotation()), [])

    def test_rejects_cross_type_subtype(self) -> None:
        annotation = valid_arch_annotation()
        annotation["confirmed_subtype"] = "plain_whorl"
        self.assertIn("Select a permitted arch subtype.", validate_annotation(ITEMS[0], annotation))

    def test_main_type_issue_requires_unclear_adjudication_and_notes(self) -> None:
        annotation = valid_arch_annotation()
        annotation["main_type_issue"] = "incorrect"
        errors = validate_annotation(ITEMS[0], annotation)
        self.assertIn("Use subtype 'unclear' when the main type is disputed.", errors)
        self.assertIn("A main-type issue must be sent for adjudication.", errors)
        self.assertIn("Explain the main-type issue in the review notes.", errors)

    def test_progress_and_filters(self) -> None:
        annotations = {"A0001": valid_arch_annotation()}
        counts = progress_counts(ITEMS, annotations)
        self.assertEqual(counts["completed"], 1)
        self.assertEqual(counts["pending"], 1)
        self.assertEqual(counts["warnings"], 1)
        self.assertEqual(
            [item["review_id"] for item in filtered_items(ITEMS, annotations)],
            ["W0001"],
        )
        self.assertEqual(
            [
                item["review_id"]
                for item in filtered_items(
                    ITEMS, annotations, status="all", warnings_only=True
                )
            ],
            ["W0001"],
        )

    def test_export_is_sorted_and_contains_annotations(self) -> None:
        annotation = valid_arch_annotation()
        annotation.update({"reviewed_at": "2026-08-04T10:00:00Z", "revision": 1})
        payload = build_export_csv(reversed(ITEMS), {"A0001": annotation})
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        self.assertEqual([row["review_id"] for row in rows], ["A0001", "W0001"])
        self.assertEqual(rows[0]["confirmed_subtype"], "plain_arch")
        self.assertEqual(rows[1]["confirmed_subtype"], "")


if __name__ == "__main__":
    unittest.main()
