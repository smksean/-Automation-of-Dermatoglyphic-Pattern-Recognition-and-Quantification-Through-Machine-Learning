from __future__ import annotations

import unittest

from scripts.upload_subtype_review_to_supabase import select_review_rows


class SupabaseUploadSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"review_id": "W0002", "current_main_type": "whorl"},
            {"review_id": "A0002", "current_main_type": "arch"},
            {"review_id": "W0001", "current_main_type": "whorl"},
            {"review_id": "A0001", "current_main_type": "arch"},
        ]

    def test_balanced_pilot_is_stable(self) -> None:
        selected = select_review_rows(self.rows, pilot_per_type=1)
        self.assertEqual(
            [row["review_id"] for row in selected], ["A0001", "W0001"]
        )

    def test_full_selection_preserves_rows(self) -> None:
        self.assertIs(select_review_rows(self.rows), self.rows)

    def test_rejects_oversized_pilot(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 2 are available"):
            select_review_rows(self.rows, pilot_per_type=3)


if __name__ == "__main__":
    unittest.main()
