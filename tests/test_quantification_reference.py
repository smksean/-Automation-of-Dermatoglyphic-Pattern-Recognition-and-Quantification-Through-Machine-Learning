import unittest

import pandas as pd

from scripts.build_quantification_reference import (
    pattern_intensity_index,
    select_canonical_impressions,
)


class QuantificationReferenceTests(unittest.TestCase):
    def test_pattern_intensity_weights_loops_and_whorls(self):
        patterns = ["arch"] * 2 + ["left_slant_loop"] * 3 + ["right_slant_loop"] + ["whorl"] * 4
        self.assertEqual(pattern_intensity_index(patterns), 12)

    def test_pattern_intensity_rejects_incomplete_or_unclassifiable_sets(self):
        with self.assertRaises(ValueError):
            pattern_intensity_index(["whorl"] * 9)
        with self.assertRaises(ValueError):
            pattern_intensity_index(["whorl"] * 9 + ["unclassifiable"])

    def test_canonical_selection_is_uniform(self):
        rows = pd.DataFrame(
            [
                {"subject_id": "1", "finger_position": "01", "collection_type": "baseline", "device": "V", "resolution": 1000, "capture_type": "roll"},
                {"subject_id": "1", "finger_position": "01", "collection_type": "challengers", "device": "C", "resolution": 500, "capture_type": "roll"},
            ]
        )
        selected = select_canonical_impressions(rows)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["device"], "V")


if __name__ == "__main__":
    unittest.main()
