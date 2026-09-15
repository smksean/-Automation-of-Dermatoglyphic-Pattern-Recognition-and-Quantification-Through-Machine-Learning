import unittest

from scripts.restore_sd302_2026 import (
    broad_class,
    expected_png_member,
    label_from_parts,
    legacy_png_path,
    review_record_key,
    split_repeated_field,
    subtype_name,
)


class RestoreSd302Tests(unittest.TestCase):
    def test_repeated_pattern_entries_are_preserved(self):
        entries = split_repeated_field("WU\x1fPW\x1eWU\x1fDL")
        self.assertEqual([label_from_parts(parts) for parts in entries], ["WU+PW", "WU+DL"])

    def test_pattern_mappings(self):
        self.assertEqual(broad_class("AU+TA"), "arch")
        self.assertEqual(subtype_name("AU+TA"), "tented_arch")
        self.assertEqual(broad_class("RS"), "right_slant_loop")
        self.assertIsNone(subtype_name("RS"))

    def test_challenger_image_member(self):
        metadata = {
            "subject_id": "00002307",
            "device": "C",
            "resolution": "500",
            "capture_type": "roll",
            "finger_position": "04",
        }
        self.assertEqual(
            expected_png_member(metadata, "challengers"),
            ("sd302a.zip", "images/challengers/C/roll/png/00002307_C_roll_04.png"),
        )

    def test_segmented_slap_image_member(self):
        metadata = {
            "subject_id": "00002307",
            "device": "V",
            "resolution": "1000",
            "capture_type": "slap",
            "finger_position": "04",
        }
        archive, member = expected_png_member(metadata, "baseline")
        self.assertEqual(archive, "sd302b.zip")
        self.assertIn("/slap-segmented/png/", member)

    def test_legacy_review_key_uses_windows_relative_path(self):
        path = legacy_png_path(
            "sd302a.zip",
            "images/challengers/C/roll/png/00002307_C_roll_04.png",
        )
        self.assertEqual(
            path,
            r"sd302a\images\challengers\C\roll\png\00002307_C_roll_04.png",
        )
        self.assertEqual(len(review_record_key("00002307", "04", "challengers", path)), 16)


if __name__ == "__main__":
    unittest.main()
