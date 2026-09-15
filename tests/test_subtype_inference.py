from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from broad_classifier.subtype_inference import subtype_available


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1]


class SubtypeInferenceTests(unittest.TestCase):
    def test_availability_can_be_checked_per_broad_class(self) -> None:
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_directory:
            artifact_dir = Path(temporary_directory)
            (artifact_dir / "arch_subtype_classifier.pkl").touch()

            self.assertTrue(subtype_available(artifact_dir, broad_class="arch"))
            self.assertFalse(subtype_available(artifact_dir, broad_class="whorl"))
            self.assertFalse(subtype_available(artifact_dir))

    def test_unsupported_broad_class_is_unavailable(self) -> None:
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_directory:
            artifact_dir = Path(temporary_directory)
            self.assertFalse(
                subtype_available(artifact_dir, broad_class="left_slant_loop")
            )


if __name__ == "__main__":
    unittest.main()
