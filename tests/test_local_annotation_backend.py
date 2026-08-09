from __future__ import annotations

import csv
import shutil
import unittest
from pathlib import Path

from annotation_app.app_logic import build_export_csv
from annotation_app.backend import AnnotationBackendError, LocalAnnotationBackend


class LocalAnnotationBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path("annotation_exports/test_local_annotation_backend").resolve()
        workspace = Path.cwd().resolve()
        if workspace not in root.parents:
            raise RuntimeError("Test directory must remain inside the workspace")
        if root.exists():
            shutil.rmtree(root)
        self.package_dir = root / "package"
        self.state_dir = root / "state"
        image_dir = self.package_dir / "images" / "arch"
        image_dir.mkdir(parents=True)
        (image_dir / "A0001.png").write_bytes(b"test-png-bytes")

        with (self.package_dir / "subtype_labeling_template.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "review_id",
                    "record_key",
                    "image_file",
                    "current_main_type",
                    "source_primary_code",
                    "recorded_pattern_codes",
                    "alternative_type_warning",
                    "permitted_subtypes",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "review_id": "A0001",
                    "record_key": "record-key",
                    "image_file": "images/arch/A0001.png",
                    "current_main_type": "arch",
                    "source_primary_code": "AU",
                    "recorded_pattern_codes": "AU",
                    "alternative_type_warning": "no",
                    "permitted_subtypes": "plain_arch|tented_arch|unclear",
                }
            )
        self.backend = LocalAnnotationBackend(self.package_dir, self.state_dir)

    def tearDown(self) -> None:
        root = self.package_dir.parent.resolve()
        if Path.cwd().resolve() in root.parents and root.exists():
            shutil.rmtree(root)

    def test_local_save_revision_event_and_csv(self) -> None:
        items = self.backend.list_review_items()
        self.assertEqual(len(items), 1)
        self.assertEqual(
            self.backend.download_image(items[0]["image_path"]), b"test-png-bytes"
        )
        proposed = {
            "confirmed_subtype": "plain_arch",
            "confidence": "high",
            "review_action": "accept",
            "main_type_issue": "",
            "review_notes": "",
            "reviewer_id": "local-reviewer",
        }
        saved = self.backend.save_annotation("A0001", proposed, expected_revision=0)
        self.assertEqual(saved["revision"], 1)
        self.assertEqual(len(self.backend.list_annotations()), 1)
        self.assertEqual(
            len(self.backend.events_path.read_text(encoding="utf-8").splitlines()), 1
        )

        annotations = {"A0001": saved}
        self.backend.write_latest_csv(build_export_csv(items, annotations))
        self.assertIn(
            "plain_arch", self.backend.export_path.read_text(encoding="utf-8-sig")
        )

        with self.assertRaises(AnnotationBackendError):
            self.backend.save_annotation("A0001", proposed, expected_revision=0)

    def test_rejects_unsafe_image_path(self) -> None:
        with self.assertRaises(AnnotationBackendError):
            self.backend.download_image("../outside.png")


if __name__ == "__main__":
    unittest.main()
