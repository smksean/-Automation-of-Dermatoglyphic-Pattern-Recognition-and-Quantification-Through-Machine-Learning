"""Supabase access layer for the private annotation app."""

from __future__ import annotations

import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Protocol

from supabase import Client, create_client

from annotation_app.app_logic import validate_annotation


class AnnotationBackendError(RuntimeError):
    """Raised when the shared annotation backend cannot complete an operation."""


class AnnotationBackend(Protocol):
    def list_review_items(self) -> list[dict[str, Any]]: ...

    def list_annotations(self) -> list[dict[str, Any]]: ...

    def download_image(self, image_path: str) -> bytes: ...

    def save_annotation(
        self,
        review_id: str,
        annotation: Mapping[str, Any],
        expected_revision: int,
    ) -> dict[str, Any]: ...

    def write_latest_csv(self, csv_bytes: bytes) -> None: ...


class LocalAnnotationBackend:
    """Package-backed backend for localhost review before cloud deployment."""

    def __init__(self, package_dir: Path, state_dir: Path) -> None:
        self.package_dir = package_dir.resolve()
        self.state_dir = state_dir.resolve()
        self.template_path = self.package_dir / "subtype_labeling_template.csv"
        self.annotations_path = self.state_dir / "annotations.json"
        self.events_path = self.state_dir / "annotation_events.jsonl"
        self.export_path = self.state_dir / "subtype_labeling_latest.csv"
        if not self.template_path.is_file():
            raise AnnotationBackendError(
                f"Local review template was not found: {self.template_path}"
            )
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def list_review_items(self) -> list[dict[str, Any]]:
        try:
            with self.template_path.open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            return [
                {
                    "review_id": row["review_id"],
                    "record_key": row["record_key"],
                    "image_path": row["image_file"],
                    "current_main_type": row["current_main_type"],
                    "source_primary_code": row["source_primary_code"],
                    "recorded_pattern_codes": row["recorded_pattern_codes"],
                    "alternative_type_warning": row["alternative_type_warning"]
                    == "yes",
                    "permitted_subtypes": row["permitted_subtypes"].split("|"),
                }
                for row in rows
            ]
        except Exception as exc:
            raise AnnotationBackendError(
                f"Could not load the local review template: {exc}"
            ) from exc

    def _load_annotation_map(self) -> dict[str, dict[str, Any]]:
        if not self.annotations_path.exists():
            return {}
        try:
            data = json.loads(self.annotations_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("annotations.json must contain an object")
            return {str(key): dict(value) for key, value in data.items()}
        except Exception as exc:
            raise AnnotationBackendError(
                f"Could not load local annotations: {exc}"
            ) from exc

    def list_annotations(self) -> list[dict[str, Any]]:
        return list(self._load_annotation_map().values())

    def download_image(self, image_path: str) -> bytes:
        path = (self.package_dir / image_path).resolve()
        try:
            path.relative_to(self.package_dir)
        except ValueError as exc:
            raise AnnotationBackendError("Unsafe local image path was rejected.") from exc
        if not path.is_file():
            raise AnnotationBackendError(f"Local review image was not found: {path}")
        return path.read_bytes()

    def _write_annotations(self, annotations: Mapping[str, Any]) -> None:
        temporary = self.annotations_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(annotations, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.annotations_path)

    def save_annotation(
        self,
        review_id: str,
        annotation: Mapping[str, Any],
        expected_revision: int,
    ) -> dict[str, Any]:
        items_by_id = {
            str(item["review_id"]): item for item in self.list_review_items()
        }
        item = items_by_id.get(review_id)
        if item is None:
            raise AnnotationBackendError(f"Unknown local review ID: {review_id}")
        errors = validate_annotation(item, annotation)
        if errors:
            raise AnnotationBackendError(" ".join(errors))

        annotation_map = self._load_annotation_map()
        existing = annotation_map.get(review_id, {})
        current_revision = int(existing.get("revision") or 0)
        if current_revision != expected_revision:
            raise AnnotationBackendError(
                f"Revision conflict for {review_id}: expected {expected_revision}, "
                f"current {current_revision}. Refresh before saving."
            )

        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        saved = {
            "review_id": review_id,
            "confirmed_subtype": annotation["confirmed_subtype"],
            "confidence": annotation["confidence"],
            "review_action": annotation["review_action"],
            "main_type_issue": annotation.get("main_type_issue") or None,
            "review_notes": annotation.get("review_notes") or None,
            "reviewer_id": annotation["reviewer_id"],
            "first_reviewed_at": existing.get("first_reviewed_at", now),
            "reviewed_at": now,
            "revision": current_revision + 1,
        }
        annotation_map[review_id] = saved
        self._write_annotations(annotation_map)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(saved, sort_keys=True) + "\n")
        return saved

    def write_latest_csv(self, csv_bytes: bytes) -> None:
        temporary = self.export_path.with_suffix(".csv.tmp")
        temporary.write_bytes(csv_bytes)
        os.replace(temporary, self.export_path)


class SupabaseAnnotationBackend:
    def __init__(
        self,
        url: str,
        service_role_key: str,
        bucket: str,
        export_path: str = "exports/subtype_labeling_latest.csv",
    ) -> None:
        self.client: Client = create_client(url, service_role_key)
        self.bucket = bucket
        self.export_path = export_path

    def list_review_items(self) -> list[dict[str, Any]]:
        try:
            response = (
                self.client.table("review_items")
                .select(
                    "review_id,record_key,image_path,current_main_type,"
                    "source_primary_code,recorded_pattern_codes,"
                    "alternative_type_warning,permitted_subtypes"
                )
                .order("review_id")
                .execute()
            )
            return list(response.data or [])
        except Exception as exc:  # SDK exceptions vary by transport version.
            raise AnnotationBackendError(f"Could not load review items: {exc}") from exc

    def list_annotations(self) -> list[dict[str, Any]]:
        try:
            response = self.client.table("annotations").select("*").execute()
            return list(response.data or [])
        except Exception as exc:
            raise AnnotationBackendError(f"Could not load annotations: {exc}") from exc

    def download_image(self, image_path: str) -> bytes:
        try:
            return self.client.storage.from_(self.bucket).download(image_path)
        except Exception as exc:
            raise AnnotationBackendError(
                f"Could not load private image {image_path}: {exc}"
            ) from exc

    def save_annotation(
        self,
        review_id: str,
        annotation: Mapping[str, Any],
        expected_revision: int,
    ) -> dict[str, Any]:
        params = {
            "p_review_id": review_id,
            "p_confirmed_subtype": annotation["confirmed_subtype"],
            "p_confidence": annotation["confidence"],
            "p_review_action": annotation["review_action"],
            "p_main_type_issue": annotation.get("main_type_issue") or None,
            "p_review_notes": annotation.get("review_notes") or None,
            "p_reviewer_id": annotation["reviewer_id"],
            "p_expected_revision": expected_revision,
        }
        try:
            response = self.client.rpc("save_subtype_annotation", params).execute()
            data = response.data
            if isinstance(data, list):
                return dict(data[0]) if data else {}
            return dict(data or {})
        except Exception as exc:
            raise AnnotationBackendError(
                "Could not save the annotation. Refresh first if another reviewer "
                f"may have edited this record. Details: {exc}"
            ) from exc

    def write_latest_csv(self, csv_bytes: bytes) -> None:
        try:
            self.client.storage.from_(self.bucket).upload(
                path=self.export_path,
                file=csv_bytes,
                file_options={
                    "content-type": "text/csv; charset=utf-8",
                    "cache-control": "no-cache",
                    "upsert": "true",
                },
            )
        except Exception as exc:
            raise AnnotationBackendError(
                f"Annotation saved, but the latest shared CSV was not updated: {exc}"
            ) from exc
