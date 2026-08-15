"""Secure, cached delivery of broad-classifier checkpoint files."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Final, Iterable
from urllib.request import Request, urlopen


MODEL_SOURCE_COMMIT: Final[str] = (
    "4fcea1af0c88e63f086f40332445996cf31d8df6"
)
MODEL_BASE_URL: Final[str] = (
    "https://media.githubusercontent.com/media/smksean/"
    "-Automation-of-Dermatoglyphic-Pattern-Recognition-and-Quantification-"
    "Through-Machine-Learning/"
    f"{MODEL_SOURCE_COMMIT}/models/efficientnet_320_cv"
)
DOWNLOAD_CHUNK_BYTES: Final[int] = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 120


class CheckpointDownloadError(RuntimeError):
    """Raised when a deployment checkpoint cannot be retrieved or verified."""


@dataclass(frozen=True)
class CheckpointAsset:
    filename: str
    sha256: str
    size_bytes: int

    @property
    def url(self) -> str:
        return f"{MODEL_BASE_URL}/{self.filename}"


CHECKPOINT_ASSETS: Final[tuple[CheckpointAsset, ...]] = (
    CheckpointAsset(
        "efficientnet_b0_320_fold_1.pt",
        "677e9157cbd04a0198e0f8f0f814e96bb8de59c142706e0be2d1cda4735125c4",
        16_356_313,
    ),
    CheckpointAsset(
        "efficientnet_b0_320_fold_2.pt",
        "81444e30d8bea26d00807cd5f77aee7eeb3dcf04636b38f208d042ed07412109",
        16_356_313,
    ),
    CheckpointAsset(
        "efficientnet_b0_320_fold_3.pt",
        "da7a71027acc7fb067c79f16cb3c93561acd377b6c24a54af76447340c273a4b",
        16_356_313,
    ),
    CheckpointAsset(
        "efficientnet_b0_320_fold_4.pt",
        "a412992a9ee5856f81f1b7d6925f8912ea4ce30eae36a412e0e747f7be629cce",
        16_356_313,
    ),
    CheckpointAsset(
        "efficientnet_b0_320_fold_5.pt",
        "bd6e89b2c7c8da1ab2111c18738d042b295c3b2170442f3b061bee56fc29fc72",
        16_356_313,
    ),
)


def file_sha256(path: Path) -> str:
    """Calculate a file digest without loading the whole checkpoint into memory."""
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_is_valid(path: Path, asset: CheckpointAsset) -> bool:
    """Return whether a local checkpoint has the frozen size and digest."""
    return (
        path.is_file()
        and path.stat().st_size == asset.size_bytes
        and file_sha256(path) == asset.sha256
    )


def _download_checkpoint(
    destination: Path,
    asset: CheckpointAsset,
    opener: Callable[..., object],
) -> None:
    temporary_path = destination.with_name(f".{destination.name}.download")
    request = Request(
        asset.url,
        headers={"User-Agent": "broad-fingerprint-classifier/0.1"},
    )
    digest = sha256()
    downloaded_bytes = 0

    try:
        with opener(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with temporary_path.open("wb") as output:
                while True:
                    block = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not block:
                        break
                    downloaded_bytes += len(block)
                    if downloaded_bytes > asset.size_bytes:
                        raise CheckpointDownloadError(
                            f"Downloaded checkpoint is larger than expected: {asset.filename}"
                        )
                    digest.update(block)
                    output.write(block)

        if downloaded_bytes != asset.size_bytes:
            raise CheckpointDownloadError(
                f"Incomplete checkpoint download: {asset.filename} "
                f"({downloaded_bytes} of {asset.size_bytes} bytes)"
            )
        if digest.hexdigest() != asset.sha256:
            raise CheckpointDownloadError(
                f"Checkpoint verification failed: {asset.filename}"
            )
        temporary_path.replace(destination)
    except CheckpointDownloadError:
        temporary_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise CheckpointDownloadError(
            f"Could not download deployment checkpoint {asset.filename}: {exc}"
        ) from exc


def ensure_checkpoints(
    model_directory: Path,
    assets: Iterable[CheckpointAsset] = CHECKPOINT_ASSETS,
    opener: Callable[..., object] = urlopen,
) -> Path:
    """Download missing checkpoints once and verify every file before use."""
    model_directory = model_directory.resolve()
    model_directory.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        if Path(asset.filename).name != asset.filename:
            raise CheckpointDownloadError("Unsafe checkpoint filename in manifest.")
        destination = model_directory / asset.filename
        if not checkpoint_is_valid(destination, asset):
            _download_checkpoint(destination, asset, opener)

    return model_directory
