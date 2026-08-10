from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import urllib.request


HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# The official bundle is several megabytes. This sanity floor catches HTML error
# pages, proxy messages, and otherwise truncated downloads without depending on
# a version-specific hash.
_MIN_MODEL_BYTES = 1_000_000


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "adaptive-hand-gesture-recognition-v3.6"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def is_valid_model_asset(path: str | Path, minimum_bytes: int = _MIN_MODEL_BYTES) -> bool:
    path = Path(path)
    try:
        return path.is_file() and path.stat().st_size >= int(minimum_bytes)
    except OSError:
        return False


def ensure_hand_landmarker_model(
    path: str | Path,
    *,
    url: str = HAND_LANDMARKER_MODEL_URL,
    minimum_bytes: int = _MIN_MODEL_BYTES,
    downloader=None,
) -> Path:
    """Return a usable Hand Landmarker task bundle, downloading once if needed.

    The model is runtime data rather than source code, so the normal V3 app stores
    it under ``data/v3/models`` (already ignored by the repository's .gitignore).
    ``downloader`` is injectable to keep this helper fully unit-testable offline.
    """

    destination = Path(path)
    if is_valid_model_asset(destination, minimum_bytes=minimum_bytes):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    download_fn = downloader or _download

    # Download atomically so a cancelled first run never leaves a corrupt model
    # that later looks like a valid local asset.
    with tempfile.NamedTemporaryFile(
        prefix=destination.stem + "-",
        suffix=".part",
        dir=destination.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        download_fn(url, temp_path)
        if not is_valid_model_asset(temp_path, minimum_bytes=minimum_bytes):
            size = temp_path.stat().st_size if temp_path.exists() else 0
            raise RuntimeError(
                "Downloaded MediaPipe Hand Landmarker model is incomplete "
                f"({size} bytes)."
            )
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return destination
