from pathlib import Path

import pytest

from adaptive_gesture.tracking.model_assets import (
    ensure_hand_landmarker_model,
    is_valid_model_asset,
)


def test_existing_model_is_reused_without_downloading(tmp_path):
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"x" * 64)
    calls = []

    def downloader(_url, _path):
        calls.append(True)

    result = ensure_hand_landmarker_model(
        model,
        minimum_bytes=32,
        downloader=downloader,
    )

    assert result == model
    assert calls == []


def test_model_download_is_atomic_and_validated(tmp_path):
    model = tmp_path / "nested" / "hand_landmarker.task"

    def downloader(_url, path: Path):
        path.write_bytes(b"m" * 128)

    result = ensure_hand_landmarker_model(
        model,
        minimum_bytes=100,
        downloader=downloader,
    )

    assert result == model
    assert is_valid_model_asset(model, minimum_bytes=100)
    assert model.read_bytes() == b"m" * 128
    assert not list(model.parent.glob("*.part"))


def test_invalid_download_is_removed(tmp_path):
    model = tmp_path / "hand_landmarker.task"

    def downloader(_url, path: Path):
        path.write_bytes(b"tiny")

    with pytest.raises(RuntimeError):
        ensure_hand_landmarker_model(
            model,
            minimum_bytes=100,
            downloader=downloader,
        )

    assert not model.exists()
    assert not list(tmp_path.glob("*.part"))
