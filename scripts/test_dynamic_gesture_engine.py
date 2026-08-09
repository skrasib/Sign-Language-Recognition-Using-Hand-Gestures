"""Offline smoke test for the dynamic landmark-trajectory engine.

This does not need a webcam. Run from the repository root with:
    python scripts/test_dynamic_gesture_engine.py
"""

from pathlib import Path
import math
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from adaptive_gesture.features.dynamic_features import (  # noqa: E402
    DynamicObservation,
    prepare_dynamic_trajectory,
)
from adaptive_gesture.learning.dynamic_learner import DynamicGestureLearner  # noqa: E402
from adaptive_gesture.storage.dynamic_gesture_store import DynamicGestureStore  # noqa: E402


def make_demo(kind: str, duration: float, seed: int):
    rng = np.random.default_rng(seed)
    observations = []

    for index, t in enumerate(np.linspace(0.0, 1.0, 34)):
        if kind == "right":
            x, y = 0.42 * t, 0.0
        elif kind == "left":
            x, y = -0.42 * t, 0.0
        elif kind == "circle":
            x = 0.25 * (math.cos(2.0 * math.pi * t) - 1.0)
            y = 0.25 * math.sin(2.0 * math.pi * t)
        else:
            x, y = 0.0, -0.42 * t

        pose = rng.normal(0.0, 0.004, 63).astype(np.float32)
        anchor = np.array([0.5 + x, 0.5 + y], dtype=np.float32)
        anchor += rng.normal(0.0, 0.0002, 2).astype(np.float32)

        observations.append(
            DynamicObservation(
                timestamp=float(t * duration),
                hand_signature="Right",
                pose_vector=pose,
                anchor_xy=anchor,
                palm_scale=0.10,
            )
        )

    return prepare_dynamic_trajectory(observations)


def main():
    learner = DynamicGestureLearner()

    for name in ("right", "left", "circle"):
        learner.learn_gesture(
            name,
            [
                make_demo(name, 0.9, hash((name, 1)) & 0xFFFF),
                make_demo(name, 1.0, hash((name, 2)) & 0xFFFF),
                make_demo(name, 1.15, hash((name, 3)) & 0xFFFF),
            ],
        )

    for expected in ("right", "left", "circle"):
        prediction = learner.predict(
            make_demo(expected, 1.05, hash((expected, 9)) & 0xFFFF)
        )
        assert prediction.accepted, prediction
        assert prediction.label == expected, prediction

    unknown = learner.predict(make_demo("up", 1.0, 999))
    assert not unknown.accepted, unknown

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dynamic.json"
        store = DynamicGestureStore(path)
        store.save(learner)

        restored = DynamicGestureLearner()
        count = store.load_into(restored)
        assert count == 3
        assert set(restored.gestures) == {"right", "left", "circle"}

    print("Dynamic gesture engine smoke test: PASS")


if __name__ == "__main__":
    main()
