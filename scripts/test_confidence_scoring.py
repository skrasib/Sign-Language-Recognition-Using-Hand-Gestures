"""Offline smoke test for static and dynamic confidence scoring.

Run from the repository root with:
    python scripts/test_confidence_scoring.py
"""

from pathlib import Path
import math
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from adaptive_gesture.features.dynamic_features import (  # noqa: E402
    DynamicObservation,
    prepare_dynamic_trajectory,
)
from adaptive_gesture.learning.dynamic_learner import DynamicGestureLearner  # noqa: E402
from adaptive_gesture.learning.online_learner import OnlineGestureLearner  # noqa: E402


def make_static(center: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        np.full(63, center, dtype=np.float32)
        + rng.normal(0.0, 0.004, 63).astype(np.float32)
        for _ in range(12)
    ]


def make_dynamic(kind: str, seed: int):
    rng = np.random.default_rng(seed)
    observations = []
    for t in np.linspace(0.0, 1.0, 34):
        if kind == "right":
            x, y = 0.42 * t, 0.0
        elif kind == "left":
            x, y = -0.42 * t, 0.0
        else:
            x = 0.23 * (math.cos(2.0 * math.pi * t) - 1.0)
            y = 0.23 * math.sin(2.0 * math.pi * t)

        pose = rng.normal(0.0, 0.004, 63).astype(np.float32)
        observations.append(
            DynamicObservation(
                timestamp=float(t),
                hand_signature="Right",
                pose_vector=pose,
                anchor_xy=np.array([0.5 + x, 0.5 + y], dtype=np.float32),
                palm_scale=0.10,
            )
        )
    return prepare_dynamic_trajectory(observations)


def main():
    static = OnlineGestureLearner()
    a = make_static(0.0, 1)
    b = make_static(0.25, 2)
    static.learn_gesture("A", a, hand_signature="Right")
    static.learn_gesture("B", b, hand_signature="Right")

    known = static.predict(a[0], hand_signature="Right")
    assert known.accepted
    assert known.label == "A"
    assert known.confidence is not None
    assert 0.0 <= known.confidence <= 1.0

    unknown = static.predict(np.full(63, 1.2, dtype=np.float32), hand_signature="Right")
    assert not unknown.accepted
    assert unknown.confidence is not None
    assert 0.0 <= unknown.confidence <= 1.0

    dynamic = DynamicGestureLearner()
    for name in ("right", "left"):
        dynamic.learn_gesture(
            name,
            [
                make_dynamic(name, hash((name, 1)) & 0xFFFF),
                make_dynamic(name, hash((name, 2)) & 0xFFFF),
                make_dynamic(name, hash((name, 3)) & 0xFFFF),
            ],
        )

    dynamic_known = dynamic.predict(make_dynamic("right", 999))
    assert dynamic_known.accepted
    assert dynamic_known.confidence is not None
    assert 0.0 <= dynamic_known.confidence <= 1.0

    print("Confidence scoring smoke test: PASS")
    print(f"Static known confidence index: {known.confidence * 100:.1f}%")
    print(f"Static UNKNOWN confidence index: {unknown.confidence * 100:.1f}%")
    print(f"Dynamic known confidence index: {dynamic_known.confidence * 100:.1f}%")


if __name__ == "__main__":
    main()
