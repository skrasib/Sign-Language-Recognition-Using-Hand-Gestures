"""Small synthetic smoke test for the V3.2 EVT open-set gate."""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from adaptive_gesture.learning.evt_open_set import EVTOpenSetGestureLearner


def cluster(center: float) -> list[np.ndarray]:
    offsets = (-0.04, -0.02, 0.0, 0.015, 0.03, 0.05)
    return [np.asarray([center + d, 0.25 * d], dtype=np.float32) for d in offsets]


def main() -> None:
    learner = EVTOpenSetGestureLearner(
        evt_tail_size=6,
        evt_min_negatives=3,
        inclusion_threshold=0.35,
        minimum_threshold=0.05,
    )
    learner.learn_gesture("A", cluster(0.0), hand_signature="Right")
    learner.learn_gesture("B", cluster(2.0), hand_signature="Right")

    known = learner.predict(np.asarray([0.01, 0.0], dtype=np.float32), hand_signature="Right")
    unknown = learner.predict(np.asarray([0.0, 3.0], dtype=np.float32), hand_signature="Right")

    assert known.accepted and known.label == "A"
    assert not unknown.accepted and unknown.label == "UNKNOWN"
    assert known.open_set_method == "evt_evm"
    assert unknown.rejection_reason == "evt_open_set"

    print("V3.2 EVT open-set smoke test: PASS")
    print(f"Known inclusion:   {known.open_set_score:.3f}")
    print(f"Unknown inclusion: {unknown.open_set_score:.3f}")


if __name__ == "__main__":
    main()
