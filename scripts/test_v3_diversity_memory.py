"""Small camera-free smoke test for the V3.4 diversity-aware exemplar memory."""

from __future__ import annotations

import numpy as np

from adaptive_gesture.learning.exemplar_memory import (
    select_boundary_diverse_negatives,
    select_diverse_exemplars,
)
from adaptive_gesture.learning.evt_open_set import EVTOpenSetGestureLearner


def vector(value: float, dimension: int = 48) -> np.ndarray:
    return np.full(dimension, value, dtype=np.float32)


def main() -> None:
    ordered = [vector(v) for v in (-1.0, -0.8, -0.2, 0.0, 0.1, 0.7, 0.8, 0.9, 1.0)]
    selected, report = select_diverse_exemplars(ordered, budget=4)
    selected_values = [round(float(sample[0]), 3) for sample in selected]

    print("Positive memory")
    print("  selected:", selected_values)
    print("  FIFO coverage radius:", round(report.coverage_radius_before or 0.0, 4))
    print("  V3.4 coverage radius:", round(report.coverage_radius_after or 0.0, 4))

    positives = [vector(0.0), vector(0.05)]
    negatives = [vector(v) for v in (0.10, 0.13, 0.16, 0.25, 0.5, 0.9, 1.5, 2.0)]
    selected_negatives, _ = select_boundary_diverse_negatives(
        negatives,
        positive_samples=positives,
        budget=4,
    )
    print("Hard-negative memory")
    print("  selected:", [round(float(sample[0]), 3) for sample in selected_negatives])

    learner = EVTOpenSetGestureLearner(
        feedback_duplicate_threshold=1e-6,
        exemplar_memory_strategy="diversity",
        hard_negative_memory_strategy="boundary_diversity",
        evt_tail_size=6,
        evt_min_negatives=3,
        inclusion_threshold=0.35,
    )
    learner.learn_gesture(
        "A",
        [vector(-0.2), vector(-0.1), vector(0.0), vector(0.1)],
        hand_signature="Right",
    )
    learner.learn_gesture(
        "B",
        [vector(1.8), vector(1.9), vector(2.0), vector(2.1)],
        hand_signature="Right",
    )
    for value in (0.2, -0.3, 0.35, -0.45, 0.5):
        learner.update_gesture("A", vector(value), max_samples=5)

    assert learner.gestures["A"].sample_count == 5
    assert min(float(s[0]) for s in learner.gestures["A"].samples) <= -0.3
    assert max(float(s[0]) for s in learner.gestures["A"].samples) >= 0.35

    known = learner.predict(vector(0.02), hand_signature="Right")
    assert known.accepted and known.label == "A"

    print("V3.4 diversity-aware exemplar memory smoke test: PASS")


if __name__ == "__main__":
    main()
