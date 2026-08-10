import numpy as np

from adaptive_gesture.learning.exemplar_memory import (
    select_boundary_diverse_negatives,
    select_diverse_exemplars,
)
from adaptive_gesture.learning.online_learner import OnlineGestureLearner


def _v(value: float, dimension: int = 4) -> np.ndarray:
    return np.full(dimension, value, dtype=np.float32)


def test_diverse_memory_covers_old_and_new_variations_better_than_fifo():
    # Deliberately ordered so the newest samples all lie on the positive side.
    samples = [_v(value) for value in (-1.0, -0.8, -0.2, 0.0, 0.1, 0.7, 0.8, 0.9, 1.0)]
    selected, report = select_diverse_exemplars(samples, budget=4)
    values = sorted(float(sample[0]) for sample in selected)

    assert len(selected) == 4
    assert values[0] <= -0.8
    assert values[-1] >= 0.9
    assert report.coverage_radius_after <= report.coverage_radius_before
    assert report.strategy == "diversity_kcenter"


def test_diverse_selection_is_deterministic():
    samples = [_v(value) for value in (-1.0, -0.4, 0.0, 0.2, 0.9, 1.2)]
    first, _ = select_diverse_exemplars(samples, budget=3)
    second, _ = select_diverse_exemplars(samples, budget=3)
    assert [sample.tolist() for sample in first] == [sample.tolist() for sample in second]


def test_online_learner_uses_diversity_instead_of_fifo_when_memory_is_full():
    learner = OnlineGestureLearner(
        feedback_duplicate_threshold=1e-6,
        exemplar_memory_strategy="diversity",
    )
    learner.learn_gesture(
        "Pose",
        [_v(-1.0), _v(-0.5), _v(0.0), _v(0.5)],
        hand_signature="Right",
    )

    # A new positive would make five examples, but the budget is four.
    learner.update_gesture("Pose", _v(1.0), max_samples=4)
    values = [float(sample[0]) for sample in learner.gestures["Pose"].samples]

    assert len(values) == 4
    # FIFO would have discarded -1.0. The diversity core-set retains the old
    # extreme because it still covers a meaningful variation of the class.
    assert any(np.isclose(value, -1.0) for value in values)
    assert any(np.isclose(value, 1.0) for value in values)


def test_fifo_strategy_remains_available_for_ablation():
    learner = OnlineGestureLearner(
        feedback_duplicate_threshold=1e-6,
        exemplar_memory_strategy="fifo",
    )
    learner.learn_gesture(
        "Pose",
        [_v(-1.0), _v(-0.5), _v(0.0), _v(0.5)],
        hand_signature="Right",
    )
    learner.update_gesture("Pose", _v(1.0), max_samples=4)
    values = [float(sample[0]) for sample in learner.gestures["Pose"].samples]

    assert len(values) == 4
    assert not any(np.isclose(value, -1.0) for value in values)
    assert any(np.isclose(value, 1.0) for value in values)


def test_hard_negative_memory_keeps_boundary_evidence_and_diversity():
    positives = [_v(0.0), _v(0.05)]
    negatives = [_v(value) for value in (0.10, 0.13, 0.16, 0.25, 0.5, 0.9, 1.5, 2.0)]

    selected, report = select_boundary_diverse_negatives(
        negatives,
        positive_samples=positives,
        budget=4,
    )
    values = [float(sample[0]) for sample in selected]

    assert len(values) == 4
    # Half of the memory is protected for the hardest (closest) negatives.
    assert any(np.isclose(value, 0.10) for value in values)
    assert any(np.isclose(value, 0.13) for value in values)
    # Remaining slots cover a wider corrected region rather than only near copies.
    assert max(values) >= 0.5
    assert report.strategy == "boundary_diversity"
