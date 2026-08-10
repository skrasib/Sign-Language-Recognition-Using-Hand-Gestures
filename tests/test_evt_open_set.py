import numpy as np

from adaptive_gesture.learning.evt_open_set import (
    EVTOpenSetGestureLearner,
    ExtremeVectorModel,
    fit_weibull_zero_location,
)


def _cluster(center, offsets):
    center = np.asarray(center, dtype=np.float32)
    return [center + np.asarray(offset, dtype=np.float32) for offset in offsets]


def _make_learner(**kwargs):
    return EVTOpenSetGestureLearner(
        radius_multiplier=2.5,
        minimum_threshold=0.05,
        prototype_multiplier=2.2,
        max_prototypes=3,
        evt_tail_size=6,
        evt_min_negatives=3,
        inclusion_threshold=0.35,
        **kwargs,
    )


def test_weibull_fit_is_finite_and_positive():
    shape, scale = fit_weibull_zero_location(
        np.asarray([0.18, 0.20, 0.21, 0.25, 0.31], dtype=np.float64)
    )
    assert np.isfinite(shape)
    assert np.isfinite(scale)
    assert shape > 0
    assert scale > 0


def test_radial_inclusion_decreases_with_distance():
    model = ExtremeVectorModel(
        vector=np.asarray([0.0, 0.0], dtype=np.float32),
        shape=2.0,
        scale=1.0,
        tail_size=5,
        nearest_negative_margin=0.5,
    )
    near = model.inclusion(np.asarray([0.1, 0.0], dtype=np.float32))
    middle = model.inclusion(np.asarray([0.7, 0.0], dtype=np.float32))
    far = model.inclusion(np.asarray([2.0, 0.0], dtype=np.float32))
    assert near > middle > far


def test_evt_accepts_known_and_rejects_far_unknown():
    learner = _make_learner()
    offsets = [
        [-0.05, 0.00],
        [-0.02, 0.03],
        [0.00, 0.00],
        [0.02, -0.03],
        [0.05, 0.01],
        [0.01, 0.04],
    ]
    learner.learn_gesture("A", _cluster([0.0, 0.0], offsets), hand_signature="Right")
    learner.learn_gesture("B", _cluster([2.0, 0.0], offsets), hand_signature="Right")

    known = learner.predict(np.asarray([0.01, 0.02], dtype=np.float32), hand_signature="Right")
    unknown = learner.predict(np.asarray([0.0, 3.0], dtype=np.float32), hand_signature="Right")

    assert known.accepted
    assert known.label == "A"
    assert known.open_set_method == "evt_evm"
    assert known.open_set_score >= learner.inclusion_threshold

    assert not unknown.accepted
    assert unknown.label == learner.UNKNOWN_LABEL
    assert unknown.rejection_reason == "evt_open_set"
    assert unknown.open_set_score < learner.inclusion_threshold


def test_single_class_uses_radius_fallback_until_negative_evidence_exists():
    learner = _make_learner()
    samples = _cluster(
        [0.0, 0.0],
        [[-0.03, 0.0], [0.0, 0.0], [0.03, 0.0], [0.0, 0.03]],
    )
    learner.learn_gesture("Only", samples, hand_signature="Right")
    prediction = learner.predict(np.asarray([0.01, 0.0], dtype=np.float32), hand_signature="Right")
    assert getattr(prediction, "open_set_method") == "radius_fallback"


def test_new_class_rebuilds_evt_boundaries_incrementally():
    learner = _make_learner()
    a = _cluster([0.0, 0.0], [[-0.04, 0.0], [0.0, 0.0], [0.04, 0.0], [0.0, 0.04]])
    b = _cluster([1.5, 0.0], [[-0.04, 0.0], [0.0, 0.0], [0.04, 0.0], [0.0, 0.04]])

    learner.learn_gesture("A", a, hand_signature="Right")
    assert learner.extreme_models["A"] == []

    learner.learn_gesture("B", b, hand_signature="Right")
    assert learner.extreme_models["A"]
    assert learner.extreme_models["B"]


def test_hard_negative_becomes_open_set_boundary_evidence():
    learner = _make_learner()
    samples = _cluster(
        [0.0, 0.0],
        [[-0.03, 0.0], [0.0, 0.0], [0.03, 0.0], [0.0, 0.03]],
    )
    learner.learn_gesture("A", samples, hand_signature="Right")

    # Three explicit rejected poses are enough to fit an EVT boundary even when
    # no second learned class exists for this hand configuration.
    learner.add_hard_negative("A", np.asarray([0.45, 0.00], dtype=np.float32))
    learner.add_hard_negative("A", np.asarray([0.50, 0.03], dtype=np.float32))
    learner.add_hard_negative("A", np.asarray([0.55, -0.02], dtype=np.float32))

    assert learner.extreme_models["A"]
    near = learner.predict(np.asarray([0.01, 0.0], dtype=np.float32), hand_signature="Right")
    rejected = learner.predict(np.asarray([0.50, 0.0], dtype=np.float32), hand_signature="Right")
    assert near.accepted
    assert not rejected.accepted
