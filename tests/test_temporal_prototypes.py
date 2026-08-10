import numpy as np

from adaptive_gesture.features.dynamic_features import DynamicTrajectory
from adaptive_gesture.learning.dtw import (
    trajectory_dtw_alignment,
    trajectory_dtw_distance,
)
from adaptive_gesture.learning.dynamic_learner import DynamicGestureLearner
from adaptive_gesture.learning.temporal_prototypes import (
    build_dtw_barycenter,
    build_temporal_prototypes,
)


def make_curve(
    bend: float = 0.25,
    warp: float = 1.0,
    shape_offset: float = 0.0,
) -> DynamicTrajectory:
    frames = 48
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    warped = np.power(t, warp).astype(np.float32)

    shape = np.zeros((frames, 63), dtype=np.float32)
    shape[:, 8] = shape_offset + 0.03 * np.sin(np.pi * warped)

    x = warped
    y = bend * np.sin(np.pi * warped)
    motion = np.column_stack([x, y]).astype(np.float32)
    motion -= motion[:1]
    velocity = np.diff(motion, axis=0, prepend=motion[:1]).astype(np.float32)

    return DynamicTrajectory(
        hand_signature="Right",
        shape_sequence=shape,
        motion_sequence=motion,
        velocity_sequence=velocity,
        duration_seconds=1.0 + 0.1 * abs(warp - 1.0),
        raw_frame_count=55,
        motion_extent=float(np.max(np.linalg.norm(motion, axis=1))),
        shape_extent=float(np.max(np.abs(shape[:, 8] - shape[0, 8]))),
    )


def test_dtw_alignment_returns_valid_monotonic_path():
    first = make_curve(warp=1.0)
    second = make_curve(warp=1.25)

    distance, path = trajectory_dtw_alignment(first, second)

    assert np.isfinite(distance)
    assert path[0] == (0, 0)
    assert path[-1] == (first.length - 1, second.length - 1)
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(path, path[1:]))


def test_dtw_barycenter_preserves_dynamic_trajectory_contract():
    templates = [
        make_curve(warp=0.85, shape_offset=-0.002),
        make_curve(warp=1.00, shape_offset=0.000),
        make_curve(warp=1.20, shape_offset=0.002),
    ]

    prototype = build_dtw_barycenter(templates, iterations=3)

    assert prototype.hand_signature == "Right"
    assert prototype.shape_sequence.shape == templates[0].shape_sequence.shape
    assert prototype.motion_sequence.shape == templates[0].motion_sequence.shape
    assert prototype.velocity_sequence.shape == templates[0].velocity_sequence.shape
    assert np.allclose(prototype.motion_sequence[0], 0.0, atol=1e-6)
    assert np.all(np.isfinite(prototype.shape_sequence))
    assert np.all(np.isfinite(prototype.motion_sequence))


def test_temporal_prototype_is_representative_of_training_group():
    templates = [
        make_curve(warp=0.8),
        make_curve(warp=1.0),
        make_curve(warp=1.25),
    ]
    prototype = build_dtw_barycenter(templates, iterations=4)

    distances = [trajectory_dtw_distance(prototype, item) for item in templates]
    assert max(distances) < 0.06


def test_multiple_temporal_modes_can_create_two_prototypes():
    templates = [
        make_curve(bend=0.25, warp=0.9),
        make_curve(bend=0.25, warp=1.0),
        make_curve(bend=0.25, warp=1.1),
        make_curve(bend=-0.25, warp=0.9),
        make_curve(bend=-0.25, warp=1.0),
        make_curve(bend=-0.25, warp=1.1),
    ]

    prototypes, info = build_temporal_prototypes(
        templates,
        max_prototypes=2,
        iterations=2,
    )

    assert len(prototypes) == 2
    assert info.prototype_count == 2
    assert sorted(info.cluster_sizes) == [3, 3]


def test_dynamic_learner_v35_uses_temporal_prototype_by_default():
    learner = DynamicGestureLearner(
        minimum_templates=3,
        minimum_threshold=0.02,
        threshold_multiplier=1.8,
    )
    templates = [
        make_curve(warp=0.9),
        make_curve(warp=1.0),
        make_curve(warp=1.1),
    ]

    gesture = learner.learn_gesture("Arc Right", templates)
    prediction = learner.predict(make_curve(warp=1.05))

    assert gesture.prototype_count == 1
    assert prediction.accepted
    assert prediction.label == "Arc Right"


def test_nearest_template_strategy_remains_available_for_ablation():
    learner = DynamicGestureLearner(
        minimum_templates=3,
        minimum_threshold=0.02,
        threshold_multiplier=1.8,
        temporal_prototype_strategy="templates",
    )
    gesture = learner.learn_gesture(
        "Arc Right",
        [make_curve(warp=0.9), make_curve(warp=1.0), make_curve(warp=1.1)],
    )

    assert gesture.prototype_count == 0
    assert learner.predict(make_curve(warp=1.05)).accepted
