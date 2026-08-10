import numpy as np

from adaptive_gesture.features.dynamic_features import DynamicTrajectory
from adaptive_gesture.learning.dtw import trajectory_dtw_distance
from adaptive_gesture.learning.dynamic_learner import DynamicGestureLearner


def make_swipe(direction: float = 1.0, offset: float = 0.0) -> DynamicTrajectory:
    frames = 32
    shape = np.zeros((frames, 63), dtype=np.float32)
    shape[:, 8] = offset

    x = np.linspace(0.0, direction, frames, dtype=np.float32) + offset
    motion = np.column_stack([x - x[0], np.zeros(frames, dtype=np.float32)]).astype(
        np.float32
    )
    velocity = np.diff(motion, axis=0, prepend=motion[:1]).astype(np.float32)

    return DynamicTrajectory(
        hand_signature="Right",
        shape_sequence=shape,
        motion_sequence=motion,
        velocity_sequence=velocity,
        duration_seconds=1.0 + abs(offset) * 0.1,
        raw_frame_count=frames,
        motion_extent=1.0,
        shape_extent=abs(offset),
    )


def test_dtw_identical_trajectory_has_zero_distance():
    trajectory = make_swipe()
    assert trajectory_dtw_distance(trajectory, trajectory) == 0.0


def test_dynamic_runtime_learning_and_direction_discrimination():
    learner = DynamicGestureLearner(
        minimum_templates=3,
        minimum_threshold=0.02,
        threshold_multiplier=1.8,
    )

    right_templates = [
        make_swipe(1.0, 0.000),
        make_swipe(1.0, 0.005),
        make_swipe(1.0, -0.005),
    ]
    learner.learn_gesture("Swipe Right", right_templates)

    prediction = learner.predict(make_swipe(1.0, 0.002))
    opposite = learner.predict(make_swipe(-1.0, 0.0))

    assert prediction.accepted
    assert prediction.label == "Swipe Right"
    assert prediction.confidence is not None

    assert not opposite.accepted
    assert opposite.label == learner.UNKNOWN_LABEL
