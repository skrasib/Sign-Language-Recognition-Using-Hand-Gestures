from pathlib import Path

import numpy as np

from adaptive_gesture.features.dynamic_features import DynamicTrajectory
from adaptive_gesture.learning.dynamic_learner import DynamicGestureLearner
from adaptive_gesture.learning.online_learner import OnlineGestureLearner
from adaptive_gesture.storage.dynamic_gesture_store import DynamicGestureStore
from adaptive_gesture.storage.gesture_store import GestureStore


def static_samples(center: float):
    return [
        np.full(63, center + index * 0.0005, dtype=np.float32)
        for index in range(12)
    ]


def dynamic_template(offset: float) -> DynamicTrajectory:
    frames = 24
    shape = np.zeros((frames, 63), dtype=np.float32)
    motion = np.column_stack(
        [
            np.linspace(0.0, 1.0 + offset, frames, dtype=np.float32),
            np.zeros(frames, dtype=np.float32),
        ]
    )
    velocity = np.diff(motion, axis=0, prepend=motion[:1]).astype(np.float32)
    return DynamicTrajectory(
        hand_signature="Right",
        shape_sequence=shape,
        motion_sequence=motion,
        velocity_sequence=velocity,
        duration_seconds=1.0,
        raw_frame_count=frames,
        motion_extent=1.0,
        shape_extent=0.0,
    )


def test_static_memory_round_trip(tmp_path: Path):
    path = tmp_path / "gesture_memory.json"
    learner = OnlineGestureLearner()
    learner.learn_gesture("Victory", static_samples(0.0), hand_signature="Right")
    learner.add_hard_negative("Victory", np.full(63, 0.8, dtype=np.float32))

    GestureStore(path).save(learner)

    restored = OnlineGestureLearner()
    count = GestureStore(path).load_into(restored)

    assert count == 1
    assert "Victory" in restored.gestures
    assert restored.gestures["Victory"].negative_count == 1


def test_dynamic_memory_round_trip(tmp_path: Path):
    path = tmp_path / "dynamic_gesture_memory.json"
    learner = DynamicGestureLearner(minimum_templates=3)
    learner.learn_gesture(
        "Swipe Right",
        [dynamic_template(0.0), dynamic_template(0.01), dynamic_template(-0.01)],
    )

    DynamicGestureStore(path).save(learner)

    restored = DynamicGestureLearner(minimum_templates=3)
    count = DynamicGestureStore(path).load_into(restored)

    assert count == 1
    assert "Swipe Right" in restored.gestures
    assert restored.gestures["Swipe Right"].template_count == 3
