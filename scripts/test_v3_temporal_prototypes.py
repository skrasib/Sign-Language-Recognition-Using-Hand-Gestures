"""Small deterministic smoke test for the V3.5 temporal-prototype engine."""

import numpy as np

from adaptive_gesture.features.dynamic_features import DynamicTrajectory
from adaptive_gesture.learning.dtw import trajectory_dtw_distance
from adaptive_gesture.learning.dynamic_learner import DynamicGestureLearner


def make_arc(bend: float, warp: float) -> DynamicTrajectory:
    frames = 48
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    tw = np.power(t, warp).astype(np.float32)

    shape = np.zeros((frames, 63), dtype=np.float32)
    shape[:, 8] = 0.025 * np.sin(np.pi * tw)

    motion = np.column_stack([tw, bend * np.sin(np.pi * tw)]).astype(np.float32)
    motion -= motion[:1]
    velocity = np.diff(motion, axis=0, prepend=motion[:1]).astype(np.float32)

    return DynamicTrajectory(
        hand_signature="Right",
        shape_sequence=shape,
        motion_sequence=motion,
        velocity_sequence=velocity,
        duration_seconds=1.0,
        raw_frame_count=52,
        motion_extent=float(np.max(np.linalg.norm(motion, axis=1))),
        shape_extent=float(np.max(np.abs(shape[:, 8]))),
    )


def main() -> None:
    learner = DynamicGestureLearner(
        minimum_templates=3,
        minimum_threshold=0.02,
        threshold_multiplier=1.8,
        temporal_prototype_strategy="dtw_barycenter",
        max_temporal_prototypes=2,
        prototype_iterations=4,
    )

    templates = [
        make_arc(0.25, 0.85),
        make_arc(0.25, 1.00),
        make_arc(0.25, 1.20),
    ]
    gesture = learner.learn_gesture("Arc Right", templates)

    print("V3.5 temporal prototype smoke test")
    print(f"  demonstrations: {gesture.template_count}")
    print(f"  temporal prototypes: {gesture.prototype_count}")
    print(f"  threshold: {gesture.threshold:.5f}")

    prototype = gesture.temporal_prototypes[0]
    print("  demo->prototype DTW:")
    for index, template in enumerate(templates, start=1):
        print(f"    demo {index}: {trajectory_dtw_distance(template, prototype):.5f}")

    prediction = learner.predict(make_arc(0.25, 1.08))
    opposite = learner.predict(make_arc(-0.55, 1.00))

    print(
        f"  in-class query: {prediction.label}, accepted={prediction.accepted}, "
        f"distance={prediction.distance:.5f}"
    )
    print(
        f"  different motion: {opposite.label}, accepted={opposite.accepted}, "
        f"distance={opposite.distance:.5f}"
    )

    assert gesture.prototype_count == 1
    assert prediction.accepted and prediction.label == "Arc Right"
    assert not opposite.accepted
    print("V3.5 temporal prototype smoke test: PASS")


if __name__ == "__main__":
    main()
