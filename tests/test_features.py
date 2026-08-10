from dataclasses import dataclass

import numpy as np

from adaptive_gesture.features.hand_features import build_frame_features
from adaptive_gesture.features.normalizer import normalize_hand_landmarks
from adaptive_gesture.features.similarity import feature_distance


@dataclass
class FakeHand:
    handedness: str
    image_landmarks: np.ndarray
    world_landmarks: np.ndarray | None = None
    handedness_score: float = 0.99


def make_landmarks(x_offset: float = 0.0) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    for index in range(21):
        points[index] = [x_offset + 0.01 * index, 0.015 * index, 0.003 * index]
    points[5] = [x_offset + 0.08, 0.03, 0.01]
    points[9] = [x_offset + 0.10, 0.04, 0.01]
    points[17] = [x_offset + 0.07, 0.06, 0.01]
    return points


def test_normalization_is_translation_invariant():
    first = make_landmarks(0.0)
    second = first + np.array([0.35, -0.20, 0.10], dtype=np.float32)

    first_features = normalize_hand_landmarks(first)
    second_features = normalize_hand_landmarks(second)

    assert first_features.shape == (63,)
    assert np.allclose(first_features, second_features, atol=1e-5)


def test_one_and_two_hand_feature_dimensions():
    right = FakeHand("Right", make_landmarks(0.20))
    left = FakeHand("Left", make_landmarks(0.55))

    one = build_frame_features([right], representation="coordinate")
    two = build_frame_features([left, right], representation="coordinate")
    one_hybrid = build_frame_features([right], representation="hybrid")
    two_hybrid = build_frame_features([left, right], representation="hybrid")
    one_angle = build_frame_features([right], representation="angle")
    two_angle = build_frame_features([left, right], representation="angle")

    assert one is not None
    assert one.vector.shape == (63,)
    assert one.hand_count == 1
    assert one.hand_signature == "Right"

    assert two is not None
    assert two.vector.shape == (129,)
    assert two.hand_count == 2
    assert two.hand_signature == "Both"

    assert one_hybrid is not None
    assert one_hybrid.vector.shape == (83,)
    assert one_hybrid.representation == "hybrid"

    assert two_hybrid is not None
    assert two_hybrid.vector.shape == (169,)

    assert one_angle is not None
    assert one_angle.vector.shape == (20,)

    assert two_angle is not None
    assert two_angle.vector.shape == (43,)


def test_feature_distance_zero_for_identical_vectors():
    vector = normalize_hand_landmarks(make_landmarks())
    assert feature_distance(vector, vector.copy()) == 0.0
