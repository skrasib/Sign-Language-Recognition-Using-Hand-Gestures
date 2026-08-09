from dataclasses import dataclass

import numpy as np

from adaptive_gesture.features.normalizer import (
    WRIST,
    calculate_palm_scale_2d,
    normalize_hand_landmarks,
)


@dataclass
class FrameFeatureSet:
    vector: np.ndarray
    hand_signature: str
    hand_count: int


def _local_features(hand) -> np.ndarray:
    landmarks = (
        hand.world_landmarks
        if hand.world_landmarks is not None
        else hand.image_landmarks
    )
    return normalize_hand_landmarks(landmarks)


def _ordered_two_hands(hands):
    """
    Prefer MediaPipe handedness labels. If labels are ambiguous, use image
    wrist x-position as a deterministic fallback.
    """
    left = next((hand for hand in hands if hand.handedness == "Left"), None)
    right = next((hand for hand in hands if hand.handedness == "Right"), None)

    if left is not None and right is not None and left is not right:
        return left, right

    ordered = sorted(
        hands[:2],
        key=lambda hand: float(hand.image_landmarks[WRIST, 0]),
    )
    return ordered[0], ordered[1]


def build_frame_features(hands) -> FrameFeatureSet | None:
    """
    Build the runtime feature representation.

    One hand:
        63 normalized local XYZ features.

    Two hands:
        63 left-local + 63 right-local + 3 relative hand-position features
        = 129 features.

    The relative features preserve coarse spatial relationships between the
    two hands while local hand shapes remain translation/scale normalized.
    """
    if not hands:
        return None

    if len(hands) == 1:
        hand = hands[0]
        return FrameFeatureSet(
            vector=_local_features(hand),
            hand_signature=hand.handedness,
            hand_count=1,
        )

    left, right = _ordered_two_hands(hands)

    left_features = _local_features(left)
    right_features = _local_features(right)

    left_image = np.asarray(left.image_landmarks, dtype=np.float32)
    right_image = np.asarray(right.image_landmarks, dtype=np.float32)

    average_palm_scale = (
        calculate_palm_scale_2d(left_image)
        + calculate_palm_scale_2d(right_image)
    ) / 2.0

    wrist_delta_xy = (
        right_image[WRIST, :2] - left_image[WRIST, :2]
    ) / average_palm_scale

    wrist_distance = float(np.linalg.norm(wrist_delta_xy))

    relative_features = np.array(
        [wrist_delta_xy[0], wrist_delta_xy[1], wrist_distance],
        dtype=np.float32,
    )

    vector = np.concatenate(
        [left_features, right_features, relative_features]
    ).astype(np.float32)

    return FrameFeatureSet(
        vector=vector,
        hand_signature="Both",
        hand_count=2,
    )
