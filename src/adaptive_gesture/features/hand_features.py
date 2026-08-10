from dataclasses import dataclass
from typing import Literal

import numpy as np

from adaptive_gesture.features.geometry_features import (
    build_hybrid_descriptor,
    normalized_angle_descriptor,
)
from adaptive_gesture.features.normalizer import (
    WRIST,
    calculate_palm_scale_2d,
    normalize_hand_landmarks,
)


FeatureRepresentation = Literal["coordinate", "angle", "hybrid"]


@dataclass
class FrameFeatureSet:
    vector: np.ndarray
    hand_signature: str
    hand_count: int
    representation: FeatureRepresentation = "coordinate"


def _source_landmarks(hand) -> np.ndarray:
    return (
        hand.world_landmarks
        if hand.world_landmarks is not None
        else hand.image_landmarks
    )


def _local_features(
    hand,
    representation: FeatureRepresentation,
) -> np.ndarray:
    landmarks = _source_landmarks(hand)

    if representation == "coordinate":
        return normalize_hand_landmarks(landmarks)
    if representation == "angle":
        return normalized_angle_descriptor(landmarks)
    if representation == "hybrid":
        return build_hybrid_descriptor(landmarks)

    raise ValueError(f"Unsupported feature representation: {representation}")


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


def build_frame_features(
    hands,
    representation: FeatureRepresentation = "coordinate",
) -> FrameFeatureSet | None:
    """Build a static runtime feature representation.

    ``coordinate`` keeps the original V2 representation:
        one hand: 63 local XYZ values
        two hands: 63 + 63 + 3 relative geometry = 129

    ``angle`` uses geometry-aware joint angles:
        one hand: 20 angles
        two hands: 20 + 20 + 3 relative geometry = 43

    ``hybrid`` is the V3 research representation:
        one hand: 83 coordinate + angle values
        two hands: 83 + 83 + 3 relative geometry = 169

    The final three two-hand values preserve coarse relative wrist position,
    while each local hand descriptor remains independent of absolute screen
    position.
    """
    if not hands:
        return None

    if representation not in ("coordinate", "angle", "hybrid"):
        raise ValueError(f"Unsupported feature representation: {representation}")

    if len(hands) == 1:
        hand = hands[0]
        return FrameFeatureSet(
            vector=_local_features(hand, representation),
            hand_signature=hand.handedness,
            hand_count=1,
            representation=representation,
        )

    left, right = _ordered_two_hands(hands)

    left_features = _local_features(left, representation)
    right_features = _local_features(right, representation)

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
        representation=representation,
    )
