"""Geometry-aware hand descriptors for the V3 research branch.

The 20-angle topology is inspired by the public implementation accompanying
Chamachot & Lertniponphan, CVPR Workshops 2026, but integrated independently
into this project's live, user-defined few-shot recognition pipeline.

This module deliberately keeps geometry extraction separate from the learner so
we can later run ablations between coordinate, angle, and hybrid descriptors.
"""

from __future__ import annotations

import numpy as np

from adaptive_gesture.features.normalizer import normalize_hand_landmarks


# 15 within-finger flexion angles + 5 palm/finger-spread angles = 20.
# Each triplet (A, B, C) measures the angle at pivot B between BA and BC.
ANGLE_TRIPLETS: tuple[tuple[int, int, int], ...] = (
    # Thumb
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    # Index
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    # Middle
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    # Ring
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    # Pinky
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
    # Palm / finger spread
    (5, 0, 9),
    (9, 0, 13),
    (13, 0, 17),
    (1, 0, 5),
    (17, 0, 1),
)

ANGLE_DIMENSION = len(ANGLE_TRIPLETS)
COORDINATE_DIMENSION = 63
HYBRID_DIMENSION = COORDINATE_DIMENSION + ANGLE_DIMENSION


def _validate_landmarks(landmarks: np.ndarray) -> np.ndarray:
    array = np.asarray(landmarks, dtype=np.float32)
    if array.shape != (21, 3):
        raise ValueError(
            f"Expected hand landmarks with shape (21, 3), got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Hand landmarks contain NaN or infinite values.")
    return array


def compute_joint_angles(landmarks: np.ndarray) -> np.ndarray:
    """Return 20 geometry-aware inter-joint angles in radians.

    The descriptor is inherently invariant to global translation, uniform
    scaling, and proper 3D rotation because each value depends only on the
    normalized dot product between two local displacement vectors.
    """
    points = _validate_landmarks(landmarks)
    angles = np.empty(ANGLE_DIMENSION, dtype=np.float32)

    for index, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        ba = points[a] - points[b]
        bc = points[c] - points[b]

        denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
        if denominator < 1e-8:
            # Degenerate landmarks should be rare with MediaPipe. Returning a
            # neutral zero value is preferable to propagating NaNs into online
            # memory and distance calculations.
            angles[index] = 0.0
            continue

        cosine = float(np.dot(ba, bc) / denominator)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        angles[index] = np.arccos(cosine)

    return angles


def normalized_angle_descriptor(landmarks: np.ndarray) -> np.ndarray:
    """Return angles scaled from radians to [0, 1] for direct metric use."""
    return (compute_joint_angles(landmarks) / np.pi).astype(np.float32)


def _zscore_block(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    mean = float(np.mean(vector))
    std = float(np.std(vector))
    if std < 1e-8:
        return (vector - mean).astype(np.float32)
    return ((vector - mean) / std).astype(np.float32)


def build_hybrid_descriptor(landmarks: np.ndarray) -> np.ndarray:
    """Build an 83-D coordinate + geometry descriptor.

    The coordinate block preserves pose/orientation information already useful
    to the V2 recognizer. The angle block contributes rotation/translation/
    scale-invariant hand-shape information. Each block is standardized
    independently before concatenation so one feature family does not dominate
    only because of numerical scale.

    This is intentionally a *hybrid metric descriptor*, not yet the learned
    128-D MLP embedding from the reference paper. A learned encoder is planned
    as a later V3 stage so the representation change can first be evaluated in
    isolation.
    """
    points = _validate_landmarks(landmarks)
    coordinate = normalize_hand_landmarks(points)
    angles = compute_joint_angles(points)

    coordinate = _zscore_block(coordinate)
    angles = _zscore_block(angles)

    hybrid = np.concatenate([coordinate, angles]).astype(np.float32)
    if hybrid.shape != (HYBRID_DIMENSION,):
        raise RuntimeError(f"Unexpected hybrid descriptor shape: {hybrid.shape}")
    return hybrid
