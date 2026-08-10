import numpy as np

from adaptive_gesture.features.geometry_features import (
    ANGLE_DIMENSION,
    HYBRID_DIMENSION,
    build_hybrid_descriptor,
    compute_joint_angles,
    normalized_angle_descriptor,
)


def make_hand() -> np.ndarray:
    # Simple non-degenerate synthetic hand with five articulated fingers.
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = [0.0, 0.0, 0.0]

    chains = {
        1: ([0.25, 0.10, 0.02], [0.40, 0.23, 0.05], [0.52, 0.36, 0.08], [0.61, 0.49, 0.10]),
        5: ([0.18, 0.34, 0.01], [0.19, 0.58, 0.04], [0.20, 0.79, 0.07], [0.21, 0.97, 0.10]),
        9: ([0.02, 0.38, 0.00], [0.02, 0.67, 0.03], [0.03, 0.92, 0.05], [0.03, 1.14, 0.07]),
        13: ([-0.14, 0.35, 0.00], [-0.16, 0.61, 0.02], [-0.17, 0.83, 0.04], [-0.18, 1.01, 0.06]),
        17: ([-0.28, 0.28, -0.01], [-0.32, 0.49, 0.01], [-0.35, 0.67, 0.03], [-0.38, 0.82, 0.05]),
    }
    for start, chain in chains.items():
        for offset, value in enumerate(chain):
            points[start + offset] = value
    return points


def rotation_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def test_geometry_angle_dimension_and_range():
    angles = compute_joint_angles(make_hand())
    assert angles.shape == (ANGLE_DIMENSION,)
    assert ANGLE_DIMENSION == 20
    assert np.all(np.isfinite(angles))
    assert np.all(angles >= 0.0)
    assert np.all(angles <= np.pi)


def test_angle_descriptor_is_similarity_transform_invariant():
    original = make_hand()
    transformed = 2.75 * (original @ rotation_z(0.83).T)
    transformed += np.array([4.0, -3.2, 1.7], dtype=np.float32)

    first = compute_joint_angles(original)
    second = compute_joint_angles(transformed)

    assert np.allclose(first, second, atol=1e-5)


def test_normalized_angles_are_in_unit_interval():
    descriptor = normalized_angle_descriptor(make_hand())
    assert descriptor.shape == (20,)
    assert np.all(descriptor >= 0.0)
    assert np.all(descriptor <= 1.0)


def test_hybrid_descriptor_dimension_and_finiteness():
    hybrid = build_hybrid_descriptor(make_hand())
    assert hybrid.shape == (HYBRID_DIMENSION,)
    assert HYBRID_DIMENSION == 83
    assert np.all(np.isfinite(hybrid))
