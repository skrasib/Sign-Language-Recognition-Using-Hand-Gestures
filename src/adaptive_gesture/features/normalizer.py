import numpy as np


WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert 21 XYZ hand landmarks into a translation- and
    scale-normalized representation.

    Parameters
    ----------
    landmarks:
        NumPy array with shape (21, 3).

    Returns
    -------
    np.ndarray:
        Flattened feature vector containing 63 values.
    """

    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.shape != (21, 3):
        raise ValueError(
            f"Expected landmarks with shape (21, 3), got {landmarks.shape}"
        )

    normalized = landmarks.copy()

    # ---------------------------------------------------------
    # 1. Translation normalization
    #
    # Move the wrist to (0, 0, 0).
    #
    # This means the location of the hand inside the camera
    # frame should no longer matter.
    # ---------------------------------------------------------

    wrist = normalized[WRIST].copy()

    normalized -= wrist

    # ---------------------------------------------------------
    # 2. Scale normalization
    #
    # Estimate palm size using three stable MCP joints.
    #
    # This reduces the effect of:
    # - distance from the camera
    # - different physical hand sizes
    # ---------------------------------------------------------

    palm_points = normalized[
        [
            INDEX_MCP,
            MIDDLE_MCP,
            PINKY_MCP,
        ]
    ]

    palm_distances = np.linalg.norm(
        palm_points,
        axis=1,
    )

    palm_scale = float(np.mean(palm_distances))

    if palm_scale < 1e-6:
        raise ValueError("Palm scale is too small to normalize landmarks.")

    normalized /= palm_scale

    # 21 × XYZ → 63 features
    return normalized.flatten().astype(np.float32)