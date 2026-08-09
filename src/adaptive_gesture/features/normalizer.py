import numpy as np


WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert 21 XYZ hand landmarks into a translation- and scale-normalized
    63-dimensional representation.
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.shape != (21, 3):
        raise ValueError(
            f"Expected landmarks with shape (21, 3), got {landmarks.shape}"
        )

    normalized = landmarks.copy()

    # Translation invariance: wrist becomes origin.
    wrist = normalized[WRIST].copy()
    normalized -= wrist

    # Scale invariance: normalize by stable palm MCP geometry.
    palm_points = normalized[[INDEX_MCP, MIDDLE_MCP, PINKY_MCP]]
    palm_distances = np.linalg.norm(palm_points, axis=1)
    palm_scale = float(np.mean(palm_distances))

    if palm_scale < 1e-6:
        raise ValueError("Palm scale is too small to normalize landmarks.")

    normalized /= palm_scale
    return normalized.flatten().astype(np.float32)


def calculate_palm_scale_2d(image_landmarks: np.ndarray) -> float:
    """Palm scale in normalized image coordinates, used for two-hand geometry."""
    landmarks = np.asarray(image_landmarks, dtype=np.float32)
    if landmarks.shape != (21, 3):
        raise ValueError(
            f"Expected image landmarks with shape (21, 3), got {landmarks.shape}"
        )

    wrist_xy = landmarks[WRIST, :2]
    palm_xy = landmarks[[INDEX_MCP, MIDDLE_MCP, PINKY_MCP], :2]
    distances = np.linalg.norm(palm_xy - wrist_xy, axis=1)
    scale = float(np.mean(distances))
    return max(scale, 1e-6)
