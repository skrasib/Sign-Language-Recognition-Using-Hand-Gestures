from dataclasses import dataclass

import numpy as np

from adaptive_gesture.features.hand_features import build_frame_features
from adaptive_gesture.features.normalizer import WRIST, calculate_palm_scale_2d


@dataclass
class DynamicObservation:
    """One temporally sampled hand observation used for a dynamic gesture."""

    timestamp: float
    hand_signature: str
    pose_vector: np.ndarray
    anchor_xy: np.ndarray
    palm_scale: float


@dataclass
class DynamicTrajectory:
    """
    Privacy-preserving representation of one gesture demonstration.

    shape_sequence:
        Translation/scale-normalized hand-shape features for every resampled
        time step.

    motion_sequence:
        Wrist displacement from the beginning of the gesture, normalized by
        palm scale. This deliberately preserves motion direction while removing
        absolute screen position and user/camera scale.

    velocity_sequence:
        First temporal difference of the normalized wrist trajectory.
    """

    hand_signature: str
    shape_sequence: np.ndarray
    motion_sequence: np.ndarray
    velocity_sequence: np.ndarray
    duration_seconds: float
    raw_frame_count: int
    motion_extent: float
    shape_extent: float

    @property
    def length(self) -> int:
        return int(self.shape_sequence.shape[0])


def _ordered_two_hands(hands):
    left = next((hand for hand in hands if hand.handedness == "Left"), None)
    right = next((hand for hand in hands if hand.handedness == "Right"), None)

    if left is not None and right is not None and left is not right:
        return left, right

    ordered = sorted(
        hands[:2],
        key=lambda hand: float(hand.image_landmarks[WRIST, 0]),
    )
    return ordered[0], ordered[1]


def build_dynamic_observation(hands, timestamp: float) -> DynamicObservation | None:
    """Build a dynamic observation from the already-stabilized tracker output."""
    frame_features = build_frame_features(hands, representation="coordinate")
    if frame_features is None:
        return None

    if frame_features.hand_count == 1:
        hand = hands[0]
        image = np.asarray(hand.image_landmarks, dtype=np.float32)
        anchor = image[WRIST, :2].astype(np.float32)
        palm_scale = calculate_palm_scale_2d(image)
    else:
        left, right = _ordered_two_hands(hands)
        left_image = np.asarray(left.image_landmarks, dtype=np.float32)
        right_image = np.asarray(right.image_landmarks, dtype=np.float32)

        anchor = np.concatenate(
            [left_image[WRIST, :2], right_image[WRIST, :2]]
        ).astype(np.float32)
        palm_scale = (
            calculate_palm_scale_2d(left_image)
            + calculate_palm_scale_2d(right_image)
        ) / 2.0

    return DynamicObservation(
        timestamp=float(timestamp),
        hand_signature=frame_features.hand_signature,
        pose_vector=np.asarray(frame_features.vector, dtype=np.float32).copy(),
        anchor_xy=anchor,
        palm_scale=max(float(palm_scale), 1e-6),
    )


def observation_motion_score(
    previous: DynamicObservation,
    current: DynamicObservation,
    pose_weight: float = 0.35,
) -> float:
    """
    Motion energy between two consecutive observations.

    Wrist displacement is measured in palm-size units. A smaller pose-change
    term also allows articulated hand motion to contribute to segmentation.
    """
    if previous.hand_signature != current.hand_signature:
        return float("inf")
    if previous.pose_vector.shape != current.pose_vector.shape:
        return float("inf")
    if previous.anchor_xy.shape != current.anchor_xy.shape:
        return float("inf")

    scale = max((previous.palm_scale + current.palm_scale) / 2.0, 1e-6)
    anchor_delta = (current.anchor_xy - previous.anchor_xy) / scale

    if anchor_delta.size == 2:
        anchor_motion = float(np.linalg.norm(anchor_delta))
    else:
        # Two hands: use the more active wrist so one stationary hand does not
        # hide meaningful motion of the other.
        per_hand = anchor_delta.reshape(-1, 2)
        anchor_motion = float(np.max(np.linalg.norm(per_hand, axis=1)))

    pose_motion = float(
        np.sqrt(np.mean(np.square(current.pose_vector - previous.pose_vector)))
    )

    return anchor_motion + pose_weight * pose_motion


def _moving_average(sequence: np.ndarray, window: int = 3) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape[0] < 3 or window <= 1:
        return sequence.copy()

    window = min(int(window), sequence.shape[0])
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return sequence.copy()

    pad = window // 2
    padded = np.pad(sequence, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)

    smoothed = np.empty_like(sequence, dtype=np.float32)
    for dimension in range(sequence.shape[1]):
        smoothed[:, dimension] = np.convolve(
            padded[:, dimension],
            kernel,
            mode="valid",
        )
    return smoothed


def _resample(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape[0] == target_frames:
        return sequence.copy()

    old_t = np.linspace(0.0, 1.0, sequence.shape[0], dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)

    output = np.empty((target_frames, sequence.shape[1]), dtype=np.float32)
    for dimension in range(sequence.shape[1]):
        output[:, dimension] = np.interp(
            new_t,
            old_t,
            sequence[:, dimension],
        )
    return output


def _trim_motion(
    observations: list[DynamicObservation],
    threshold: float = 0.020,
    padding_frames: int = 2,
) -> list[DynamicObservation]:
    if len(observations) < 3:
        return observations

    scores = np.array(
        [
            observation_motion_score(observations[i - 1], observations[i])
            for i in range(1, len(observations))
        ],
        dtype=np.float32,
    )

    moving = np.where(np.isfinite(scores) & (scores >= threshold))[0]
    if moving.size == 0:
        return observations

    # score index k describes transition k -> k+1.
    start = max(0, int(moving[0]) - padding_frames)
    end = min(len(observations), int(moving[-1]) + 2 + padding_frames)
    return observations[start:end]


def prepare_dynamic_trajectory(
    observations: list[DynamicObservation],
    target_frames: int = 48,
    minimum_frames: int = 10,
    maximum_frames: int = 140,
    minimum_motion_extent: float = 0.10,
) -> DynamicTrajectory:
    """
    Convert a raw live demonstration into a position/scale-normalized temporal
    trajectory suitable for multivariate DTW.

    No images or video frames are retained.
    """
    if len(observations) < minimum_frames:
        raise ValueError(
            f"Dynamic gesture is too short. Need at least {minimum_frames} sampled frames."
        )

    if len(observations) > maximum_frames:
        observations = observations[-maximum_frames:]

    signature = observations[0].hand_signature
    pose_shape = observations[0].pose_vector.shape
    anchor_shape = observations[0].anchor_xy.shape

    for observation in observations:
        if observation.hand_signature != signature:
            raise ValueError("Hand configuration changed during the gesture.")
        if observation.pose_vector.shape != pose_shape:
            raise ValueError("Pose feature dimension changed during the gesture.")
        if observation.anchor_xy.shape != anchor_shape:
            raise ValueError("Motion anchor dimension changed during the gesture.")

    observations = _trim_motion(observations)
    if len(observations) < minimum_frames:
        raise ValueError("Not enough motion remained after trimming idle frames.")

    timestamps = np.array([obs.timestamp for obs in observations], dtype=np.float64)
    duration = float(max(timestamps[-1] - timestamps[0], 1e-3))

    poses = np.stack([obs.pose_vector for obs in observations]).astype(np.float32)
    anchors = np.stack([obs.anchor_xy for obs in observations]).astype(np.float32)

    # Robust scale from the beginning of the motion rather than one potentially
    # noisy frame.
    initial_count = min(5, len(observations))
    base_scale = float(
        np.median([obs.palm_scale for obs in observations[:initial_count]])
    )
    base_scale = max(base_scale, 1e-6)

    motion = (anchors - anchors[0]) / base_scale

    # Light temporal smoothing before resampling reduces landmark jitter while
    # retaining the overall path shape.
    poses = _moving_average(poses, window=3)
    motion = _moving_average(motion, window=3)

    # Validate that the demonstration actually contains meaningful movement.
    if motion.shape[1] == 2:
        motion_extent = float(np.max(np.linalg.norm(motion, axis=1)))
    else:
        reshaped = motion.reshape(motion.shape[0], -1, 2)
        motion_extent = float(np.max(np.linalg.norm(reshaped, axis=2)))

    pose_delta = poses - poses[0]
    shape_extent = float(
        np.max(np.sqrt(np.mean(np.square(pose_delta), axis=1)))
    )

    # Wrist translation is the primary signal for swipes/waves/circles. Shape
    # articulation can still make a small-motion dynamic gesture valid.
    combined_extent = motion_extent + 1.5 * shape_extent
    if combined_extent < minimum_motion_extent:
        raise ValueError(
            "Too little movement detected. Perform the dynamic gesture more clearly."
        )

    target_frames = max(16, int(target_frames))
    pose_resampled = _resample(poses, target_frames)
    motion_resampled = _resample(motion, target_frames)

    velocity = np.diff(
        motion_resampled,
        axis=0,
        prepend=motion_resampled[:1],
    ).astype(np.float32)

    return DynamicTrajectory(
        hand_signature=signature,
        shape_sequence=pose_resampled.astype(np.float32),
        motion_sequence=motion_resampled.astype(np.float32),
        velocity_sequence=velocity,
        duration_seconds=duration,
        raw_frame_count=len(observations),
        motion_extent=motion_extent,
        shape_extent=shape_extent,
    )
