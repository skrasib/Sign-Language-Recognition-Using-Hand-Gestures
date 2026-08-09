import math

import numpy as np

from adaptive_gesture.features.dynamic_features import DynamicTrajectory


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b))))


def dynamic_local_cost(
    first: DynamicTrajectory,
    i: int,
    second: DynamicTrajectory,
    j: int,
    shape_weight: float = 0.25,
    motion_weight: float = 0.55,
    velocity_weight: float = 0.20,
) -> float:
    """Weighted multivariate frame cost for dynamic hand gestures."""
    shape_distance = _rmse(first.shape_sequence[i], second.shape_sequence[j])
    motion_distance = _rmse(first.motion_sequence[i], second.motion_sequence[j])
    velocity_distance = _rmse(
        first.velocity_sequence[i],
        second.velocity_sequence[j],
    )

    total_weight = shape_weight + motion_weight + velocity_weight
    if total_weight <= 0:
        raise ValueError("DTW feature weights must sum to a positive value.")

    return (
        shape_weight * shape_distance
        + motion_weight * motion_distance
        + velocity_weight * velocity_distance
    ) / total_weight


def trajectory_dtw_distance(
    first: DynamicTrajectory,
    second: DynamicTrajectory,
    window_ratio: float = 0.20,
    shape_weight: float = 0.25,
    motion_weight: float = 0.55,
    velocity_weight: float = 0.20,
) -> float:
    """
    Exact, Sakoe-Chiba-constrained multivariate DTW with path-length
    normalization.

    The implementation intentionally uses only NumPy, avoiding a new compiled
    dependency in the existing project environment.
    """
    if first.hand_signature != second.hand_signature:
        return float("inf")
    if first.shape_sequence.shape[1] != second.shape_sequence.shape[1]:
        return float("inf")
    if first.motion_sequence.shape[1] != second.motion_sequence.shape[1]:
        return float("inf")

    n = first.length
    m = second.length
    if n == 0 or m == 0:
        return float("inf")

    window = max(
        abs(n - m),
        int(math.ceil(max(n, m) * max(float(window_ratio), 0.0))),
    )

    inf = float("inf")
    previous_cost = np.full(m + 1, inf, dtype=np.float64)
    previous_steps = np.zeros(m + 1, dtype=np.int32)
    previous_cost[0] = 0.0

    for i in range(1, n + 1):
        current_cost = np.full(m + 1, inf, dtype=np.float64)
        current_steps = np.zeros(m + 1, dtype=np.int32)

        j_start = max(1, i - window)
        j_end = min(m, i + window)

        for j in range(j_start, j_end + 1):
            predecessors = (
                (previous_cost[j - 1], previous_steps[j - 1]),  # diagonal
                (previous_cost[j], previous_steps[j]),          # vertical
                (current_cost[j - 1], current_steps[j - 1]),    # horizontal
            )
            predecessor_cost, predecessor_steps = min(
                predecessors,
                key=lambda item: item[0],
            )

            if not np.isfinite(predecessor_cost):
                continue

            local = dynamic_local_cost(
                first,
                i - 1,
                second,
                j - 1,
                shape_weight=shape_weight,
                motion_weight=motion_weight,
                velocity_weight=velocity_weight,
            )

            current_cost[j] = predecessor_cost + local
            current_steps[j] = predecessor_steps + 1

        previous_cost = current_cost
        previous_steps = current_steps

    final_cost = float(previous_cost[m])
    path_steps = int(previous_steps[m])

    if not np.isfinite(final_cost) or path_steps <= 0:
        return float("inf")

    return final_cost / float(path_steps)
