from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_gesture.features.dynamic_features import DynamicTrajectory
from adaptive_gesture.learning.dtw import (
    trajectory_dtw_alignment,
    trajectory_dtw_distance,
)


@dataclass(frozen=True)
class TemporalPrototypeBuildInfo:
    template_count: int
    prototype_count: int
    cluster_sizes: tuple[int, ...]
    iterations: int


def _trajectory_medoid(templates: list[DynamicTrajectory]) -> DynamicTrajectory:
    if not templates:
        raise ValueError("Cannot choose a temporal medoid from an empty list.")
    if len(templates) == 1:
        return templates[0]

    totals = []
    for i, candidate in enumerate(templates):
        total = 0.0
        for j, other in enumerate(templates):
            if i == j:
                continue
            distance = trajectory_dtw_distance(candidate, other)
            if not np.isfinite(distance):
                total = float("inf")
                break
            total += float(distance)
        totals.append(total)

    return templates[int(np.argmin(np.asarray(totals, dtype=np.float64)))]


def _prototype_metadata(
    templates: list[DynamicTrajectory],
    shape: np.ndarray,
    motion: np.ndarray,
) -> tuple[float, int, float, float]:
    duration = float(np.median([item.duration_seconds for item in templates]))
    raw_frames = int(round(np.median([item.raw_frame_count for item in templates])))

    if motion.shape[1] == 2:
        motion_extent = float(np.max(np.linalg.norm(motion, axis=1)))
    else:
        reshaped = motion.reshape(motion.shape[0], -1, 2)
        motion_extent = float(np.max(np.linalg.norm(reshaped, axis=2)))

    delta = shape - shape[0]
    shape_extent = float(np.max(np.sqrt(np.mean(np.square(delta), axis=1))))
    return duration, raw_frames, motion_extent, shape_extent


def _aligned_template_means(
    reference: DynamicTrajectory,
    template: DynamicTrajectory,
) -> tuple[np.ndarray, np.ndarray]:
    _, path = trajectory_dtw_alignment(reference, template)
    if not path:
        raise ValueError("Could not align dynamic trajectories while building prototype.")

    shape = np.empty_like(reference.shape_sequence, dtype=np.float64)
    motion = np.empty_like(reference.motion_sequence, dtype=np.float64)

    buckets: list[list[int]] = [[] for _ in range(reference.length)]
    for reference_index, template_index in path:
        buckets[reference_index].append(template_index)

    # A valid DTW path should touch every reference index. The fallback below is
    # defensive and keeps prototype reconstruction numerically stable.
    last_non_empty = [0]
    for i, indices in enumerate(buckets):
        if indices:
            last_non_empty[0] = indices[-1]
        else:
            buckets[i] = [last_non_empty[0]]

    for reference_index, indices in enumerate(buckets):
        shape[reference_index] = np.mean(
            template.shape_sequence[indices], axis=0, dtype=np.float64
        )
        motion[reference_index] = np.mean(
            template.motion_sequence[indices], axis=0, dtype=np.float64
        )

    return shape, motion


def build_dtw_barycenter(
    templates: list[DynamicTrajectory],
    iterations: int = 4,
) -> DynamicTrajectory:
    """
    Build a lightweight DTW-aligned temporal barycenter.

    This follows the central DBA idea: start from a real medoid, align each
    sequence to the current center with DTW, then update every center time step
    from the observations aligned to it. It is intentionally implemented in
    NumPy and uses the project's existing multivariate DTW cost.
    """
    if not templates:
        raise ValueError("At least one dynamic template is required.")

    signature = templates[0].hand_signature
    shape_dim = templates[0].shape_sequence.shape[1]
    motion_dim = templates[0].motion_sequence.shape[1]
    length = templates[0].length

    for template in templates:
        if template.hand_signature != signature:
            raise ValueError("Temporal prototype templates must use one hand signature.")
        if template.shape_sequence.shape[1] != shape_dim:
            raise ValueError("Temporal prototype shape dimensions do not match.")
        if template.motion_sequence.shape[1] != motion_dim:
            raise ValueError("Temporal prototype motion dimensions do not match.")
        if template.length != length:
            raise ValueError("V3.5 expects pre-resampled trajectories of equal length.")

    medoid = _trajectory_medoid(templates)
    current = DynamicTrajectory(
        hand_signature=signature,
        shape_sequence=np.asarray(medoid.shape_sequence, dtype=np.float32).copy(),
        motion_sequence=np.asarray(medoid.motion_sequence, dtype=np.float32).copy(),
        velocity_sequence=np.asarray(medoid.velocity_sequence, dtype=np.float32).copy(),
        duration_seconds=float(medoid.duration_seconds),
        raw_frame_count=int(medoid.raw_frame_count),
        motion_extent=float(medoid.motion_extent),
        shape_extent=float(medoid.shape_extent),
    )

    iterations = max(1, int(iterations))
    for _ in range(iterations):
        aligned_shapes = []
        aligned_motions = []
        for template in templates:
            shape, motion = _aligned_template_means(current, template)
            aligned_shapes.append(shape)
            aligned_motions.append(motion)

        new_shape = np.mean(np.stack(aligned_shapes, axis=0), axis=0).astype(np.float32)
        new_motion = np.mean(np.stack(aligned_motions, axis=0), axis=0).astype(np.float32)

        # Motion features are defined relative to gesture start. Numerical
        # averaging can introduce a tiny nonzero start, so restore that invariant.
        new_motion = new_motion - new_motion[:1]
        new_velocity = np.diff(
            new_motion,
            axis=0,
            prepend=new_motion[:1],
        ).astype(np.float32)

        duration, raw_frames, motion_extent, shape_extent = _prototype_metadata(
            templates,
            new_shape,
            new_motion,
        )
        current = DynamicTrajectory(
            hand_signature=signature,
            shape_sequence=new_shape,
            motion_sequence=new_motion,
            velocity_sequence=new_velocity,
            duration_seconds=duration,
            raw_frame_count=raw_frames,
            motion_extent=motion_extent,
            shape_extent=shape_extent,
        )

    return current


def _farthest_pair(templates: list[DynamicTrajectory]) -> tuple[int, int]:
    best_pair = (0, 1)
    best_distance = -1.0
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            distance = trajectory_dtw_distance(templates[i], templates[j])
            if np.isfinite(distance) and distance > best_distance:
                best_pair = (i, j)
                best_distance = float(distance)
    return best_pair


def _two_cluster_partition(
    templates: list[DynamicTrajectory],
) -> tuple[list[DynamicTrajectory], list[DynamicTrajectory]] | None:
    if len(templates) < 5:
        return None

    first_seed, second_seed = _farthest_pair(templates)
    cluster_a: list[DynamicTrajectory] = []
    cluster_b: list[DynamicTrajectory] = []

    for template in templates:
        distance_a = trajectory_dtw_distance(template, templates[first_seed])
        distance_b = trajectory_dtw_distance(template, templates[second_seed])
        if distance_a <= distance_b:
            cluster_a.append(template)
        else:
            cluster_b.append(template)

    # Do not manufacture a second temporal mode from a single outlier.
    if len(cluster_a) < 2 or len(cluster_b) < 2:
        return None
    return cluster_a, cluster_b


def build_temporal_prototypes(
    templates: list[DynamicTrajectory],
    max_prototypes: int = 2,
    iterations: int = 4,
) -> tuple[list[DynamicTrajectory], TemporalPrototypeBuildInfo]:
    """Build one or, when supported by enough data, two DTW barycenters."""
    if not templates:
        return [], TemporalPrototypeBuildInfo(0, 0, (), max(1, int(iterations)))

    max_prototypes = max(1, int(max_prototypes))
    clusters: list[list[DynamicTrajectory]] = [templates]

    if max_prototypes >= 2:
        partition = _two_cluster_partition(templates)
        if partition is not None:
            clusters = [partition[0], partition[1]]

    prototypes = [
        build_dtw_barycenter(cluster, iterations=iterations)
        for cluster in clusters[:max_prototypes]
    ]
    info = TemporalPrototypeBuildInfo(
        template_count=len(templates),
        prototype_count=len(prototypes),
        cluster_sizes=tuple(len(cluster) for cluster in clusters[:max_prototypes]),
        iterations=max(1, int(iterations)),
    )
    return prototypes, info
