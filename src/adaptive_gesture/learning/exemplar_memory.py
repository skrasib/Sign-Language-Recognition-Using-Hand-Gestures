"""Bounded diversity-aware exemplar memory for V3.4.

The live recognizer learns from a small number of numerical hand descriptors rather
than stored images.  Once feedback accumulates beyond a class memory budget, V2/V3
historically kept the most recent examples (FIFO).  V3.4 instead keeps a compact
core-set that covers the geometry of the class in the current feature space.

This is deliberately lightweight and deterministic:

* positive exemplars: class medoid + farthest-first (k-center-style) traversal;
* hard negatives: retain a pool of the negatives nearest the positive class
  (the most useful boundary evidence), then apply farthest-first selection inside
  that pool so the EVT/open-set learner sees diverse mistakes rather than many
  near-duplicates.

The implementation is inspired by core-set selection and exemplar-memory work; it
is not claimed to reproduce a particular paper's complete training algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MemorySelectionReport:
    """Small diagnostic record useful for tests and later ablation logging."""

    input_count: int
    retained_count: int
    dropped_count: int
    strategy: str
    coverage_radius_before: float | None = None
    coverage_radius_after: float | None = None


def _prepare_matrix(samples: list[np.ndarray]) -> np.ndarray:
    if not samples:
        return np.empty((0, 0), dtype=np.float32)

    arrays = [np.asarray(sample, dtype=np.float32).reshape(-1) for sample in samples]
    dimension = arrays[0].shape[0]
    if any(array.shape != (dimension,) for array in arrays):
        raise ValueError("All memory exemplars must have the same feature dimension.")
    return np.stack(arrays, axis=0)


def _pairwise_rmse(matrix: np.ndarray) -> np.ndarray:
    """Pairwise distance matching the project's feature-distance scale (RMSE)."""
    if matrix.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)
    differences = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.mean(differences * differences, axis=2)).astype(np.float32)


def _coverage_radius(distance_matrix: np.ndarray, selected_indices: list[int]) -> float:
    if distance_matrix.size == 0 or not selected_indices:
        return 0.0
    nearest = np.min(distance_matrix[:, selected_indices], axis=1)
    return float(np.max(nearest))


def _farthest_first_indices(
    distance_matrix: np.ndarray,
    budget: int,
    seed_indices: list[int] | None = None,
    candidate_indices: list[int] | None = None,
) -> list[int]:
    """Deterministic farthest-first traversal used as a small k-center approximation."""
    count = distance_matrix.shape[0]
    if count == 0 or budget <= 0:
        return []

    candidates = (
        list(range(count)) if candidate_indices is None else list(dict.fromkeys(candidate_indices))
    )
    candidates = [index for index in candidates if 0 <= index < count]
    if not candidates:
        return []

    selected: list[int] = []
    for index in seed_indices or []:
        if index in candidates and index not in selected:
            selected.append(index)
            if len(selected) >= budget:
                return selected

    if not selected:
        selected.append(candidates[0])

    while len(selected) < min(budget, len(candidates)):
        remaining = [index for index in candidates if index not in selected]
        if not remaining:
            break

        # Choose the point whose nearest selected exemplar is farthest away.
        # np.argmax is deterministic and therefore gives reproducible memories.
        nearest = np.asarray(
            [float(np.min(distance_matrix[index, selected])) for index in remaining],
            dtype=np.float64,
        )
        chosen = remaining[int(np.argmax(nearest))]
        selected.append(chosen)

    return selected


def select_diverse_exemplars(
    samples: list[np.ndarray],
    budget: int,
) -> tuple[list[np.ndarray], MemorySelectionReport]:
    """Select a compact class core-set using a medoid + farthest-first traversal.

    The medoid anchors the class centre.  Farthest-first additions then retain
    natural extremes/variations so a fixed memory budget covers more of the
    gesture than a recency-only FIFO buffer.
    """
    budget = max(1, int(budget))
    matrix = _prepare_matrix(samples)
    count = matrix.shape[0]

    if count <= budget:
        retained = [row.copy() for row in matrix]
        return retained, MemorySelectionReport(
            input_count=count,
            retained_count=count,
            dropped_count=0,
            strategy="diversity_kcenter",
            coverage_radius_before=0.0,
            coverage_radius_after=0.0,
        )

    distances = _pairwise_rmse(matrix)

    # The medoid is the observed sample with the smallest mean distance to the
    # rest of the class.  Unlike a synthetic mean vector it is a real exemplar.
    medoid = int(np.argmin(np.mean(distances, axis=1)))
    selected_indices = _farthest_first_indices(
        distances,
        budget=budget,
        seed_indices=[medoid],
    )

    # For comparison, FIFO would retain only the newest ``budget`` points.
    fifo_indices = list(range(count - budget, count))
    before = _coverage_radius(distances, fifo_indices)
    after = _coverage_radius(distances, selected_indices)

    retained = [matrix[index].copy() for index in selected_indices]
    return retained, MemorySelectionReport(
        input_count=count,
        retained_count=len(retained),
        dropped_count=count - len(retained),
        strategy="diversity_kcenter",
        coverage_radius_before=before,
        coverage_radius_after=after,
    )


def select_boundary_diverse_negatives(
    negatives: list[np.ndarray],
    positive_samples: list[np.ndarray],
    budget: int,
    candidate_multiplier: int = 3,
) -> tuple[list[np.ndarray], MemorySelectionReport]:
    """Keep hard negatives that are both boundary-relevant and diverse.

    A negative close to any positive exemplar is more informative for the current
    open-set boundary than one far away.  We therefore build a candidate pool from
    the closest negatives, then use farthest-first traversal to avoid keeping many
    copies of the same correction.
    """
    budget = max(1, int(budget))
    negative_matrix = _prepare_matrix(negatives)
    count = negative_matrix.shape[0]

    if count <= budget:
        retained = [row.copy() for row in negative_matrix]
        return retained, MemorySelectionReport(
            input_count=count,
            retained_count=count,
            dropped_count=0,
            strategy="boundary_diversity",
        )

    positive_matrix = _prepare_matrix(positive_samples)
    if positive_matrix.shape[0] == 0:
        # Defensive fallback; normal learner classes always contain positives.
        return select_diverse_exemplars(negatives, budget)

    if positive_matrix.shape[1] != negative_matrix.shape[1]:
        raise ValueError("Positive and negative exemplars must share a feature dimension.")

    # Distance from every negative to its nearest positive class exemplar.
    differences = negative_matrix[:, None, :] - positive_matrix[None, :, :]
    nearest_positive = np.sqrt(np.mean(differences * differences, axis=2)).min(axis=1)
    hardness_order = np.argsort(nearest_positive, kind="stable")

    pool_size = min(count, max(budget, int(candidate_multiplier) * budget))
    candidate_indices = hardness_order[:pool_size].tolist()

    # Reserve roughly half the memory for the genuinely hardest corrections.
    # Duplicate filtering in OnlineGestureLearner prevents this boundary half
    # from becoming a stack of virtually identical points.  The remaining half
    # is filled by k-center-style diversity within the hard candidate pool.
    protected_count = max(1, budget // 2)
    protected_hard = hardness_order[:protected_count].tolist()

    negative_distances = _pairwise_rmse(negative_matrix)
    selected_indices = _farthest_first_indices(
        negative_distances,
        budget=budget,
        seed_indices=protected_hard,
        candidate_indices=candidate_indices,
    )

    retained = [negative_matrix[index].copy() for index in selected_indices]
    return retained, MemorySelectionReport(
        input_count=count,
        retained_count=len(retained),
        dropped_count=count - len(retained),
        strategy="boundary_diversity",
    )
