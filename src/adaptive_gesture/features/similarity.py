import numpy as np


def feature_distance(
    features_a: np.ndarray,
    features_b: np.ndarray,
) -> float:
    """RMSE distance between two flat feature vectors."""
    a = np.asarray(features_a, dtype=np.float32)
    b = np.asarray(features_b, dtype=np.float32)

    if a.shape != b.shape:
        raise ValueError(
            f"Feature shapes must match. Got {a.shape} and {b.shape}."
        )
    if a.ndim != 1:
        raise ValueError(f"Expected flat feature vectors, got shape {a.shape}.")

    return float(np.sqrt(np.mean(np.square(a - b))))


def create_prototype(samples: list[np.ndarray]) -> np.ndarray:
    """Robust median prototype for a collection of samples."""
    if not samples:
        raise ValueError("At least one sample is required.")

    stacked = np.stack(samples).astype(np.float32)
    return np.median(stacked, axis=0).astype(np.float32)


def calculate_reference_spread(
    samples: list[np.ndarray],
    prototype: np.ndarray,
) -> float:
    """95th percentile of sample-to-prototype distances."""
    if not samples:
        raise ValueError("At least one sample is required.")

    distances = np.array(
        [feature_distance(sample, prototype) for sample in samples],
        dtype=np.float32,
    )
    return float(np.percentile(distances, 95))


def calculate_local_sample_radius(
    samples: list[np.ndarray],
    percentile: float = 95.0,
) -> float:
    """
    Estimate the local density of a gesture class.

    For every sample, find its nearest other positive sample. The requested
    percentile of those distances becomes a local class radius.
    """
    if not samples:
        raise ValueError("At least one sample is required.")
    if len(samples) == 1:
        return 1e-4

    nearest_distances = []
    for i, sample in enumerate(samples):
        distances = [
            feature_distance(sample, other)
            for j, other in enumerate(samples)
            if i != j
        ]
        nearest_distances.append(min(distances))

    return float(np.percentile(nearest_distances, percentile))
