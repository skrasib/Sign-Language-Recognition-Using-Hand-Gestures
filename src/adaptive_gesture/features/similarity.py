import numpy as np


def feature_distance(
    features_a: np.ndarray,
    features_b: np.ndarray,
) -> float:
    """
    Calculate RMSE distance between two feature vectors.

    Smaller distance = more similar hand geometry.
    """

    a = np.asarray(
        features_a,
        dtype=np.float32,
    )

    b = np.asarray(
        features_b,
        dtype=np.float32,
    )

    if a.shape != b.shape:
        raise ValueError(
            f"Feature shapes must match. "
            f"Got {a.shape} and {b.shape}."
        )

    if a.ndim != 1:
        raise ValueError(
            f"Expected flat feature vectors, "
            f"got shape {a.shape}."
        )

    return float(
        np.sqrt(
            np.mean(
                np.square(a - b)
            )
        )
    )


def create_prototype(
    samples: list[np.ndarray],
) -> np.ndarray:
    """
    Create a robust prototype from several gesture samples.
    """

    if not samples:
        raise ValueError(
            "At least one sample is required."
        )

    stacked = np.stack(
        samples
    ).astype(
        np.float32
    )

    return np.median(
        stacked,
        axis=0,
    ).astype(
        np.float32
    )


def calculate_reference_spread(
    samples: list[np.ndarray],
    prototype: np.ndarray,
) -> float:
    """
    Measure sample-to-prototype variation.
    """

    if not samples:
        raise ValueError(
            "At least one sample is required."
        )

    distances = np.array(
        [
            feature_distance(
                sample,
                prototype,
            )
            for sample in samples
        ],
        dtype=np.float32,
    )

    return float(
        np.percentile(
            distances,
            95,
        )
    )


def calculate_local_sample_radius(
    samples: list[np.ndarray],
    percentile: float = 95.0,
) -> float:
    """
    Estimate the local density of a gesture class.

    For every training sample, find its nearest OTHER
    sample. The requested percentile of those distances
    becomes the class's local sample radius.
    """

    if not samples:
        raise ValueError(
            "At least one sample is required."
        )

    if len(samples) == 1:
        return 1e-4

    nearest_distances = []

    for i, sample in enumerate(
        samples
    ):

        distances = []

        for j, other in enumerate(
            samples
        ):

            if i == j:
                continue

            distances.append(
                feature_distance(
                    sample,
                    other,
                )
            )

        nearest_distances.append(
            min(distances)
        )

    return float(
        np.percentile(
            nearest_distances,
            percentile,
        )
    )