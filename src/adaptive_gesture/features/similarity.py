import numpy as np


def feature_distance(
    features_a: np.ndarray,
    features_b: np.ndarray,
) -> float:
    """
    Calculate the RMSE distance between two feature vectors.

    Smaller distance = more similar hand geometry.
    """

    a = np.asarray(features_a, dtype=np.float32)
    b = np.asarray(features_b, dtype=np.float32)

    if a.shape != b.shape:
        raise ValueError(
            f"Feature shapes must match. Got {a.shape} and {b.shape}."
        )

    if a.ndim != 1:
        raise ValueError(
            f"Expected flat feature vectors, got shape {a.shape}."
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
    Create one robust prototype from several gesture samples.

    Median is used rather than mean so occasional noisy MediaPipe
    frames have less influence on the reference gesture.
    """

    if not samples:
        raise ValueError("At least one sample is required.")

    stacked = np.stack(samples).astype(np.float32)

    return np.median(
        stacked,
        axis=0,
    ).astype(np.float32)


def calculate_reference_spread(
    samples: list[np.ndarray],
    prototype: np.ndarray,
) -> float:
    """
    Estimate how much natural variation exists inside a gesture.

    We use the 95th percentile of sample-to-prototype distances
    rather than the maximum, because one poor tracking frame
    should not define the entire gesture boundary.
    """

    if not samples:
        raise ValueError("At least one sample is required.")

    distances = np.array(
        [
            feature_distance(sample, prototype)
            for sample in samples
        ],
        dtype=np.float32,
    )

    return float(
        np.percentile(distances, 95)
    )