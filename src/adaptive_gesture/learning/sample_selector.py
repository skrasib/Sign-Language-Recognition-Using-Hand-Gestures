from collections import deque
from dataclasses import dataclass

import numpy as np

from adaptive_gesture.features.similarity import feature_distance


@dataclass
class SampleSelectionStats:
    observed: int = 0
    accepted: int = 0
    duplicates: int = 0
    unstable: int = 0


class SmartSampleSelector:
    """
    Select informative samples from a live camera stream.

    Goals:
    - reject frames while the hand is moving too much;
    - keep a few stable baseline samples;
    - ignore excessive near-duplicate frames;
    - retain natural variation in the demonstrated gesture.
    """

    def __init__(
        self,
        target_samples: int = 12,
        minimum_samples: int = 6,
        stability_window: int = 5,
        stability_threshold: float = 0.035,
        duplicate_threshold: float = 0.018,
        baseline_samples: int = 4,
    ):
        self.target_samples = target_samples
        self.minimum_samples = minimum_samples

        self.stability_window = stability_window
        self.stability_threshold = stability_threshold
        self.duplicate_threshold = duplicate_threshold
        self.baseline_samples = baseline_samples

        self.recent_features = deque(
            maxlen=stability_window
        )

        self.samples: list[np.ndarray] = []

        self.stats = SampleSelectionStats()

        self.last_stability = None
        self.last_min_distance = None

    @property
    def ready(self) -> bool:
        """
        Enough useful samples exist for learning.
        """
        return (
            len(self.samples)
            >= self.minimum_samples
        )

    @property
    def complete(self) -> bool:
        """
        Target number of useful samples reached.
        """
        return (
            len(self.samples)
            >= self.target_samples
        )

    def _calculate_stability(self) -> float | None:

        if (
            len(self.recent_features)
            < self.stability_window
        ):
            return None

        distances = []

        recent = list(
            self.recent_features
        )

        for index in range(
            1,
            len(recent),
        ):
            distances.append(
                feature_distance(
                    recent[index - 1],
                    recent[index],
                )
            )

        if not distances:
            return None

        return float(
            np.mean(distances)
        )

    def consider(
        self,
        features: np.ndarray,
    ) -> bool:
        """
        Inspect one incoming frame.

        Returns True only when the frame was accepted as
        a useful training sample.
        """

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        self.stats.observed += 1

        self.recent_features.append(
            features.copy()
        )

        stability = (
            self._calculate_stability()
        )

        self.last_stability = stability

        # We need several consecutive frames before
        # judging whether the pose is stable.
        if stability is None:
            self.stats.unstable += 1
            return False

        # Reject frames captured while the user is
        # still moving into/out of the gesture.
        if (
            stability
            > self.stability_threshold
        ):
            self.stats.unstable += 1
            return False

        # -------------------------------------------------
        # Capture a small stable baseline.
        #
        # These samples help us estimate normal tracking
        # noise and within-gesture spread.
        # -------------------------------------------------

        if (
            len(self.samples)
            < self.baseline_samples
        ):
            self.samples.append(
                features.copy()
            )

            self.stats.accepted += 1
            return True

        # -------------------------------------------------
        # After baseline capture, retain only samples that
        # contribute some new natural variation.
        # -------------------------------------------------

        distances = [
            feature_distance(
                features,
                sample,
            )
            for sample in self.samples
        ]

        min_distance = min(
            distances
        )

        self.last_min_distance = (
            min_distance
        )

        if (
            min_distance
            < self.duplicate_threshold
        ):
            self.stats.duplicates += 1
            return False

        self.samples.append(
            features.copy()
        )

        self.stats.accepted += 1

        return True

    def reset(self) -> None:

        self.recent_features.clear()
        self.samples.clear()

        self.stats = (
            SampleSelectionStats()
        )

        self.last_stability = None
        self.last_min_distance = None