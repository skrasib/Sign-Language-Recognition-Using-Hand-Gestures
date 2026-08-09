from dataclasses import dataclass, field

import numpy as np

from adaptive_gesture.features.similarity import (
    calculate_reference_spread,
    create_prototype,
    feature_distance,
)


@dataclass
class GestureClass:
    name: str
    prototype: np.ndarray
    spread: float
    samples: list[np.ndarray] = field(default_factory=list)
    handedness: str | None = None

    @property
    def sample_count(self) -> int:
        return len(self.samples)


@dataclass
class Prediction:
    label: str
    distance: float | None
    relative_distance: float | None
    accepted: bool
    threshold: float | None


class OnlineGestureLearner:
    """
    Runtime few-shot gesture learner.

    New classes can be introduced without retraining any existing
    gesture classes.
    """

    UNKNOWN_LABEL = "UNKNOWN"

    def __init__(
        self,
        rejection_multiplier: float = 3.0,
        minimum_threshold: float = 0.05,
    ):
        self.gestures: dict[str, GestureClass] = {}

        self.rejection_multiplier = rejection_multiplier
        self.minimum_threshold = minimum_threshold

    def learn_gesture(
        self,
        name: str,
        samples: list[np.ndarray],
        handedness: str | None = None,
    ) -> GestureClass:
        """
        Learn a completely new gesture from live samples.
        """

        name = name.strip()

        if not name:
            raise ValueError("Gesture name cannot be empty.")

        if not samples:
            raise ValueError(
                "Cannot learn a gesture without samples."
            )

        samples = [
            np.asarray(sample, dtype=np.float32)
            for sample in samples
        ]

        prototype = create_prototype(samples)

        spread = calculate_reference_spread(
            samples,
            prototype,
        )

        gesture = GestureClass(
            name=name,
            prototype=prototype,
            spread=max(spread, 1e-4),
            samples=[
                sample.copy()
                for sample in samples
            ],
            handedness=handedness,
        )

        self.gestures[name] = gesture

        return gesture

    def update_gesture(
        self,
        name: str,
        sample: np.ndarray,
        max_samples: int = 100,
    ) -> GestureClass:
        """
        Incrementally improve an existing gesture from one new
        user-confirmed example.

        This is the foundation for interactive feedback learning.
        """

        if name not in self.gestures:
            raise KeyError(
                f"Unknown gesture: {name}"
            )

        gesture = self.gestures[name]

        gesture.samples.append(
            np.asarray(
                sample,
                dtype=np.float32,
            ).copy()
        )

        # Keep bounded memory.
        if len(gesture.samples) > max_samples:
            gesture.samples.pop(0)

        gesture.prototype = create_prototype(
            gesture.samples
        )

        gesture.spread = max(
            calculate_reference_spread(
                gesture.samples,
                gesture.prototype,
            ),
            1e-4,
        )

        return gesture

    def predict(
        self,
        features: np.ndarray,
        handedness: str | None = None,
    ) -> Prediction:
        """
        Find the closest learned gesture.

        If the closest gesture is still too far away from its
        learned distribution, return UNKNOWN.
        """

        if not self.gestures:
            return Prediction(
                label=self.UNKNOWN_LABEL,
                distance=None,
                relative_distance=None,
                accepted=False,
                threshold=None,
            )

        candidates = []

        for gesture in self.gestures.values():

            # For our first implementation, preserve handedness.
            if (
                handedness is not None
                and gesture.handedness is not None
                and handedness != gesture.handedness
            ):
                continue

            distance = feature_distance(
                features,
                gesture.prototype,
            )

            threshold = max(
                gesture.spread
                * self.rejection_multiplier,
                self.minimum_threshold,
            )

            relative_distance = (
                distance / gesture.spread
            )

            candidates.append(
                (
                    gesture,
                    distance,
                    threshold,
                    relative_distance,
                )
            )

        if not candidates:
            return Prediction(
                label=self.UNKNOWN_LABEL,
                distance=None,
                relative_distance=None,
                accepted=False,
                threshold=None,
            )

        (
            best_gesture,
            best_distance,
            best_threshold,
            best_relative_distance,
        ) = min(
            candidates,
            key=lambda item: item[1],
        )

        accepted = (
            best_distance <= best_threshold
        )

        return Prediction(
            label=(
                best_gesture.name
                if accepted
                else self.UNKNOWN_LABEL
            ),
            distance=best_distance,
            relative_distance=best_relative_distance,
            accepted=accepted,
            threshold=best_threshold,
        )

    def list_gestures(self) -> list[str]:
        return list(
            self.gestures.keys()
        )