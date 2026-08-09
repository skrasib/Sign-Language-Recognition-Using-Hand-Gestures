from dataclasses import dataclass, field

import numpy as np

from adaptive_gesture.features.similarity import (
    calculate_local_sample_radius,
    calculate_reference_spread,
    create_prototype,
    feature_distance,
)


@dataclass
class GestureClass:
    name: str

    prototype: np.ndarray

    # Global sample-to-prototype variation.
    spread: float

    # Local nearest-neighbor variation.
    sample_radius: float

    samples: list[np.ndarray] = field(
        default_factory=list
    )

    # Examples the user explicitly told us are
    # NOT this gesture.
    hard_negatives: list[np.ndarray] = field(
        default_factory=list
    )

    handedness: str | None = None

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def negative_count(self) -> int:
        return len(self.hard_negatives)


@dataclass
class Prediction:
    label: str
    accepted: bool

    nearest_label: str | None = None

    # Distance to the closest positive exemplar.
    distance: float | None = None

    # Distance to the class prototype.
    prototype_distance: float | None = None

    relative_distance: float | None = None

    threshold: float | None = None

    hard_negative_distance: float | None = None

    rejection_reason: str | None = None


class OnlineGestureLearner:
    """
    Online few-shot gesture learner.

    Supports:
    - adding previously unknown classes at runtime;
    - exemplar-based open-set recognition;
    - incremental positive feedback;
    - hard-negative feedback;
    - bounded runtime memory.
    """

    UNKNOWN_LABEL = "UNKNOWN"

    def __init__(
        self,
        radius_multiplier: float = 2.5,
        minimum_threshold: float = 0.035,
    ):
        self.gestures: dict[str, GestureClass] = {}

        self.radius_multiplier = (
            radius_multiplier
        )

        self.minimum_threshold = (
            minimum_threshold
        )

    # =====================================================
    # Internal class reconstruction
    # =====================================================

    def _rebuild_gesture(
        self,
        gesture: GestureClass,
    ) -> None:

        gesture.prototype = (
            create_prototype(
                gesture.samples
            )
        )

        gesture.spread = max(
            calculate_reference_spread(
                gesture.samples,
                gesture.prototype,
            ),
            1e-4,
        )

        gesture.sample_radius = max(
            calculate_local_sample_radius(
                gesture.samples
            ),
            1e-4,
        )

    # =====================================================
    # Learn completely new class
    # =====================================================

    def learn_gesture(
        self,
        name: str,
        samples: list[np.ndarray],
        handedness: str | None = None,
    ) -> GestureClass:

        name = name.strip()

        if not name:
            raise ValueError(
                "Gesture name cannot be empty."
            )

        if name in self.gestures:
            raise ValueError(
                f"Gesture '{name}' already exists."
            )

        if not samples:
            raise ValueError(
                "Cannot learn a gesture "
                "without samples."
            )

        prepared_samples = [
            np.asarray(
                sample,
                dtype=np.float32,
            ).copy()
            for sample in samples
        ]

        prototype = create_prototype(
            prepared_samples
        )

        spread = max(
            calculate_reference_spread(
                prepared_samples,
                prototype,
            ),
            1e-4,
        )

        sample_radius = max(
            calculate_local_sample_radius(
                prepared_samples
            ),
            1e-4,
        )

        gesture = GestureClass(
            name=name,
            prototype=prototype,
            spread=spread,
            sample_radius=sample_radius,
            samples=prepared_samples,
            handedness=handedness,
        )

        self.gestures[name] = gesture

        return gesture

    # =====================================================
    # Positive feedback
    # =====================================================

    def update_gesture(
        self,
        name: str,
        sample: np.ndarray,
        max_samples: int = 40,
    ) -> GestureClass:
        """
        Add a user-confirmed positive example.

        Existing classes are NOT retrained.
        Only this gesture's memory is updated.
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

        # Keep runtime memory bounded.
        if len(gesture.samples) > max_samples:
            gesture.samples.pop(0)

        self._rebuild_gesture(
            gesture
        )

        return gesture

    # =====================================================
    # Negative feedback
    # =====================================================

    def add_hard_negative(
        self,
        name: str,
        sample: np.ndarray,
        max_negatives: int = 30,
    ) -> GestureClass:
        """
        Teach the system that this particular hand pose
        must NOT be classified as `name`.
        """

        if name not in self.gestures:
            raise KeyError(
                f"Unknown gesture: {name}"
            )

        gesture = self.gestures[name]

        gesture.hard_negatives.append(
            np.asarray(
                sample,
                dtype=np.float32,
            ).copy()
        )

        if (
            len(gesture.hard_negatives)
            > max_negatives
        ):
            gesture.hard_negatives.pop(0)

        return gesture

    # =====================================================
    # User correction
    # =====================================================

    def apply_correction(
        self,
        predicted_label: str | None,
        actual_label: str,
        sample: np.ndarray,
    ) -> None:
        """
        Example:

        predicted = Victory
        actual    = Three

        The current sample becomes:
        - positive evidence for Three
        - negative evidence for Victory
        """

        if actual_label not in self.gestures:
            raise KeyError(
                f"Unknown actual gesture: "
                f"{actual_label}"
            )

        if (
            predicted_label is not None
            and predicted_label
            in self.gestures
            and predicted_label
            != actual_label
        ):
            self.add_hard_negative(
                predicted_label,
                sample,
            )

        self.update_gesture(
            actual_label,
            sample,
        )

    def mark_unknown(
        self,
        predicted_label: str | None,
        sample: np.ndarray,
    ) -> None:
        """
        User explicitly says the pose belongs to none of
        the currently learned gestures.
        """

        if (
            predicted_label is not None
            and predicted_label
            in self.gestures
        ):
            self.add_hard_negative(
                predicted_label,
                sample,
            )

    # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        features: np.ndarray,
        handedness: str | None = None,
    ) -> Prediction:

        if not self.gestures:

            return Prediction(
                label=self.UNKNOWN_LABEL,
                accepted=False,
                rejection_reason=(
                    "no_gestures"
                ),
            )

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        candidates = []

        for gesture in (
            self.gestures.values()
        ):

            if (
                handedness is not None
                and gesture.handedness
                is not None
                and handedness
                != gesture.handedness
            ):
                continue

            # ---------------------------------------------
            # Positive exemplar distance
            # ---------------------------------------------

            positive_distances = [
                feature_distance(
                    features,
                    sample,
                )
                for sample
                in gesture.samples
            ]

            nearest_positive = min(
                positive_distances
            )

            prototype_distance = (
                feature_distance(
                    features,
                    gesture.prototype,
                )
            )

            threshold = max(
                gesture.sample_radius
                * self.radius_multiplier,
                self.minimum_threshold,
            )

            # ---------------------------------------------
            # User-taught negative examples
            # ---------------------------------------------

            nearest_negative = None

            if gesture.hard_negatives:

                negative_distances = [
                    feature_distance(
                        features,
                        negative,
                    )
                    for negative
                    in gesture.hard_negatives
                ]

                nearest_negative = min(
                    negative_distances
                )

            candidates.append(
                (
                    gesture,
                    nearest_positive,
                    prototype_distance,
                    threshold,
                    nearest_negative,
                )
            )

        if not candidates:

            return Prediction(
                label=self.UNKNOWN_LABEL,
                accepted=False,
                rejection_reason=(
                    "handedness"
                ),
            )

        (
            best_gesture,
            best_distance,
            prototype_distance,
            threshold,
            negative_distance,
        ) = min(
            candidates,
            key=lambda item: item[1],
        )

        accepted = (
            best_distance
            <= threshold
        )

        rejection_reason = None

        if not accepted:
            rejection_reason = (
                "outside_positive_region"
            )

        # -------------------------------------------------
        # Hard-negative veto
        #
        # If the sample resembles something the user
        # explicitly rejected MORE than it resembles the
        # gesture's positive examples, reject it.
        # -------------------------------------------------

        if (
            accepted
            and negative_distance is not None
            and negative_distance
            < best_distance
        ):
            accepted = False

            rejection_reason = (
                "hard_negative"
            )

        relative_distance = (
            best_distance
            / max(
                best_gesture.sample_radius,
                1e-4,
            )
        )

        return Prediction(
            label=(
                best_gesture.name
                if accepted
                else self.UNKNOWN_LABEL
            ),
            accepted=accepted,
            nearest_label=(
                best_gesture.name
            ),
            distance=best_distance,
            prototype_distance=(
                prototype_distance
            ),
            relative_distance=(
                relative_distance
            ),
            threshold=threshold,
            hard_negative_distance=(
                negative_distance
            ),
            rejection_reason=(
                rejection_reason
            ),
        )

    def list_gestures(
        self,
    ) -> list[str]:

        return list(
            self.gestures.keys()
        )