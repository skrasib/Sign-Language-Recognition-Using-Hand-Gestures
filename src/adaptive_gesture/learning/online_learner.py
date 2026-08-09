from dataclasses import dataclass, field

import numpy as np

from adaptive_gesture.features.similarity import (
    calculate_local_sample_radius,
    calculate_reference_spread,
    create_prototype,
    feature_distance,
)

from adaptive_gesture.learning.confidence import (
    accepted_confidence,
    hard_negative_confidence,
    outside_region_confidence,
)


@dataclass
class PrototypeCluster:
    prototype: np.ndarray
    radius: float
    sample_count: int


@dataclass
class GestureClass:
    name: str
    hand_signature: str
    feature_dimension: int
    prototype: np.ndarray
    spread: float
    sample_radius: float
    prototypes: list[PrototypeCluster] = field(default_factory=list)
    samples: list[np.ndarray] = field(default_factory=list)
    hard_negatives: list[np.ndarray] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def negative_count(self) -> int:
        return len(self.hard_negatives)

    @property
    def prototype_count(self) -> int:
        return len(self.prototypes)

    # Backward-friendly name for older UI code.
    @property
    def handedness(self) -> str:
        return self.hand_signature


@dataclass
class Prediction:
    label: str
    accepted: bool
    nearest_label: str | None = None
    distance: float | None = None
    prototype_distance: float | None = None
    relative_distance: float | None = None
    threshold: float | None = None
    hard_negative_distance: float | None = None
    second_best_label: str | None = None
    second_best_relative_distance: float | None = None
    confidence: float | None = None
    rejection_reason: str | None = None


class OnlineGestureLearner:
    """
    Online few-shot gesture learner with:
    - runtime class addition;
    - adaptive multi-prototype class representations;
    - open-set rejection;
    - positive and hard-negative feedback;
    - class management without global retraining.
    """

    UNKNOWN_LABEL = "UNKNOWN"

    def __init__(
        self,
        radius_multiplier: float = 2.5,
        minimum_threshold: float = 0.035,
        prototype_multiplier: float = 2.2,
        prototype_split_multiplier: float = 2.5,
        max_prototypes: int = 3,
        feedback_duplicate_threshold: float = 0.012,
        hard_negative_margin: float = 1.05,
    ):
        self.gestures: dict[str, GestureClass] = {}
        self.radius_multiplier = radius_multiplier
        self.minimum_threshold = minimum_threshold
        self.prototype_multiplier = prototype_multiplier
        self.prototype_split_multiplier = prototype_split_multiplier
        self.max_prototypes = max_prototypes
        self.feedback_duplicate_threshold = feedback_duplicate_threshold
        self.hard_negative_margin = hard_negative_margin

    # =====================================================
    # Validation / helpers
    # =====================================================

    @staticmethod
    def _prepare_samples(samples: list[np.ndarray]) -> list[np.ndarray]:
        if not samples:
            raise ValueError("At least one sample is required.")

        prepared = [np.asarray(sample, dtype=np.float32).copy() for sample in samples]
        first_shape = prepared[0].shape

        if len(first_shape) != 1:
            raise ValueError("Gesture samples must be flat feature vectors.")

        for sample in prepared:
            if sample.shape != first_shape:
                raise ValueError("All gesture samples must have the same feature shape.")

        return prepared

    def _build_prototypes(
        self,
        samples: list[np.ndarray],
        sample_radius: float,
    ) -> list[PrototypeCluster]:
        """Greedy adaptive clustering followed by median-center refinement."""
        if not samples:
            return []

        centers = [create_prototype(samples)]
        split_threshold = max(
            sample_radius * self.prototype_split_multiplier,
            self.minimum_threshold * 1.25,
        )

        # Add a new center only when the current prototypes cannot adequately
        # represent a naturally different positive variation.
        while len(centers) < self.max_prototypes and len(samples) >= len(centers) + 2:
            nearest_distances = [
                min(feature_distance(sample, center) for center in centers)
                for sample in samples
            ]
            farthest_index = int(np.argmax(nearest_distances))
            farthest_distance = nearest_distances[farthest_index]

            if farthest_distance <= split_threshold:
                break

            centers.append(samples[farthest_index].copy())

            # A few robust k-median-like refinement rounds.
            for _ in range(4):
                groups: list[list[np.ndarray]] = [[] for _ in centers]
                for sample in samples:
                    index = int(
                        np.argmin([feature_distance(sample, center) for center in centers])
                    )
                    groups[index].append(sample)

                new_centers = []
                for old_center, group in zip(centers, groups):
                    new_centers.append(
                        create_prototype(group) if group else old_center
                    )
                centers = new_centers

        groups = [[] for _ in centers]
        for sample in samples:
            index = int(
                np.argmin([feature_distance(sample, center) for center in centers])
            )
            groups[index].append(sample)

        clusters: list[PrototypeCluster] = []
        for center, group in zip(centers, groups):
            if not group:
                continue

            distances = [feature_distance(sample, center) for sample in group]
            radius = float(np.percentile(distances, 95)) if distances else 1e-4
            clusters.append(
                PrototypeCluster(
                    prototype=np.asarray(center, dtype=np.float32),
                    radius=max(radius, 1e-4),
                    sample_count=len(group),
                )
            )

        return clusters

    def _rebuild_gesture(self, gesture: GestureClass) -> None:
        gesture.prototype = create_prototype(gesture.samples)
        gesture.spread = max(
            calculate_reference_spread(gesture.samples, gesture.prototype),
            1e-4,
        )
        gesture.sample_radius = max(
            calculate_local_sample_radius(gesture.samples),
            1e-4,
        )
        gesture.prototypes = self._build_prototypes(
            gesture.samples,
            gesture.sample_radius,
        )
        gesture.feature_dimension = int(gesture.samples[0].shape[0])

    @staticmethod
    def _remove_nearby_negatives(
        gesture: GestureClass,
        positive_sample: np.ndarray,
        threshold: float,
    ) -> None:
        gesture.hard_negatives = [
            negative
            for negative in gesture.hard_negatives
            if feature_distance(positive_sample, negative) >= threshold
        ]

    # =====================================================
    # New class / class management
    # =====================================================

    def learn_gesture(
        self,
        name: str,
        samples: list[np.ndarray],
        hand_signature: str | None = None,
        handedness: str | None = None,
    ) -> GestureClass:
        name = name.strip()
        if not name:
            raise ValueError("Gesture name cannot be empty.")
        if name in self.gestures:
            raise ValueError(f"Gesture '{name}' already exists.")

        signature = hand_signature or handedness or "Unknown"
        prepared = self._prepare_samples(samples)

        gesture = GestureClass(
            name=name,
            hand_signature=signature,
            feature_dimension=int(prepared[0].shape[0]),
            prototype=create_prototype(prepared),
            spread=1e-4,
            sample_radius=1e-4,
            samples=prepared,
        )
        self._rebuild_gesture(gesture)
        self.gestures[name] = gesture
        return gesture

    def rename_gesture(self, old_name: str, new_name: str) -> GestureClass:
        if old_name not in self.gestures:
            raise KeyError(f"Unknown gesture: {old_name}")

        new_name = new_name.strip()
        if not new_name:
            raise ValueError("New gesture name cannot be empty.")
        if new_name != old_name and new_name in self.gestures:
            raise ValueError(f"Gesture '{new_name}' already exists.")

        gesture = self.gestures.pop(old_name)
        gesture.name = new_name
        self.gestures[new_name] = gesture
        return gesture

    def delete_gesture(self, name: str) -> None:
        if name not in self.gestures:
            raise KeyError(f"Unknown gesture: {name}")
        del self.gestures[name]

    def clear(self) -> None:
        self.gestures.clear()

    def add_samples_to_gesture(
        self,
        name: str,
        samples: list[np.ndarray],
        hand_signature: str,
        max_samples: int = 60,
    ) -> GestureClass:
        if name not in self.gestures:
            raise KeyError(f"Unknown gesture: {name}")

        gesture = self.gestures[name]
        prepared = self._prepare_samples(samples)

        if hand_signature != gesture.hand_signature:
            raise ValueError(
                f"Gesture '{name}' expects {gesture.hand_signature} input, "
                f"but received {hand_signature}."
            )
        if prepared[0].shape[0] != gesture.feature_dimension:
            raise ValueError("Feature dimension does not match the existing gesture.")

        for sample in prepared:
            if not gesture.samples or min(
                feature_distance(sample, existing) for existing in gesture.samples
            ) >= self.feedback_duplicate_threshold:
                gesture.samples.append(sample)
                self._remove_nearby_negatives(
                    gesture,
                    sample,
                    self.feedback_duplicate_threshold,
                )

        if len(gesture.samples) > max_samples:
            gesture.samples = gesture.samples[-max_samples:]

        self._rebuild_gesture(gesture)
        return gesture

    def replace_gesture_samples(
        self,
        name: str,
        samples: list[np.ndarray],
        hand_signature: str,
        keep_negatives: bool = True,
    ) -> GestureClass:
        if name not in self.gestures:
            raise KeyError(f"Unknown gesture: {name}")

        gesture = self.gestures[name]
        prepared = self._prepare_samples(samples)

        gesture.samples = prepared
        gesture.hand_signature = hand_signature
        gesture.feature_dimension = int(prepared[0].shape[0])
        if not keep_negatives:
            gesture.hard_negatives = []
        else:
            gesture.hard_negatives = [
                negative
                for negative in gesture.hard_negatives
                if negative.shape[0] == gesture.feature_dimension
            ]

        self._rebuild_gesture(gesture)
        return gesture

    # =====================================================
    # Feedback learning
    # =====================================================

    def update_gesture(
        self,
        name: str,
        sample: np.ndarray,
        max_samples: int = 60,
    ) -> GestureClass:
        if name not in self.gestures:
            raise KeyError(f"Unknown gesture: {name}")

        gesture = self.gestures[name]
        sample = np.asarray(sample, dtype=np.float32).copy()

        if sample.shape != (gesture.feature_dimension,):
            raise ValueError("Feedback sample has the wrong feature dimension.")

        # Do not let repeated Correct clicks flood the class with duplicates.
        if gesture.samples:
            nearest = min(feature_distance(sample, existing) for existing in gesture.samples)
            if nearest < self.feedback_duplicate_threshold:
                return gesture

        gesture.samples.append(sample)
        self._remove_nearby_negatives(
            gesture,
            sample,
            self.feedback_duplicate_threshold,
        )

        if len(gesture.samples) > max_samples:
            gesture.samples.pop(0)

        self._rebuild_gesture(gesture)
        return gesture

    def add_hard_negative(
        self,
        name: str,
        sample: np.ndarray,
        max_negatives: int = 40,
    ) -> GestureClass:
        if name not in self.gestures:
            raise KeyError(f"Unknown gesture: {name}")

        gesture = self.gestures[name]
        sample = np.asarray(sample, dtype=np.float32).copy()

        if sample.shape != (gesture.feature_dimension,):
            raise ValueError("Negative sample has the wrong feature dimension.")

        # Ignore duplicate negative corrections.
        if gesture.hard_negatives:
            nearest = min(
                feature_distance(sample, negative)
                for negative in gesture.hard_negatives
            )
            if nearest < self.feedback_duplicate_threshold:
                return gesture

        gesture.hard_negatives.append(sample)
        if len(gesture.hard_negatives) > max_negatives:
            gesture.hard_negatives.pop(0)

        return gesture

    def apply_correction(
        self,
        predicted_label: str | None,
        actual_label: str,
        sample: np.ndarray,
    ) -> None:
        if actual_label not in self.gestures:
            raise KeyError(f"Unknown actual gesture: {actual_label}")

        sample = np.asarray(sample, dtype=np.float32)
        actual = self.gestures[actual_label]
        if sample.shape != (actual.feature_dimension,):
            raise ValueError(
                f"'{actual_label}' uses {actual.hand_signature} input and cannot "
                "learn from the captured hand configuration."
            )

        if (
            predicted_label is not None
            and predicted_label in self.gestures
            and predicted_label != actual_label
        ):
            predicted = self.gestures[predicted_label]
            if sample.shape == (predicted.feature_dimension,):
                self.add_hard_negative(predicted_label, sample)

        self.update_gesture(actual_label, sample)

    def mark_unknown(
        self,
        predicted_label: str | None,
        sample: np.ndarray,
    ) -> None:
        if predicted_label is None or predicted_label not in self.gestures:
            return

        gesture = self.gestures[predicted_label]
        sample = np.asarray(sample, dtype=np.float32)
        if sample.shape == (gesture.feature_dimension,):
            self.add_hard_negative(predicted_label, sample)

    # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        features: np.ndarray,
        hand_signature: str | None = None,
        handedness: str | None = None,
    ) -> Prediction:
        if not self.gestures:
            return Prediction(
                label=self.UNKNOWN_LABEL,
                accepted=False,
                rejection_reason="no_gestures",
            )

        features = np.asarray(features, dtype=np.float32)
        signature = hand_signature or handedness
        candidates = []

        for gesture in self.gestures.values():
            if signature is not None and gesture.hand_signature != signature:
                continue
            if features.shape != (gesture.feature_dimension,):
                continue

            nearest_positive = min(
                feature_distance(features, sample) for sample in gesture.samples
            )
            positive_threshold = max(
                gesture.sample_radius * self.radius_multiplier,
                self.minimum_threshold,
            )
            positive_ratio = nearest_positive / positive_threshold

            if gesture.prototypes:
                prototype_candidates = []
                for cluster in gesture.prototypes:
                    distance = feature_distance(features, cluster.prototype)
                    threshold = max(
                        max(cluster.radius, gesture.sample_radius)
                        * self.prototype_multiplier,
                        self.minimum_threshold,
                    )
                    prototype_candidates.append((distance, threshold, distance / threshold))

                prototype_distance, prototype_threshold, prototype_ratio = min(
                    prototype_candidates,
                    key=lambda item: item[2],
                )
            else:
                prototype_distance = feature_distance(features, gesture.prototype)
                prototype_threshold = positive_threshold
                prototype_ratio = prototype_distance / prototype_threshold

            # Either a nearby real exemplar OR a nearby adaptive prototype may
            # explain the pose. The normalized score allows fair comparison across
            # classes with different natural variability.
            score = min(positive_ratio, prototype_ratio)

            nearest_negative = None
            if gesture.hard_negatives:
                nearest_negative = min(
                    feature_distance(features, negative)
                    for negative in gesture.hard_negatives
                )

            candidates.append(
                {
                    "gesture": gesture,
                    "score": score,
                    "nearest_positive": nearest_positive,
                    "positive_threshold": positive_threshold,
                    "prototype_distance": prototype_distance,
                    "prototype_threshold": prototype_threshold,
                    "nearest_negative": nearest_negative,
                }
            )

        if not candidates:
            return Prediction(
                label=self.UNKNOWN_LABEL,
                accepted=False,
                rejection_reason="hand_configuration",
            )

        candidates.sort(key=lambda item: item["score"])
        best = candidates[0]
        gesture = best["gesture"]

        second_best_label = None
        second_best_relative_distance = None
        if len(candidates) > 1:
            second = candidates[1]
            second_best_label = second["gesture"].name
            second_best_relative_distance = float(second["score"])

        accepted = best["score"] <= 1.0
        reason = None if accepted else "outside_positive_region"

        nearest_negative = best["nearest_negative"]
        if (
            accepted
            and nearest_negative is not None
            and nearest_negative <= best["nearest_positive"] * self.hard_negative_margin
        ):
            accepted = False
            reason = "hard_negative"

        if accepted:
            confidence_result = accepted_confidence(
                relative_distance=best["score"],
                second_best_relative_distance=second_best_relative_distance,
                positive_distance=best["nearest_positive"],
                hard_negative_distance=nearest_negative,
            )
        elif reason == "hard_negative":
            confidence_result = hard_negative_confidence(
                positive_distance=best["nearest_positive"],
                hard_negative_distance=nearest_negative,
                threshold=best["positive_threshold"],
            )
        else:
            confidence_result = outside_region_confidence(best["score"])

        return Prediction(
            label=gesture.name if accepted else self.UNKNOWN_LABEL,
            accepted=accepted,
            nearest_label=gesture.name,
            distance=best["nearest_positive"],
            prototype_distance=best["prototype_distance"],
            relative_distance=best["score"],
            threshold=best["positive_threshold"],
            hard_negative_distance=nearest_negative,
            second_best_label=second_best_label,
            second_best_relative_distance=second_best_relative_distance,
            confidence=confidence_result.score,
            rejection_reason=reason,
        )

    def list_gestures(self) -> list[str]:
        return list(self.gestures.keys())
