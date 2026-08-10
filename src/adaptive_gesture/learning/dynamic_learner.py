from dataclasses import dataclass, field

import numpy as np

from adaptive_gesture.features.dynamic_features import DynamicTrajectory
from adaptive_gesture.learning.dtw import trajectory_dtw_distance
from adaptive_gesture.learning.temporal_prototypes import build_temporal_prototypes

from adaptive_gesture.learning.confidence import (
    accepted_confidence,
    ambiguous_confidence,
    outside_region_confidence,
)


@dataclass
class DynamicGestureClass:
    name: str
    hand_signature: str
    templates: list[DynamicTrajectory] = field(default_factory=list)
    temporal_prototypes: list[DynamicTrajectory] = field(default_factory=list)
    reference_distance: float = 0.0
    threshold: float = 0.0
    median_duration: float = 0.0

    @property
    def template_count(self) -> int:
        return len(self.templates)

    @property
    def prototype_count(self) -> int:
        return len(self.temporal_prototypes)


@dataclass
class DynamicPrediction:
    label: str
    accepted: bool
    nearest_label: str | None = None
    distance: float | None = None
    threshold: float | None = None
    relative_distance: float | None = None
    second_best_label: str | None = None
    second_best_distance: float | None = None
    second_best_relative_distance: float | None = None
    confidence: float | None = None
    rejection_reason: str | None = None


class DynamicGestureLearner:
    """
    Few-shot, class-incremental dynamic gesture learner.

    V3.5 adds DTW-aligned temporal barycenter prototypes. Raw video is never
    stored: only normalized landmark trajectories are retained. The previous
    nearest-template strategy remains available for controlled ablation tests.
    """

    UNKNOWN_LABEL = "UNKNOWN"

    def __init__(
        self,
        minimum_templates: int = 3,
        threshold_multiplier: float = 1.60,
        minimum_threshold: float = 0.045,
        ambiguity_ratio: float = 1.12,
        max_templates: int = 6,
        duration_ratio_min: float = 0.35,
        duration_ratio_max: float = 2.85,
        temporal_prototype_strategy: str = "dtw_barycenter",
        max_temporal_prototypes: int = 2,
        prototype_iterations: int = 4,
    ):
        self.gestures: dict[str, DynamicGestureClass] = {}
        self.minimum_templates = max(2, int(minimum_templates))
        self.threshold_multiplier = float(threshold_multiplier)
        self.minimum_threshold = float(minimum_threshold)
        self.ambiguity_ratio = float(ambiguity_ratio)
        self.max_templates = max(self.minimum_templates, int(max_templates))
        self.duration_ratio_min = float(duration_ratio_min)
        self.duration_ratio_max = float(duration_ratio_max)

        strategy = str(temporal_prototype_strategy).strip().lower()
        if strategy not in {"templates", "dtw_barycenter"}:
            raise ValueError(
                "temporal_prototype_strategy must be 'templates' or 'dtw_barycenter'."
            )
        self.temporal_prototype_strategy = strategy
        self.max_temporal_prototypes = max(1, int(max_temporal_prototypes))
        self.prototype_iterations = max(1, int(prototype_iterations))

    @staticmethod
    def _validate_templates(templates: list[DynamicTrajectory]) -> str:
        if len(templates) < 2:
            raise ValueError("At least two dynamic demonstrations are required.")

        signature = templates[0].hand_signature
        shape_dimension = templates[0].shape_sequence.shape[1]
        motion_dimension = templates[0].motion_sequence.shape[1]
        velocity_dimension = templates[0].velocity_sequence.shape[1]

        for template in templates:
            if template.hand_signature != signature:
                raise ValueError("All demonstrations must use the same hand configuration.")
            if template.shape_sequence.shape[1] != shape_dimension:
                raise ValueError("Dynamic pose dimensions do not match.")
            if template.motion_sequence.shape[1] != motion_dimension:
                raise ValueError("Dynamic motion dimensions do not match.")
            if template.velocity_sequence.shape[1] != velocity_dimension:
                raise ValueError("Dynamic velocity dimensions do not match.")

        return signature

    @staticmethod
    def _pairwise_distances(templates: list[DynamicTrajectory]) -> list[float]:
        distances: list[float] = []
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                distance = trajectory_dtw_distance(templates[i], templates[j])
                if np.isfinite(distance):
                    distances.append(float(distance))
        return distances

    @staticmethod
    def _distance_to_representatives(
        trajectory: DynamicTrajectory,
        representatives: list[DynamicTrajectory],
    ) -> float:
        if not representatives:
            return float("inf")
        return min(
            trajectory_dtw_distance(trajectory, representative)
            for representative in representatives
        )

    def _rebuild(self, gesture: DynamicGestureClass) -> None:
        pairwise_distances = self._pairwise_distances(gesture.templates)

        if self.temporal_prototype_strategy == "dtw_barycenter":
            prototypes, _ = build_temporal_prototypes(
                gesture.templates,
                max_prototypes=self.max_temporal_prototypes,
                iterations=self.prototype_iterations,
            )
            gesture.temporal_prototypes = prototypes

            prototype_distances = [
                self._distance_to_representatives(template, prototypes)
                for template in gesture.templates
            ]
            prototype_distances = [
                float(distance)
                for distance in prototype_distances
                if np.isfinite(distance)
            ]

            if prototype_distances:
                prototype_reference = float(np.percentile(prototype_distances, 90))
            else:
                prototype_reference = 0.0

            # Pairwise class spread is approximately twice a center radius for
            # compact clusters. Keeping half of the old pairwise estimate stops
            # the new centroid threshold from becoming artificially tiny with
            # only three near-identical demonstrations.
            pairwise_reference = (
                0.5 * float(np.percentile(pairwise_distances, 90))
                if pairwise_distances
                else 0.0
            )
            reference = max(prototype_reference, pairwise_reference)
        else:
            gesture.temporal_prototypes = []
            if not pairwise_distances:
                reference = self.minimum_threshold / max(
                    self.threshold_multiplier, 1e-6
                )
            else:
                reference = float(np.percentile(pairwise_distances, 90))

        if reference <= 0.0:
            reference = self.minimum_threshold / max(self.threshold_multiplier, 1e-6)

        gesture.reference_distance = max(float(reference), 1e-6)
        gesture.threshold = max(
            gesture.reference_distance * self.threshold_multiplier,
            self.minimum_threshold,
        )
        gesture.median_duration = float(
            np.median([template.duration_seconds for template in gesture.templates])
        )

    def learn_gesture(
        self,
        name: str,
        templates: list[DynamicTrajectory],
    ) -> DynamicGestureClass:
        name = name.strip()
        if not name:
            raise ValueError("Dynamic gesture name cannot be empty.")
        if name in self.gestures:
            raise ValueError(f"Dynamic gesture '{name}' already exists.")
        if len(templates) < self.minimum_templates:
            raise ValueError(
                f"Need {self.minimum_templates} demonstrations to learn a dynamic gesture."
            )

        signature = self._validate_templates(templates)
        gesture = DynamicGestureClass(
            name=name,
            hand_signature=signature,
            templates=list(templates[: self.max_templates]),
        )
        self._rebuild(gesture)
        self.gestures[name] = gesture
        return gesture

    def add_template(
        self,
        name: str,
        template: DynamicTrajectory,
    ) -> DynamicGestureClass:
        if name not in self.gestures:
            raise KeyError(f"Unknown dynamic gesture: {name}")

        gesture = self.gestures[name]
        if template.hand_signature != gesture.hand_signature:
            raise ValueError(
                f"Gesture '{name}' expects {gesture.hand_signature} input, "
                f"but received {template.hand_signature}."
            )

        # Avoid filling memory with virtually identical demonstrations.
        nearest = min(
            trajectory_dtw_distance(template, existing)
            for existing in gesture.templates
        )
        if nearest < max(gesture.reference_distance * 0.35, 0.010):
            return gesture

        gesture.templates.append(template)
        if len(gesture.templates) > self.max_templates:
            # Keep a small diverse trajectory memory: remove the template with
            # the smallest average DTW distance to the others (most redundant).
            averages = []
            for i, candidate in enumerate(gesture.templates):
                distances = [
                    trajectory_dtw_distance(candidate, other)
                    for j, other in enumerate(gesture.templates)
                    if i != j
                ]
                averages.append(float(np.mean(distances)))
            remove_index = int(np.argmin(averages))
            gesture.templates.pop(remove_index)

        self._rebuild(gesture)
        return gesture

    def rename_gesture(self, old_name: str, new_name: str) -> DynamicGestureClass:
        if old_name not in self.gestures:
            raise KeyError(f"Unknown dynamic gesture: {old_name}")
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("New dynamic gesture name cannot be empty.")
        if new_name != old_name and new_name in self.gestures:
            raise ValueError(f"Dynamic gesture '{new_name}' already exists.")

        gesture = self.gestures.pop(old_name)
        gesture.name = new_name
        self.gestures[new_name] = gesture
        return gesture

    def delete_gesture(self, name: str) -> None:
        if name not in self.gestures:
            raise KeyError(f"Unknown dynamic gesture: {name}")
        del self.gestures[name]

    def clear(self) -> None:
        self.gestures.clear()

    def _gesture_representatives(
        self, gesture: DynamicGestureClass
    ) -> list[DynamicTrajectory]:
        if (
            self.temporal_prototype_strategy == "dtw_barycenter"
            and gesture.temporal_prototypes
        ):
            return gesture.temporal_prototypes
        return gesture.templates

    def predict(self, trajectory: DynamicTrajectory) -> DynamicPrediction:
        if not self.gestures:
            return DynamicPrediction(
                label=self.UNKNOWN_LABEL,
                accepted=False,
                rejection_reason="no_dynamic_gestures",
            )

        candidates = []
        for gesture in self.gestures.values():
            if gesture.hand_signature != trajectory.hand_signature:
                continue

            # Very large duration mismatch is a cheap sanity filter, while DTW
            # remains free to handle normal differences in performance speed.
            if gesture.median_duration > 1e-6:
                duration_ratio = trajectory.duration_seconds / gesture.median_duration
                if not (self.duration_ratio_min <= duration_ratio <= self.duration_ratio_max):
                    continue

            representatives = self._gesture_representatives(gesture)
            distance = self._distance_to_representatives(trajectory, representatives)
            if not np.isfinite(distance):
                continue

            candidates.append((gesture, float(distance), float(distance / gesture.threshold)))

        if not candidates:
            return DynamicPrediction(
                label=self.UNKNOWN_LABEL,
                accepted=False,
                rejection_reason="hand_configuration_or_duration",
            )

        candidates.sort(key=lambda item: item[2])
        best_gesture, best_distance, best_ratio = candidates[0]

        second_label = None
        second_distance = None
        second_ratio = None
        ambiguous = False
        if len(candidates) > 1:
            second_gesture, second_distance, second_ratio = candidates[1]
            second_label = second_gesture.name
            if best_distance > 1e-9:
                ambiguous = (second_distance / best_distance) < self.ambiguity_ratio

        accepted = best_ratio <= 1.0 and not ambiguous
        if best_ratio > 1.0:
            reason = "outside_dynamic_region"
        elif ambiguous:
            reason = "ambiguous_dynamic_match"
        else:
            reason = None

        if accepted:
            confidence_result = accepted_confidence(
                relative_distance=best_ratio,
                second_best_relative_distance=second_ratio,
            )
        elif reason == "ambiguous_dynamic_match":
            confidence_result = ambiguous_confidence(
                best_distance=best_distance,
                second_best_distance=second_distance,
                ambiguity_ratio=self.ambiguity_ratio,
            )
        else:
            confidence_result = outside_region_confidence(best_ratio)

        return DynamicPrediction(
            label=best_gesture.name if accepted else self.UNKNOWN_LABEL,
            accepted=accepted,
            nearest_label=best_gesture.name,
            distance=best_distance,
            threshold=best_gesture.threshold,
            relative_distance=best_ratio,
            second_best_label=second_label,
            second_best_distance=second_distance,
            second_best_relative_distance=second_ratio,
            confidence=confidence_result.score,
            rejection_reason=reason,
        )

    def list_gestures(self) -> list[str]:
        return list(self.gestures.keys())
