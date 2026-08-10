from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from adaptive_gesture.features.similarity import feature_distance
from adaptive_gesture.learning.online_learner import OnlineGestureLearner, Prediction


_EPS = 1e-8


@dataclass
class ExtremeVectorModel:
    """EVM-style radial inclusion model fitted around one positive exemplar."""

    vector: np.ndarray
    shape: float
    scale: float
    tail_size: int
    nearest_negative_margin: float

    def inclusion(self, query: np.ndarray) -> float:
        """Probability-of-sample-inclusion style radial score in [0, 1]."""
        distance = max(feature_distance(query, self.vector), 0.0)
        scale = max(float(self.scale), _EPS)
        shape = max(float(self.shape), 0.05)
        exponent = -((distance / scale) ** shape)
        # Prevent numerical underflow warnings while preserving a practical zero.
        exponent = max(exponent, -80.0)
        return float(math.exp(exponent))


@dataclass
class EVTPrediction(Prediction):
    """Prediction enriched with the open-set statistics used by V3.2."""

    open_set_score: float | None = None
    open_set_threshold: float | None = None
    open_set_method: str | None = None
    extreme_vector_count: int = 0


def _weibull_shape_equation(k: float, samples: np.ndarray) -> float:
    """Score equation for a two-parameter Weibull with location fixed at zero."""
    logs = np.log(samples)
    scaled = k * logs
    scaled -= float(np.max(scaled))
    weights = np.exp(scaled)
    denominator = float(np.sum(weights))
    if denominator <= _EPS:
        return float("nan")
    weighted_log = float(np.sum(weights * logs) / denominator)
    return (1.0 / k) + float(np.mean(logs)) - weighted_log


def fit_weibull_zero_location(values: np.ndarray) -> tuple[float, float]:
    """
    Fit a two-parameter Weibull using maximum-likelihood equations.

    This intentionally avoids an additional SciPy dependency. The location is fixed
    to zero, matching the positive margin-distance model used by the EVM.
    """
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    samples = np.clip(samples, 1e-6, None)

    if samples.size == 0:
        raise ValueError("At least one positive margin value is required.")

    # Degenerate tails can occur in tiny few-shot sets. A high-but-bounded shape
    # gives a stable, almost hard radial boundary without numerical explosions.
    if float(np.max(samples) - np.min(samples)) < 1e-7:
        return 12.0, float(np.mean(samples))

    lo, hi = 0.15, 30.0
    f_lo = _weibull_shape_equation(lo, samples)
    f_hi = _weibull_shape_equation(hi, samples)

    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        # Robust fallback based on log-space dispersion.
        log_std = float(np.std(np.log(samples)))
        shape = float(np.clip(1.2 / max(log_std, 1e-3), 0.25, 20.0))
    else:
        # The score equation is monotonic for positive samples. Bisection is slower
        # than Newton but much safer for few-shot / nearly-degenerate tails.
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = _weibull_shape_equation(mid, samples)
            if not np.isfinite(f_mid):
                break
            if abs(f_mid) < 1e-10:
                lo = hi = mid
                break
            if f_lo * f_mid <= 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        shape = float(np.clip(0.5 * (lo + hi), 0.25, 20.0))

    logs = np.log(samples)
    powered_logs = shape * logs
    maximum = float(np.max(powered_logs))
    log_mean_power = maximum + math.log(float(np.mean(np.exp(powered_logs - maximum))))
    scale = float(math.exp(log_mean_power / shape))
    return shape, max(scale, 1e-6)


class EVTOpenSetGestureLearner(OnlineGestureLearner):
    """
    V3.2 open-set learner inspired by the Extreme Value Machine (EVM).

    The existing online prototype/exemplar learner remains the closed-set/fallback
    engine. Once enough compatible negative evidence exists, each positive exemplar
    receives an EVT-fitted Weibull radial inclusion model based on the smallest
    half-distances to negative samples. Queries are accepted only when the best
    class inclusion score exceeds ``inclusion_threshold``.

    This preserves runtime class addition and feedback while replacing the primary
    heuristic radius gate with a distribution-driven open-set decision.
    """

    def __init__(
        self,
        *args,
        evt_tail_size: int = 10,
        evt_min_negatives: int = 3,
        inclusion_threshold: float = 0.35,
        top_k_inclusion: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.evt_tail_size = max(2, int(evt_tail_size))
        self.evt_min_negatives = max(2, int(evt_min_negatives))
        self.inclusion_threshold = float(np.clip(inclusion_threshold, 0.01, 0.99))
        self.top_k_inclusion = max(1, int(top_k_inclusion))
        self.extreme_models: dict[str, list[ExtremeVectorModel]] = {}

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _compatible_negative_samples(self, target_name: str) -> list[np.ndarray]:
        target = self.gestures[target_name]
        negatives: list[np.ndarray] = []

        # Other learned classes provide the negative margins needed by the EVM.
        for other_name, other in self.gestures.items():
            if other_name == target_name:
                continue
            if other.hand_signature != target.hand_signature:
                continue
            if other.feature_dimension != target.feature_dimension:
                continue
            negatives.extend(other.samples)

        # Explicit user corrections are particularly informative boundary points.
        negatives.extend(
            negative
            for negative in target.hard_negatives
            if negative.shape == (target.feature_dimension,)
        )
        return negatives

    def _fit_models_for_class(self, name: str) -> list[ExtremeVectorModel]:
        gesture = self.gestures[name]
        negatives = self._compatible_negative_samples(name)
        if len(negatives) < self.evt_min_negatives:
            return []

        models: list[ExtremeVectorModel] = []
        for positive in gesture.samples:
            half_distances = np.asarray(
                [0.5 * feature_distance(positive, negative) for negative in negatives],
                dtype=np.float64,
            )
            half_distances = half_distances[np.isfinite(half_distances)]
            half_distances = half_distances[half_distances > 1e-7]
            if half_distances.size < self.evt_min_negatives:
                continue

            # EVM fits the Weibull to the smallest margin estimates, i.e. the
            # negative points closest to this positive exemplar's boundary.
            tail = np.sort(half_distances)[: min(self.evt_tail_size, half_distances.size)]
            shape, scale = fit_weibull_zero_location(tail)
            models.append(
                ExtremeVectorModel(
                    vector=np.asarray(positive, dtype=np.float32).copy(),
                    shape=shape,
                    scale=scale,
                    tail_size=int(tail.size),
                    nearest_negative_margin=float(tail[0]),
                )
            )
        return models

    def rebuild_open_set_models(self) -> None:
        self.extreme_models = {
            name: self._fit_models_for_class(name)
            for name in self.gestures
        }

    # ------------------------------------------------------------------
    # Incremental updates. Any positive class can become a negative class
    # for another gesture, so class additions/positive updates rebuild the
    # compatible open-set boundaries globally. The few-shot memories are
    # deliberately small, keeping this inexpensive.
    # ------------------------------------------------------------------

    def learn_gesture(self, *args, **kwargs):
        gesture = super().learn_gesture(*args, **kwargs)
        self.rebuild_open_set_models()
        return gesture

    def rename_gesture(self, *args, **kwargs):
        gesture = super().rename_gesture(*args, **kwargs)
        self.rebuild_open_set_models()
        return gesture

    def delete_gesture(self, *args, **kwargs):
        result = super().delete_gesture(*args, **kwargs)
        self.rebuild_open_set_models()
        return result

    def clear(self):
        super().clear()
        self.extreme_models.clear()

    def add_samples_to_gesture(self, *args, **kwargs):
        gesture = super().add_samples_to_gesture(*args, **kwargs)
        self.rebuild_open_set_models()
        return gesture

    def replace_gesture_samples(self, *args, **kwargs):
        gesture = super().replace_gesture_samples(*args, **kwargs)
        self.rebuild_open_set_models()
        return gesture

    def update_gesture(self, *args, **kwargs):
        gesture = super().update_gesture(*args, **kwargs)
        self.rebuild_open_set_models()
        return gesture

    def add_hard_negative(self, *args, **kwargs):
        gesture = super().add_hard_negative(*args, **kwargs)
        self.rebuild_open_set_models()
        return gesture

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _class_inclusion(self, name: str, features: np.ndarray) -> tuple[float, int]:
        models = self.extreme_models.get(name, [])
        if not models:
            return 0.0, 0
        scores = sorted((model.inclusion(features) for model in models), reverse=True)
        k = min(self.top_k_inclusion, len(scores))
        return float(np.mean(scores[:k])), len(models)

    def predict(
        self,
        features: np.ndarray,
        hand_signature: str | None = None,
        handedness: str | None = None,
    ) -> Prediction:
        features = np.asarray(features, dtype=np.float32)
        signature = hand_signature or handedness

        if not self.gestures:
            return EVTPrediction(
                label=self.UNKNOWN_LABEL,
                accepted=False,
                rejection_reason="no_gestures",
                open_set_method="evt_evm",
            )

        compatible_names = [
            name
            for name, gesture in self.gestures.items()
            if (signature is None or gesture.hand_signature == signature)
            and features.shape == (gesture.feature_dimension,)
        ]

        if not compatible_names:
            return EVTPrediction(
                label=self.UNKNOWN_LABEL,
                accepted=False,
                rejection_reason="hand_configuration",
                open_set_method="evt_evm",
            )

        # A single class (or an early class with too little negative evidence)
        # cannot support a meaningful EVM margin fit. Preserve V3.1's proven
        # local-radius behavior until sufficient open-set evidence exists.
        if any(not self.extreme_models.get(name) for name in compatible_names):
            fallback = super().predict(
                features,
                hand_signature=hand_signature,
                handedness=handedness,
            )
            setattr(fallback, "open_set_method", "radius_fallback")
            setattr(fallback, "open_set_score", None)
            setattr(fallback, "open_set_threshold", None)
            setattr(fallback, "extreme_vector_count", 0)
            return fallback

        candidates = []
        for name in compatible_names:
            gesture = self.gestures[name]
            inclusion, model_count = self._class_inclusion(name, features)
            nearest_positive = min(
                feature_distance(features, sample) for sample in gesture.samples
            )
            prototype_distance = min(
                [feature_distance(features, cluster.prototype) for cluster in gesture.prototypes]
                or [feature_distance(features, gesture.prototype)]
            )
            nearest_negative = None
            if gesture.hard_negatives:
                nearest_negative = min(
                    feature_distance(features, negative)
                    for negative in gesture.hard_negatives
                )
            candidates.append(
                {
                    "name": name,
                    "gesture": gesture,
                    "inclusion": inclusion,
                    "models": model_count,
                    "nearest_positive": nearest_positive,
                    "prototype_distance": prototype_distance,
                    "nearest_negative": nearest_negative,
                }
            )

        candidates.sort(key=lambda item: item["inclusion"], reverse=True)
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        accepted = best["inclusion"] >= self.inclusion_threshold
        reason = None if accepted else "evt_open_set"

        # Retain the already-tested correction veto. Hard negatives are also used
        # to fit the EVT boundary, but this direct veto makes a just-corrected pose
        # take effect immediately even in a very small few-shot tail.
        nearest_negative = best["nearest_negative"]
        if (
            accepted
            and nearest_negative is not None
            and nearest_negative <= best["nearest_positive"] * self.hard_negative_margin
        ):
            accepted = False
            reason = "hard_negative"

        if accepted:
            confidence = best["inclusion"]
        elif reason == "hard_negative" and nearest_negative is not None:
            ratio = nearest_negative / max(best["nearest_positive"], 1e-6)
            confidence = float(np.clip(1.0 - ratio / self.hard_negative_margin, 0.0, 1.0))
        else:
            # For UNKNOWN the index expresses strength of rejection, not posterior
            # correctness. Values near the threshold intentionally have low score.
            confidence = float(
                np.clip(
                    1.0 - best["inclusion"] / max(self.inclusion_threshold, 1e-6),
                    0.0,
                    1.0,
                )
            )

        second_label = second["name"] if second else None
        second_relative = (1.0 - second["inclusion"]) if second else None

        return EVTPrediction(
            label=best["name"] if accepted else self.UNKNOWN_LABEL,
            accepted=accepted,
            nearest_label=best["name"],
            distance=best["nearest_positive"],
            prototype_distance=best["prototype_distance"],
            relative_distance=1.0 - best["inclusion"],
            threshold=None,
            hard_negative_distance=nearest_negative,
            second_best_label=second_label,
            second_best_relative_distance=second_relative,
            confidence=confidence,
            rejection_reason=reason,
            open_set_score=best["inclusion"],
            open_set_threshold=self.inclusion_threshold,
            open_set_method="evt_evm",
            extreme_vector_count=best["models"],
        )
