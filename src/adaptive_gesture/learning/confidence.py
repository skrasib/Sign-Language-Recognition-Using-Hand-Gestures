from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ConfidenceResult:
    """
    Bounded recognition-confidence index.

    score is in [0, 1], but it is intentionally NOT a calibrated probability.
    It combines distance-to-boundary, competing-class separation, and (when
    available) hard-negative evidence. Proper probability calibration should be
    performed later with independent evaluation/calibration data.
    """

    score: float
    fit: float
    separation: float
    evidence: float

    @property
    def percent(self) -> float:
        return self.score * 100.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _sigmoid(value: float) -> float:
    # Numerically safe enough for the small values used by this module.
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _known_fit(relative_distance: float) -> float:
    """
    Convert normalized distance-to-threshold into a smooth fit score.

    relative_distance = 0   -> very strong fit
    relative_distance ~ 1   -> weak/borderline fit
    """

    r = max(0.0, float(relative_distance))
    return _sigmoid(6.0 * (0.72 - r))


def _known_separation(
    best_relative_distance: float,
    second_best_relative_distance: float | None,
) -> float:
    """Reward a clear gap between the best and second-best known classes."""

    if second_best_relative_distance is None:
        # With only one compatible class, class separation is unknown rather
        # than perfect. Use a conservative neutral value.
        return 0.65

    best = max(0.0, float(best_relative_distance))
    second = max(0.0, float(second_best_relative_distance))
    margin = max(0.0, second - best)
    return 1.0 - math.exp(-2.5 * margin)


def accepted_confidence(
    relative_distance: float,
    second_best_relative_distance: float | None = None,
    positive_distance: float | None = None,
    hard_negative_distance: float | None = None,
) -> ConfidenceResult:
    """
    Confidence index for an accepted known gesture.

    The score is driven mainly by fit to the learned class region, with a
    smaller contribution from separation from the next-best class. Nearby
    hard-negative evidence can only reduce the result.
    """

    fit = _known_fit(relative_distance)
    separation = _known_separation(
        relative_distance,
        second_best_relative_distance,
    )

    evidence = 1.0
    if (
        positive_distance is not None
        and hard_negative_distance is not None
        and positive_distance > 1e-9
    ):
        ratio = float(hard_negative_distance) / float(positive_distance)
        # Negative evidence that is only barely farther away than the positive
        # evidence should lower confidence. A safely distant negative has no
        # penalty.
        evidence = 0.70 + 0.30 * _clamp((ratio - 1.0) / 1.5)

    score = (0.78 * fit + 0.22 * separation) * evidence
    return ConfidenceResult(
        score=_clamp(score, 0.01, 0.99),
        fit=_clamp(fit),
        separation=_clamp(separation),
        evidence=_clamp(evidence),
    )


def outside_region_confidence(
    relative_distance: float,
) -> ConfidenceResult:
    """Confidence index for UNKNOWN caused by being outside a class boundary."""

    r = max(0.0, float(relative_distance))
    outside = max(0.0, r - 1.0)

    # Immediately outside the boundary should be only moderately certain;
    # confidence rises smoothly as the observation moves farther away.
    rejection_strength = 1.0 - math.exp(-2.2 * outside)
    score = 0.52 + 0.46 * rejection_strength

    return ConfidenceResult(
        score=_clamp(score, 0.01, 0.99),
        fit=_clamp(1.0 - min(r, 1.0)),
        separation=0.50,
        evidence=_clamp(rejection_strength),
    )


def hard_negative_confidence(
    positive_distance: float | None,
    hard_negative_distance: float | None,
    threshold: float | None,
) -> ConfidenceResult:
    """Confidence index for UNKNOWN caused by learned hard-negative evidence."""

    if (
        positive_distance is None
        or hard_negative_distance is None
        or threshold is None
        or threshold <= 1e-9
    ):
        return ConfidenceResult(0.70, 0.0, 0.5, 0.7)

    advantage = (
        float(positive_distance) - float(hard_negative_distance)
    ) / float(threshold)
    strength = _clamp(advantage / 0.75)
    score = 0.65 + 0.30 * strength

    return ConfidenceResult(
        score=_clamp(score, 0.01, 0.99),
        fit=0.0,
        separation=0.5,
        evidence=_clamp(strength),
    )


def ambiguous_confidence(
    best_distance: float | None,
    second_best_distance: float | None,
    ambiguity_ratio: float,
) -> ConfidenceResult:
    """
    Confidence index for UNKNOWN caused by two dynamic classes being too close.

    A near tie gives stronger evidence that the result should remain UNKNOWN.
    """

    if (
        best_distance is None
        or second_best_distance is None
        or best_distance <= 1e-9
        or ambiguity_ratio <= 1.0
    ):
        return ConfidenceResult(0.65, 0.0, 0.0, 0.65)

    ratio = float(second_best_distance) / float(best_distance)
    # ratio == 1: exact tie (strong ambiguity)
    # ratio == ambiguity_ratio: just at the ambiguity boundary
    ambiguity_strength = _clamp(
        (float(ambiguity_ratio) - ratio)
        / (float(ambiguity_ratio) - 1.0)
    )
    score = 0.55 + 0.40 * ambiguity_strength

    return ConfidenceResult(
        score=_clamp(score, 0.01, 0.99),
        fit=0.0,
        separation=_clamp(1.0 - ambiguity_strength),
        evidence=_clamp(ambiguity_strength),
    )
