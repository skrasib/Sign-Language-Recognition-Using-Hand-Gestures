from dataclasses import dataclass
from typing import Any


@dataclass
class StabilizationResult:
    """
    Result returned by PredictionStabilizer.update().

    prediction:
        The currently confirmed/stable prediction. This can be None while the
        first prediction is still being confirmed.

    candidate_label / candidate_count / required_count:
        Describe a new label that is currently being checked before it is
        allowed to replace the confirmed label.
    """

    prediction: Any | None
    candidate_label: str | None = None
    candidate_count: int = 0
    required_count: int = 0
    changed: bool = False

    @property
    def pending(self) -> bool:
        return (
            self.candidate_label is not None
            and self.candidate_count > 0
            and self.required_count > 0
        )


class PredictionStabilizer:
    """
    Small temporal voting layer for frame-by-frame gesture predictions.

    A new known gesture must be observed consistently for several recognition
    updates before it becomes the visible confirmed prediction. UNKNOWN can use
    a slightly shorter confirmation period so the recognizer does not remain
    stuck on an old known gesture for too long.

    This class does not modify the learner or its distances. It only stabilizes
    the sequence of already-produced predictions.
    """

    def __init__(
        self,
        confirm_frames: int = 3,
        unknown_confirm_frames: int = 2,
        unknown_label: str = "UNKNOWN",
    ):
        self.confirm_frames = max(1, int(confirm_frames))
        self.unknown_confirm_frames = max(1, int(unknown_confirm_frames))
        self.unknown_label = str(unknown_label)

        self._confirmed_prediction = None
        self._candidate_label: str | None = None
        self._candidate_prediction = None
        self._candidate_count = 0

    @staticmethod
    def _label(prediction) -> str | None:
        if prediction is None:
            return None
        return getattr(prediction, "label", None)

    def _required_frames(self, label: str | None) -> int:
        if label == self.unknown_label:
            return self.unknown_confirm_frames
        return self.confirm_frames

    def _clear_candidate(self) -> None:
        self._candidate_label = None
        self._candidate_prediction = None
        self._candidate_count = 0

    def reset(self) -> None:
        """Forget temporal state, for example after teaching or no-hand state."""
        self._confirmed_prediction = None
        self._clear_candidate()

    @property
    def confirmed_prediction(self):
        return self._confirmed_prediction

    def update(self, prediction) -> StabilizationResult:
        if prediction is None:
            self.reset()
            return StabilizationResult(prediction=None)

        label = self._label(prediction)

        # ----------------------------------------------------
        # The raw prediction agrees with the already-confirmed label.
        # Refresh its distances/metrics and cancel any competing candidate.
        # ----------------------------------------------------
        if (
            self._confirmed_prediction is not None
            and label == self._label(self._confirmed_prediction)
        ):
            self._confirmed_prediction = prediction
            self._clear_candidate()
            return StabilizationResult(
                prediction=self._confirmed_prediction,
                changed=False,
            )

        # ----------------------------------------------------
        # New/different candidate. It must persist for N updates.
        # ----------------------------------------------------
        if label == self._candidate_label:
            self._candidate_count += 1
            self._candidate_prediction = prediction
        else:
            self._candidate_label = label
            self._candidate_prediction = prediction
            self._candidate_count = 1

        required = self._required_frames(label)

        if self._candidate_count >= required:
            previous_label = self._label(self._confirmed_prediction)
            self._confirmed_prediction = self._candidate_prediction
            new_label = self._label(self._confirmed_prediction)
            changed = previous_label != new_label
            self._clear_candidate()

            return StabilizationResult(
                prediction=self._confirmed_prediction,
                changed=changed,
            )

        # Until the candidate is confirmed, keep showing the previous stable
        # prediction. On startup this is None, so the UI can explicitly show a
        # short "STABILIZING" state instead of flickering between labels.
        return StabilizationResult(
            prediction=self._confirmed_prediction,
            candidate_label=self._candidate_label,
            candidate_count=self._candidate_count,
            required_count=required,
            changed=False,
        )
