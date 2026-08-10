from dataclasses import dataclass


ONE_HAND_MODE = "one"
TWO_HAND_MODE = "two"


def required_hand_count(mode: str) -> int:
    """Return the exact number of hands required by a teaching mode."""
    if mode == ONE_HAND_MODE:
        return 1
    if mode == TWO_HAND_MODE:
        return 2
    raise ValueError(f"Unsupported teaching hand mode: {mode!r}")


def hand_mode_from_signature(hand_signature: str | None) -> str:
    """Map the learner's Left/Right/Both signature to the UI hand-count mode."""
    return TWO_HAND_MODE if hand_signature == "Both" else ONE_HAND_MODE


def describe_hand_requirement(required_count: int, exact_signature: str | None = None) -> str:
    if required_count == 2:
        return "2 hands"
    if exact_signature in {"Left", "Right"}:
        return f"1 {exact_signature.lower()} hand"
    return "1 hand"


@dataclass(frozen=True)
class HandReadiness:
    """Snapshot returned by :class:`HandReadinessGate`."""

    ready: bool
    matched_results: int
    required_results: int
    hand_signature: str | None
    hold_seconds: float
    required_hold_seconds: float


class HandReadinessGate:
    """
    Confirm that the requested hand configuration is genuinely present.

    Tkinter may render the same asynchronous MediaPipe result several times, so
    readiness is counted only when ``result_token`` changes.  A configuration
    must satisfy both a minimum number of fresh tracker results and a minimum
    wall-clock hold time.  This prevents a one-frame second-hand detection from
    arming a two-hand teaching session.

    ``max_motion_score`` is optional.  Static teaching leaves it disabled because
    Smart Capture already checks pose stability.  Dynamic teaching supplies a
    small threshold so a new demonstration is armed only after the user has
    returned to a reasonably still starting pose.
    """

    def __init__(
        self,
        required_count: int,
        *,
        expected_signature: str | None = None,
        confirm_results: int = 5,
        hold_seconds: float = 0.30,
        max_motion_score: float | None = None,
    ):
        if int(required_count) not in (1, 2):
            raise ValueError("required_count must be 1 or 2")
        if expected_signature not in (None, "Left", "Right", "Both"):
            raise ValueError("expected_signature must be Left, Right, Both, or None")
        if expected_signature == "Both" and int(required_count) != 2:
            raise ValueError("Both requires required_count=2")
        if expected_signature in {"Left", "Right"} and int(required_count) != 1:
            raise ValueError("Left/Right requires required_count=1")

        self.required_count = int(required_count)
        self.expected_signature = expected_signature
        self.confirm_results = max(1, int(confirm_results))
        self.required_hold_seconds = max(0.0, float(hold_seconds))
        self.max_motion_score = (
            None if max_motion_score is None else max(0.0, float(max_motion_score))
        )
        self.reset()

    def reset(self) -> None:
        self.matched_results = 0
        self.match_started_at: float | None = None
        self.last_result_token = None
        self.candidate_signature: str | None = None

    def _restart_candidate(self, hand_signature: str | None, now: float) -> None:
        self.matched_results = 0
        self.match_started_at = None
        self.last_result_token = None
        self.candidate_signature = hand_signature

    def _configuration_matches(
        self,
        hand_count: int,
        hand_signature: str | None,
    ) -> bool:
        if int(hand_count) != self.required_count:
            return False
        if self.required_count == 2 and hand_signature != "Both":
            return False
        if self.required_count == 1 and hand_signature not in {"Left", "Right"}:
            return False
        if self.expected_signature is not None and hand_signature != self.expected_signature:
            return False
        return True

    def update(
        self,
        *,
        hand_count: int,
        hand_signature: str | None,
        result_token,
        now: float,
        motion_score: float | None = None,
    ) -> HandReadiness:
        now = float(now)

        if not self._configuration_matches(hand_count, hand_signature):
            self.reset()
            return self.snapshot(now)

        # For a generic one-hand teaching request, make sure Left/Right remains
        # consistent throughout the confirmation window.  We only permanently
        # lock the gesture signature after the gate reports ready.
        if self.required_count == 1 and self.expected_signature is None:
            if self.candidate_signature is None:
                self.candidate_signature = hand_signature
            elif hand_signature != self.candidate_signature:
                self._restart_candidate(hand_signature, now)

        if self.max_motion_score is not None:
            if motion_score is None or float(motion_score) > self.max_motion_score:
                # Keep the candidate handedness, but require a fresh stillness
                # window before declaring the pose ready.
                candidate = self.candidate_signature
                self.reset()
                self.candidate_signature = candidate
                return self.snapshot(now)

        # Ignore duplicate renders of the same async landmark callback.
        if result_token is None or result_token == self.last_result_token:
            return self.snapshot(now)

        self.last_result_token = result_token
        if self.match_started_at is None:
            self.match_started_at = now
        self.matched_results += 1
        return self.snapshot(now)

    def snapshot(self, now: float) -> HandReadiness:
        if self.match_started_at is None:
            elapsed = 0.0
        else:
            elapsed = max(0.0, float(now) - self.match_started_at)

        signature = self.expected_signature or self.candidate_signature
        ready = (
            self.matched_results >= self.confirm_results
            and elapsed >= self.required_hold_seconds
        )
        return HandReadiness(
            ready=ready,
            matched_results=self.matched_results,
            required_results=self.confirm_results,
            hand_signature=signature,
            hold_seconds=elapsed,
            required_hold_seconds=self.required_hold_seconds,
        )
