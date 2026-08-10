from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Generic, TypeVar


T = TypeVar("T")


class MonotonicTimestamp:
    """Generate strictly increasing millisecond timestamps for MediaPipe Tasks."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_ms = -1

    def next_ms(self, now_ns: int | None = None) -> int:
        candidate = int((time.monotonic_ns() if now_ns is None else now_ns) // 1_000_000)
        with self._lock:
            if candidate <= self._last_ms:
                candidate = self._last_ms + 1
            self._last_ms = candidate
            return candidate


@dataclass(frozen=True)
class AsyncTrackingDiagnostics:
    submitted_frames: int
    result_callbacks: int
    latest_timestamp_ms: int | None
    latest_latency_ms: float | None
    latest_result_age_ms: float | None


class LatestAsyncResult(Generic[T]):
    """Thread-safe bridge from a MediaPipe callback thread to the Tk camera loop."""

    def __init__(self, initial: T):
        self._lock = threading.Lock()
        self._value = initial
        self._submitted_frames = 0
        self._result_callbacks = 0
        self._latest_timestamp_ms: int | None = None
        self._latest_latency_ms: float | None = None
        self._published_monotonic: float | None = None

    def note_submission(self) -> None:
        with self._lock:
            self._submitted_frames += 1

    def publish(self, value: T, *, timestamp_ms: int, latency_ms: float | None) -> None:
        with self._lock:
            self._value = value
            self._result_callbacks += 1
            self._latest_timestamp_ms = int(timestamp_ms)
            self._latest_latency_ms = None if latency_ms is None else float(latency_ms)
            self._published_monotonic = time.monotonic()

    def snapshot(self) -> T:
        with self._lock:
            return self._value

    def diagnostics(self) -> AsyncTrackingDiagnostics:
        with self._lock:
            age_ms = None
            if self._published_monotonic is not None:
                age_ms = max(0.0, (time.monotonic() - self._published_monotonic) * 1000.0)
            return AsyncTrackingDiagnostics(
                submitted_frames=self._submitted_frames,
                result_callbacks=self._result_callbacks,
                latest_timestamp_ms=self._latest_timestamp_ms,
                latest_latency_ms=self._latest_latency_ms,
                latest_result_age_ms=age_ms,
            )


@dataclass(frozen=True)
class ReacquisitionDecision:
    expose_result: bool
    clear_tracker_state: bool = False


class ReacquisitionGuard:
    """Suppress transient hand results immediately after a confirmed absence.

    LIVE_STREAM hand tracking can return a short burst of unstable metadata when
    a hand re-enters the frame and a new track is being established.  This guard
    does not run at application startup; it activates only after a hand has been
    seen before and then fully disappears.
    """

    def __init__(self, confirm_frames: int = 3):
        self.confirm_frames = max(1, int(confirm_frames))
        self.ever_had_hand = False
        self.awaiting_reacquisition = False
        self.reacquire_streak = 0

    def update(self, *, raw_count: int, stable_count: int) -> ReacquisitionDecision:
        raw_count = max(0, int(raw_count))
        stable_count = max(0, int(stable_count))

        if raw_count == 0:
            if self.awaiting_reacquisition:
                # The candidate new track disappeared before it was confirmed.
                self.reacquire_streak = 0
                return ReacquisitionDecision(False, clear_tracker_state=True)

            if stable_count == 0 and self.ever_had_hand:
                self.awaiting_reacquisition = True
                self.reacquire_streak = 0

            return ReacquisitionDecision(stable_count > 0)

        self.ever_had_hand = True

        if not self.awaiting_reacquisition:
            return ReacquisitionDecision(True)

        self.reacquire_streak += 1
        if self.reacquire_streak < self.confirm_frames:
            return ReacquisitionDecision(False)

        self.awaiting_reacquisition = False
        self.reacquire_streak = 0
        return ReacquisitionDecision(True)


def choose_handedness(
    *,
    previous_label: str,
    previous_score: float,
    current_label: str,
    current_score: float,
    wrist_continuous: bool,
    reacquiring: bool,
) -> tuple[str, float]:
    """Choose a stable Left/Right identity without making bad reacquisition sticky."""
    known = {"Left", "Right"}
    previous_known = previous_label in known
    current_known = current_label in known

    if reacquiring:
        # While a new track settles, use the newest Tasks estimate so the first
        # transient Unknown/wrong label cannot lock the whole track forever.
        return current_label, float(current_score)

    if not previous_known and current_known:
        # A valid label should always be allowed to repair an earlier Unknown.
        return current_label, float(current_score)

    if wrist_continuous:
        return previous_label, max(float(previous_score), float(current_score))

    return current_label, float(current_score)
