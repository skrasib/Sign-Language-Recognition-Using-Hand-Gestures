from collections import deque
from dataclasses import dataclass

from adaptive_gesture.features.dynamic_features import (
    DynamicObservation,
    observation_motion_score,
)


@dataclass
class MotionSegmentResult:
    state: str
    motion_score: float | None = None
    completed: list[DynamicObservation] | None = None


class MotionSegmenter:
    """
    Lightweight online motion onset/offset detector.

    It runs only on landmark observations. A small pre-roll keeps the true
    beginning of the movement, and hysteresis (different start/stop thresholds)
    avoids rapid start/stop oscillation around one threshold.
    """

    IDLE = "IDLE"
    MOTION = "MOTION"
    COOLDOWN = "COOLDOWN"

    def __init__(
        self,
        start_threshold: float = 0.045,
        stop_threshold: float = 0.020,
        start_confirm_frames: int = 2,
        stop_confirm_frames: int = 7,
        pre_roll_frames: int = 4,
        minimum_frames: int = 10,
        maximum_frames: int = 110,
        missing_tolerance: int = 2,
    ):
        self.start_threshold = float(start_threshold)
        self.stop_threshold = float(stop_threshold)
        self.start_confirm_frames = max(1, int(start_confirm_frames))
        self.stop_confirm_frames = max(1, int(stop_confirm_frames))
        self.minimum_frames = max(3, int(minimum_frames))
        self.maximum_frames = max(self.minimum_frames, int(maximum_frames))
        self.missing_tolerance = max(0, int(missing_tolerance))

        self.pre_roll = deque(maxlen=max(1, int(pre_roll_frames)))
        self.previous: DynamicObservation | None = None
        self.active: list[DynamicObservation] = []
        self.state = self.IDLE
        self.start_streak = 0
        self.stop_streak = 0
        self.missing_streak = 0
        self.cooldown_until = 0.0

    def reset(self) -> None:
        self.pre_roll.clear()
        self.previous = None
        self.active = []
        self.state = self.IDLE
        self.start_streak = 0
        self.stop_streak = 0
        self.missing_streak = 0
        self.cooldown_until = 0.0

    def set_cooldown(self, now: float, seconds: float = 0.65) -> None:
        self.state = self.COOLDOWN
        self.cooldown_until = float(now) + max(0.0, float(seconds))
        self.pre_roll.clear()
        self.active = []
        self.previous = None
        self.start_streak = 0
        self.stop_streak = 0
        self.missing_streak = 0

    def _finish(self) -> MotionSegmentResult:
        completed = list(self.active)
        self.active = []
        self.pre_roll.clear()
        self.state = self.IDLE
        self.start_streak = 0
        self.stop_streak = 0
        self.missing_streak = 0
        self.previous = None

        if len(completed) < self.minimum_frames:
            return MotionSegmentResult(state=self.IDLE)
        return MotionSegmentResult(state=self.IDLE, completed=completed)

    def update(
        self,
        observation: DynamicObservation | None,
        now: float,
    ) -> MotionSegmentResult:
        now = float(now)

        if self.state == self.COOLDOWN:
            if now < self.cooldown_until:
                return MotionSegmentResult(state=self.COOLDOWN)
            self.state = self.IDLE
            self.previous = None
            self.pre_roll.clear()

        if observation is None:
            self.missing_streak += 1
            if self.state == self.MOTION and self.missing_streak > self.missing_tolerance:
                return self._finish()
            if self.state == self.IDLE:
                self.previous = None
                self.pre_roll.clear()
                self.start_streak = 0
            return MotionSegmentResult(state=self.state)

        self.missing_streak = 0

        if self.previous is None:
            self.previous = observation
            self.pre_roll.append(observation)
            return MotionSegmentResult(state=self.state, motion_score=0.0)

        # Changing from one-hand to two-hand (or Left to Right) ends the current
        # segment; dynamic classes intentionally have a fixed configuration.
        if observation.hand_signature != self.previous.hand_signature:
            if self.state == self.MOTION:
                result = self._finish()
                self.previous = observation
                self.pre_roll.append(observation)
                return result
            self.pre_roll.clear()
            self.pre_roll.append(observation)
            self.previous = observation
            self.start_streak = 0
            return MotionSegmentResult(state=self.IDLE)

        score = observation_motion_score(self.previous, observation)
        self.previous = observation

        if self.state == self.IDLE:
            self.pre_roll.append(observation)
            if score >= self.start_threshold:
                self.start_streak += 1
            else:
                self.start_streak = 0

            if self.start_streak >= self.start_confirm_frames:
                self.state = self.MOTION
                self.active = list(self.pre_roll)
                self.stop_streak = 0
                return MotionSegmentResult(state=self.MOTION, motion_score=score)

            return MotionSegmentResult(state=self.IDLE, motion_score=score)

        # Active motion.
        self.active.append(observation)

        if score <= self.stop_threshold:
            self.stop_streak += 1
        else:
            self.stop_streak = 0

        if len(self.active) >= self.maximum_frames:
            return self._finish()

        if (
            self.stop_streak >= self.stop_confirm_frames
            and len(self.active) >= self.minimum_frames
        ):
            return self._finish()

        return MotionSegmentResult(state=self.MOTION, motion_score=score)
