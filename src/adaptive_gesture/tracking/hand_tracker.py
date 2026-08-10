from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .model_assets import ensure_hand_landmarker_model
from .task_runtime import (
    LatestAsyncResult,
    MonotonicTimestamp,
    ReacquisitionGuard,
    choose_handedness,
)


logger = logging.getLogger(__name__)


# Canonical MediaPipe 21-landmark hand graph. Keeping the drawing topology local
# avoids depending on the deprecated ``mp.solutions`` drawing stack.
_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class TrackedHand:
    handedness: str
    handedness_score: float
    image_landmarks: np.ndarray
    world_landmarks: np.ndarray | None
    raw_landmarks: object


class HandTracker:
    """Asynchronous MediaPipe Tasks Hand Landmarker for live webcam tracking.

    V3.6 replaces the legacy ``mp.solutions.hands.Hands.process`` call with
    ``HandLandmarker.detect_async`` in LIVE_STREAM mode. The Tk camera loop only
    submits frames and consumes the latest completed result; landmark inference
    runs through MediaPipe's asynchronous task pipeline and may intentionally
    drop input frames to keep live-stream latency low.

    The temporal stabilization introduced in V2 is preserved on top of the new
    task results: brief detection loss is held, hand-count changes use hysteresis,
    handedness identity is stabilized, and landmark coordinates are smoothed.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        min_presence_confidence: float | None = None,
        missing_hold_frames: int = 2,
        two_hand_confirm_frames: int = 3,
        one_hand_confirm_frames: int = 3,
        smoothing_alpha: float = 0.65,
        model_path: str | Path | None = None,
        auto_download_model: bool = True,
        stale_result_ms: float = 500.0,
        reacquire_confirm_frames: int = 3,
    ):
        self.max_num_hands = int(max_num_hands)
        self.missing_hold_frames = max(0, int(missing_hold_frames))
        self.two_hand_confirm_frames = max(1, int(two_hand_confirm_frames))
        self.one_hand_confirm_frames = max(1, int(one_hand_confirm_frames))
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.stale_result_ms = max(100.0, float(stale_result_ms))
        self.reacquire_confirm_frames = max(1, int(reacquire_confirm_frames))

        self._stable_hands: list[TrackedHand] = []
        self._stable_mode = 0
        self._missing_streak = 0
        self._one_hand_streak = 0
        self._two_hand_streak = 0

        # After a hand fully leaves the frame, wait for a few consecutive fresh
        # Tasks callbacks before exposing the reacquired track to recognition.
        # This avoids locking a transient first-frame handedness/world estimate.
        self._reacquisition_guard = ReacquisitionGuard(
            confirm_frames=self.reacquire_confirm_frames
        )

        self._closed = False

        self._timestamps = MonotonicTimestamp()
        self._results = LatestAsyncResult[list[TrackedHand]]([])

        requested_path = Path(model_path) if model_path is not None else Path(
            "data/v3/models/hand_landmarker.task"
        )
        if auto_download_model:
            try:
                self.model_path = ensure_hand_landmarker_model(requested_path)
            except Exception as error:
                raise RuntimeError(
                    "Could not prepare the MediaPipe Hand Landmarker model. "
                    "The first V3.6 run needs internet access, or place the official "
                    f"hand_landmarker.task at: {requested_path}"
                ) from error
        else:
            self.model_path = requested_path
            if not self.model_path.is_file():
                raise FileNotFoundError(self.model_path)

        presence_confidence = (
            float(min_detection_confidence)
            if min_presence_confidence is None
            else float(min_presence_confidence)
        )

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=float(min_tracking_confidence),
            result_callback=self._on_task_result,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        logger.info(
            "MediaPipe Tasks HandLandmarker ready: mode=LIVE_STREAM hands=%s model=%s mediapipe=%s",
            self.max_num_hands,
            self.model_path,
            getattr(mp, "__version__", "unknown"),
        )

    # ========================================================
    # MediaPipe Tasks extraction / callback
    # ========================================================

    @staticmethod
    def _category_name(category) -> str:
        return str(
            getattr(category, "category_name", None)
            or getattr(category, "display_name", None)
            or "Unknown"
        )

    def _extract_task_hands(self, result) -> list[TrackedHand]:
        landmark_sets = getattr(result, "hand_landmarks", None) or []
        if not landmark_sets:
            return []

        world_sets = getattr(result, "hand_world_landmarks", None) or []
        handedness_sets = getattr(result, "handedness", None) or []
        tracked_hands: list[TrackedHand] = []

        for index, landmarks in enumerate(landmark_sets):
            image_landmarks = np.asarray(
                [[lm.x, lm.y, lm.z] for lm in landmarks],
                dtype=np.float32,
            )

            world_landmarks = None
            if index < len(world_sets) and world_sets[index]:
                world_landmarks = np.asarray(
                    [[lm.x, lm.y, lm.z] for lm in world_sets[index]],
                    dtype=np.float32,
                )

            handedness = "Unknown"
            handedness_score = 0.0
            if index < len(handedness_sets) and handedness_sets[index]:
                category = handedness_sets[index][0]
                handedness = self._category_name(category)
                handedness_score = float(getattr(category, "score", 0.0) or 0.0)

            tracked_hands.append(
                TrackedHand(
                    handedness=handedness,
                    handedness_score=handedness_score,
                    image_landmarks=image_landmarks,
                    world_landmarks=world_landmarks,
                    raw_landmarks=landmarks,
                )
            )

        return tracked_hands

    def _on_task_result(self, result, _output_image, timestamp_ms: int) -> None:
        if self._closed:
            return

        try:
            raw_hands = self._extract_task_hands(result)
            stable_hands = self._stabilize(raw_hands)
            stable_hands = self._apply_reacquisition_gate(raw_hands, stable_hands)
            now_ms = time.monotonic_ns() / 1_000_000.0
            latency_ms = max(0.0, now_ms - float(timestamp_ms))
            self._results.publish(
                list(stable_hands),
                timestamp_ms=timestamp_ms,
                latency_ms=latency_ms,
            )
        except Exception:
            # Do not crash MediaPipe's callback dispatcher because of a malformed
            # result. The main UI can continue consuming the last valid result.
            logger.exception("Failed to process asynchronous HandLandmarker result")

    # ========================================================
    # Temporal stabilization (preserved from V2/V3.1-V3.5)
    # ========================================================

    @staticmethod
    def _wrist_distance(hand_a: TrackedHand, hand_b: TrackedHand) -> float:
        return float(
            np.linalg.norm(
                hand_a.image_landmarks[0, :2]
                - hand_b.image_landmarks[0, :2]
            )
        )

    def _smooth_hand(
        self,
        current: TrackedHand,
        previous: TrackedHand | None,
    ) -> TrackedHand:
        if previous is None:
            return current

        alpha = self.smoothing_alpha
        image_landmarks = (
            alpha * current.image_landmarks
            + (1.0 - alpha) * previous.image_landmarks
        ).astype(np.float32)

        if current.world_landmarks is not None and previous.world_landmarks is not None:
            world_landmarks = (
                alpha * current.world_landmarks
                + (1.0 - alpha) * previous.world_landmarks
            ).astype(np.float32)
        else:
            world_landmarks = current.world_landmarks

        handedness, handedness_score = choose_handedness(
            previous_label=previous.handedness,
            previous_score=previous.handedness_score,
            current_label=current.handedness,
            current_score=current.handedness_score,
            wrist_continuous=self._wrist_distance(current, previous) < 0.30,
            reacquiring=self._reacquisition_guard.awaiting_reacquisition,
        )

        return TrackedHand(
            handedness=handedness,
            handedness_score=handedness_score,
            image_landmarks=image_landmarks,
            world_landmarks=world_landmarks,
            raw_landmarks=current.raw_landmarks,
        )


    def _apply_reacquisition_gate(
        self,
        raw_hands: list[TrackedHand],
        stable_hands: list[TrackedHand],
    ) -> list[TrackedHand]:
        """Keep transient reacquisition results away from the recognizer."""
        was_awaiting = self._reacquisition_guard.awaiting_reacquisition
        decision = self._reacquisition_guard.update(
            raw_count=len(raw_hands),
            stable_count=len(stable_hands),
        )

        if decision.clear_tracker_state:
            self._stable_hands = []
            self._stable_mode = 0
            return []

        if (
            not was_awaiting
            and self._reacquisition_guard.awaiting_reacquisition
        ):
            logger.debug("Hand absence confirmed; waiting for clean reacquisition")

        if (
            was_awaiting
            and not self._reacquisition_guard.awaiting_reacquisition
            and decision.expose_result
        ):
            logger.debug(
                "Hand reacquired after %s stable callbacks",
                self.reacquire_confirm_frames,
            )

        return stable_hands if decision.expose_result else []

    def _select_primary_hand(self, current_hands: list[TrackedHand]) -> TrackedHand:
        if not current_hands:
            raise ValueError("No hands available.")

        if self._stable_hands:
            previous = self._stable_hands[0]
            return min(
                current_hands,
                key=lambda hand: self._wrist_distance(hand, previous),
            )

        return max(
            current_hands,
            key=lambda hand: hand.handedness_score,
        )

    def _match_two_hands(
        self,
        current_hands: list[TrackedHand],
    ) -> list[TrackedHand]:
        current = current_hands[:2]

        if len(self._stable_hands) != 2:
            return sorted(
                current,
                key=lambda hand: float(hand.image_landmarks[0, 0]),
            )

        previous_a, previous_b = self._stable_hands
        current_a, current_b = current

        direct_cost = (
            self._wrist_distance(current_a, previous_a)
            + self._wrist_distance(current_b, previous_b)
        )
        swapped_cost = (
            self._wrist_distance(current_b, previous_a)
            + self._wrist_distance(current_a, previous_b)
        )

        if swapped_cost < direct_cost:
            return [current_b, current_a]

        return [current_a, current_b]

    def _stabilize(self, raw_hands: list[TrackedHand]) -> list[TrackedHand]:
        hand_count = len(raw_hands)

        if hand_count == 0:
            self._missing_streak += 1
            self._one_hand_streak = 0
            self._two_hand_streak = 0

            if self._stable_hands and self._missing_streak <= self.missing_hold_frames:
                return self._stable_hands

            self._stable_hands = []
            self._stable_mode = 0
            return []

        self._missing_streak = 0

        if hand_count == 1:
            self._one_hand_streak += 1
            self._two_hand_streak = 0

            current = raw_hands[0]

            if (
                self._stable_mode == 2
                and len(self._stable_hands) == 2
                and self._one_hand_streak < self.one_hand_confirm_frames
            ):
                distances = [
                    self._wrist_distance(current, previous)
                    for previous in self._stable_hands
                ]
                match_index = int(np.argmin(distances))
                updated = list(self._stable_hands)
                updated[match_index] = self._smooth_hand(
                    current,
                    self._stable_hands[match_index],
                )
                self._stable_hands = updated
                return self._stable_hands

            previous = None
            if self._stable_hands:
                previous = min(
                    self._stable_hands,
                    key=lambda hand: self._wrist_distance(current, hand),
                )

            stable = self._smooth_hand(current, previous)
            self._stable_hands = [stable]
            self._stable_mode = 1
            return self._stable_hands

        self._two_hand_streak += 1
        self._one_hand_streak = 0

        if (
            self._stable_mode == 1
            and self._two_hand_streak < self.two_hand_confirm_frames
        ):
            primary = self._select_primary_hand(raw_hands)
            stable = self._smooth_hand(
                primary,
                self._stable_hands[0] if self._stable_hands else None,
            )
            self._stable_hands = [stable]
            return self._stable_hands

        matched = self._match_two_hands(raw_hands)

        previous_hands = self._stable_hands if len(self._stable_hands) == 2 else []
        stable_pair = []
        for index, current in enumerate(matched):
            previous = previous_hands[index] if previous_hands else None
            stable_pair.append(self._smooth_hand(current, previous))

        self._stable_hands = stable_pair
        self._stable_mode = 2
        return self._stable_hands

    # ========================================================
    # Public API
    # ========================================================

    @property
    def backend_name(self) -> str:
        return "MediaPipe Tasks HandLandmarker LIVE_STREAM"

    def process(self, frame_bgr: np.ndarray) -> list[TrackedHand]:
        """Submit a frame asynchronously and return the latest completed result."""
        if self._closed:
            return []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = self._timestamps.next_ms()

        try:
            self.landmarker.detect_async(mp_image, timestamp_ms)
            self._results.note_submission()
        except Exception:
            logger.exception("MediaPipe HandLandmarker detect_async submission failed")

        diagnostics = self._results.diagnostics()
        if (
            diagnostics.latest_result_age_ms is not None
            and diagnostics.latest_result_age_ms > self.stale_result_ms
        ):
            return []

        return list(self._results.snapshot())

    def diagnostics(self):
        return self._results.diagnostics()

    def draw(self, frame_bgr: np.ndarray, hands: list[TrackedHand]) -> None:
        """Draw stabilized landmarks without using deprecated mp.solutions APIs."""
        height, width = frame_bgr.shape[:2]

        for hand in hands:
            points: list[tuple[int, int]] = []
            for x, y, _z in hand.image_landmarks:
                px = int(np.clip(x, 0.0, 1.0) * max(1, width - 1))
                py = int(np.clip(y, 0.0, 1.0) * max(1, height - 1))
                points.append((px, py))

            for start, end in _HAND_CONNECTIONS:
                if start < len(points) and end < len(points):
                    cv2.line(
                        frame_bgr,
                        points[start],
                        points[end],
                        (80, 200, 120),
                        2,
                        cv2.LINE_AA,
                    )

            for index, point in enumerate(points):
                radius = 4 if index in (4, 8, 12, 16, 20) else 3
                cv2.circle(
                    frame_bgr,
                    point,
                    radius,
                    (245, 245, 245),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    frame_bgr,
                    point,
                    radius,
                    (40, 120, 80),
                    1,
                    cv2.LINE_AA,
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.landmarker.close()
        finally:
            logger.info("MediaPipe Tasks HandLandmarker closed")
