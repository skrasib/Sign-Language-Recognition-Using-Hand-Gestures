from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


@dataclass
class TrackedHand:
    handedness: str
    handedness_score: float
    image_landmarks: np.ndarray
    world_landmarks: np.ndarray | None
    raw_landmarks: object


class HandTracker:
    """
    Real-time MediaPipe tracker supporting up to two hands.

    The tracker adds a small temporal-stability layer on top of MediaPipe.
    This prevents one-hand mode from rapidly switching between 0/1/2 hands
    when MediaPipe briefly misses a hand or produces a transient second
    detection. Landmark coordinates are also lightly smoothed.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        missing_hold_frames: int = 2,
        two_hand_confirm_frames: int = 3,
        one_hand_confirm_frames: int = 3,
        smoothing_alpha: float = 0.65,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.max_num_hands = max_num_hands
        self.missing_hold_frames = max(0, int(missing_hold_frames))
        self.two_hand_confirm_frames = max(1, int(two_hand_confirm_frames))
        self.one_hand_confirm_frames = max(1, int(one_hand_confirm_frames))
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))

        self._stable_hands: list[TrackedHand] = []
        self._stable_mode = 0
        self._missing_streak = 0
        self._one_hand_streak = 0
        self._two_hand_streak = 0

    # ========================================================
    # Raw MediaPipe extraction
    # ========================================================

    def _extract_raw_hands(self, results) -> list[TrackedHand]:
        if not results.multi_hand_landmarks:
            return []

        tracked_hands: list[TrackedHand] = []

        for index, landmarks in enumerate(results.multi_hand_landmarks):
            image_landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                dtype=np.float32,
            )

            world_landmarks = None
            if (
                results.multi_hand_world_landmarks
                and index < len(results.multi_hand_world_landmarks)
            ):
                world = results.multi_hand_world_landmarks[index]
                world_landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in world.landmark],
                    dtype=np.float32,
                )

            handedness = "Unknown"
            handedness_score = 0.0
            if results.multi_handedness and index < len(results.multi_handedness):
                classification = results.multi_handedness[index].classification[0]
                handedness = classification.label
                handedness_score = float(classification.score)

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

    # ========================================================
    # Temporal stabilization
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

        # A physical tracked hand cannot change handedness from one frame to
        # the next. Preserve the previous stable identity when the wrist is
        # spatially continuous, avoiding Left/Right signature flicker.
        if self._wrist_distance(current, previous) < 0.30:
            handedness = previous.handedness
            handedness_score = max(
                previous.handedness_score,
                current.handedness_score,
            )
        else:
            handedness = current.handedness
            handedness_score = current.handedness_score

        return TrackedHand(
            handedness=handedness,
            handedness_score=handedness_score,
            image_landmarks=image_landmarks,
            world_landmarks=world_landmarks,
            raw_landmarks=current.raw_landmarks,
        )

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

        # ----------------------------------------------------
        # Brief total detection loss: hold the previous result
        # for only a couple of frames instead of flashing off.
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # One raw hand
        # ----------------------------------------------------
        if hand_count == 1:
            self._one_hand_streak += 1
            self._two_hand_streak = 0

            current = raw_hands[0]

            # If we were in stable two-hand mode, do not instantly collapse
            # because one hand was missed for a single frame.
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

        # ----------------------------------------------------
        # Two raw hands
        # ----------------------------------------------------
        self._two_hand_streak += 1
        self._one_hand_streak = 0

        # A transient second detection while one real hand is visible should
        # not switch the whole application into 129-D two-hand mode.
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

    def process(self, frame_bgr: np.ndarray) -> list[TrackedHand]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        raw_hands = self._extract_raw_hands(results)
        return self._stabilize(raw_hands)

    def draw(self, frame_bgr: np.ndarray, hands: list[TrackedHand]) -> None:
        """Draw the temporally stabilized landmarks, not the raw detections."""
        for hand in hands:
            landmark_list = landmark_pb2.NormalizedLandmarkList()

            for x, y, z in hand.image_landmarks:
                landmark = landmark_list.landmark.add()
                landmark.x = float(x)
                landmark.y = float(y)
                landmark.z = float(z)

            self.mp_drawing.draw_landmarks(
                frame_bgr,
                landmark_list,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

    def close(self) -> None:
        self.hands.close()
