from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class TrackedHand:
    """
    Result produced for one detected hand.
    """

    handedness: str
    handedness_score: float

    # MediaPipe normalized image coordinates.
    # Shape: (21, 3)
    image_landmarks: np.ndarray

    # MediaPipe 3D world coordinates.
    # Shape: (21, 3)
    world_landmarks: np.ndarray | None

    # Original MediaPipe landmark object.
    raw_landmarks: object


class HandTracker:
    """
    Real-time MediaPipe hand tracker.

    Unlike the original FYP, this tracker:
    - supports up to two hands;
    - preserves X, Y and Z;
    - exposes MediaPipe world landmarks;
    - separates tracking from classification.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
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

    def process(self, frame_bgr: np.ndarray) -> list[TrackedHand]:
        """
        Detect hands in a BGR OpenCV frame.
        """

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe documentation recommends marking the image
        # non-writeable while inference is performed.
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return []

        tracked_hands: list[TrackedHand] = []

        for index, landmarks in enumerate(results.multi_hand_landmarks):
            image_landmarks = np.array(
                [
                    [landmark.x, landmark.y, landmark.z]
                    for landmark in landmarks.landmark
                ],
                dtype=np.float32,
            )

            world_landmarks = None

            if (
                results.multi_hand_world_landmarks
                and index < len(results.multi_hand_world_landmarks)
            ):
                world = results.multi_hand_world_landmarks[index]

                world_landmarks = np.array(
                    [
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in world.landmark
                    ],
                    dtype=np.float32,
                )

            handedness = "Unknown"
            handedness_score = 0.0

            if results.multi_handedness and index < len(results.multi_handedness):
                classification = results.multi_handedness[index].classification[0]

                handedness = classification.label
                handedness_score = classification.score

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

    def draw(self, frame_bgr: np.ndarray, hands: list[TrackedHand]) -> None:
        """
        Draw MediaPipe landmarks on the supplied frame.
        """

        for hand in hands:
            self.mp_drawing.draw_landmarks(
                frame_bgr,
                hand.raw_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

    def close(self) -> None:
        self.hands.close()