from pathlib import Path
import sys
import time

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from adaptive_gesture.features.normalizer import (
    normalize_hand_landmarks,
)

from adaptive_gesture.learning.online_learner import (
    OnlineGestureLearner,
)

from adaptive_gesture.tracking.hand_tracker import (
    HandTracker,
)


TEACH_SAMPLE_COUNT = 60


def extract_features(hand):
    landmarks = (
        hand.world_landmarks
        if hand.world_landmarks is not None
        else hand.image_landmarks
    )

    return normalize_hand_landmarks(
        landmarks
    )


def main():
    tracker = HandTracker(
        max_num_hands=1
    )

    learner = OnlineGestureLearner(
        rejection_multiplier=3.0,
        minimum_threshold=0.05,
    )

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    teaching = False
    teaching_name = None
    teaching_samples = []
    teaching_handedness = None

    last_prediction = None

    print()
    print("ONLINE FEW-SHOT LEARNING")
    print("------------------------")
    print("T = teach new gesture")
    print("F = confirm current pose as correct prediction")
    print("L = list learned gestures")
    print("Q = quit")
    print()

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(
                frame,
                1,
            )

            hands = tracker.process(frame)

            tracker.draw(
                frame,
                hands,
            )

            current_features = None
            current_hand = None

            if hands:

                current_hand = hands[0]

                current_features = (
                    extract_features(
                        current_hand
                    )
                )

            # ---------------------------------------------
            # Teaching mode
            # ---------------------------------------------

            if teaching:

                cv2.putText(
                    frame,
                    f"TEACHING: {teaching_name}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )

                if current_features is not None:

                    if teaching_handedness is None:
                        teaching_handedness = (
                            current_hand.handedness
                        )

                    # Only accept samples from the
                    # same hand used at teaching start.
                    if (
                        current_hand.handedness
                        == teaching_handedness
                    ):
                        teaching_samples.append(
                            current_features.copy()
                        )

                progress = min(
                    len(teaching_samples)
                    / TEACH_SAMPLE_COUNT,
                    1.0,
                )

                cv2.putText(
                    frame,
                    (
                        f"Samples: "
                        f"{len(teaching_samples)}"
                        f"/{TEACH_SAMPLE_COUNT}"
                    ),
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                bar_width = 450

                cv2.rectangle(
                    frame,
                    (20, 100),
                    (20 + bar_width, 125),
                    (255, 255, 255),
                    1,
                )

                cv2.rectangle(
                    frame,
                    (20, 100),
                    (
                        20
                        + int(
                            bar_width
                            * progress
                        ),
                        125,
                    ),
                    (255, 255, 255),
                    -1,
                )

                if (
                    len(teaching_samples)
                    >= TEACH_SAMPLE_COUNT
                ):

                    gesture = (
                        learner.learn_gesture(
                            name=teaching_name,
                            samples=teaching_samples,
                            handedness=(
                                teaching_handedness
                            ),
                        )
                    )

                    print()
                    print(
                        f"Learned: {gesture.name}"
                    )

                    print(
                        "Samples:",
                        gesture.sample_count,
                    )

                    print(
                        "Spread:",
                        round(
                            gesture.spread,
                            5,
                        ),
                    )

                    print(
                        "Known gestures:",
                        learner.list_gestures(),
                    )

                    print()

                    teaching = False
                    teaching_name = None
                    teaching_samples = []
                    teaching_handedness = None

            # ---------------------------------------------
            # Recognition mode
            # ---------------------------------------------

            elif current_features is not None:

                prediction = learner.predict(
                    current_features,
                    handedness=(
                        current_hand.handedness
                    ),
                )

                last_prediction = (
                    prediction
                )

                cv2.putText(
                    frame,
                    (
                        f"Prediction: "
                        f"{prediction.label}"
                    ),
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )

                if (
                    prediction.distance
                    is not None
                ):

                    cv2.putText(
                        frame,
                        (
                            f"Distance: "
                            f"{prediction.distance:.4f}"
                        ),
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Threshold: "
                            f"{prediction.threshold:.4f}"
                        ),
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            else:

                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            known = (
                ", ".join(
                    learner.list_gestures()
                )
                or "None"
            )

            cv2.putText(
                frame,
                f"Known: {known}",
                (
                    20,
                    frame.shape[0] - 55,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "T: Teach | L: List | Q: Quit",
                (
                    20,
                    frame.shape[0] - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "V2 - Online Few-Shot Learning",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            if (
                key == ord("t")
                and not teaching
            ):

                print()
                print(
                    "Enter gesture name in terminal:"
                )

                name = input(
                    "> "
                ).strip()

                if not name:
                    print(
                        "Gesture name cannot be empty."
                    )
                    continue

                if (
                    name
                    in learner.gestures
                ):

                    print(
                        "That gesture already exists."
                    )

                    continue

                print()
                print(
                    f"Get ready to teach '{name}'."
                )

                print(
                    "Hold the gesture naturally."
                )

                time.sleep(1)

                teaching = True
                teaching_name = name
                teaching_samples = []
                teaching_handedness = None

            if key == ord("l"):

                print()
                print(
                    "Known gestures:"
                )

                for name in (
                    learner.list_gestures()
                ):
                    gesture = (
                        learner.gestures[name]
                    )

                    print(
                        f"- {name}: "
                        f"{gesture.sample_count} samples, "
                        f"spread={gesture.spread:.5f}"
                    )

    finally:

        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()