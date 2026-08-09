from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from adaptive_gesture.features.normalizer import (
    normalize_hand_landmarks,
)

from adaptive_gesture.features.similarity import (
    calculate_reference_spread,
    create_prototype,
    feature_distance,
)

from adaptive_gesture.tracking.hand_tracker import (
    HandTracker,
)


REFERENCE_SAMPLE_COUNT = 45
MIN_REFERENCE_SPREAD = 1e-4


def main():
    tracker = HandTracker(max_num_hands=1)

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

    reference_samples = []
    prototype = None
    reference_spread = None

    capturing_reference = False

    print()
    print("FEATURE SIMILARITY TEST")
    print("-----------------------")
    print("R = record reference gesture")
    print("C = clear reference")
    print("Q = quit")
    print()
    print(
        "For this experiment, use ONE hand "
        "and keep using the same hand."
    )

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print(
                    "Could not read webcam frame."
                )
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

            if hands:
                hand = hands[0]

                source_landmarks = (
                    hand.world_landmarks
                    if hand.world_landmarks is not None
                    else hand.image_landmarks
                )

                current_features = (
                    normalize_hand_landmarks(
                        source_landmarks
                    )
                )

                cv2.putText(
                    frame,
                    (
                        f"Tracking: {hand.handedness} "
                        f"({hand.handedness_score:.2f})"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

            # -------------------------------------------------
            # Capture reference samples
            # -------------------------------------------------

            if (
                capturing_reference
                and current_features is not None
            ):
                reference_samples.append(
                    current_features.copy()
                )

                progress = (
                    len(reference_samples)
                    / REFERENCE_SAMPLE_COUNT
                )

                cv2.putText(
                    frame,
                    (
                        "Recording reference: "
                        f"{len(reference_samples)}"
                        f"/{REFERENCE_SAMPLE_COUNT}"
                    ),
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                bar_width = 400

                cv2.rectangle(
                    frame,
                    (20, 95),
                    (20 + bar_width, 115),
                    (255, 255, 255),
                    1,
                )

                cv2.rectangle(
                    frame,
                    (20, 95),
                    (
                        20 + int(
                            bar_width * progress
                        ),
                        115,
                    ),
                    (255, 255, 255),
                    -1,
                )

                if (
                    len(reference_samples)
                    >= REFERENCE_SAMPLE_COUNT
                ):
                    prototype = create_prototype(
                        reference_samples
                    )

                    reference_spread = (
                        calculate_reference_spread(
                            reference_samples,
                            prototype,
                        )
                    )

                    reference_spread = max(
                        reference_spread,
                        MIN_REFERENCE_SPREAD,
                    )

                    capturing_reference = False

                    print()
                    print("Reference captured.")
                    print(
                        "Samples:",
                        len(reference_samples),
                    )
                    print(
                        "Reference spread (P95):",
                        round(reference_spread, 5),
                    )
                    print()

            # -------------------------------------------------
            # Compare current hand with reference
            # -------------------------------------------------

            if (
                prototype is not None
                and current_features is not None
            ):
                distance = feature_distance(
                    current_features,
                    prototype,
                )

                relative_distance = (
                    distance
                    / reference_spread
                )

                cv2.putText(
                    frame,
                    (
                        f"RMSE distance: "
                        f"{distance:.4f}"
                    ),
                    (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Reference spread: "
                        f"{reference_spread:.4f}"
                    ),
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Relative distance: "
                        f"{relative_distance:.2f}x"
                    ),
                    (20, 225),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        "Lower relative distance "
                        "= more similar"
                    ),
                    (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            elif not capturing_reference:
                cv2.putText(
                    frame,
                    "Press R to record a reference gesture",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                "R: Record | C: Clear | Q: Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "V2 - Feature Similarity Experiment",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            if key == ord("r"):
                if not hands:
                    print(
                        "Cannot record reference: "
                        "no hand detected."
                    )
                else:
                    reference_samples = []
                    prototype = None
                    reference_spread = None

                    capturing_reference = True

                    print()
                    print(
                        "Recording new reference..."
                    )

            if key == ord("c"):
                reference_samples = []
                prototype = None
                reference_spread = None
                capturing_reference = False

                print()
                print(
                    "Reference cleared."
                )

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()