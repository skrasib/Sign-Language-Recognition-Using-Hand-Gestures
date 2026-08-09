from pathlib import Path
import sys

import cv2
import numpy as np


# Allow scripts/ to import our src/ package before we set up
# the final pyproject package configuration.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from adaptive_gesture.features.normalizer import normalize_hand_landmarks
from adaptive_gesture.tracking.hand_tracker import HandTracker


def main():
    tracker = HandTracker(max_num_hands=2)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Your webcam defaults to 2560 × 1440.
    # We don't need that resolution for landmark tracking.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_counter = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Selfie-style view.
            frame = cv2.flip(frame, 1)

            hands = tracker.process(frame)

            tracker.draw(frame, hands)

            y = 35

            cv2.putText(
                frame,
                f"Hands detected: {len(hands)}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            y += 35

            for index, hand in enumerate(hands):
                source_landmarks = (
                    hand.world_landmarks
                    if hand.world_landmarks is not None
                    else hand.image_landmarks
                )

                features = normalize_hand_landmarks(source_landmarks)

                text = (
                    f"Hand {index + 1}: "
                    f"{hand.handedness} "
                    f"({hand.handedness_score:.2f}) "
                    f"| features={len(features)}"
                )

                cv2.putText(
                    frame,
                    text,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                y += 30

                # Print a feature snapshot about once per second.
                if frame_counter % 30 == 0:
                    print()
                    print(
                        f"{hand.handedness} hand:"
                    )
                    print(
                        "Shape:",
                        features.shape,
                    )
                    print(
                        "First 10 normalized features:",
                        np.round(features[:10], 3),
                    )

            cv2.imshow(
                "V2 - 3D Hand Feature Test",
                frame,
            )

            frame_counter += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()