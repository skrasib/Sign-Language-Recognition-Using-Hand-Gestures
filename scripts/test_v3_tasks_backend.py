from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import mediapipe as mp
from mediapipe.tasks.python import vision

from adaptive_gesture.tracking.hand_tracker import HandTracker


def main():
    model_path = PROJECT_ROOT / "data" / "v3" / "models" / "hand_landmarker.task"

    assert hasattr(vision, "HandLandmarker")
    assert hasattr(vision, "HandLandmarkerOptions")
    assert hasattr(vision, "RunningMode")

    tracker = HandTracker(
        max_num_hands=2,
        model_path=model_path,
        auto_download_model=True,
    )
    try:
        print(f"MediaPipe: {getattr(mp, '__version__', 'unknown')}")
        print(f"Backend: {tracker.backend_name}")
        print(f"Model: {tracker.model_path}")
        print("V3.6 MediaPipe Tasks backend smoke test: PASS")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
