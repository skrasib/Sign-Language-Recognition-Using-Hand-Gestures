from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from adaptive_gesture.tracking.model_assets import ensure_hand_landmarker_model


def main():
    destination = PROJECT_ROOT / "data" / "v3" / "models" / "hand_landmarker.task"
    model = ensure_hand_landmarker_model(destination)
    print(f"MediaPipe Hand Landmarker model ready: {model}")
    print(f"Size: {model.stat().st_size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
