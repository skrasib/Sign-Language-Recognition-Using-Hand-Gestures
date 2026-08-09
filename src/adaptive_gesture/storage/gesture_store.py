import json
from pathlib import Path

import numpy as np


class GestureStore:
    """
    Persistent landmark-only gesture memory.

    Format v2 supports one-hand and two-hand classes. Format v1 from the
    previous hotfix is migrated automatically on load.
    """

    FORMAT_VERSION = 2

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, learner) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        gestures = []
        for name, gesture in learner.gestures.items():
            gestures.append(
                {
                    "name": name,
                    "hand_signature": gesture.hand_signature,
                    "feature_dimension": gesture.feature_dimension,
                    "positive_samples": [
                        np.asarray(sample, dtype=np.float32).tolist()
                        for sample in gesture.samples
                    ],
                    "hard_negatives": [
                        np.asarray(sample, dtype=np.float32).tolist()
                        for sample in gesture.hard_negatives
                    ],
                }
            )

        payload = {
            "format_version": self.FORMAT_VERSION,
            "gestures": gestures,
        }

        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        temporary_path.replace(self.path)

    def load_into(self, learner) -> int:
        if not self.path.exists():
            return 0

        with self.path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        version = payload.get("format_version", 1)
        if version not in (1, 2):
            raise ValueError(f"Unsupported gesture-memory format version: {version}")

        restored = 0
        for stored in payload.get("gestures", []):
            name = str(stored.get("name", "")).strip()
            if not name:
                continue

            positive_samples = [
                np.asarray(sample, dtype=np.float32)
                for sample in stored.get("positive_samples", [])
            ]
            if not positive_samples:
                continue

            if version == 1:
                # Previous memory format stored one-hand handedness globally.
                signature = stored.get("handedness") or "Unknown"
            else:
                signature = stored.get("hand_signature") or "Unknown"

            learner.learn_gesture(
                name=name,
                samples=positive_samples,
                hand_signature=signature,
            )

            gesture = learner.gestures[name]
            for negative in stored.get("hard_negatives", []):
                negative_array = np.asarray(negative, dtype=np.float32)
                if negative_array.shape == (gesture.feature_dimension,):
                    learner.add_hard_negative(name, negative_array)

            restored += 1

        return restored

    def exists(self) -> bool:
        return self.path.exists()

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
