import json
from pathlib import Path

import numpy as np

from adaptive_gesture.features.dynamic_features import DynamicTrajectory


class DynamicGestureStore:
    """Atomic JSON persistence for landmark trajectories only; no video data."""

    FORMAT_VERSION = 1

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def _trajectory_to_dict(trajectory: DynamicTrajectory) -> dict:
        return {
            "hand_signature": trajectory.hand_signature,
            "shape_sequence": np.asarray(
                trajectory.shape_sequence, dtype=np.float32
            ).tolist(),
            "motion_sequence": np.asarray(
                trajectory.motion_sequence, dtype=np.float32
            ).tolist(),
            "velocity_sequence": np.asarray(
                trajectory.velocity_sequence, dtype=np.float32
            ).tolist(),
            "duration_seconds": float(trajectory.duration_seconds),
            "raw_frame_count": int(trajectory.raw_frame_count),
            "motion_extent": float(trajectory.motion_extent),
            "shape_extent": float(trajectory.shape_extent),
        }

    @staticmethod
    def _trajectory_from_dict(data: dict) -> DynamicTrajectory:
        shape = np.asarray(data["shape_sequence"], dtype=np.float32)
        motion = np.asarray(data["motion_sequence"], dtype=np.float32)
        velocity = np.asarray(data["velocity_sequence"], dtype=np.float32)

        if shape.ndim != 2 or motion.ndim != 2 or velocity.ndim != 2:
            raise ValueError("Stored dynamic trajectory arrays must be 2-D.")
        if not (shape.shape[0] == motion.shape[0] == velocity.shape[0]):
            raise ValueError("Stored dynamic trajectory lengths do not match.")

        return DynamicTrajectory(
            hand_signature=str(data.get("hand_signature", "Unknown")),
            shape_sequence=shape,
            motion_sequence=motion,
            velocity_sequence=velocity,
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            raw_frame_count=int(data.get("raw_frame_count", shape.shape[0])),
            motion_extent=float(data.get("motion_extent", 0.0)),
            shape_extent=float(data.get("shape_extent", 0.0)),
        )

    def save(self, learner) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": self.FORMAT_VERSION,
            "storage": "landmark_trajectories_only",
            "gestures": [
                {
                    "name": name,
                    "hand_signature": gesture.hand_signature,
                    "templates": [
                        self._trajectory_to_dict(template)
                        for template in gesture.templates
                    ],
                }
                for name, gesture in learner.gestures.items()
            ],
        }

        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        temporary.replace(self.path)

    def load_into(self, learner) -> int:
        if not self.path.exists():
            return 0

        with self.path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        version = payload.get("format_version")
        if version != self.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported dynamic gesture-memory format version: {version}"
            )

        restored = 0
        for stored in payload.get("gestures", []):
            name = str(stored.get("name", "")).strip()
            if not name:
                continue

            templates = [
                self._trajectory_from_dict(template)
                for template in stored.get("templates", [])
            ]
            if len(templates) < learner.minimum_templates:
                continue

            learner.learn_gesture(name, templates)
            restored += 1

        return restored

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
