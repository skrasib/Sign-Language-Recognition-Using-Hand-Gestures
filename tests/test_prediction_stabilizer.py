from dataclasses import dataclass

from adaptive_gesture.learning.prediction_stabilizer import PredictionStabilizer


@dataclass
class DummyPrediction:
    label: str


def test_known_prediction_requires_confirmation_frames():
    stabilizer = PredictionStabilizer(
        confirm_frames=3,
        unknown_confirm_frames=2,
        unknown_label="UNKNOWN",
    )

    first = stabilizer.update(DummyPrediction("Victory"))
    second = stabilizer.update(DummyPrediction("Victory"))
    third = stabilizer.update(DummyPrediction("Victory"))

    assert first.prediction is None
    assert first.pending
    assert second.prediction is None
    assert third.prediction is not None
    assert third.prediction.label == "Victory"


def test_unknown_releases_confirmed_label_faster():
    stabilizer = PredictionStabilizer(confirm_frames=3, unknown_confirm_frames=2)
    for _ in range(3):
        stabilizer.update(DummyPrediction("Victory"))

    first_unknown = stabilizer.update(DummyPrediction("UNKNOWN"))
    second_unknown = stabilizer.update(DummyPrediction("UNKNOWN"))

    assert first_unknown.prediction.label == "Victory"
    assert second_unknown.prediction.label == "UNKNOWN"
