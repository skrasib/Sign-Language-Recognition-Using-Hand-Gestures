import numpy as np

from adaptive_gesture.learning.online_learner import OnlineGestureLearner


def cluster(center: float, count: int = 12, dimension: int = 63):
    return [
        np.full(dimension, center + (index - count / 2) * 0.0005, dtype=np.float32)
        for index in range(count)
    ]


def test_runtime_learning_and_open_set_prediction():
    learner = OnlineGestureLearner(
        radius_multiplier=2.5,
        minimum_threshold=0.035,
        max_prototypes=3,
    )
    learner.learn_gesture("Victory", cluster(0.0), hand_signature="Right")
    learner.learn_gesture("Three", cluster(0.30), hand_signature="Right")

    known = learner.predict(np.zeros(63, dtype=np.float32), hand_signature="Right")
    unknown = learner.predict(np.full(63, 1.2, dtype=np.float32), hand_signature="Right")

    assert known.accepted
    assert known.label == "Victory"
    assert known.confidence is not None

    assert not unknown.accepted
    assert unknown.label == learner.UNKNOWN_LABEL


def test_feedback_and_gesture_management():
    learner = OnlineGestureLearner()
    learner.learn_gesture("One", cluster(0.0), hand_signature="Right")
    learner.learn_gesture("Two", cluster(0.25), hand_signature="Right")

    correction = np.full(63, 0.25, dtype=np.float32)
    learner.apply_correction(
        predicted_label="One",
        actual_label="Two",
        sample=correction,
    )

    assert learner.gestures["One"].negative_count == 1
    assert learner.gestures["Two"].sample_count >= 12

    learner.rename_gesture("Two", "Peace")
    assert "Peace" in learner.gestures
    assert "Two" not in learner.gestures

    learner.delete_gesture("Peace")
    assert "Peace" not in learner.gestures
