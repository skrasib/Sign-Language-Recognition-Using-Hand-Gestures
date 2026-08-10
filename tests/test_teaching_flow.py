import pytest

from adaptive_gesture.learning.teaching_flow import (
    HandReadinessGate,
    ONE_HAND_MODE,
    TWO_HAND_MODE,
    hand_mode_from_signature,
    required_hand_count,
)


def test_hand_mode_helpers():
    assert required_hand_count(ONE_HAND_MODE) == 1
    assert required_hand_count(TWO_HAND_MODE) == 2
    assert hand_mode_from_signature("Left") == ONE_HAND_MODE
    assert hand_mode_from_signature("Right") == ONE_HAND_MODE
    assert hand_mode_from_signature("Both") == TWO_HAND_MODE


def test_two_hand_gate_requires_fresh_stable_results():
    gate = HandReadinessGate(2, expected_signature="Both", confirm_results=3, hold_seconds=0.2)

    first = gate.update(
        hand_count=2,
        hand_signature="Both",
        result_token=10,
        now=1.0,
    )
    assert not first.ready
    assert first.matched_results == 1

    # Re-rendering one async result must not advance readiness.
    duplicate = gate.update(
        hand_count=2,
        hand_signature="Both",
        result_token=10,
        now=1.3,
    )
    assert not duplicate.ready
    assert duplicate.matched_results == 1

    gate.update(hand_count=2, hand_signature="Both", result_token=11, now=1.1)
    ready = gate.update(hand_count=2, hand_signature="Both", result_token=12, now=1.25)
    assert ready.ready
    assert ready.hand_signature == "Both"


def test_one_hand_gate_locks_consistent_handedness_candidate():
    gate = HandReadinessGate(1, confirm_results=2, hold_seconds=0.05)
    gate.update(hand_count=1, hand_signature="Right", result_token=1, now=0.0)

    # A handedness flip restarts confirmation instead of mixing signatures.
    flipped = gate.update(hand_count=1, hand_signature="Left", result_token=2, now=0.1)
    assert not flipped.ready
    assert flipped.hand_signature == "Left"

    ready = gate.update(hand_count=1, hand_signature="Left", result_token=3, now=0.2)
    assert ready.ready
    assert ready.hand_signature == "Left"


def test_wrong_hand_count_resets_gate():
    gate = HandReadinessGate(2, expected_signature="Both", confirm_results=2, hold_seconds=0.0)
    gate.update(hand_count=2, hand_signature="Both", result_token=1, now=0.0)
    reset = gate.update(hand_count=1, hand_signature="Right", result_token=2, now=0.1)
    assert not reset.ready
    assert reset.matched_results == 0


def test_dynamic_stillness_gate_rejects_motion_before_arming():
    gate = HandReadinessGate(
        2,
        expected_signature="Both",
        confirm_results=2,
        hold_seconds=0.05,
        max_motion_score=0.02,
    )

    moving = gate.update(
        hand_count=2,
        hand_signature="Both",
        result_token=1,
        now=0.0,
        motion_score=0.08,
    )
    assert not moving.ready
    assert moving.matched_results == 0

    gate.update(
        hand_count=2,
        hand_signature="Both",
        result_token=2,
        now=0.1,
        motion_score=0.01,
    )
    ready = gate.update(
        hand_count=2,
        hand_signature="Both",
        result_token=3,
        now=0.2,
        motion_score=0.01,
    )
    assert ready.ready


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        required_hand_count("three")
