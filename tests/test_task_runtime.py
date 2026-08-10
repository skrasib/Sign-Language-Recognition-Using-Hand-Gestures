from adaptive_gesture.tracking.task_runtime import LatestAsyncResult, MonotonicTimestamp


def test_timestamp_is_strictly_increasing_with_same_clock_value():
    clock = MonotonicTimestamp()
    assert clock.next_ms(1_000_000_000) == 1000
    assert clock.next_ms(1_000_000_000) == 1001
    assert clock.next_ms(999_000_000) == 1002


def test_latest_async_result_bridges_submission_and_callback():
    buffer = LatestAsyncResult([])
    buffer.note_submission()
    buffer.note_submission()
    buffer.publish(["hand"], timestamp_ms=1234, latency_ms=17.5)

    assert buffer.snapshot() == ["hand"]
    diagnostics = buffer.diagnostics()
    assert diagnostics.submitted_frames == 2
    assert diagnostics.result_callbacks == 1
    assert diagnostics.latest_timestamp_ms == 1234
    assert diagnostics.latest_latency_ms == 17.5
    assert diagnostics.latest_result_age_ms is not None
    assert diagnostics.latest_result_age_ms >= 0.0


def test_latest_async_result_replaces_old_result():
    buffer = LatestAsyncResult(["old"])
    buffer.publish(["new"], timestamp_ms=5, latency_ms=1.0)
    assert buffer.snapshot() == ["new"]


def test_reacquisition_guard_waits_for_fresh_track_after_confirmed_absence():
    from adaptive_gesture.tracking.task_runtime import ReacquisitionGuard

    guard = ReacquisitionGuard(confirm_frames=3)

    # Normal startup/first detection is immediate.
    assert guard.update(raw_count=1, stable_count=1).expose_result is True

    # Brief held misses are still exposed by the existing tracker stabilizer.
    assert guard.update(raw_count=0, stable_count=1).expose_result is True
    assert guard.update(raw_count=0, stable_count=1).expose_result is True

    # Once the tracker itself reports no stable hand, reacquisition mode begins.
    decision = guard.update(raw_count=0, stable_count=0)
    assert decision.expose_result is False
    assert guard.awaiting_reacquisition is True

    # First two fresh detections are warm-up only.
    assert guard.update(raw_count=1, stable_count=1).expose_result is False
    assert guard.update(raw_count=1, stable_count=1).expose_result is False

    # Third consecutive detection is exposed.
    assert guard.update(raw_count=1, stable_count=1).expose_result is True
    assert guard.awaiting_reacquisition is False


def test_reacquisition_guard_resets_if_candidate_track_drops():
    from adaptive_gesture.tracking.task_runtime import ReacquisitionGuard

    guard = ReacquisitionGuard(confirm_frames=3)
    guard.update(raw_count=1, stable_count=1)
    guard.update(raw_count=0, stable_count=0)

    assert guard.update(raw_count=1, stable_count=1).expose_result is False
    decision = guard.update(raw_count=0, stable_count=1)
    assert decision.expose_result is False
    assert decision.clear_tracker_state is True
    assert guard.reacquire_streak == 0


def test_handedness_reacquisition_uses_latest_label_instead_of_sticky_first_label():
    from adaptive_gesture.tracking.task_runtime import choose_handedness

    label, score = choose_handedness(
        previous_label="Unknown",
        previous_score=0.0,
        current_label="Right",
        current_score=0.93,
        wrist_continuous=True,
        reacquiring=True,
    )
    assert label == "Right"
    assert score == 0.93


def test_valid_handedness_repairs_unknown_even_outside_reacquisition():
    from adaptive_gesture.tracking.task_runtime import choose_handedness

    label, score = choose_handedness(
        previous_label="Unknown",
        previous_score=0.0,
        current_label="Left",
        current_score=0.88,
        wrist_continuous=True,
        reacquiring=False,
    )
    assert label == "Left"
    assert score == 0.88
