# V3.6.2 — Hand-Count-Aware, Hands-Free Teaching

## Motivation

Two-hand gestures are awkward to teach if the user must hold both hands in the
camera view and also use the mouse to start or stop capture. V3.6.2 separates
**configuration** from **performance**:

1. enter the gesture name;
2. choose **One hand** or **Two hands**;
3. press the teaching button once;
4. move into the camera and perform the gesture without further mouse input.

The change affects interaction/collection logic only. It does not alter the V3
geometry descriptor, metric encoder, exemplar learner, EVT rejection, temporal
prototype learner, or MediaPipe Tasks backend.

## Static teaching state machine

```text
CONFIGURE
  |  name + one/two hands
  v
WAITING_HANDS
  |  exact requested hand count must remain stable
  v
COUNTDOWN
  |  configuration must stay visible
  v
CAPTURING
  |  Smart Capture keeps stable/diverse samples
  v
COMPLETE
```

For a new one-hand class, the readiness gate locks the consistent MediaPipe
`Left` or `Right` signature only after readiness is confirmed. For a new two-hand
class, the required signature is `Both` from the start.

Improve/Retrain sessions inherit the already-saved class hand signature, so a
stored two-hand class automatically requires two hands again.

## Dynamic teaching state machine

```text
CONFIGURE
  |  name + one/two hands
  v
WAITING_HANDS / HOLD STILL
  |  exact hand count + stable starting pose
  v
READY
  |  user begins moving naturally
  v
RECORDING
  |  MotionSegmenter detects the active movement
  v
MOTION STOPS
  |  trajectory prepared automatically
  v
DEMO ACCEPTED
  |  return to start + hold still
  +---------------------> next demo

After the final demo -> learn dynamic class -> save -> complete
```

The normal dynamic runtime `MotionSegmenter` and the teaching segmenter are
separate instances so recognition state cannot leak into a teaching session.

## Readiness gate

`src/adaptive_gesture/learning/teaching_flow.py` implements a reusable
`HandReadinessGate`.

The gate requires:

- exactly the selected hand count;
- a compatible `Left` / `Right` / `Both` signature;
- multiple **fresh** MediaPipe Tasks callback results;
- a short minimum hold time;
- for dynamic teaching, a low motion score before a demo can arm.

Repeated Tkinter renders of the same asynchronous tracking result do not advance
readiness. This prevents a single transient second-hand detection from starting a
two-hand capture.

## Dynamic interruption behavior

If a required hand disappears or the hand configuration changes during a dynamic
demonstration, that attempt is discarded and the same demo automatically re-arms.
A partially captured movement is not silently accepted as a complete gesture.

## Persistence

No persistence schema changes are required. Learned classes continue to store the
same `Left`, `Right`, or `Both` hand signature and the same numerical
feature/trajectory data as before.

## Research/evaluation note

This is primarily a usability and acquisition-reliability improvement. A future
study can compare teaching completion time, failed demonstrations, accidental
captures, and user interaction count before/after the hands-free state machine.
