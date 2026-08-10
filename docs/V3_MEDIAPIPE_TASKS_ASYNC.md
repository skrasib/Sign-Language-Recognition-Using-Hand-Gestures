# V3.6 — MediaPipe Tasks Live-Stream Tracking

## Goal

V3.6 modernizes only the camera-to-landmark layer. The V3.1–V3.5 learning stack is preserved.

Previous path:

```text
OpenCV frame
  -> mp.solutions.hands.Hands.process(...)
  -> blocking landmark result
  -> temporal stabilizer
  -> V3 learning stack
```

V3.6 path:

```text
OpenCV frame
  -> MediaPipe Tasks HandLandmarker.detect_async(...)
  -> LIVE_STREAM worker / callback
  -> latest completed landmark result
  -> preserved temporal stabilizer
  -> V3.1 hybrid geometry
  -> V3.3 metric embedding
  -> V3.4 exemplar memory
  -> V3.2 EVT open-set decision
  -> V3.5 temporal prototypes for dynamics
```

MediaPipe documents LIVE_STREAM as the mode intended for camera/live-stream input. `detect_async` returns immediately and results arrive through a callback. MediaPipe may intentionally drop input frames when necessary to reduce live-stream latency rather than guaranteeing a result for every submitted frame.

Official API references:

- https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker
- https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerOptions
- https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/hand_landmarker.py

## What changed

### 1. MediaPipe Tasks Hand Landmarker

`src/adaptive_gesture/tracking/hand_tracker.py` now creates a `HandLandmarker` with:

```text
RunningMode.LIVE_STREAM
num_hands = 2
result_callback = asynchronous callback
```

The normal Tkinter camera loop no longer waits for `Hands.process()`.

### 2. Model bundle management

The Tasks API requires the official Hand Landmarker `.task` bundle.

V3.6 automatically downloads the official float16 model bundle on first use to:

```text
data/v3/models/hand_landmarker.task
```

The source URL used by the official MediaPipe samples is:

```text
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

`data/` is already ignored by Git, so the binary model does not become part of normal commits.

The download is written to a temporary `.part` file first and atomically renamed only after a basic size check. An interrupted first download therefore does not leave a seemingly valid model file.

### 3. Strict timestamps

MediaPipe live-stream input requires monotonically increasing millisecond timestamps.

`task_runtime.MonotonicTimestamp` guarantees strictly increasing timestamps even if two UI iterations occur within the same millisecond.

### 4. Callback-to-UI bridge

The Hand Landmarker callback runs separately from Tk's UI loop. `LatestAsyncResult` is a small thread-safe bridge that stores:

- latest stabilized result;
- submitted frame count;
- callback count;
- latest result timestamp;
- approximate callback latency;
- age of the latest result.

Tk only consumes the latest completed result.

### 5. Existing anti-flicker stabilization remains

The V2/V3 hand stability behavior was not discarded. It still performs:

- brief missing-result hold;
- one-hand/two-hand hysteresis;
- wrist-based hand identity matching;
- handedness stabilization;
- landmark exponential smoothing.

The difference is that stabilization is now applied inside the asynchronous Tasks callback.

### 6. No legacy drawing dependency

V3.6 draws the canonical 21-landmark hand graph with OpenCV rather than relying on `mp.solutions.drawing_utils`.

This removes the runtime dependency on the legacy Solutions API from the V3 tracker.

### 7. Dynamic stream de-duplication

Because the camera UI can render faster than the asynchronous Hand Landmarker returns new results, consecutive UI frames can briefly contain the same landmark result.

V3.6 records the callback timestamp and allows the dynamic sampler to consume a landmark result only once. This avoids inserting duplicate temporal observations solely because the UI loop was faster than the tracker.

This is a tracking integration change, not a redesign of V3.5 dynamic gesture representation. A deeper redesign of dynamic gestures can be handled separately later.

## What did NOT change

V3.6 does not change:

- geometry-aware features;
- metric encoder training;
- static gesture memory format;
- diversity-aware exemplar selection;
- EVT open-set scoring;
- prediction stabilization;
- static feedback learning;
- temporal prototype algorithm;
- dynamic memory persistence format.

This keeps V3.6 usable as a controlled tracking-layer experiment.

## Research comparison

A future latency experiment can compare:

```text
Legacy synchronous tracker
vs
Tasks LIVE_STREAM asynchronous tracker
```

Useful measurements include:

- camera/UI frame rate;
- tracker callback rate;
- callback latency;
- static recognition latency;
- dynamic trajectory sample rate;
- hand-detection continuity.

Do not describe asynchronous input-frame dropping as an error by default. The MediaPipe Tasks API explicitly allows it in LIVE_STREAM mode as a latency-control behavior.

## First-run behavior

The first V3.6 launch may take longer because the Hand Landmarker model is downloaded once. Later runs reuse the local model.

Optional manual preparation:

```powershell
uv run python scripts\download_v3_hand_landmarker_model.py
```

Normal app usage does not require this command because the application downloads the model automatically when missing.
