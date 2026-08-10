# Adaptive Real-Time Hand Gesture Recognition Using Online Few-Shot Learning

This repository contains an **adaptive real-time hand gesture recognition system** that can learn new user-defined gestures while the application is running.

Unlike a conventional gesture classifier that requires a fixed dataset and an offline retraining step whenever a new class is added, this system is designed around **online few-shot learning**. A user can demonstrate a new gesture inside the application, provide only a small number of useful examples, and begin using that gesture immediately.

The current implementation supports both **static hand poses** and **dynamic hand movements**, one-hand and two-hand gestures, open-set `UNKNOWN` rejection, interactive feedback, persistent personalized gesture memory, prediction stabilization, and confidence scoring.

> **Current scope:** this project recognizes user-defined hand gestures and gesture trajectories. It is not yet a full continuous natural-language sign-language translator with sentence-level linguistic modeling, facial expression modeling, or body-pose grammar.

---

## What makes this project different?

The central idea is:

> **Teach the recognizer a new gesture in seconds, without preparing an offline image/video dataset and without retraining a global classifier.**

The system learns from structured MediaPipe hand landmarks rather than storing raw camera footage. The adaptive memory evolves as the user teaches, confirms, corrects, or rejects predictions.

### Core research contributions implemented in the project

- **Online few-shot class creation** — new gestures are learned during runtime.
- **Smart Capture** — only stable and informative frames are retained instead of blindly saving every camera frame.
- **Normalized landmark representation** — reduces sensitivity to hand position and scale.
- **Adaptive exemplar/prototype memory** — recognition is based on learned examples and prototypes rather than full-model retraining.
- **Multi-prototype classes** — one gesture can retain multiple natural variations.
- **Open-set recognition** — unfamiliar poses can be rejected as `UNKNOWN`.
- **Interactive positive/negative feedback** — `Correct` and `Wrong` actions update gesture memory immediately.
- **Hard-negative learning** — examples explicitly rejected by the user are remembered as evidence against incorrect classes.
- **One-hand and two-hand gestures** — both static configurations are supported.
- **Dynamic gesture learning** — motions such as swipes, waves, and circles can be taught live.
- **Trajectory-based recognition with DTW** — dynamic gestures are compared as temporal landmark trajectories.
- **Prediction stabilization** — prevents frame-to-frame label flicker before a prediction is confirmed.
- **Confidence scoring** — gives an interpretable confidence index based on recognition evidence.
- **Persistent gesture memory** — learned gestures survive application restarts.
- **Privacy-oriented storage** — the adaptive memory stores numerical landmark representations rather than raw videos/images.

---

# System overview

The application has two recognition paths that share the same camera and hand-tracking pipeline.

```text
                         Webcam
                            |
                            v
                  OpenCV frame capture
                            |
                            v
                   MediaPipe Hands
                 21 landmarks per hand
                            |
              +-------------+-------------+
              |                           |
              v                           v
       STATIC GESTURES              DYNAMIC GESTURES
              |                           |
      normalized geometry          landmark trajectory
              |                           |
      Smart Capture / memory       motion segmentation
              |                           |
       exemplar + prototypes       temporal resampling
              |                           |
      open-set recognition          constrained DTW
              |                           |
              +-------------+-------------+
                            |
                            v
                 stabilized prediction
                            |
                            v
                    confidence index
                            |
                            v
                     user feedback
                            |
                            v
                 persistent adaptation
```

---

# How learning works

## Static gestures

When a user teaches a static gesture, the application does **not** train a neural network or retrain a global classifier.

Instead:

1. MediaPipe extracts 21 hand landmarks.
2. The landmarks are normalized into a geometric feature representation.
3. Smart Capture filters unstable and duplicate frames.
4. A small number of useful samples are retained.
5. The class stores exemplars and prototype representations.
6. A local acceptance region is estimated from the learned samples.
7. Recognition starts immediately.

This makes the system closer to **online few-shot exemplar/prototype learning** than conventional batch training.

## Dynamic gestures

Dynamic gestures are represented as sequences of landmark-derived features:

```text
frame 1 -> frame 2 -> frame 3 -> ... -> frame T
```

A user provides a few live demonstrations. The application stores numerical trajectories, performs temporal normalization/resampling, and compares new movements to learned examples using **Dynamic Time Warping (DTW)**.

No video dataset is required for adding a new movement class.

---

# Technology stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Package / environment manager | `uv` |
| Main desktop frontend | Tkinter / `ttk` |
| Experimental desktop frontend | PySide6 / Qt (WIP) |
| Experimental web frontend | Next.js-based design lab (WIP) |
| Camera capture | OpenCV |
| Hand tracking | MediaPipe Hands |
| Numerical processing | NumPy |
| Image display | Pillow |
| Static recognition | Custom online exemplar / multi-prototype learner |
| Dynamic recognition | Custom multivariate DTW pipeline |
| Persistence | JSON landmark memory |
| Tests | pytest |
| CI | GitHub Actions |

---

# Repository structure

The most important parts of the current V2 implementation are:

```text
Sign-Language-Recognition-Using-Hand-Gestures/
|
|-- scripts/
|   |-- interactive_online_learning.py        # Primary Tkinter application
|   |-- interactive_online_learning_qt.py     # Experimental Qt frontend (WIP)
|   |-- test_confidence_scoring.py
|   |-- test_dynamic_gesture_engine.py
|   `-- ...
|
|-- src/adaptive_gesture/
|   |-- features/
|   |   |-- normalizer.py
|   |   |-- similarity.py
|   |   |-- hand_features.py
|   |   `-- dynamic_features.py
|   |
|   |-- learning/
|   |   |-- online_learner.py
|   |   |-- sample_selector.py
|   |   |-- prediction_stabilizer.py
|   |   |-- confidence.py
|   |   |-- dtw.py
|   |   |-- dynamic_learner.py
|   |   `-- motion_segmenter.py
|   |
|   |-- storage/
|   |   |-- gesture_store.py
|   |   `-- dynamic_gesture_store.py
|   |
|   |-- tracking/
|   |   `-- hand_tracker.py
|   |
|   `-- utils/
|
|-- tests/                                     # Automated unit/integration tests
|-- frontend-design-lab/                       # Experimental web UI (WIP)
|-- data/                                      # Local learned gesture memory (ignored by Git)
|-- logs/                                      # Runtime logs (ignored by Git)
|-- FRONTEND_WIP.md
|-- pyproject.toml
|-- uv.lock
|-- .python-version
`-- README.md
```

> The two experimental frontends are intentionally marked **WIP**. The Tkinter application is currently the primary supported interface.

---

# Quick start

## 1. Install Git

If Git is not installed, install it from the official Git distribution for your operating system.

Verify:

```powershell
git --version
```

## 2. Install `uv`

This project uses **Astral uv** for Python version management, dependency resolution, virtual environments, and reproducible installs.

After installing `uv`, verify:

```powershell
uv --version
```

## 3. Clone the repository

```powershell
git clone https://github.com/skrasib/Sign-Language-Recognition-Using-Hand-Gestures.git
cd Sign-Language-Recognition-Using-Hand-Gestures
```

## 4. Switch to the V2 branch

The adaptive online-learning implementation currently lives on:

```powershell
git switch v2-online-few-shot
```

If the branch is not available locally yet:

```powershell
git fetch origin
git switch -c v2-online-few-shot --track origin/v2-online-few-shot
```

## 5. Install the required Python version

The project is pinned to Python **3.11.9**.

```powershell
uv python install 3.11.9
```

The repository contains `.python-version`, so `uv` will use the configured Python version for the project.

## 6. Install the application dependencies

For the primary Tkinter application plus development/test tools:

```powershell
uv sync --extra dev
```

This creates/updates the local `.venv` automatically from `pyproject.toml` and `uv.lock`.

You do **not** need to manually activate the environment when using `uv run`.

## 7. Run the automated tests

On most systems:

```powershell
uv run pytest
```

If Windows denies access to pytest's normal temporary directory, use the project-local test directory:

```powershell
uv run pytest --basetemp=.pytest_tmp
```

The current automated suite is expected to pass before normal use.

## 8. Start the primary application

```powershell
uv run python scripts\interactive_online_learning.py
```

Allow access to your webcam if Windows or your security software asks for permission.

---

# First-time usage

The application is intentionally able to start with **no predefined gesture vocabulary**.

You create the vocabulary yourself.

## Teach your first static gesture

1. Launch the application.
2. Open the **Teach** section.
3. Select/static teaching if the interface asks for the gesture type.
4. Enter a gesture name, for example:

   ```text
   Victory
   ```

5. Click **Teach Gesture**.
6. Position your hand naturally in front of the webcam.
7. Hold the intended gesture steadily.
8. Make very small natural variations while keeping the same semantic gesture.
9. Smart Capture will automatically:
   - ignore unstable frames;
   - ignore near-duplicates;
   - keep useful examples.
10. Once enough useful samples have been collected, finish teaching or allow the automatic completion threshold to finish it.
11. Return to the live recognition view.
12. Show the gesture again.

The new class should now be recognized immediately.

No separate training script, notebook, offline image collection process, or application restart is required.

---

# Smart Capture

A webcam can produce dozens of almost-identical frames while a user holds one pose. Saving all of those frames does not necessarily create useful diversity.

Smart Capture therefore evaluates candidate samples and tries to retain only examples that are:

- sufficiently stable;
- not near-duplicates of already accepted samples;
- representative of natural variation in the demonstrated gesture.

During teaching, the UI displays statistics such as:

```text
Frames observed
Useful samples
Duplicates ignored
Unstable samples ignored
Current stability
```

This is why the number of retained training samples can be much smaller than the number of frames observed by the webcam.

---

# Static recognition and `UNKNOWN`

The system is **open-set aware**.

That means it does not have to force every hand pose into one of the learned classes.

For each compatible class, the recognizer evaluates how close the current features are to learned positive examples/prototypes and whether the sample falls inside the learned acceptance region.

If the evidence is insufficient, the result becomes:

```text
UNKNOWN
```

This is especially important for a customizable system because users may show poses that have never been taught.

---

# Interactive feedback

The live UI includes feedback controls so the system can continue adapting after the initial teaching phase.

## Correct prediction

If the prediction is correct:

```text
[ Correct ]
```

The current example can be added as additional positive evidence for that class.

## Wrong prediction

If the application predicts the wrong learned class:

```text
[ Wrong ]
```

Choose the actual gesture and apply the correction.

Conceptually, the captured example becomes:

```text
positive evidence  -> actual class
negative evidence  -> incorrectly predicted class
```

## Unknown pose

If a pose belongs to none of the learned classes, use:

```text
This Gesture Is Unknown
```

When appropriate, the example becomes a **hard negative** for the incorrectly predicted class.

This allows class boundaries to improve through normal interaction without retraining all existing classes.

---

# Multi-prototype learning

A single gesture can legitimately look different across natural wrist rotations, small orientation changes, or different comfortable hand configurations.

Instead of forcing all variation into one single class center, the adaptive learner can maintain several prototypes for a gesture.

Conceptually:

```text
Victory
|-- prototype A: front-facing
|-- prototype B: slight wrist rotation
`-- prototype C: another stable variant
```

This increases within-class flexibility while preserving open-set rejection.

---

# One-hand and two-hand gestures

The system supports both:

```text
one hand  -> normalized single-hand representation
both hands -> combined two-hand representation
```

For two-hand gestures, the representation preserves information from both hand shapes as well as their relative configuration.

When teaching a two-hand gesture, keep both hands visible throughout the Smart Capture phase.

---

# Teach a dynamic gesture

Dynamic gestures are movements rather than single fixed poses.

Examples include:

- Swipe Left
- Swipe Right
- Wave
- Circle
- other user-defined one-hand or two-hand motions

## Recommended teaching flow

1. Open the **Dynamic Gestures** section.
2. Enter a name such as:

   ```text
   Swipe Right
   ```

3. Start dynamic teaching.
4. Start the first demonstration.
5. Perform the movement once from a clear start state to a clear end state.
6. Stop the demonstration.
7. Repeat for the requested number of demonstrations.
8. Finish learning.
9. Return to the live view.
10. Perform the movement naturally.

The system will detect the movement segment and compare it with the learned trajectories.

---

# Dynamic recognition pipeline

Dynamic recognition uses a different mechanism from static pose recognition.

```text
Hand tracking
    |
    v
Temporal landmark features
    |
    v
Motion detection / segmentation
    |
    v
Trajectory smoothing / resampling
    |
    v
Constrained multivariate DTW
    |
    v
Known gesture or UNKNOWN
```

**Dynamic Time Warping (DTW)** allows two demonstrations of the same movement to be aligned even when one is performed somewhat faster or slower than the other.

The implementation also applies rejection/ambiguity checks so unrelated motion is not automatically forced into the nearest dynamic class.

---

# Prediction stabilization

Raw frame-by-frame recognition can flicker because tracking noise may briefly change the nearest class.

The application therefore separates:

```text
raw candidate prediction
        |
        v
stabilization logic
        |
        v
confirmed UI prediction
```

A class must remain sufficiently consistent before it replaces the currently confirmed result.

This improves visual stability without changing the underlying class memory.

---

# Confidence index

The UI reports a **confidence index**, not a calibrated probability.

The score is derived from recognition evidence such as:

- distance to the learned positive region;
- separation from competing classes;
- negative/hard-negative evidence;
- dynamic-match ambiguity for trajectory recognition.

For example:

```text
Victory
Confidence: 92%
```

should be interpreted as a relative confidence indicator produced by the current recognizer, **not** as a statement that the statistical probability of correctness is exactly 92%.

Formal probability calibration would require a dedicated held-out evaluation dataset.

---

# Gesture persistence

Learned gesture information survives application restarts.

The application stores local adaptive memory under the project's data directory, including files such as:

```text
data/gesture_memory.json
data/dynamic_gesture_memory.json
```

These files are local runtime data and are ignored by Git.

The stored information contains numerical gesture representations such as normalized landmarks, positive exemplars, negative examples, and dynamic trajectories.

The application does **not** need to save raw webcam videos in order to remember learned gestures.

---

# Gesture management

The application provides management controls for the learned gesture vocabulary.

Depending on gesture type and current UI section, available actions include:

- rename a gesture;
- delete a gesture;
- improve an existing static gesture with additional examples;
- retrain/reset a gesture representation;
- clear stored gesture memory;
- manage dynamic gesture demonstrations.

Because changes are persisted, they remain available the next time the application is opened.

---

# Frontends

## Tkinter — primary frontend

Run:

```powershell
uv run python scripts\interactive_online_learning.py
```

This is currently the most tested and supported frontend.

## PySide6 / Qt — WIP experimental frontend

Install the optional Qt dependency group:

```powershell
uv sync --extra dev --extra qt
```

Run:

```powershell
uv run python scripts\interactive_online_learning_qt.py
```

The Qt interface is experimental and may not always expose every feature exactly like the primary Tkinter application.

## Web design lab — WIP

The repository also contains:

```text
frontend-design-lab/
```

This is an experimental web-interface/design environment and is **not currently the supported runtime frontend for the Python recognition engine**.

See `FRONTEND_WIP.md` for the current frontend status.

---

# Running tests

Install the development dependency group:

```powershell
uv sync --extra dev
```

Run:

```powershell
uv run pytest --basetemp=.pytest_tmp
```

The test suite covers core numerical behavior and integration paths such as:

- feature calculations;
- static learner behavior;
- prediction stabilization;
- persistence round trips;
- dynamic gesture memory;
- confidence-scoring logic;
- other core backend components.

The test suite intentionally does not replace real webcam/user evaluation.

---

# Logs

Runtime logs are written under:

```text
logs/
```

For example:

```text
logs/adaptive_gesture.log
```

Logs can be useful when debugging startup, persistence, webcam, or unexpected runtime problems.

The log directory is ignored by Git.

---

# Troubleshooting

## Camera does not open

First verify that:

- another application is not currently using the webcam;
- Windows camera privacy settings allow desktop applications to use the camera;
- you are running the primary app only once;
- the Qt and Tkinter versions are not open simultaneously.

Then launch again:

```powershell
uv run python scripts\interactive_online_learning.py
```

## `cv2.VideoCapture` is missing

If OpenCV imports but camera APIs such as `VideoCapture` are missing, repair the declared OpenCV package:

```powershell
uv sync --extra dev --reinstall-package opencv-contrib-python
```

If you also use the experimental Qt frontend:

```powershell
uv sync --extra dev --extra qt --reinstall-package opencv-contrib-python
```

Verify:

```powershell
uv run python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'VideoCapture')); print(hasattr(cv2, 'CAP_DSHOW'))"
```

On the current Windows setup, the final two values should be `True`.

## Pytest temporary-folder permission error on Windows

If pytest reports `WinError 5` under the user's Windows temporary directory, run:

```powershell
uv run pytest --basetemp=.pytest_tmp
```

The `.pytest_tmp/` directory is ignored by Git.

## Rebuild the virtual environment

If the environment becomes inconsistent:

```powershell
Remove-Item -Recurse -Force .venv
uv sync --extra dev
```

For the optional Qt frontend:

```powershell
uv sync --extra dev --extra qt
```

---

# Development workflow

The recommended development workflow is:

```powershell
# Synchronize dependencies
uv sync --extra dev

# Run tests
uv run pytest --basetemp=.pytest_tmp

# Run application
uv run python scripts\interactive_online_learning.py
```

Before committing:

```powershell
git status
```

Local runtime artifacts such as learned gesture memory, logs, caches, and virtual environments should remain untracked.

---

# Research direction

The system is designed around the following central research question:

> **Can an interactive gesture recognizer improve continuously through natural user feedback?**

Related areas investigated by the implementation include:

- how few informative examples are required to create a useful class;
- whether active sample selection is better than consecutive-frame capture;
- whether adaptive prototypes improve robustness to natural gesture variation;
- how effectively unseen gestures can be rejected;
- how user corrections affect personalized recognition;
- how static and temporal user-defined gestures can coexist in one adaptive system;
- whether numerical landmark memory can provide practical personalization without retaining raw user video.

---

# Current evaluation status

The recognition pipeline and its major interaction mechanisms are implemented and covered by automated backend tests and manual runtime testing.

A larger formal evaluation remains a separate research phase. Planned evaluation includes:

- multiple participants;
- multiple static and dynamic classes;
- accuracy, precision, recall, and F1;
- confusion matrices;
- open-set/UNKNOWN evaluation;
- before/after-feedback experiments;
- few-shot sample-count ablations;
- Smart Capture ablations;
- one-hand versus two-hand analysis;
- DTW threshold/window analysis;
- confidence calibration analysis;
- latency, FPS, and storage measurements.

Until such a benchmark is completed, the repository should not be interpreted as claiming a universal sign-language recognition accuracy figure.

---

# Important limitations

- Recognition is based primarily on hand landmarks.
- Full sign languages can depend on facial expression, body posture, signing space, grammar, and context beyond the hands.
- The current system is best described as an **adaptive hand gesture recognizer**, not a complete continuous sign-language translation engine.
- Personalized classes may behave differently across users, cameras, viewpoints, and lighting conditions.
- Confidence values are confidence indices rather than calibrated probabilities.
- The experimental Qt and web frontends are still work in progress.

---

# Recommended demo flow

For a complete demonstration of the system:

1. Start with an empty/new gesture vocabulary.
2. Teach a static one-hand gesture.
3. Show that it is recognized immediately.
4. Present an unseen pose and demonstrate `UNKNOWN` rejection.
5. Teach a second similar gesture.
6. Demonstrate `Correct` and `Wrong` feedback.
7. Restart the application and show persistence.
8. Teach or demonstrate a two-hand static gesture.
9. Teach a dynamic gesture such as `Swipe Right`.
10. Demonstrate dynamic recognition.
11. Show prediction stabilization and the confidence index.
12. Open the gesture library and demonstrate management controls.

This sequence highlights the main research contribution: **the gesture vocabulary is created and refined through interaction rather than through a separate offline training workflow.**

---

# Contributing / testing on another machine

When testing the project on another user's computer, that user should clone the repository and create their own local gesture memory rather than copying another person's runtime database.

Recommended setup:

```powershell
git clone https://github.com/skrasib/Sign-Language-Recognition-Using-Hand-Gestures.git
cd Sign-Language-Recognition-Using-Hand-Gestures
git switch v2-online-few-shot
uv python install 3.11.9
uv sync --extra dev
uv run pytest --basetemp=.pytest_tmp
uv run python scripts\interactive_online_learning.py
```

The new user can then teach their own static and dynamic gesture vocabulary through the application UI.

---

## Project status

**V2 is under active research and evaluation.**

The major adaptive-learning features are implemented. Current work is focused on reproducible evaluation, usability refinement, frontend experimentation, and thesis-level experimental validation.
