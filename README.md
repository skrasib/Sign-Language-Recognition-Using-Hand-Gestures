# Adaptive Real-Time Sign & Gesture Recognition

A real-time, personalized hand-gesture recognition system built around **MediaPipe landmarks**, **online few-shot learning**, **open-set recognition**, and **runtime personalization**.

This repository started as a conventional static hand-gesture classifier and has evolved into a research-oriented system that can learn **new user-defined static and dynamic gestures while the application is running**, without collecting an offline image/video dataset and without retraining a conventional classifier every time a new gesture is added.

> **Current research branch:** `v3-geometry-aware-research`  
> **Stable V2 branch:** `v2-online-few-shot`

---

## Table of Contents

- [Project Overview](#project-overview)
- [What Makes This Project Different?](#what-makes-this-project-different)
- [V2 and V3](#v2-and-v3)
- [Current V3 Architecture](#current-v3-architecture)
- [V3 Feature Evolution](#v3-feature-evolution)
- [How Gesture Data Is Represented](#how-gesture-data-is-represented)
- [Static Gesture Recognition](#static-gesture-recognition)
- [Dynamic Gesture Recognition](#dynamic-gesture-recognition)
- [One-Hand and Two-Hand Gestures](#one-hand-and-two-hand-gestures)
- [Hands-Free Teaching Workflow](#hands-free-teaching-workflow)
- [Open-Set Recognition](#open-set-recognition)
- [Online Feedback and Adaptation](#online-feedback-and-adaptation)
- [Confidence Index](#confidence-index)
- [Data Storage and Privacy](#data-storage-and-privacy)
- [User Interface](#user-interface)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [First-Time Usage Guide](#first-time-usage-guide)
- [Testing](#testing)
- [Optional Qt Frontend](#optional-qt-frontend)
- [Experimental Web Frontend](#experimental-web-frontend)
- [Troubleshooting](#troubleshooting)
- [Research Direction](#research-direction)
- [Current Limitations](#current-limitations)
- [Version Summary](#version-summary)

---

# Project Overview

The goal of this project is to make gesture recognition **adaptive and personalized**.

Traditional gesture-recognition systems often follow this workflow:

```text
Collect many images/videos
        ↓
Create dataset
        ↓
Train classifier/model
        ↓
Deploy application
        ↓
Need a new gesture?
        ↓
Collect more data and retrain
```

This project instead focuses on:

```text
Open application
        ↓
Show a new gesture
        ↓
Give it a name
        ↓
Provide only a few demonstrations
        ↓
Gesture becomes usable immediately
```

The application learns from **hand landmarks** rather than storing camera footage.

The current V3 system supports:

- personalized online few-shot learning;
- one-hand static gestures;
- two-hand static gestures;
- one-hand dynamic gestures;
- two-hand dynamic gestures;
- geometry-aware hand descriptors;
- learned metric embeddings;
- multi-prototype gesture representation;
- open-set `UNKNOWN` rejection;
- diversity-aware exemplar memory;
- hard-negative feedback;
- temporal prototypes for dynamic gestures;
- prediction stabilization;
- confidence scoring;
- persistent local gesture memory;
- asynchronous MediaPipe Tasks tracking;
- hands-free teaching;
- responsive/scalable desktop UI.

---

# What Makes This Project Different?

The system is designed around a simple research question:

> **Can a real-time gesture recognizer continuously learn personalized gestures from only a few live demonstrations and improve through natural user feedback?**

A new gesture does not require a complete model-training pipeline.

Instead, the application builds a compact representation of the gesture from landmark-based examples and uses metric/prototype-based recognition.

At runtime:

```text
Camera
   ↓
MediaPipe hand landmarks
   ↓
Geometry-aware feature representation
   ↓
Learned metric embedding
   ↓
Few-shot exemplar/prototype memory
   ↓
Open-set decision
   ↓
Known gesture or UNKNOWN
```

For dynamic gestures, landmark trajectories are used instead of individual poses.

---

# V2 and V3

This repository currently contains two important research stages.

## V2 — Online Few-Shot Baseline

Branch:

```text
v2-online-few-shot
```

V2 represents the stable online few-shot system developed before the V3 research expansion.

It includes:

- normalized MediaPipe landmarks;
- online static gesture learning;
- Smart Capture;
- one-hand and two-hand gestures;
- multi-prototypes;
- open-set rejection;
- hard-negative feedback;
- prediction stabilization;
- dynamic gestures using DTW;
- persistent landmark-only memory;
- confidence index;
- gesture management;
- Tkinter interface.

V2 is preserved as a stable baseline.

---

## V3 — Research-Oriented Expansion

Branch:

```text
v3-geometry-aware-research
```

V3 builds on V2 and introduces several additional research components:

```text
V3.1   Geometry-aware hybrid features
V3.2   EVT-inspired open-set recognition
V3.3   Learned metric embedding
V3.4   Diversity-aware exemplar memory
V3.5   DTW-aligned temporal prototypes
V3.6   MediaPipe Tasks asynchronous tracking
V3.6.1 Robust hand reacquisition
V3.6.2 Hands-free one/two-hand teaching
V3.6.3 Responsive/scalable desktop UI
```

---

# Current V3 Architecture

The static pipeline is approximately:

```text
┌──────────────────────────┐
│        Camera Frame      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ MediaPipe Tasks          │
│ HandLandmarker           │
│ LIVE_STREAM / async      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Tracking Stabilization   │
│ • hand-count hysteresis  │
│ • smoothing              │
│ • handedness stability   │
│ • reacquisition logic    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ 21 hand landmarks        │
│ x, y, z coordinates      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Geometry-Aware Features  │
│ • normalized XYZ         │
│ • joint angles           │
│ • two-hand geometry      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Learned Metric Encoder   │
│ hybrid representation    │
│ → compact embedding      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Online Gesture Memory    │
│ • exemplars              │
│ • multi-prototypes       │
│ • diversity selection    │
│ • hard negatives         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ EVT-Inspired Open Set    │
│ inclusion / rejection    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Prediction Stabilizer    │
└────────────┬─────────────┘
             ↓
      Gesture / UNKNOWN
```

Dynamic recognition uses a parallel temporal branch:

```text
Tracked landmarks
      ↓
Motion segmentation
      ↓
Trajectory representation
      ↓
DTW alignment
      ↓
Temporal prototype(s)
      ↓
Dynamic gesture / UNKNOWN
```

---

# V3 Feature Evolution

## V3.1 — Geometry-Aware Hybrid Features

The original landmark representation primarily described **where joints are located**.

V3.1 additionally describes **how the hand is shaped**.

MediaPipe provides 21 landmarks:

```text
21 landmarks × x,y,z = 63 coordinate values
```

V3 adds joint-angle information.

For one hand:

```text
Normalized XYZ coordinates
        +
Geometry-aware joint angles
        ↓
Hybrid descriptor
```

This improves robustness to variations such as:

- hand rotation;
- hand tilt;
- small viewpoint changes;
- distance changes;
- natural pose variation.

The idea is inspired by geometry-aware few-shot sign-recognition research, while the implementation is adapted to this project's online personalized-learning setting.

---

## V3.2 — EVT-Inspired Open-Set Recognition

A real-world recognizer should not force every hand pose into one of the learned classes.

Example:

```text
Learned:
Victory
Three
Fist

Input:
completely different pose
```

Desired result:

```text
UNKNOWN
```

V3.2 adds an Extreme Value Theory-inspired open-set layer that models the boundary around known gesture examples using negative evidence.

This works together with:

- competing gesture classes;
- hard negatives;
- learned metric distances;
- class-specific acceptance behavior.

The current implementation is **EVM-inspired**, not a complete reproduction of the canonical Extreme Value Machine algorithm.

---

## V3.3 — Learned Metric Embedding

Instead of comparing the full handcrafted feature vector directly, V3.3 learns a compact feature space.

Conceptually:

```text
Hybrid geometry
      ↓
Small learned metric encoder
      ↓
Compact embedding
      ↓
Prototype/exemplar comparison
```

The metric-learning objective encourages:

```text
same gesture   → close together
different      → farther apart
```

After the encoder is learned, normal runtime gesture teaching still remains lightweight:

```text
New gesture
   ↓
Frozen encoder
   ↓
Few embeddings
   ↓
New prototype/exemplar memory
```

A full classifier is not retrained every time the user adds a gesture.

---

## V3.4 — Diversity-Aware Exemplar Memory

Gesture memory is bounded.

A simple FIFO system eventually removes the oldest examples, even when an old sample represents a useful variation.

V3.4 instead prefers a representative subset.

It keeps:

- central examples;
- diverse variations;
- useful boundary examples.

Conceptually:

```text
Many stored examples
       ↓
Select representative core set
       ↓
Keep useful diversity
```

Hard-negative memory also prioritizes:

- negatives near a class boundary;
- diverse mistakes;
- non-redundant negative evidence.

---

## V3.5 — Temporal Prototypes

Dynamic gestures are trajectories rather than single hand poses.

Instead of relying only on individual stored demonstrations:

```text
Demo 1
Demo 2
Demo 3
```

V3.5 uses DTW alignment to construct a representative motion prototype.

```text
Demo 1 ─┐
Demo 2 ─┼─ DTW alignment → temporal prototype
Demo 3 ─┘
```

This is a lightweight **DTW-barycenter / DBA-style** temporal representation.

The implementation does not currently claim to be Soft-DTW.

Multiple temporal prototypes may be used when a gesture contains clearly different motion styles.

---

## V3.6 — MediaPipe Tasks Asynchronous Tracking

V3.6 modernizes the landmark-extraction pipeline.

Instead of synchronously blocking the UI while processing every frame, the application uses MediaPipe Tasks `HandLandmarker` in live-stream mode.

```text
Camera / UI
    │
    ├──────────── continues rendering
    │
    └── async frame submission
             ↓
       HandLandmarker
             ↓
          callback
             ↓
       latest landmarks
```

Benefits include:

- cleaner separation between UI and tracking;
- lower risk of UI blocking;
- modern MediaPipe Tasks API;
- more suitable live-stream architecture.

---

## V3.6.1 — Hand Reacquisition

A tracking issue occurred when a hand completely left the camera and later returned.

V3.6.1 adds a short reacquisition procedure so stale or unstable hand identity does not poison the recognition state.

```text
Hand disappears
      ↓
tracking state cleared
      ↓
hand returns
      ↓
short fresh-result warm-up
      ↓
recognition resumes
```

---

## V3.6.2 — Hands-Free Teaching

Two-hand gestures are difficult to teach if the user must continuously operate a mouse.

The application now allows the user to configure the gesture first:

```text
Gesture name
Static / Dynamic
One hand / Two hands
```

After pressing Start once, capture becomes hands-free.

For a two-hand gesture:

```text
Start Teaching
     ↓
WAITING FOR 2 HANDS
     ↓
both hands detected consistently
     ↓
HOLD STEADY
     ↓
READY
     ↓
CAPTURING
```

The system does not begin recording a two-hand gesture when only one hand is detected.

Dynamic gestures can progress through multiple demonstrations without repeatedly touching the mouse.

---

## V3.6.3 — Responsive UI

The primary Tkinter interface is responsive rather than relying on a single hard-coded desktop size.

The interface now adapts to:

- different screen resolutions;
- maximize/restore;
- smaller window sizes;
- different camera-panel sizes.

The camera maintains aspect ratio, and right-side pages can scroll vertically when necessary.

---

# How Gesture Data Is Represented

The application does **not train directly on the RGB camera image**.

The camera is used by MediaPipe to locate the hands.

For each hand, MediaPipe returns:

```text
21 landmarks
```

Each landmark contains:

```text
x
y
z
```

Example:

```text
Wrist       → (x, y, z)
Thumb tip   → (x, y, z)
Index tip   → (x, y, z)
...
Pinky tip   → (x, y, z)
```

The project then converts those landmarks into a normalized numerical representation.

This means recognition operates primarily on:

```text
hand geometry
```

rather than:

```text
raw RGB pixels
```

---

# Static Gesture Recognition

Static gestures are poses such as:

```text
Victory
Fist
Three
OK
Point
```

The teaching process is roughly:

```text
Show gesture
      ↓
MediaPipe landmarks
      ↓
Hybrid geometry
      ↓
Smart Capture
      ↓
Metric embeddings
      ↓
Exemplars / prototypes
      ↓
Immediately available for recognition
```

---

## Smart Capture

Camera frames are highly redundant.

If the user holds a pose for several seconds, hundreds of nearly identical frames could be produced.

Smart Capture attempts to keep only useful examples.

It filters:

- unstable frames;
- highly redundant duplicates;
- unnecessary near-identical observations.

Conceptually:

```text
48 observed frames

→ unstable frames removed
→ duplicates removed

→ useful diverse samples retained
```

This keeps teaching fast and memory compact.

---

# Dynamic Gesture Recognition

Dynamic gestures contain movement.

Examples:

```text
Swipe Left
Swipe Right
Circle
Wave
Salute motion
```

The application stores numerical landmark trajectories instead of video.

A dynamic demonstration contains:

```text
frame 1 landmarks
frame 2 landmarks
frame 3 landmarks
...
frame N landmarks
```

The system uses:

- motion segmentation;
- temporal normalization;
- trajectory features;
- DTW alignment;
- temporal prototypes;
- open-set rejection.

Because DTW aligns sequences in time, a gesture can still match when the user performs it somewhat faster or slower.

---

# One-Hand and Two-Hand Gestures

Each taught gesture has a required hand configuration.

Examples:

```text
Victory
required_hands = 1

Heart
required_hands = 2
```

This information is used during both teaching and recognition.

For teaching:

```text
One-hand gesture
→ capture only after one hand is ready

Two-hand gesture
→ capture only after both hands are ready
```

For recognition, hand-count metadata helps prevent unnecessary comparisons between incompatible gesture classes.

Two-hand features additionally encode relationships between the hands.

---

# Hands-Free Teaching Workflow

## Static — One Hand

```text
1. Open Teach
2. Enter gesture name
3. Select One hand
4. Press Start Teaching
5. Put hand in view
6. Wait for READY
7. Hold gesture
8. Smart Capture collects useful examples
9. Gesture is learned
```

---

## Static — Two Hands

```text
1. Open Teach
2. Enter gesture name
3. Select Two hands
4. Press Start Teaching
5. App displays WAITING FOR 2 HANDS
6. Put both hands in view
7. Keep them stable briefly
8. App begins capture automatically
9. Gesture is learned
```

If one hand disappears during capture, capture is paused instead of silently learning an incorrect one-hand sample.

---

## Dynamic — Hands-Free

```text
1. Open Dynamic
2. Enter gesture name
3. Select One hand or Two hands
4. Press Start Hands-Free Teaching once
5. Wait for required hand count
6. Hold start position
7. App displays READY — START MOVING
8. Perform movement
9. Stop naturally
10. Demo is accepted automatically
11. Return to starting state
12. Repeat until all demonstrations are complete
```

---

# Open-Set Recognition

The application supports:

```text
KNOWN
```

and:

```text
UNKNOWN
```

This matters because a gesture recognizer should not classify every arbitrary hand pose as the nearest known class.

V3 combines:

- learned metric distance;
- positive class geometry;
- negative class evidence;
- hard negatives;
- EVT-inspired inclusion scoring;
- ambiguity handling.

---

# Online Feedback and Adaptation

The application supports natural correction.

## Correct Prediction

If the system recognizes:

```text
Victory
```

and the prediction is correct:

```text
✓ Correct
```

the current example can become additional positive evidence for `Victory`.

---

## Wrong Prediction

Suppose:

```text
Prediction: Victory
Actual: Three
```

The user can choose:

```text
✕ Wrong
→ Actual gesture: Three
```

That example becomes:

```text
positive evidence for Three
+
hard-negative evidence for Victory
```

This lets the recognition boundary adapt to the specific user.

The system can therefore become more personalized through interaction.

---

# Confidence Index

The UI displays a:

```text
Confidence index: XX%
```

This should **not currently be interpreted as a calibrated probability**.

It is a combined recognition-confidence index derived from signals such as:

- distance relative to the learned class boundary;
- separation from competing classes;
- hard-negative evidence;
- dynamic ambiguity;
- open-set evidence.

Formal probability calibration is part of future evaluation work.

---

# Data Storage and Privacy

The recognition system does not need to store training photos or videos.

Persistent gesture memory contains numerical representations such as:

```text
normalized landmark features
metric embeddings
gesture metadata
hard-negative examples
dynamic landmark trajectories
```

Camera frames are used for live processing and are not required as the learned gesture database.

Typical V3 runtime data is stored under:

```text
data/v3/
```

Examples may include:

```text
gesture_memory_hybrid.json
gesture_memory_metric_v33.json
dynamic_gesture_memory.json
metric_encoder_v33.npz
models/hand_landmarker.task
```

These runtime files are intended to remain local and are excluded from version control.

---

# User Interface

The primary interface is currently built with **Tkinter**.

Main sections:

```text
Live
Teach
Library
Dynamic
About
```

## Live

Shows:

- camera feed;
- MediaPipe landmarks;
- current recognized gesture;
- confidence index;
- open-set information;
- correct/wrong feedback controls.

## Teach

Used to teach static gestures.

Supports:

- gesture naming;
- one/two-hand selection;
- hands-free readiness;
- Smart Capture;
- improve/retrain flows.

## Library

Used to manage learned gestures.

Typical actions include:

```text
Improve
Retrain
Rename
Delete
Clear
```

## Dynamic

Used to:

- teach dynamic gestures;
- select one/two-hand input;
- perform hands-free demonstrations;
- view demo/prototype information;
- manage learned dynamic gestures.

## About

Contains project and privacy information.

---

# Repository Structure

The exact structure may evolve, but the important V3 components are organized approximately as follows:

```text
Sign-Language-Recognition-Using-Hand-Gestures/
│
├── scripts/
│   ├── interactive_online_learning.py
│   ├── interactive_online_learning_v3.py
│   ├── interactive_online_learning_qt.py
│   ├── train_v3_metric_encoder.py
│   ├── test_v3_metric_embedding.py
│   ├── test_v3_open_set.py
│   ├── test_v3_diversity_memory.py
│   ├── test_v3_temporal_prototypes.py
│   ├── test_v3_tasks_backend.py
│   └── download_v3_hand_landmarker_model.py
│
├── src/
│   └── adaptive_gesture/
│       ├── features/
│       │   ├── normalizer.py
│       │   ├── similarity.py
│       │   ├── hand_features.py
│       │   ├── geometry_features.py
│       │   └── dynamic_features.py
│       │
│       ├── learning/
│       │   ├── online_learner.py
│       │   ├── sample_selector.py
│       │   ├── prediction_stabilizer.py
│       │   ├── confidence.py
│       │   ├── evt_open_set.py
│       │   ├── metric_embedding.py
│       │   ├── metric_runtime.py
│       │   ├── exemplar_memory.py
│       │   ├── dtw.py
│       │   ├── dynamic_learner.py
│       │   ├── motion_segmenter.py
│       │   ├── temporal_prototypes.py
│       │   └── teaching_flow.py
│       │
│       ├── storage/
│       │   ├── gesture_store.py
│       │   └── dynamic_gesture_store.py
│       │
│       ├── tracking/
│       │   ├── hand_tracker.py
│       │   ├── task_runtime.py
│       │   └── model_assets.py
│       │
│       ├── ui/
│       └── utils/
│
├── tests/
├── docs/
├── frontend-design-lab/
├── data/                  # local runtime data, ignored
├── logs/                  # local logs, ignored
│
├── pyproject.toml
├── uv.lock
├── .python-version
└── README.md
```

---

# Requirements

Recommended environment:

```text
Windows 10/11
Python 3.11.9
Webcam
```

The project uses `uv` for Python environment and dependency management.

Install `uv` first if necessary:

```powershell
winget install --id=astral-sh.uv -e
```

Verify:

```powershell
uv --version
```

---

# Installation

## 1. Clone the Repository

```powershell
git clone https://github.com/skrasib/Sign-Language-Recognition-Using-Hand-Gestures.git
cd Sign-Language-Recognition-Using-Hand-Gestures
```

---

## 2. Choose the Branch

For the current V3 research system:

```powershell
git switch v3-geometry-aware-research
```

For the stable V2 baseline:

```powershell
git switch v2-online-few-shot
```

---

## 3. Install Python 3.11.9

```powershell
uv python install 3.11.9
```

The repository also contains:

```text
.python-version
```

to document the intended Python version.

---

## 4. Create / Sync the Environment

For the primary Tkinter application and development tools:

```powershell
uv sync --extra dev
```

This automatically creates the `.venv` environment if required.

You normally do **not** need to activate the environment manually when using `uv run`.

---

## 5. Optional Qt Dependencies

The Qt frontend is experimental and not the primary application.

To install it too:

```powershell
uv sync --extra dev --extra qt
```

---

# Running the Application

## Current V3 Application

```powershell
uv run python scripts\interactive_online_learning_v3.py
```

This is the recommended application on:

```text
v3-geometry-aware-research
```

---

## V2 Application

```powershell
uv run python scripts\interactive_online_learning.py
```

---

# First-Time Usage Guide

## 1. Start the Application

```powershell
uv run python scripts\interactive_online_learning_v3.py
```

The webcam should open automatically.

On the first V3.6 run, the application may need the MediaPipe Hand Landmarker model.

It is stored locally at approximately:

```text
data/v3/models/hand_landmarker.task
```

The helper logic can download/reuse this model.

---

## 2. Teach Your First Static Gesture

Open:

```text
Teach
```

Enter a name, for example:

```text
Victory
```

Choose:

```text
One hand
```

Press:

```text
Start Teaching
```

Place the hand in view.

The application waits until the required hand configuration is stable and then begins Smart Capture.

Once enough useful samples are collected, the gesture becomes available for recognition.

---

## 3. Teach a Two-Hand Gesture

Example:

```text
Heart
```

Choose:

```text
Two hands
```

Press Start once.

The app should display:

```text
WAITING FOR 2 HANDS
```

It will not start teaching from only one visible hand.

Bring both hands into view and hold them stable.

Capture then begins automatically.

---

## 4. Test Recognition

Open:

```text
Live
```

Show one of the taught gestures.

The interface should display:

```text
Gesture: Victory
Confidence index: ...
```

An unrecognized pose should ideally appear as:

```text
UNKNOWN
```

---

## 5. Correct the System

When correct:

```text
✓ Correct
```

When incorrect:

```text
✕ Wrong
```

Then specify the actual gesture or mark the input as unknown.

This feedback updates the personalized gesture memory.

---

## 6. Teach a Dynamic Gesture

Open:

```text
Dynamic
```

Enter a gesture name.

Choose:

```text
One hand
```

or:

```text
Two hands
```

Press:

```text
Start Hands-Free Teaching
```

Wait until:

```text
READY — START MOVING
```

Perform the motion naturally and stop.

The application automatically progresses through the required demonstrations.

---

# Testing

Run the complete test suite:

```powershell
uv run pytest --basetemp=.pytest_tmp
```

The explicit `--basetemp` is recommended on Windows because some systems restrict access to pytest's default temporary directory.

The test suite covers components such as:

- feature normalization;
- geometry-aware descriptors;
- EVT/open-set logic;
- metric embedding;
- persistence;
- exemplar memory;
- temporal prototypes;
- MediaPipe Tasks runtime helpers;
- teaching readiness;
- responsive UI layout helpers.

---

# Optional Qt Frontend

An experimental PySide6 frontend is preserved at:

```text
scripts/interactive_online_learning_qt.py
```

Install Qt dependencies:

```powershell
uv sync --extra dev --extra qt
```

Run:

```powershell
uv run python scripts\interactive_online_learning_qt.py
```

This frontend is currently **experimental / WIP**.

The Tkinter application remains the primary implementation.

---

# Experimental Web Frontend

The repository also contains:

```text
frontend-design-lab/
```

This is a separate experimental frontend design environment.

It is not currently the primary runtime interface and may require its own Node-based setup.

The backend/recognition architecture is intentionally kept sufficiently modular so different frontends can be explored later.

---

# Troubleshooting

## `cv2.VideoCapture` Does Not Exist

Symptom:

```text
AttributeError: module 'cv2' has no attribute 'VideoCapture'
```

This may occur if multiple OpenCV wheels previously shared the same `cv2` namespace and an uninstall left the environment partially broken.

Repair the declared OpenCV package:

```powershell
uv sync --extra dev --extra qt --reinstall-package opencv-contrib-python
```

Verify:

```powershell
uv run python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'VideoCapture')); print(hasattr(cv2, 'CAP_DSHOW'))"
```

Expected:

```text
True
True
```

---

## Pytest Permission Error on Windows

Example:

```text
PermissionError:
C:\Users\<user>\AppData\Local\Temp\pytest-of-...
```

Run:

```powershell
uv run pytest --basetemp=.pytest_tmp
```

---

## PowerShell Refuses to Activate `.venv`

Activation is not required when using `uv run`.

Instead of:

```powershell
.venv\Scripts\Activate.ps1
```

simply use:

```powershell
uv run python ...
```

If manual activation is desired:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

This changes policy only for the current PowerShell process.

---

## Webcam Does Not Open

Check another application is not currently using the camera.

You can also test OpenCV directly:

```powershell
uv run python -c "import cv2; c=cv2.VideoCapture(0, cv2.CAP_DSHOW); print(c.isOpened()); c.release()"
```

---

## First V3.6 Launch Cannot Find Hand Landmarker Model

Use the helper script:

```powershell
uv run python scripts\download_v3_hand_landmarker_model.py
```

Then test the backend:

```powershell
uv run python scripts\test_v3_tasks_backend.py
```

---

## Recognition Becomes UNKNOWN After Hand Leaves Frame

The current V3 branch contains reacquisition logic designed to clear stale tracking identity when a hand disappears and later returns.

If this happens unexpectedly, verify that you are using the latest:

```text
v3-geometry-aware-research
```

branch.

---

# Research Direction

The current V3 system is intended as a research prototype rather than a claim of a completed universal sign-language translator.

Important future evaluation directions include:

## Few-Shot Learning

Compare:

```text
1 shot
3 shots
6 shots
12 shots
```

for unseen gestures.

---

## Ablation Studies

Potential comparisons:

```text
V2 normalized coordinates
vs
V3.1 geometry-aware hybrid features
```

```text
radius-based open-set rejection
vs
V3.2 EVT-inspired rejection
```

```text
raw/hybrid feature space
vs
V3.3 metric embedding
```

```text
FIFO exemplar memory
vs
V3.4 diversity-aware memory
```

```text
nearest dynamic demonstration
vs
V3.5 temporal prototypes
```

---

## Multi-User Evaluation

Evaluate:

- different users;
- different hand sizes;
- different camera distances;
- different viewing angles;
- user-specific personalization;
- cross-user generalization.

---

## Open-Set Evaluation

Measure how well the system distinguishes:

```text
known gesture
```

from:

```text
previously unseen gesture
```

Potential metrics include:

- known-class accuracy;
- unknown rejection rate;
- false acceptance rate;
- false rejection rate;
- AUROC / open-set curves.

---

## Feedback Adaptation

Evaluate recognition:

```text
before user feedback
```

versus:

```text
after positive / corrective feedback
```

to determine whether online personalization produces measurable improvements.

---

## Runtime Evaluation

Measure:

- camera FPS;
- landmark latency;
- recognition latency;
- memory usage;
- storage size;
- effect of asynchronous tracking;
- one-hand vs two-hand cost;
- static vs dynamic cost.

---

## Confidence Calibration

The current confidence value is an index rather than a calibrated probability.

Future work can investigate:

- held-out calibration;
- reliability diagrams;
- Expected Calibration Error;
- conformal prediction;
- calibrated open-set confidence.

---

# Current Limitations

The current project should not yet be interpreted as a complete continuous sign-language translation system.

Current limitations include:

- primary focus is isolated user-defined gestures/signs;
- continuous sentence-level sign segmentation is not implemented;
- facial expressions and other non-manual sign-language cues are not currently modeled;
- body pose/context is not currently part of the main classifier;
- the learned metric encoder still requires stronger formal source/target evaluation;
- EVT thresholds require formal validation;
- dynamic gesture structure still needs deeper refinement;
- confidence is not a calibrated probability;
- formal multi-user benchmark results are still required;
- MediaPipe landmark quality depends on visibility, lighting, camera angle, and occlusion.

---

# Version Summary

| Version | Main Contribution |
|---|---|
| V2 | Online few-shot personalized gesture-learning baseline |
| V3.1 | Geometry-aware hybrid coordinate + angle descriptors |
| V3.2 | EVT-inspired open-set / UNKNOWN recognition |
| V3.3 | Learned metric embedding before prototype recognition |
| V3.4 | Diversity-aware positive and hard-negative exemplar memory |
| V3.5 | DTW-aligned temporal prototypes for dynamic gestures |
| V3.6 | MediaPipe Tasks live-stream asynchronous hand tracking |
| V3.6.1 | Robust hand disappearance and reacquisition handling |
| V3.6.2 | Hands-free one-hand/two-hand teaching workflow |
| V3.6.3 | Responsive/scalable Tkinter desktop interface |

---

# Recommended Current Command

For most users testing the latest research version:

```powershell
git switch v3-geometry-aware-research
uv sync --extra dev
uv run pytest --basetemp=.pytest_tmp
uv run python scripts\interactive_online_learning_v3.py
```

Then teach a gesture directly from the UI.

---

# Project Status

The current V3 branch is feature-rich enough for controlled experiments.

The main next phase is:

```text
formal evaluation
→ ablation studies
→ multi-user testing
→ dynamic-gesture refinement
→ publication-oriented analysis
```

rather than continuously adding new recognition components.

---

## License

See the repository license file, if present, for usage terms.

---

## Acknowledgements

This project builds on open-source and research ideas from areas including:

- MediaPipe hand landmark tracking;
- few-shot metric learning;
- prototypical recognition;
- geometry-aware landmark representations;
- open-set recognition;
- Extreme Value Theory;
- exemplar-based incremental learning;
- Dynamic Time Warping;
- temporal averaging/prototypes.

Specific academic references should be maintained in the project's research documentation and any resulting thesis/paper.
