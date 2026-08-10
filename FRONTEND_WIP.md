# Frontend Experiments

The recognition and adaptive-learning backend is currently being tested with multiple frontend approaches.

## Current Primary Frontend

### Tkinter
File:

`scripts/interactive_online_learning.py`

The Tkinter frontend is currently the primary working interface and supports the complete implemented recognition pipeline.

## Work-in-Progress Frontends

### Web Frontend
Directory:

`frontend-design-lab/`

Status: **WIP / Experimental**

This frontend explores a modern web-based user interface for the gesture-recognition system. It is currently a design and integration experiment and is not yet the default application frontend.

### PySide6 / Qt Frontend
File:

`scripts/interactive_online_learning_qt.py`

Status: **WIP / Experimental**

This frontend explores a modern native desktop interface using PySide6/Qt while reusing the existing gesture-recognition backend.

It is not yet considered the primary frontend.

## Current Direction

The recognition backend is kept independent from the frontend so that different UI technologies can be evaluated without modifying the core learning and recognition algorithms.

The final frontend framework has not yet been selected.