from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)


from adaptive_gesture.features.normalizer import (
    normalize_hand_landmarks,
)

from adaptive_gesture.learning.online_learner import (
    OnlineGestureLearner,
)

from adaptive_gesture.learning.sample_selector import (
    SmartSampleSelector,
)

from adaptive_gesture.tracking.hand_tracker import (
    HandTracker,
)


# ============================================================
# Main application
# ============================================================


class InteractiveGestureApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Adaptive Real-Time Hand Gesture Recognition"
        )

        self.root.geometry(
            "1450x900"
        )

        self.root.minsize(
            1150,
            760,
        )

        # ====================================================
        # Core recognition components
        # ====================================================

        self.tracker = HandTracker(
            max_num_hands=1
        )

        self.learner = OnlineGestureLearner(
            radius_multiplier=2.5,
            minimum_threshold=0.035,
        )

        self.selector = None

        # ====================================================
        # Camera
        # ====================================================

        self.cap = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW,
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam."
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720,
        )

        # ====================================================
        # Current frame / prediction state
        # ====================================================

        self.current_features = None
        self.current_hand = None
        self.current_prediction = None

        # ====================================================
        # Teaching state
        # ====================================================

        self.teaching = False
        self.teaching_name = None
        self.teaching_handedness = None
        self.prepare_until = None

        # ====================================================
        # Feedback state
        #
        # We freeze the feature vector when the user clicks
        # "Wrong" so moving the hand afterward does not change
        # the example being corrected.
        # ====================================================

        self.feedback_features = None
        self.feedback_predicted_label = None
        self.feedback_nearest_label = None
        self.feedback_handedness = None

        # ====================================================
        # Tkinter variables
        # ====================================================

        self.gesture_name_var = tk.StringVar()

        self.prediction_var = tk.StringVar(
            value="UNKNOWN"
        )

        self.distance_var = tk.StringVar(
            value="Distance: —"
        )

        self.threshold_var = tk.StringVar(
            value="Threshold: —"
        )

        self.relative_distance_var = tk.StringVar(
            value="Relative distance: —"
        )

        self.rejection_reason_var = tk.StringVar(
            value=""
        )

        self.tracking_var = tk.StringVar(
            value="No hand detected"
        )

        self.status_var = tk.StringVar(
            value=(
                "Enter a gesture name and "
                "select Teach Gesture."
            )
        )

        # Smart Capture statistics

        self.observed_var = tk.StringVar(
            value="0"
        )

        self.accepted_var = tk.StringVar(
            value="0"
        )

        self.duplicate_var = tk.StringVar(
            value="0"
        )

        self.unstable_var = tk.StringVar(
            value="0"
        )

        self.stability_var = tk.StringVar(
            value="—"
        )

        # Feedback correction selection

        self.actual_gesture_var = tk.StringVar()

        # ====================================================
        # Build UI
        # ====================================================

        self.build_ui()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close,
        )

        self.update_camera()

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        self.root.columnconfigure(
            0,
            weight=3,
        )

        self.root.columnconfigure(
            1,
            weight=2,
        )

        self.root.rowconfigure(
            0,
            weight=1,
        )

        # ====================================================
        # LEFT SIDE — Camera
        # ====================================================

        camera_frame = ttk.Frame(
            self.root,
            padding=12,
        )

        camera_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        camera_frame.rowconfigure(
            0,
            weight=1,
        )

        camera_frame.columnconfigure(
            0,
            weight=1,
        )

        self.video_label = ttk.Label(
            camera_frame,
            anchor="center",
        )

        self.video_label.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.tracking_label = ttk.Label(
            camera_frame,
            textvariable=self.tracking_var,
            font=(
                "Segoe UI",
                11,
            ),
        )

        self.tracking_label.grid(
            row=1,
            column=0,
            pady=(10, 0),
        )

        # ====================================================
        # RIGHT SIDE — Main control panel
        # ====================================================

        panel = ttk.Frame(
            self.root,
            padding=18,
        )

        panel.grid(
            row=0,
            column=1,
            sticky="nsew",
        )

        panel.columnconfigure(
            0,
            weight=1,
        )

        # ====================================================
        # Current prediction
        # ====================================================

        ttk.Label(
            panel,
            text="Current Prediction",
            font=(
                "Segoe UI",
                11,
                "bold",
            ),
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        self.prediction_label = ttk.Label(
            panel,
            textvariable=self.prediction_var,
            font=(
                "Segoe UI",
                27,
                "bold",
            ),
        )

        self.prediction_label.grid(
            row=1,
            column=0,
            sticky="w",
            pady=(3, 4),
        )

        ttk.Label(
            panel,
            textvariable=self.distance_var,
        ).grid(
            row=2,
            column=0,
            sticky="w",
        )

        ttk.Label(
            panel,
            textvariable=self.threshold_var,
        ).grid(
            row=3,
            column=0,
            sticky="w",
        )

        ttk.Label(
            panel,
            textvariable=self.relative_distance_var,
        ).grid(
            row=4,
            column=0,
            sticky="w",
        )

        ttk.Label(
            panel,
            textvariable=self.rejection_reason_var,
        ).grid(
            row=5,
            column=0,
            sticky="w",
            pady=(2, 0),
        )

        ttk.Separator(
            panel
        ).grid(
            row=6,
            column=0,
            sticky="ew",
            pady=14,
        )

        # ====================================================
        # Teach new gesture
        # ====================================================

        ttk.Label(
            panel,
            text="Teach a New Gesture",
            font=(
                "Segoe UI",
                12,
                "bold",
            ),
        ).grid(
            row=7,
            column=0,
            sticky="w",
        )

        ttk.Label(
            panel,
            text="Gesture name",
        ).grid(
            row=8,
            column=0,
            sticky="w",
            pady=(8, 3),
        )

        self.gesture_entry = ttk.Entry(
            panel,
            textvariable=self.gesture_name_var,
        )

        self.gesture_entry.grid(
            row=9,
            column=0,
            sticky="ew",
        )

        # Allow Enter to start teaching.

        self.gesture_entry.bind(
            "<Return>",
            lambda event: self.start_teaching(),
        )

        teaching_buttons = ttk.Frame(
            panel
        )

        teaching_buttons.grid(
            row=10,
            column=0,
            sticky="ew",
            pady=(10, 0),
        )

        for column in range(3):
            teaching_buttons.columnconfigure(
                column,
                weight=1,
            )

        self.teach_button = ttk.Button(
            teaching_buttons,
            text="Teach Gesture",
            command=self.start_teaching,
        )

        self.teach_button.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )

        self.finish_button = ttk.Button(
            teaching_buttons,
            text="Finish Learning",
            command=self.finish_teaching,
            state="disabled",
        )

        self.finish_button.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=4,
        )

        self.cancel_button = ttk.Button(
            teaching_buttons,
            text="Cancel",
            command=self.cancel_teaching,
            state="disabled",
        )

        self.cancel_button.grid(
            row=0,
            column=2,
            sticky="ew",
            padx=(4, 0),
        )

        self.progress = ttk.Progressbar(
            panel,
            maximum=12,
            value=0,
        )

        self.progress.grid(
            row=11,
            column=0,
            sticky="ew",
            pady=(12, 6),
        )

        self.status_label = ttk.Label(
            panel,
            textvariable=self.status_var,
            wraplength=450,
        )

        self.status_label.grid(
            row=12,
            column=0,
            sticky="w",
            pady=(0, 10),
        )

        # ====================================================
        # Smart Capture statistics
        # ====================================================

        stats_frame = ttk.LabelFrame(
            panel,
            text="Smart Capture",
            padding=10,
        )

        stats_frame.grid(
            row=13,
            column=0,
            sticky="ew",
        )

        for column in range(2):
            stats_frame.columnconfigure(
                column,
                weight=1,
            )

        self.add_stat_row(
            stats_frame,
            0,
            "Frames observed",
            self.observed_var,
        )

        self.add_stat_row(
            stats_frame,
            1,
            "Useful samples",
            self.accepted_var,
        )

        self.add_stat_row(
            stats_frame,
            2,
            "Duplicates ignored",
            self.duplicate_var,
        )

        self.add_stat_row(
            stats_frame,
            3,
            "Unstable ignored",
            self.unstable_var,
        )

        self.add_stat_row(
            stats_frame,
            4,
            "Current stability",
            self.stability_var,
        )

        ttk.Separator(
            panel
        ).grid(
            row=14,
            column=0,
            sticky="ew",
            pady=14,
        )

        # ====================================================
        # Interactive feedback
        # ====================================================

        feedback_frame = ttk.LabelFrame(
            panel,
            text="Interactive Feedback",
            padding=10,
        )

        feedback_frame.grid(
            row=15,
            column=0,
            sticky="ew",
        )

        feedback_frame.columnconfigure(
            0,
            weight=1,
        )

        feedback_frame.columnconfigure(
            1,
            weight=1,
        )

        self.correct_button = ttk.Button(
            feedback_frame,
            text="✓ Correct",
            command=self.confirm_prediction,
        )

        self.correct_button.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )

        self.wrong_button = ttk.Button(
            feedback_frame,
            text="✕ Wrong",
            command=self.begin_correction,
        )

        self.wrong_button.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(4, 0),
        )

        # ----------------------------------------------------
        # Correction controls
        # ----------------------------------------------------

        self.correction_frame = ttk.Frame(
            feedback_frame
        )

        self.correction_frame.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )

        self.correction_frame.columnconfigure(
            0,
            weight=1,
        )

        ttk.Label(
            self.correction_frame,
            text="What was the actual gesture?",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        self.actual_gesture_combo = ttk.Combobox(
            self.correction_frame,
            textvariable=self.actual_gesture_var,
            state="readonly",
        )

        self.actual_gesture_combo.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(4, 7),
        )

        self.apply_correction_button = ttk.Button(
            self.correction_frame,
            text="Apply Correction",
            command=self.apply_feedback_correction,
        )

        self.apply_correction_button.grid(
            row=2,
            column=0,
            sticky="ew",
        )

        self.mark_unknown_button = ttk.Button(
            self.correction_frame,
            text="This Gesture Is Unknown",
            command=self.mark_feedback_unknown,
        )

        self.mark_unknown_button.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(5, 0),
        )

        self.cancel_feedback_button = ttk.Button(
            self.correction_frame,
            text="Cancel Feedback",
            command=self.cancel_feedback,
        )

        self.cancel_feedback_button.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(5, 0),
        )

        # Hidden until Wrong is clicked.

        self.correction_frame.grid_remove()

        ttk.Separator(
            panel
        ).grid(
            row=16,
            column=0,
            sticky="ew",
            pady=14,
        )

        # ====================================================
        # Gesture library
        # ====================================================

        ttk.Label(
            panel,
            text="Learned Gestures",
            font=(
                "Segoe UI",
                12,
                "bold",
            ),
        ).grid(
            row=17,
            column=0,
            sticky="w",
        )

        self.gesture_table = ttk.Treeview(
            panel,
            columns=(
                "gesture",
                "positive",
                "negative",
                "spread",
                "radius",
                "hand",
            ),
            show="headings",
            height=7,
        )

        self.gesture_table.heading(
            "gesture",
            text="Gesture",
        )

        self.gesture_table.heading(
            "positive",
            text="+",
        )

        self.gesture_table.heading(
            "negative",
            text="-",
        )

        self.gesture_table.heading(
            "spread",
            text="Spread",
        )

        self.gesture_table.heading(
            "radius",
            text="Radius",
        )

        self.gesture_table.heading(
            "hand",
            text="Hand",
        )

        self.gesture_table.column(
            "gesture",
            width=120,
            anchor="w",
        )

        self.gesture_table.column(
            "positive",
            width=45,
            anchor="center",
        )

        self.gesture_table.column(
            "negative",
            width=45,
            anchor="center",
        )

        self.gesture_table.column(
            "spread",
            width=75,
            anchor="center",
        )

        self.gesture_table.column(
            "radius",
            width=75,
            anchor="center",
        )

        self.gesture_table.column(
            "hand",
            width=70,
            anchor="center",
        )

        self.gesture_table.grid(
            row=18,
            column=0,
            sticky="nsew",
            pady=(7, 0),
        )

        panel.rowconfigure(
            18,
            weight=1,
        )

    # ========================================================
    # UI helper
    # ========================================================

    def add_stat_row(
        self,
        parent,
        row,
        label,
        variable,
    ):

        ttk.Label(
            parent,
            text=label,
        ).grid(
            row=row,
            column=0,
            sticky="w",
            pady=2,
        )

        ttk.Label(
            parent,
            textvariable=variable,
        ).grid(
            row=row,
            column=1,
            sticky="e",
            pady=2,
        )

    # ========================================================
    # Feature extraction
    # ========================================================

    def extract_features(
        self,
        hand,
    ):

        landmarks = (
            hand.world_landmarks
            if hand.world_landmarks is not None
            else hand.image_landmarks
        )

        return normalize_hand_landmarks(
            landmarks
        )

    # ========================================================
    # Teaching
    # ========================================================

    def start_teaching(self):

        if self.teaching:
            return

        # Exit correction mode if currently open.

        self.finish_feedback()

        name = (
            self.gesture_name_var
            .get()
            .strip()
        )

        if not name:

            self.status_var.set(
                "Enter a name for the gesture first."
            )

            self.gesture_entry.focus_set()

            return

        if name in self.learner.gestures:

            self.status_var.set(
                f"'{name}' already exists."
            )

            return

        self.selector = SmartSampleSelector()

        self.progress[
            "maximum"
        ] = self.selector.target_samples

        self.teaching = True

        self.teaching_name = name
        self.teaching_handedness = None

        # Give the user time to position the hand.

        self.prepare_until = (
            time.monotonic()
            + 2.0
        )

        self.progress[
            "value"
        ] = 0

        self.reset_stats_display()

        self.teach_button.config(
            state="disabled"
        )

        self.finish_button.config(
            state="disabled"
        )

        self.cancel_button.config(
            state="normal"
        )

        self.gesture_entry.config(
            state="disabled"
        )

        self.set_feedback_buttons_enabled(
            False
        )

        self.status_var.set(
            f"Get ready to teach '{name}'. "
            "Hold the gesture naturally."
        )

    def process_teaching_frame(self):

        if (
            self.current_features is None
            or self.current_hand is None
        ):
            self.status_var.set(
                "Waiting for a hand..."
            )

            return

        now = time.monotonic()

        # ----------------------------------------------------
        # Preparation countdown
        # ----------------------------------------------------

        if (
            self.prepare_until is not None
            and now < self.prepare_until
        ):

            remaining = (
                self.prepare_until
                - now
            )

            self.status_var.set(
                f"Teaching '{self.teaching_name}' "
                f"starts in {remaining:.1f}s..."
            )

            return

        # ----------------------------------------------------
        # Lock teaching to the first detected handedness
        # ----------------------------------------------------

        if self.teaching_handedness is None:

            self.teaching_handedness = (
                self.current_hand.handedness
            )

        if (
            self.current_hand.handedness
            != self.teaching_handedness
        ):

            self.status_var.set(
                "Please continue using your "
                f"{self.teaching_handedness} hand."
            )

            return

        # ----------------------------------------------------
        # Smart sample selection
        # ----------------------------------------------------

        self.selector.consider(
            self.current_features
        )

        self.update_stats_display()

        self.progress[
            "value"
        ] = len(
            self.selector.samples
        )

        if self.selector.ready:

            self.finish_button.config(
                state="normal"
            )

            if not self.selector.complete:

                self.status_var.set(
                    "Enough useful samples to learn. "
                    "You can finish now, or make small "
                    "natural variations of the SAME "
                    "gesture to collect more examples."
                )

        else:

            self.status_var.set(
                "Keep the gesture steady. "
                "Small natural variations are useful."
            )

        # Automatically finish at the target.

        if self.selector.complete:
            self.finish_teaching()

    def finish_teaching(self):

        if not self.teaching:
            return

        if (
            self.selector is None
            or not self.selector.ready
        ):

            self.status_var.set(
                "Not enough useful samples yet."
            )

            return

        gesture = self.learner.learn_gesture(
            name=self.teaching_name,
            samples=self.selector.samples,
            handedness=self.teaching_handedness,
        )

        observed = (
            self.selector.stats.observed
        )

        accepted = (
            self.selector.stats.accepted
        )

        duplicates = (
            self.selector.stats.duplicates
        )

        unstable = (
            self.selector.stats.unstable
        )

        learned_name = gesture.name

        self.finish_teaching_state()

        self.refresh_gesture_table()

        self.status_var.set(
            f"✓ Learned '{learned_name}' from "
            f"{accepted} useful samples. "
            f"{observed} frames observed; "
            f"{duplicates} duplicates and "
            f"{unstable} unstable frames ignored."
        )

    def cancel_teaching(self):

        if not self.teaching:
            return

        self.finish_teaching_state()

        self.status_var.set(
            "Teaching cancelled."
        )

    def finish_teaching_state(self):

        self.teaching = False
        self.teaching_name = None
        self.teaching_handedness = None
        self.prepare_until = None

        self.selector = None

        self.progress[
            "value"
        ] = 0

        self.teach_button.config(
            state="normal"
        )

        self.finish_button.config(
            state="disabled"
        )

        self.cancel_button.config(
            state="disabled"
        )

        self.gesture_entry.config(
            state="normal"
        )

        self.gesture_name_var.set(
            ""
        )

        self.set_feedback_buttons_enabled(
            True
        )

        self.gesture_entry.focus_set()

    # ========================================================
    # Prediction
    # ========================================================

    def update_prediction(self):

        # ----------------------------------------------------
        # No recognition during teaching
        # ----------------------------------------------------

        if self.teaching:

            self.prediction_var.set(
                f"Teaching: {self.teaching_name}"
            )

            self.distance_var.set(
                "Distance: —"
            )

            self.threshold_var.set(
                "Threshold: —"
            )

            self.relative_distance_var.set(
                "Relative distance: —"
            )

            self.rejection_reason_var.set(
                ""
            )

            return

        # ----------------------------------------------------
        # No hand
        # ----------------------------------------------------

        if self.current_features is None:

            self.current_prediction = None

            self.prediction_var.set(
                "NO HAND"
            )

            self.distance_var.set(
                "Distance: —"
            )

            self.threshold_var.set(
                "Threshold: —"
            )

            self.relative_distance_var.set(
                "Relative distance: —"
            )

            self.rejection_reason_var.set(
                ""
            )

            return

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        prediction = self.learner.predict(
            self.current_features,
            handedness=(
                self.current_hand.handedness
            ),
        )

        self.current_prediction = prediction

        self.prediction_var.set(
            prediction.label
        )

        if prediction.distance is None:

            self.distance_var.set(
                "Distance: —"
            )

            self.threshold_var.set(
                "Threshold: —"
            )

            self.relative_distance_var.set(
                "Relative distance: —"
            )

            self.rejection_reason_var.set(
                ""
            )

            return

        self.distance_var.set(
            f"Nearest positive distance: "
            f"{prediction.distance:.4f}"
        )

        self.threshold_var.set(
            f"Acceptance threshold: "
            f"{prediction.threshold:.4f}"
        )

        if prediction.relative_distance is None:

            self.relative_distance_var.set(
                "Relative distance: —"
            )

        else:

            self.relative_distance_var.set(
                "Relative distance: "
                f"{prediction.relative_distance:.2f}x"
            )

        # ----------------------------------------------------
        # Explain UNKNOWN in the development UI
        # ----------------------------------------------------

        if prediction.accepted:

            self.rejection_reason_var.set(
                ""
            )

        else:

            if (
                prediction.rejection_reason
                == "outside_positive_region"
            ):

                nearest = (
                    prediction.nearest_label
                    or "known gesture"
                )

                self.rejection_reason_var.set(
                    f"Outside learned region "
                    f"for '{nearest}'"
                )

            elif (
                prediction.rejection_reason
                == "hard_negative"
            ):

                self.rejection_reason_var.set(
                    "Rejected using learned "
                    "negative feedback"
                )

            elif (
                prediction.rejection_reason
                == "no_gestures"
            ):

                self.rejection_reason_var.set(
                    "No gestures learned yet"
                )

            elif (
                prediction.rejection_reason
                == "handedness"
            ):

                self.rejection_reason_var.set(
                    "No compatible gesture "
                    "for this hand"
                )

            else:

                self.rejection_reason_var.set(
                    ""
                )

    # ========================================================
    # Interactive feedback
    # ========================================================

    def set_feedback_buttons_enabled(
        self,
        enabled,
    ):

        state = (
            "normal"
            if enabled
            else "disabled"
        )

        self.correct_button.config(
            state=state
        )

        self.wrong_button.config(
            state=state
        )

    # --------------------------------------------------------
    # CORRECT
    # --------------------------------------------------------

    def confirm_prediction(self):

        if self.teaching:
            return

        if (
            self.current_features is None
            or self.current_prediction is None
        ):

            self.status_var.set(
                "No gesture available to confirm."
            )

            return

        prediction = (
            self.current_prediction
        )

        if not prediction.accepted:

            self.status_var.set(
                "The current pose is UNKNOWN. "
                "There is no known prediction "
                "to confirm."
            )

            return

        # Add the current pose as new positive evidence
        # for the predicted class.

        gesture = self.learner.update_gesture(
            prediction.label,
            self.current_features.copy(),
        )

        self.status_var.set(
            f"✓ Confirmed '{gesture.name}'. "
            "The recognizer learned this "
            "example as additional positive "
            "evidence."
        )

        self.refresh_gesture_table()

    # --------------------------------------------------------
    # WRONG
    # --------------------------------------------------------

    def begin_correction(self):

        if self.teaching:
            return

        if (
            self.current_features is None
            or self.current_prediction is None
        ):

            self.status_var.set(
                "No current pose available "
                "for correction."
            )

            return

        # Freeze exactly the pose that was present when
        # Wrong was clicked.

        self.feedback_features = (
            self.current_features.copy()
        )

        self.feedback_handedness = (
            self.current_hand.handedness
            if self.current_hand is not None
            else None
        )

        prediction = (
            self.current_prediction
        )

        self.feedback_nearest_label = (
            prediction.nearest_label
        )

        # Only count it as an actual wrong prediction if
        # the recognizer ACCEPTED the class.
        #
        # If it already said UNKNOWN, correcting it to a
        # known class should create a positive example but
        # should not create a hard negative for some merely
        # nearby class.

        if prediction.accepted:

            self.feedback_predicted_label = (
                prediction.label
            )

        else:

            self.feedback_predicted_label = None

        gesture_names = (
            self.learner.list_gestures()
        )

        self.actual_gesture_combo[
            "values"
        ] = gesture_names

        self.actual_gesture_var.set(
            ""
        )

        self.correction_frame.grid()

        self.set_feedback_buttons_enabled(
            False
        )

        if prediction.accepted:

            self.status_var.set(
                "Correction mode: captured "
                f"prediction was "
                f"'{prediction.label}'. "
                "Select the actual gesture, "
                "or mark this pose as unknown."
            )

        else:

            nearest_text = (
                f" Closest known class was "
                f"'{prediction.nearest_label}'."
                if prediction.nearest_label
                else ""
            )

            self.status_var.set(
                "Correction mode: the system "
                "classified this pose as "
                f"UNKNOWN.{nearest_text}"
            )

    # --------------------------------------------------------
    # Apply known-class correction
    # --------------------------------------------------------

    def apply_feedback_correction(self):

        if self.feedback_features is None:
            return

        actual_label = (
            self.actual_gesture_var
            .get()
            .strip()
        )

        if not actual_label:

            self.status_var.set(
                "Select the actual gesture first."
            )

            return

        predicted_label = (
            self.feedback_predicted_label
        )

        self.learner.apply_correction(
            predicted_label=predicted_label,
            actual_label=actual_label,
            sample=self.feedback_features,
        )

        if (
            predicted_label is not None
            and predicted_label
            != actual_label
        ):

            message = (
                f"✓ Learned correction: "
                f"this example is "
                f"'{actual_label}', not "
                f"'{predicted_label}'. "
                f"It was added as positive "
                f"evidence for '{actual_label}' "
                f"and negative evidence for "
                f"'{predicted_label}'."
            )

        else:

            message = (
                f"✓ Learned that this UNKNOWN "
                f"example belongs to "
                f"'{actual_label}'."
            )

        self.finish_feedback()

        self.refresh_gesture_table()

        self.status_var.set(
            message
        )

    # --------------------------------------------------------
    # Mark as unknown
    # --------------------------------------------------------

    def mark_feedback_unknown(self):

        if self.feedback_features is None:
            return

        predicted_label = (
            self.feedback_predicted_label
        )

        # Only add a hard negative if the system actually
        # accepted a known gesture incorrectly.

        self.learner.mark_unknown(
            predicted_label=predicted_label,
            sample=self.feedback_features,
        )

        self.finish_feedback()

        self.refresh_gesture_table()

        if predicted_label:

            self.status_var.set(
                f"✓ Learned that this pose is "
                f"NOT '{predicted_label}'. "
                "It has been stored as a "
                "hard-negative example."
            )

        else:

            self.status_var.set(
                "The pose was already UNKNOWN. "
                "No known class required a "
                "negative correction."
            )

    # --------------------------------------------------------
    # Cancel feedback
    # --------------------------------------------------------

    def cancel_feedback(self):

        self.finish_feedback()

        self.status_var.set(
            "Feedback cancelled."
        )

    def finish_feedback(self):

        self.feedback_features = None
        self.feedback_predicted_label = None
        self.feedback_nearest_label = None
        self.feedback_handedness = None

        self.actual_gesture_var.set(
            ""
        )

        if hasattr(
            self,
            "correction_frame",
        ):

            self.correction_frame.grid_remove()

        if (
            hasattr(
                self,
                "correct_button",
            )
            and not self.teaching
        ):

            self.set_feedback_buttons_enabled(
                True
            )

    # ========================================================
    # Smart Capture statistics
    # ========================================================

    def reset_stats_display(self):

        self.observed_var.set(
            "0"
        )

        self.accepted_var.set(
            "0"
        )

        self.duplicate_var.set(
            "0"
        )

        self.unstable_var.set(
            "0"
        )

        self.stability_var.set(
            "—"
        )

    def update_stats_display(self):

        if self.selector is None:
            return

        stats = (
            self.selector.stats
        )

        self.observed_var.set(
            str(stats.observed)
        )

        self.accepted_var.set(
            str(stats.accepted)
        )

        self.duplicate_var.set(
            str(stats.duplicates)
        )

        self.unstable_var.set(
            str(stats.unstable)
        )

        if (
            self.selector.last_stability
            is None
        ):

            self.stability_var.set(
                "—"
            )

        else:

            self.stability_var.set(
                f"{self.selector.last_stability:.4f}"
            )

    # ========================================================
    # Gesture table
    # ========================================================

    def refresh_gesture_table(self):

        for item in (
            self.gesture_table.get_children()
        ):

            self.gesture_table.delete(
                item
            )

        for (
            name,
            gesture,
        ) in self.learner.gestures.items():

            self.gesture_table.insert(
                "",
                "end",
                values=(
                    name,
                    gesture.sample_count,
                    gesture.negative_count,
                    f"{gesture.spread:.4f}",
                    f"{gesture.sample_radius:.4f}",
                    gesture.handedness
                    or "Any",
                ),
            )

        # Refresh correction dropdown as well.

        self.actual_gesture_combo[
            "values"
        ] = self.learner.list_gestures()

    # ========================================================
    # Camera loop
    # ========================================================

    def update_camera(self):

        success, frame = (
            self.cap.read()
        )

        if success:

            frame = cv2.flip(
                frame,
                1,
            )

            hands = self.tracker.process(
                frame
            )

            self.tracker.draw(
                frame,
                hands,
            )

            self.current_features = None
            self.current_hand = None

            # ------------------------------------------------
            # Current hand
            # ------------------------------------------------

            if hands:

                self.current_hand = (
                    hands[0]
                )

                self.current_features = (
                    self.extract_features(
                        self.current_hand
                    )
                )

                self.tracking_var.set(
                    f"Tracking "
                    f"{self.current_hand.handedness} "
                    f"hand — "
                    f"{self.current_hand.handedness_score:.0%}"
                )

            else:

                self.tracking_var.set(
                    "No hand detected"
                )

            # ------------------------------------------------
            # Teaching
            # ------------------------------------------------

            if self.teaching:
                self.process_teaching_frame()

            # ------------------------------------------------
            # Recognition
            # ------------------------------------------------

            self.update_prediction()

            # ------------------------------------------------
            # Camera image → Tkinter
            # ------------------------------------------------

            display_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(
                display_frame
            )

            image.thumbnail(
                (
                    850,
                    650,
                )
            )

            photo = ImageTk.PhotoImage(
                image=image
            )

            self.video_label.configure(
                image=photo
            )

            # Keep reference so Tkinter doesn't garbage
            # collect the image.

            self.video_label.image = (
                photo
            )

        self.root.after(
            15,
            self.update_camera,
        )

    # ========================================================
    # Cleanup
    # ========================================================

    def close(self):

        if (
            self.cap is not None
            and self.cap.isOpened()
        ):

            self.cap.release()

        self.tracker.close()

        cv2.destroyAllWindows()

        self.root.destroy()


# ============================================================
# Entry point
# ============================================================


def main():

    root = tk.Tk()

    InteractiveGestureApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()