from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk


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


class InteractiveGestureApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Adaptive Real-Time Hand Gesture Recognition"
        )

        self.root.geometry(
            "1350x780"
        )

        self.root.minsize(
            1100,
            700,
        )

        # -------------------------------------------------
        # Core recognition components
        # -------------------------------------------------

        self.tracker = HandTracker(
            max_num_hands=1
        )

        self.learner = (
            OnlineGestureLearner(
                rejection_multiplier=3.0,
                minimum_threshold=0.05,
            )
        )

        self.selector = None

        # -------------------------------------------------
        # Camera
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Runtime state
        # -------------------------------------------------

        self.current_features = None
        self.current_hand = None
        self.current_prediction = None

        self.teaching = False
        self.teaching_name = None
        self.teaching_handedness = None

        self.prepare_until = None

        # -------------------------------------------------
        # Tkinter variables
        # -------------------------------------------------

        self.gesture_name_var = (
            tk.StringVar()
        )

        self.prediction_var = (
            tk.StringVar(
                value="UNKNOWN"
            )
        )

        self.distance_var = (
            tk.StringVar(
                value="Distance: —"
            )
        )

        self.threshold_var = (
            tk.StringVar(
                value="Threshold: —"
            )
        )

        self.tracking_var = (
            tk.StringVar(
                value="No hand detected"
            )
        )

        self.status_var = (
            tk.StringVar(
                value=(
                    "Enter a gesture name "
                    "and select Teach Gesture."
                )
            )
        )

        self.observed_var = (
            tk.StringVar(
                value="0"
            )
        )

        self.accepted_var = (
            tk.StringVar(
                value="0"
            )
        )

        self.duplicate_var = (
            tk.StringVar(
                value="0"
            )
        )

        self.unstable_var = (
            tk.StringVar(
                value="0"
            )
        )

        self.stability_var = (
            tk.StringVar(
                value="—"
            )
        )

        # -------------------------------------------------
        # Build UI
        # -------------------------------------------------

        self.build_ui()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close,
        )

        self.update_camera()

    # =====================================================
    # UI
    # =====================================================

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

        # -------------------------------------------------
        # Camera side
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Right control panel
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

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
            pady=(3, 3),
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

        ttk.Separator(
            panel
        ).grid(
            row=4,
            column=0,
            sticky="ew",
            pady=18,
        )

        # -------------------------------------------------
        # Teach gesture
        # -------------------------------------------------

        ttk.Label(
            panel,
            text="Teach a New Gesture",
            font=(
                "Segoe UI",
                12,
                "bold",
            ),
        ).grid(
            row=5,
            column=0,
            sticky="w",
        )

        ttk.Label(
            panel,
            text="Gesture name",
        ).grid(
            row=6,
            column=0,
            sticky="w",
            pady=(10, 3),
        )

        self.gesture_entry = ttk.Entry(
            panel,
            textvariable=(
                self.gesture_name_var
            ),
        )

        self.gesture_entry.grid(
            row=7,
            column=0,
            sticky="ew",
        )

        button_frame = ttk.Frame(
            panel
        )

        button_frame.grid(
            row=8,
            column=0,
            sticky="ew",
            pady=(10, 0),
        )

        button_frame.columnconfigure(
            0,
            weight=1,
        )

        button_frame.columnconfigure(
            1,
            weight=1,
        )

        button_frame.columnconfigure(
            2,
            weight=1,
        )

        self.teach_button = ttk.Button(
            button_frame,
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
            button_frame,
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
            button_frame,
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
            row=9,
            column=0,
            sticky="ew",
            pady=(14, 6),
        )

        self.status_label = ttk.Label(
            panel,
            textvariable=self.status_var,
            wraplength=430,
        )

        self.status_label.grid(
            row=10,
            column=0,
            sticky="w",
            pady=(0, 10),
        )

        # -------------------------------------------------
        # Smart capture metrics
        # -------------------------------------------------

        stats_frame = ttk.LabelFrame(
            panel,
            text="Smart Capture",
            padding=10,
        )

        stats_frame.grid(
            row=11,
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
            row=12,
            column=0,
            sticky="ew",
            pady=18,
        )

        # -------------------------------------------------
        # Gesture library
        # -------------------------------------------------

        ttk.Label(
            panel,
            text="Learned Gestures",
            font=(
                "Segoe UI",
                12,
                "bold",
            ),
        ).grid(
            row=13,
            column=0,
            sticky="w",
        )

        self.gesture_table = ttk.Treeview(
            panel,
            columns=(
                "samples",
                "spread",
                "hand",
            ),
            show="headings",
            height=7,
        )

        self.gesture_table.heading(
            "samples",
            text="Gesture / Samples",
        )

        self.gesture_table.heading(
            "spread",
            text="Spread",
        )

        self.gesture_table.heading(
            "hand",
            text="Hand",
        )

        self.gesture_table.column(
            "samples",
            width=180,
        )

        self.gesture_table.column(
            "spread",
            width=90,
        )

        self.gesture_table.column(
            "hand",
            width=80,
        )

        self.gesture_table.grid(
            row=14,
            column=0,
            sticky="nsew",
            pady=(7, 0),
        )

        panel.rowconfigure(
            14,
            weight=1,
        )

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

    # =====================================================
    # Feature extraction
    # =====================================================

    def extract_features(
        self,
        hand,
    ):

        landmarks = (
            hand.world_landmarks
            if hand.world_landmarks
            is not None
            else hand.image_landmarks
        )

        return (
            normalize_hand_landmarks(
                landmarks
            )
        )

    # =====================================================
    # Teaching
    # =====================================================

    def start_teaching(self):

        if self.teaching:
            return

        name = (
            self.gesture_name_var
            .get()
            .strip()
        )

        if not name:

            self.status_var.set(
                "Enter a name for the "
                "gesture first."
            )

            return

        if name in self.learner.gestures:

            self.status_var.set(
                f"'{name}' already exists."
            )

            return

        self.selector = (
            SmartSampleSelector()
        )

        self.teaching = True

        self.teaching_name = name
        self.teaching_handedness = None

        # Give the user two seconds to move
        # into the desired pose.
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

        self.status_var.set(
            f"Get ready to teach '{name}'. "
            "Hold the gesture naturally."
        )

    def process_teaching_frame(self):

        if (
            self.current_features
            is None
            or self.current_hand
            is None
        ):
            return

        now = time.monotonic()

        if (
            self.prepare_until
            is not None
            and now
            < self.prepare_until
        ):

            remaining = (
                self.prepare_until
                - now
            )

            self.status_var.set(
                f"Teaching "
                f"'{self.teaching_name}' "
                f"starts in "
                f"{remaining:.1f}s..."
            )

            return

        if (
            self.teaching_handedness
            is None
        ):
            self.teaching_handedness = (
                self.current_hand.handedness
            )

        if (
            self.current_hand.handedness
            != self.teaching_handedness
        ):

            self.status_var.set(
                "Please continue using "
                f"your "
                f"{self.teaching_handedness} "
                "hand."
            )

            return

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
                    "Enough samples to learn. "
                    "You can finish now, or "
                    "slightly vary the same "
                    "gesture to collect more "
                    "useful examples."
                )

        else:

            self.status_var.set(
                "Keep the gesture steady. "
                "Small natural variations "
                "are useful."
            )

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

        gesture = (
            self.learner.learn_gesture(
                name=self.teaching_name,
                samples=self.selector.samples,
                handedness=(
                    self.teaching_handedness
                ),
            )
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

        self.status_var.set(
            f"Learned '{gesture.name}' "
            f"from {accepted} useful samples. "
            f"{observed} frames observed, "
            f"{duplicates} duplicates and "
            f"{unstable} unstable frames ignored."
        )

        self.finish_teaching_state()

        self.refresh_gesture_table()

    def cancel_teaching(self):

        if not self.teaching:
            return

        self.status_var.set(
            "Teaching cancelled."
        )

        self.finish_teaching_state()

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

        self.gesture_entry.focus_set()

    # =====================================================
    # Prediction
    # =====================================================

    def update_prediction(self):

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

            return

        if self.current_features is None:

            self.prediction_var.set(
                "NO HAND"
            )

            self.distance_var.set(
                "Distance: —"
            )

            self.threshold_var.set(
                "Threshold: —"
            )

            return

        prediction = (
            self.learner.predict(
                self.current_features,
                handedness=(
                    self.current_hand.handedness
                ),
            )
        )

        self.current_prediction = (
            prediction
        )

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

            return

        self.distance_var.set(
            f"Distance: "
            f"{prediction.distance:.4f}"
        )

        self.threshold_var.set(
            f"Threshold: "
            f"{prediction.threshold:.4f}"
        )

    # =====================================================
    # Statistics
    # =====================================================

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

    # =====================================================
    # Gesture list
    # =====================================================

    def refresh_gesture_table(self):

        for item in (
            self.gesture_table
            .get_children()
        ):
            self.gesture_table.delete(
                item
            )

        for name, gesture in (
            self.learner.gestures.items()
        ):

            self.gesture_table.insert(
                "",
                "end",
                values=(
                    (
                        f"{name} "
                        f"({gesture.sample_count})"
                    ),
                    f"{gesture.spread:.4f}",
                    gesture.handedness
                    or "Any",
                ),
            )

    # =====================================================
    # Camera loop
    # =====================================================

    def update_camera(self):

        success, frame = (
            self.cap.read()
        )

        if success:

            frame = cv2.flip(
                frame,
                1,
            )

            hands = (
                self.tracker.process(
                    frame
                )
            )

            self.tracker.draw(
                frame,
                hands,
            )

            self.current_features = None
            self.current_hand = None

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

            if self.teaching:

                self.process_teaching_frame()

            self.update_prediction()

            # ---------------------------------------------
            # Display camera frame inside Tkinter
            # ---------------------------------------------

            display_frame = (
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )
            )

            image = Image.fromarray(
                display_frame
            )

            # Keep camera reasonably sized.
            image.thumbnail(
                (
                    850,
                    650,
                )
            )

            photo = (
                ImageTk.PhotoImage(
                    image=image
                )
            )

            self.video_label.configure(
                image=photo
            )

            self.video_label.image = (
                photo
            )

        self.root.after(
            15,
            self.update_camera,
        )

    # =====================================================
    # Cleanup
    # =====================================================

    def close(self):

        if (
            self.cap
            and self.cap.isOpened()
        ):
            self.cap.release()

        self.tracker.close()

        cv2.destroyAllWindows()

        self.root.destroy()


def main():

    root = tk.Tk()

    app = InteractiveGestureApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()