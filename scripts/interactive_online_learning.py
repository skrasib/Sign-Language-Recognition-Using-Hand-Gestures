from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from adaptive_gesture.features.hand_features import build_frame_features
from adaptive_gesture.learning.online_learner import OnlineGestureLearner
from adaptive_gesture.learning.sample_selector import SmartSampleSelector
from adaptive_gesture.storage.gesture_store import GestureStore
from adaptive_gesture.tracking.hand_tracker import HandTracker


class InteractiveGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive Real-Time Hand Gesture Recognition")
        self.root.geometry("1450x900")
        self.root.minsize(1150, 760)

        # Core engine.
        self.tracker = HandTracker(max_num_hands=2)
        self.learner = OnlineGestureLearner(
            radius_multiplier=2.5,
            minimum_threshold=0.035,
            prototype_multiplier=2.2,
            max_prototypes=3,
        )
        self.selector = None

        # Persistent numerical gesture memory.
        self.gesture_store = GestureStore(
            PROJECT_ROOT / "data" / "gesture_memory.json"
        )
        self.restore_error = None
        try:
            self.restored_gestures = self.gesture_store.load_into(self.learner)
        except Exception as error:
            self.restored_gestures = 0
            self.restore_error = str(error)

        # Camera.
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        # Runtime input state.
        self.current_feature_set = None
        self.current_features = None
        self.current_hand_signature = None
        self.current_prediction = None
        self.current_hands = []

        # Keep the camera preview smooth. Recognition/UI metrics do not need
        # to be recalculated at camera-frame rate. Updating those values less
        # frequently also prevents Tkinter from redrawing several changing
        # labels on every frame in one-hand mode.
        self.prediction_update_interval = 0.08
        self.last_prediction_update = 0.0

        # Teaching state.
        self.teaching = False
        self.teaching_mode = None  # new | improve | retrain
        self.teaching_name = None
        self.teaching_signature = None
        self.required_teaching_signature = None
        self.prepare_until = None

        # Feedback state.
        self.feedback_features = None
        self.feedback_predicted_label = None
        self.feedback_nearest_label = None
        self.feedback_hand_signature = None

        # Tk variables.
        self.gesture_name_var = tk.StringVar()
        self.prediction_var = tk.StringVar(value="UNKNOWN")
        self.distance_var = tk.StringVar(value="Distance: —")
        self.threshold_var = tk.StringVar(value="Threshold: —")
        self.relative_distance_var = tk.StringVar(value="Relative distance: —")
        self.rejection_reason_var = tk.StringVar(value="")
        self.tracking_var = tk.StringVar(value="No hand detected")
        self.status_var = tk.StringVar(value="")

        self.observed_var = tk.StringVar(value="0")
        self.accepted_var = tk.StringVar(value="0")
        self.duplicate_var = tk.StringVar(value="0")
        self.unstable_var = tk.StringVar(value="0")
        self.stability_var = tk.StringVar(value="—")
        self.input_mode_var = tk.StringVar(value="—")

        self.actual_gesture_var = tk.StringVar()

        self.build_ui()
        self.refresh_gesture_table()

        if self.restore_error:
            self.status_var.set(
                "Gesture memory could not be restored: " + self.restore_error
            )
        elif self.restored_gestures > 0:
            suffix = "s" if self.restored_gestures != 1 else ""
            self.status_var.set(
                f"✓ Restored {self.restored_gestures} learned gesture{suffix} "
                "from previous sessions."
            )
        else:
            self.status_var.set(
                "No saved gestures yet. Teach a new gesture to begin."
            )

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_camera()

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        # Camera area.
        camera_frame = ttk.Frame(self.root, padding=12)
        camera_frame.grid(row=0, column=0, sticky="nsew")
        camera_frame.rowconfigure(0, weight=1)
        camera_frame.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(camera_frame, anchor="center")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        self.tracking_label = ttk.Label(
            camera_frame,
            textvariable=self.tracking_var,
            font=("Segoe UI", 11),
        )
        self.tracking_label.grid(row=1, column=0, pady=(10, 0))

        # Right-side tabs keep the application usable on smaller screens.
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)

        self.live_tab = ttk.Frame(notebook, padding=16)
        self.library_tab = ttk.Frame(notebook, padding=16)
        notebook.add(self.live_tab, text="Live")
        notebook.add(self.library_tab, text="Gesture Library")

        self.live_tab.columnconfigure(0, weight=1)
        self.library_tab.columnconfigure(0, weight=1)
        self.library_tab.rowconfigure(1, weight=1)

        self.build_live_tab(self.live_tab)
        self.build_library_tab(self.library_tab)

    def build_live_tab(self, panel):
        ttk.Label(
            panel,
            text="Current Prediction",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            panel,
            textvariable=self.prediction_var,
            font=("Segoe UI", 27, "bold"),
        ).grid(row=1, column=0, sticky="w", pady=(3, 4))

        ttk.Label(panel, textvariable=self.distance_var).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(panel, textvariable=self.threshold_var).grid(
            row=3, column=0, sticky="w"
        )
        ttk.Label(panel, textvariable=self.relative_distance_var).grid(
            row=4, column=0, sticky="w"
        )
        ttk.Label(panel, textvariable=self.rejection_reason_var).grid(
            row=5, column=0, sticky="w", pady=(2, 0)
        )

        ttk.Separator(panel).grid(row=6, column=0, sticky="ew", pady=14)

        # New gesture teaching.
        ttk.Label(
            panel,
            text="Teach a New Gesture",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=7, column=0, sticky="w")

        ttk.Label(panel, text="Gesture name").grid(
            row=8, column=0, sticky="w", pady=(8, 3)
        )

        self.gesture_entry = ttk.Entry(panel, textvariable=self.gesture_name_var)
        self.gesture_entry.grid(row=9, column=0, sticky="ew")
        self.gesture_entry.bind("<Return>", lambda event: self.start_new_teaching())

        teaching_buttons = ttk.Frame(panel)
        teaching_buttons.grid(row=10, column=0, sticky="ew", pady=(10, 0))
        for column in range(3):
            teaching_buttons.columnconfigure(column, weight=1)

        self.teach_button = ttk.Button(
            teaching_buttons,
            text="Teach Gesture",
            command=self.start_new_teaching,
        )
        self.teach_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.finish_button = ttk.Button(
            teaching_buttons,
            text="Finish Learning",
            command=self.finish_teaching,
            state="disabled",
        )
        self.finish_button.grid(row=0, column=1, sticky="ew", padx=4)

        self.cancel_button = ttk.Button(
            teaching_buttons,
            text="Cancel",
            command=self.cancel_teaching,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        self.progress = ttk.Progressbar(panel, maximum=12, value=0)
        self.progress.grid(row=11, column=0, sticky="ew", pady=(12, 6))

        ttk.Label(
            panel,
            textvariable=self.status_var,
            wraplength=440,
        ).grid(row=12, column=0, sticky="w", pady=(0, 10))

        # Smart capture.
        stats_frame = ttk.LabelFrame(panel, text="Smart Capture", padding=10)
        stats_frame.grid(row=13, column=0, sticky="ew")
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(1, weight=1)

        self.add_stat_row(stats_frame, 0, "Frames observed", self.observed_var)
        self.add_stat_row(stats_frame, 1, "Useful samples", self.accepted_var)
        self.add_stat_row(stats_frame, 2, "Duplicates ignored", self.duplicate_var)
        self.add_stat_row(stats_frame, 3, "Unstable ignored", self.unstable_var)
        self.add_stat_row(stats_frame, 4, "Current stability", self.stability_var)
        self.add_stat_row(stats_frame, 5, "Input mode", self.input_mode_var)

        ttk.Separator(panel).grid(row=14, column=0, sticky="ew", pady=14)

        # Interactive feedback.
        feedback_frame = ttk.LabelFrame(
            panel,
            text="Interactive Feedback",
            padding=10,
        )
        feedback_frame.grid(row=15, column=0, sticky="ew")
        feedback_frame.columnconfigure(0, weight=1)
        feedback_frame.columnconfigure(1, weight=1)

        self.correct_button = ttk.Button(
            feedback_frame,
            text="✓ Correct",
            command=self.confirm_prediction,
        )
        self.correct_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.wrong_button = ttk.Button(
            feedback_frame,
            text="✕ Wrong",
            command=self.begin_correction,
        )
        self.wrong_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.correction_frame = ttk.Frame(feedback_frame)
        self.correction_frame.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )
        self.correction_frame.columnconfigure(0, weight=1)

        ttk.Label(
            self.correction_frame,
            text="What was the actual gesture?",
        ).grid(row=0, column=0, sticky="w")

        self.actual_gesture_combo = ttk.Combobox(
            self.correction_frame,
            textvariable=self.actual_gesture_var,
            state="readonly",
        )
        self.actual_gesture_combo.grid(row=1, column=0, sticky="ew", pady=(4, 7))

        ttk.Button(
            self.correction_frame,
            text="Apply Correction",
            command=self.apply_feedback_correction,
        ).grid(row=2, column=0, sticky="ew")

        ttk.Button(
            self.correction_frame,
            text="This Gesture Is Unknown",
            command=self.mark_feedback_unknown,
        ).grid(row=3, column=0, sticky="ew", pady=(5, 0))

        ttk.Button(
            self.correction_frame,
            text="Cancel Feedback",
            command=self.cancel_feedback,
        ).grid(row=4, column=0, sticky="ew", pady=(5, 0))

        self.correction_frame.grid_remove()

    def build_library_tab(self, panel):
        ttk.Label(
            panel,
            text="Learned Gestures",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.gesture_table = ttk.Treeview(
            panel,
            columns=(
                "gesture",
                "positive",
                "negative",
                "prototypes",
                "spread",
                "radius",
                "input",
            ),
            show="headings",
            height=15,
        )
        headings = {
            "gesture": "Gesture",
            "positive": "+",
            "negative": "-",
            "prototypes": "P",
            "spread": "Spread",
            "radius": "Radius",
            "input": "Input",
        }
        widths = {
            "gesture": 130,
            "positive": 40,
            "negative": 40,
            "prototypes": 40,
            "spread": 70,
            "radius": 70,
            "input": 70,
        }
        for column, heading in headings.items():
            self.gesture_table.heading(column, text=heading)
            self.gesture_table.column(
                column,
                width=widths[column],
                anchor="w" if column == "gesture" else "center",
            )

        self.gesture_table.grid(row=1, column=0, sticky="nsew")

        management = ttk.LabelFrame(panel, text="Gesture Management", padding=10)
        management.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        for column in range(2):
            management.columnconfigure(column, weight=1)

        ttk.Button(
            management,
            text="Rename",
            command=self.rename_selected_gesture,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 5))

        ttk.Button(
            management,
            text="Delete",
            command=self.delete_selected_gesture,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 5))

        ttk.Button(
            management,
            text="Improve (+ samples)",
            command=self.improve_selected_gesture,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=5)

        ttk.Button(
            management,
            text="Retrain (replace samples)",
            command=self.retrain_selected_gesture,
        ).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=5)

        ttk.Button(
            management,
            text="Clear All Gesture Memory",
            command=self.clear_all_gestures,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        ttk.Label(
            panel,
            text=(
                "P = number of adaptive prototypes. Improve adds new useful "
                "examples; Retrain replaces positive examples while preserving "
                "compatible negative feedback."
            ),
            wraplength=440,
        ).grid(row=3, column=0, sticky="w", pady=(10, 0))

    def add_stat_row(self, parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, textvariable=variable).grid(
            row=row, column=1, sticky="e", pady=2
        )

    # ========================================================
    # Persistence
    # ========================================================

    def save_gesture_memory(self):
        try:
            self.gesture_store.save(self.learner)
            return True
        except Exception as error:
            self.status_var.set(
                "Warning: learning succeeded but gesture memory could not be "
                f"saved: {error}"
            )
            return False

    # ========================================================
    # Gesture management
    # ========================================================

    def get_selected_gesture_name(self):
        selection = self.gesture_table.selection()
        if not selection:
            messagebox.showinfo(
                "Select Gesture",
                "Select a gesture in the Gesture Library first.",
                parent=self.root,
            )
            return None

        values = self.gesture_table.item(selection[0], "values")
        return values[0] if values else None

    def rename_selected_gesture(self):
        name = self.get_selected_gesture_name()
        if not name:
            return

        new_name = simpledialog.askstring(
            "Rename Gesture",
            f"New name for '{name}':",
            initialvalue=name,
            parent=self.root,
        )
        if new_name is None:
            return

        try:
            gesture = self.learner.rename_gesture(name, new_name)
            self.save_gesture_memory()
            self.refresh_gesture_table()
            self.status_var.set(f"✓ Renamed '{name}' to '{gesture.name}'.")
        except Exception as error:
            messagebox.showerror("Rename Failed", str(error), parent=self.root)

    def delete_selected_gesture(self):
        name = self.get_selected_gesture_name()
        if not name:
            return

        if not messagebox.askyesno(
            "Delete Gesture",
            f"Delete '{name}' and all of its learned positive/negative memory?",
            parent=self.root,
        ):
            return

        self.learner.delete_gesture(name)
        self.save_gesture_memory()
        self.refresh_gesture_table()
        self.status_var.set(f"Deleted '{name}'.")

    def clear_all_gestures(self):
        if not self.learner.gestures:
            self.status_var.set("Gesture memory is already empty.")
            return

        if not messagebox.askyesno(
            "Clear All Gesture Memory",
            "Delete ALL learned gestures and feedback memory? This cannot be undone.",
            parent=self.root,
        ):
            return

        self.learner.clear()
        self.save_gesture_memory()
        self.refresh_gesture_table()
        self.status_var.set("All gesture memory cleared.")

    def improve_selected_gesture(self):
        name = self.get_selected_gesture_name()
        if not name:
            return
        gesture = self.learner.gestures[name]
        self.begin_teaching(
            name=name,
            mode="improve",
            required_signature=gesture.hand_signature,
        )

    def retrain_selected_gesture(self):
        name = self.get_selected_gesture_name()
        if not name:
            return

        gesture = self.learner.gestures[name]
        if not messagebox.askyesno(
            "Retrain Gesture",
            f"Replace the positive examples for '{name}' with a new live "
            "demonstration? Compatible negative feedback will be preserved.",
            parent=self.root,
        ):
            return

        self.begin_teaching(
            name=name,
            mode="retrain",
            required_signature=gesture.hand_signature,
        )

    # ========================================================
    # Teaching
    # ========================================================

    def start_new_teaching(self):
        name = self.gesture_name_var.get().strip()
        if not name:
            self.status_var.set("Enter a name for the gesture first.")
            self.gesture_entry.focus_set()
            return
        if name in self.learner.gestures:
            self.status_var.set(
                f"'{name}' already exists. Use Improve or Retrain in Gesture Library."
            )
            return
        self.begin_teaching(name=name, mode="new", required_signature=None)

    def begin_teaching(self, name, mode, required_signature):
        if self.teaching:
            return

        self.finish_feedback()
        self.selector = SmartSampleSelector()
        self.progress["maximum"] = self.selector.target_samples

        self.teaching = True
        self.teaching_mode = mode
        self.teaching_name = name
        self.teaching_signature = None
        self.required_teaching_signature = required_signature
        self.prepare_until = time.monotonic() + 2.0

        self.progress["value"] = 0
        self.reset_stats_display()
        self.teach_button.config(state="disabled")
        self.finish_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.gesture_entry.config(state="disabled")
        self.set_feedback_buttons_enabled(False)

        action = {
            "new": "teach",
            "improve": "improve",
            "retrain": "retrain",
        }[mode]
        expected = (
            f" Use {required_signature} input."
            if required_signature is not None
            else " One or two hands are supported."
        )
        self.status_var.set(
            f"Get ready to {action} '{name}'. Hold the gesture naturally.{expected}"
        )

    def process_teaching_frame(self):
        if self.current_feature_set is None:
            self.status_var.set("Waiting for a hand...")
            return

        now = time.monotonic()
        if self.prepare_until is not None and now < self.prepare_until:
            remaining = self.prepare_until - now
            self.status_var.set(
                f"Teaching '{self.teaching_name}' starts in {remaining:.1f}s..."
            )
            return

        current_signature = self.current_feature_set.hand_signature

        if self.required_teaching_signature is not None:
            if current_signature != self.required_teaching_signature:
                self.status_var.set(
                    f"'{self.teaching_name}' expects {self.required_teaching_signature} "
                    f"input. Current input is {current_signature}."
                )
                return
            self.teaching_signature = self.required_teaching_signature

        elif self.teaching_signature is None:
            self.teaching_signature = current_signature
            # Two-hand vectors include relative geometry and naturally vary more.
            if current_signature == "Both":
                self.selector.stability_threshold = 0.050
                self.selector.duplicate_threshold = 0.025

        elif current_signature != self.teaching_signature:
            self.status_var.set(
                f"Continue using {self.teaching_signature} input for this gesture."
            )
            return

        self.selector.consider(self.current_features)
        self.update_stats_display()
        self.progress["value"] = len(self.selector.samples)

        if self.selector.ready:
            self.finish_button.config(state="normal")
            if not self.selector.complete:
                self.status_var.set(
                    "Enough useful samples to learn. You may finish now, or make "
                    "small natural variations of the SAME gesture."
                )
        else:
            self.status_var.set(
                "Keep the gesture steady. Small natural variations are useful."
            )

        if self.selector.complete:
            self.finish_teaching()

    def finish_teaching(self):
        if not self.teaching:
            return
        if self.selector is None or not self.selector.ready:
            self.status_var.set("Not enough useful samples yet.")
            return

        observed = self.selector.stats.observed
        accepted = self.selector.stats.accepted
        duplicates = self.selector.stats.duplicates
        unstable = self.selector.stats.unstable
        samples = [sample.copy() for sample in self.selector.samples]
        signature = self.teaching_signature
        name = self.teaching_name
        mode = self.teaching_mode

        try:
            if mode == "new":
                gesture = self.learner.learn_gesture(
                    name=name,
                    samples=samples,
                    hand_signature=signature,
                )
                action_text = "Learned"
            elif mode == "improve":
                gesture = self.learner.add_samples_to_gesture(
                    name=name,
                    samples=samples,
                    hand_signature=signature,
                )
                action_text = "Improved"
            elif mode == "retrain":
                gesture = self.learner.replace_gesture_samples(
                    name=name,
                    samples=samples,
                    hand_signature=signature,
                    keep_negatives=True,
                )
                action_text = "Retrained"
            else:
                raise ValueError("Unknown teaching mode.")

            saved = self.save_gesture_memory()
        except Exception as error:
            self.status_var.set(f"Learning failed: {error}")
            return

        self.finish_teaching_state()
        self.refresh_gesture_table()

        message = (
            f"✓ {action_text} '{gesture.name}' from {accepted} useful samples. "
            f"{observed} frames observed; {duplicates} duplicates and {unstable} "
            f"unstable frames ignored. {gesture.prototype_count} adaptive "
            f"prototype(s) active."
        )
        if saved:
            message += " Gesture memory saved."
        self.status_var.set(message)

    def cancel_teaching(self):
        if not self.teaching:
            return
        self.finish_teaching_state()
        self.status_var.set("Teaching cancelled.")

    def finish_teaching_state(self):
        self.teaching = False
        self.teaching_mode = None
        self.teaching_name = None
        self.teaching_signature = None
        self.required_teaching_signature = None
        self.prepare_until = None
        self.selector = None
        self.progress["value"] = 0

        self.teach_button.config(state="normal")
        self.finish_button.config(state="disabled")
        self.cancel_button.config(state="disabled")
        self.gesture_entry.config(state="normal")
        self.gesture_name_var.set("")
        self.set_feedback_buttons_enabled(True)
        self.gesture_entry.focus_set()

    # ========================================================
    # Prediction
    # ========================================================

    def update_prediction(self):
        if self.teaching:
            self.prediction_var.set(f"Teaching: {self.teaching_name}")
            self.distance_var.set("Distance: —")
            self.threshold_var.set("Threshold: —")
            self.relative_distance_var.set("Relative distance: —")
            self.rejection_reason_var.set("")
            return

        if self.current_features is None:
            self.current_prediction = None
            self.prediction_var.set("NO HAND")
            self.distance_var.set("Distance: —")
            self.threshold_var.set("Threshold: —")
            self.relative_distance_var.set("Relative distance: —")
            self.rejection_reason_var.set("")
            return

        prediction = self.learner.predict(
            self.current_features,
            hand_signature=self.current_hand_signature,
        )
        self.current_prediction = prediction
        self.prediction_var.set(prediction.label)

        if prediction.distance is None:
            self.distance_var.set("Distance: —")
            self.threshold_var.set("Threshold: —")
            self.relative_distance_var.set("Relative distance: —")
        else:
            self.distance_var.set(
                f"Nearest positive distance: {prediction.distance:.4f}"
            )
            self.threshold_var.set(
                f"Acceptance threshold: {prediction.threshold:.4f}"
            )
            self.relative_distance_var.set(
                f"Relative score: {prediction.relative_distance:.2f}x"
                if prediction.relative_distance is not None
                else "Relative score: —"
            )

        if prediction.accepted:
            self.rejection_reason_var.set("")
        elif prediction.rejection_reason == "outside_positive_region":
            nearest = prediction.nearest_label or "known gesture"
            self.rejection_reason_var.set(
                f"UNKNOWN — outside learned region for '{nearest}'"
            )
        elif prediction.rejection_reason == "hard_negative":
            self.rejection_reason_var.set(
                "UNKNOWN — rejected using learned negative feedback"
            )
        elif prediction.rejection_reason == "no_gestures":
            self.rejection_reason_var.set("No gestures learned yet")
        elif prediction.rejection_reason == "hand_configuration":
            self.rejection_reason_var.set(
                f"No learned gesture matches {self.current_hand_signature} input"
            )
        else:
            self.rejection_reason_var.set("")

    # ========================================================
    # Feedback
    # ========================================================

    def set_feedback_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.correct_button.config(state=state)
        self.wrong_button.config(state=state)

    def confirm_prediction(self):
        if self.teaching:
            return
        if self.current_features is None or self.current_prediction is None:
            self.status_var.set("No gesture available to confirm.")
            return
        if not self.current_prediction.accepted:
            self.status_var.set(
                "The current pose is UNKNOWN. There is no known prediction to confirm."
            )
            return

        before = self.learner.gestures[self.current_prediction.label].sample_count
        try:
            gesture = self.learner.update_gesture(
                self.current_prediction.label,
                self.current_features.copy(),
            )
            saved = self.save_gesture_memory()
        except Exception as error:
            self.status_var.set(f"Feedback failed: {error}")
            return

        added = gesture.sample_count > before
        self.refresh_gesture_table()
        if added:
            message = (
                f"✓ Confirmed '{gesture.name}'. A useful positive example was added "
                "and adaptive prototypes were updated."
            )
        else:
            message = (
                f"✓ Confirmed '{gesture.name}'. The pose was already represented, "
                "so duplicate positive memory was not added."
            )
        if saved:
            message += " Memory saved."
        self.status_var.set(message)

    def begin_correction(self):
        if self.teaching:
            return
        if self.current_features is None or self.current_prediction is None:
            self.status_var.set("No current pose available for correction.")
            return

        self.feedback_features = self.current_features.copy()
        self.feedback_hand_signature = self.current_hand_signature
        prediction = self.current_prediction
        self.feedback_nearest_label = prediction.nearest_label
        self.feedback_predicted_label = prediction.label if prediction.accepted else None

        # Only show known gestures compatible with the frozen hand configuration.
        compatible = [
            name
            for name, gesture in self.learner.gestures.items()
            if gesture.hand_signature == self.feedback_hand_signature
            and gesture.feature_dimension == self.feedback_features.shape[0]
        ]
        self.actual_gesture_combo["values"] = compatible
        self.actual_gesture_var.set("")
        self.correction_frame.grid()
        self.set_feedback_buttons_enabled(False)

        if prediction.accepted:
            self.status_var.set(
                f"Correction mode: captured prediction was '{prediction.label}'. "
                "Select the actual gesture or mark it Unknown."
            )
        else:
            nearest_text = (
                f" Closest learned class was '{prediction.nearest_label}'."
                if prediction.nearest_label
                else ""
            )
            self.status_var.set(
                "Correction mode: captured pose was UNKNOWN." + nearest_text
            )

    def apply_feedback_correction(self):
        if self.feedback_features is None:
            self.status_var.set("No captured feedback example.")
            return

        actual_label = self.actual_gesture_var.get().strip()
        if not actual_label:
            self.status_var.set("Select the actual gesture first.")
            return

        predicted_label = self.feedback_predicted_label
        try:
            self.learner.apply_correction(
                predicted_label=predicted_label,
                actual_label=actual_label,
                sample=self.feedback_features,
            )
            saved = self.save_gesture_memory()
        except Exception as error:
            self.status_var.set(f"Correction failed: {error}")
            return

        if predicted_label and predicted_label != actual_label:
            message = (
                f"✓ Learned correction: this is '{actual_label}', not "
                f"'{predicted_label}'. Positive and hard-negative memory were updated."
            )
        else:
            message = (
                f"✓ Learned that this previously UNKNOWN example belongs to "
                f"'{actual_label}'."
            )
        if saved:
            message += " Memory saved."

        self.finish_feedback()
        self.refresh_gesture_table()
        self.status_var.set(message)

    def mark_feedback_unknown(self):
        if self.feedback_features is None:
            self.status_var.set("No captured feedback example.")
            return

        predicted_label = self.feedback_predicted_label
        try:
            self.learner.mark_unknown(
                predicted_label=predicted_label,
                sample=self.feedback_features,
            )
            saved = self.save_gesture_memory()
        except Exception as error:
            self.status_var.set(f"Negative feedback failed: {error}")
            return

        self.finish_feedback()
        self.refresh_gesture_table()

        if predicted_label:
            message = (
                f"✓ Learned that this pose is NOT '{predicted_label}'. "
                "A non-duplicate hard-negative example was stored."
            )
        else:
            message = (
                "The captured pose was already UNKNOWN, so no accepted class "
                "required negative correction."
            )
        if saved:
            message += " Memory saved."
        self.status_var.set(message)

    def cancel_feedback(self):
        self.finish_feedback()
        self.status_var.set("Feedback cancelled.")

    def finish_feedback(self):
        self.feedback_features = None
        self.feedback_predicted_label = None
        self.feedback_nearest_label = None
        self.feedback_hand_signature = None
        self.actual_gesture_var.set("")

        if hasattr(self, "correction_frame"):
            self.correction_frame.grid_remove()
        if hasattr(self, "correct_button") and not self.teaching:
            self.set_feedback_buttons_enabled(True)

    # ========================================================
    # Stats / library
    # ========================================================

    def reset_stats_display(self):
        self.observed_var.set("0")
        self.accepted_var.set("0")
        self.duplicate_var.set("0")
        self.unstable_var.set("0")
        self.stability_var.set("—")
        self.input_mode_var.set("—")

    def update_stats_display(self):
        if self.selector is None:
            return
        stats = self.selector.stats
        self.observed_var.set(str(stats.observed))
        self.accepted_var.set(str(stats.accepted))
        self.duplicate_var.set(str(stats.duplicates))
        self.unstable_var.set(str(stats.unstable))
        self.stability_var.set(
            "—"
            if self.selector.last_stability is None
            else f"{self.selector.last_stability:.4f}"
        )
        self.input_mode_var.set(self.teaching_signature or "—")

    def refresh_gesture_table(self):
        for item in self.gesture_table.get_children():
            self.gesture_table.delete(item)

        for name, gesture in self.learner.gestures.items():
            self.gesture_table.insert(
                "",
                "end",
                values=(
                    name,
                    gesture.sample_count,
                    gesture.negative_count,
                    gesture.prototype_count,
                    f"{gesture.spread:.4f}",
                    f"{gesture.sample_radius:.4f}",
                    gesture.hand_signature,
                ),
            )

        self.actual_gesture_combo["values"] = self.learner.list_gestures()

    # ========================================================
    # Camera loop
    # ========================================================

    def _set_stringvar_if_changed(self, variable, value):
        """Avoid unnecessary Tk redraws when displayed text is unchanged."""
        if variable.get() != value:
            variable.set(value)

    def update_camera(self):
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)

            # HandTracker now temporally stabilizes brief 0/1/2-hand changes,
            # so one physical hand does not flash between modes.
            hands = self.tracker.process(frame)
            self.tracker.draw(frame, hands)
            self.current_hands = hands

            self.current_feature_set = build_frame_features(hands)
            if self.current_feature_set is None:
                self.current_features = None
                self.current_hand_signature = None
                self._set_stringvar_if_changed(
                    self.tracking_var,
                    "No hand detected",
                )
            else:
                self.current_features = self.current_feature_set.vector
                self.current_hand_signature = self.current_feature_set.hand_signature

                if self.current_hand_signature == "Both":
                    confidence = min(
                        hand.handedness_score for hand in hands[:2]
                    )
                    tracking_text = (
                        f"Tracking both hands — {confidence:.0%} "
                        "minimum handedness confidence"
                    )
                else:
                    confidence = hands[0].handedness_score
                    tracking_text = (
                        f"Tracking {self.current_hand_signature} hand — "
                        f"{confidence:.0%}"
                    )

                self._set_stringvar_if_changed(
                    self.tracking_var,
                    tracking_text,
                )

            if self.teaching:
                self.process_teaching_frame()

            # Keep live camera rendering independent from rapidly changing
            # recognition metrics. 12.5 Hz is responsive for prediction text
            # while the camera itself stays close to 30 FPS.
            now = time.monotonic()
            if (
                self.teaching
                or now - self.last_prediction_update
                >= self.prediction_update_interval
            ):
                self.update_prediction()
                self.last_prediction_update = now

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(display_frame)
            image.thumbnail((850, 650))
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo

        self.root.after(15, self.update_camera)

    # ========================================================
    # Cleanup
    # ========================================================

    def close(self):
        self.save_gesture_memory()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.tracker.close()
        cv2.destroyAllWindows()
        self.root.destroy()


def main():
    root = tk.Tk()
    InteractiveGestureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
