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


from adaptive_gesture.features.dynamic_features import (
    build_dynamic_observation,
    prepare_dynamic_trajectory,
)
from adaptive_gesture.features.hand_features import build_frame_features
from adaptive_gesture.learning.dynamic_learner import DynamicGestureLearner
from adaptive_gesture.learning.motion_segmenter import MotionSegmenter
from adaptive_gesture.learning.online_learner import OnlineGestureLearner
from adaptive_gesture.learning.prediction_stabilizer import PredictionStabilizer
from adaptive_gesture.learning.sample_selector import SmartSampleSelector
from adaptive_gesture.storage.dynamic_gesture_store import DynamicGestureStore
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

        # Dynamic few-shot engine. Dynamic classes are intentionally separate
        # from static classes so the mature static path remains unchanged.
        self.dynamic_learner = DynamicGestureLearner(
            minimum_templates=3,
            threshold_multiplier=1.60,
            minimum_threshold=0.045,
            ambiguity_ratio=1.12,
            max_templates=6,
        )
        self.motion_segmenter = MotionSegmenter()

        # Temporal prediction stabilization. A known label must appear in
        # 3 consecutive recognition updates before it becomes confirmed.
        # UNKNOWN uses 2 updates so stale known labels are released faster.
        self.prediction_stabilizer = PredictionStabilizer(
            confirm_frames=3,
            unknown_confirm_frames=2,
            unknown_label=self.learner.UNKNOWN_LABEL,
        )

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

        # Dynamic gesture trajectories are stored separately from the existing
        # static memory. This avoids risky schema changes and stores landmarks
        # only -- never camera frames or videos.
        self.dynamic_gesture_store = DynamicGestureStore(
            PROJECT_ROOT / "data" / "dynamic_gesture_memory.json"
        )
        self.dynamic_restore_error = None
        try:
            self.restored_dynamic_gestures = (
                self.dynamic_gesture_store.load_into(self.dynamic_learner)
            )
        except Exception as error:
            self.restored_dynamic_gestures = 0
            self.dynamic_restore_error = str(error)

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
        self.current_raw_prediction = None
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

        # Dynamic teaching/runtime state. Training uses explicit Start/Stop Demo
        # controls for clean few-shot examples; recognition uses automatic
        # motion onset/offset segmentation.
        self.dynamic_teaching = False
        self.dynamic_demo_recording = False
        self.dynamic_teaching_name = None
        self.dynamic_required_signature = None
        self.dynamic_demo_observations = []
        self.dynamic_templates = []
        self.dynamic_target_demos = self.dynamic_learner.minimum_templates
        self.dynamic_sample_interval = 0.04  # ~25 Hz temporal sampling
        self.last_dynamic_sample_time = 0.0
        self.dynamic_prediction_hold_until = 0.0

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
        self.relative_distance_var = tk.StringVar(value="Relative score: —\nConfidence index: —")
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

        # Dynamic gesture UI state.
        self.dynamic_name_var = tk.StringVar()
        self.dynamic_prediction_var = tk.StringVar(value="—")
        self.dynamic_distance_var = tk.StringVar(value="Distance: —\nConfidence index: —")
        self.dynamic_runtime_state_var = tk.StringVar(value="Dynamic: IDLE")
        self.dynamic_status_var = tk.StringVar(value="")
        self.dynamic_demo_progress_var = tk.StringVar(
            value=f"Demonstrations: 0/{self.dynamic_target_demos}"
        )

        self.build_ui()
        self.refresh_gesture_table()
        self.refresh_dynamic_gesture_table()

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

        if self.dynamic_restore_error:
            self.dynamic_status_var.set(
                "Dynamic memory could not be restored: "
                + self.dynamic_restore_error
            )
        elif self.restored_dynamic_gestures > 0:
            suffix = "s" if self.restored_dynamic_gestures != 1 else ""
            self.dynamic_status_var.set(
                f"✓ Restored {self.restored_dynamic_gestures} dynamic "
                f"gesture{suffix} from previous sessions."
            )
        else:
            self.dynamic_status_var.set(
                "No dynamic gestures saved yet. Teach 3 live demonstrations."
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
        self.dynamic_tab = ttk.Frame(notebook, padding=16)
        notebook.add(self.live_tab, text="Live")
        notebook.add(self.library_tab, text="Gesture Library")
        notebook.add(self.dynamic_tab, text="Dynamic Gestures")

        self.live_tab.columnconfigure(0, weight=1)
        self.library_tab.columnconfigure(0, weight=1)
        self.library_tab.rowconfigure(1, weight=1)
        self.dynamic_tab.columnconfigure(0, weight=1)
        self.dynamic_tab.rowconfigure(8, weight=1)

        self.build_live_tab(self.live_tab)
        self.build_library_tab(self.library_tab)
        self.build_dynamic_tab(self.dynamic_tab)

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

        # Dynamic recognition is segment-based and shown separately so static
        # frame classification remains unchanged.
        dynamic_live = ttk.LabelFrame(
            panel,
            text="Dynamic Recognition",
            padding=10,
        )
        dynamic_live.grid(row=16, column=0, sticky="ew", pady=(14, 0))
        dynamic_live.columnconfigure(0, weight=1)

        ttk.Label(
            dynamic_live,
            textvariable=self.dynamic_prediction_var,
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            dynamic_live,
            textvariable=self.dynamic_distance_var,
        ).grid(row=1, column=0, sticky="w")
        ttk.Label(
            dynamic_live,
            textvariable=self.dynamic_runtime_state_var,
        ).grid(row=2, column=0, sticky="w")

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

    def build_dynamic_tab(self, panel):
        ttk.Label(
            panel,
            text="Teach a Dynamic Gesture",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            panel,
            text=(
                "Teach movements such as swipe, wave or circle. The app stores "
                "only normalized landmark trajectories -- no video."
            ),
            wraplength=440,
        ).grid(row=1, column=0, sticky="w", pady=(4, 10))

        ttk.Label(panel, text="Dynamic gesture name").grid(
            row=2, column=0, sticky="w"
        )
        self.dynamic_name_entry = ttk.Entry(
            panel, textvariable=self.dynamic_name_var
        )
        self.dynamic_name_entry.grid(row=3, column=0, sticky="ew", pady=(3, 8))

        self.dynamic_teach_button = ttk.Button(
            panel,
            text="Teach Dynamic Gesture",
            command=self.start_dynamic_teaching,
        )
        self.dynamic_teach_button.grid(row=4, column=0, sticky="ew")

        demo_controls = ttk.Frame(panel)
        demo_controls.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        for column in range(3):
            demo_controls.columnconfigure(column, weight=1)

        self.dynamic_start_demo_button = ttk.Button(
            demo_controls,
            text="Start Demo",
            command=self.start_dynamic_demo,
            state="disabled",
        )
        self.dynamic_start_demo_button.grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )

        self.dynamic_stop_demo_button = ttk.Button(
            demo_controls,
            text="Stop Demo",
            command=self.stop_dynamic_demo,
            state="disabled",
        )
        self.dynamic_stop_demo_button.grid(
            row=0, column=1, sticky="ew", padx=4
        )

        self.dynamic_cancel_button = ttk.Button(
            demo_controls,
            text="Cancel",
            command=self.cancel_dynamic_teaching,
            state="disabled",
        )
        self.dynamic_cancel_button.grid(
            row=0, column=2, sticky="ew", padx=(4, 0)
        )

        ttk.Label(
            panel,
            textvariable=self.dynamic_demo_progress_var,
        ).grid(row=6, column=0, sticky="w", pady=(8, 2))
        ttk.Label(
            panel,
            textvariable=self.dynamic_status_var,
            wraplength=440,
        ).grid(row=7, column=0, sticky="w", pady=(0, 10))

        self.dynamic_gesture_table = ttk.Treeview(
            panel,
            columns=("gesture", "templates", "threshold", "duration", "input"),
            show="headings",
            height=9,
        )
        dynamic_headings = {
            "gesture": "Gesture",
            "templates": "Demos",
            "threshold": "Threshold",
            "duration": "Median s",
            "input": "Input",
        }
        dynamic_widths = {
            "gesture": 130,
            "templates": 60,
            "threshold": 85,
            "duration": 75,
            "input": 70,
        }
        for column, heading in dynamic_headings.items():
            self.dynamic_gesture_table.heading(column, text=heading)
            self.dynamic_gesture_table.column(
                column,
                width=dynamic_widths[column],
                anchor="w" if column == "gesture" else "center",
            )
        self.dynamic_gesture_table.grid(row=8, column=0, sticky="nsew")

        management = ttk.LabelFrame(
            panel, text="Dynamic Gesture Management", padding=10
        )
        management.grid(row=9, column=0, sticky="ew", pady=(10, 0))
        for column in range(3):
            management.columnconfigure(column, weight=1)

        ttk.Button(
            management,
            text="Rename",
            command=self.rename_selected_dynamic_gesture,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            management,
            text="Delete",
            command=self.delete_selected_dynamic_gesture,
        ).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(
            management,
            text="Clear All",
            command=self.clear_all_dynamic_gestures,
        ).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        ttk.Label(
            panel,
            text=(
                "Training: press Start Demo, perform the complete movement once, "
                "then Stop Demo. Repeat three times. Live recognition starts "
                "automatically after learning and uses motion onset/offset detection."
            ),
            wraplength=440,
        ).grid(row=10, column=0, sticky="w", pady=(10, 0))

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

    def save_dynamic_gesture_memory(self):
        try:
            self.dynamic_gesture_store.save(self.dynamic_learner)
            return True
        except Exception as error:
            self.dynamic_status_var.set(
                "Warning: dynamic learning succeeded but memory could not be "
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
            self.prediction_stabilizer.reset()
            self.current_prediction = None
            self.current_raw_prediction = None
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
        self.prediction_stabilizer.reset()
        self.current_prediction = None
        self.current_raw_prediction = None
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
        self.prediction_stabilizer.reset()
        self.current_prediction = None
        self.current_raw_prediction = None
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
        if self.dynamic_teaching:
            self.status_var.set(
                "Finish or cancel dynamic-gesture teaching first."
            )
            return

        self.finish_feedback()
        self.prediction_stabilizer.reset()
        self.current_prediction = None
        self.current_raw_prediction = None
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
        self.prediction_stabilizer.reset()
        self.current_prediction = None
        self.current_raw_prediction = None
        self.progress["value"] = 0

        self.teach_button.config(state="normal")
        self.finish_button.config(state="disabled")
        self.cancel_button.config(state="disabled")
        self.gesture_entry.config(state="normal")
        self.gesture_name_var.set("")
        self.set_feedback_buttons_enabled(True)
        self.gesture_entry.focus_set()

    # ========================================================
    # Dynamic gesture learning / management
    # ========================================================

    def start_dynamic_teaching(self):
        if self.teaching:
            self.dynamic_status_var.set(
                "Finish or cancel static-gesture teaching first."
            )
            return
        if self.dynamic_teaching:
            return

        name = self.dynamic_name_var.get().strip()
        if not name:
            self.dynamic_status_var.set("Enter a dynamic gesture name first.")
            self.dynamic_name_entry.focus_set()
            return
        if name in self.dynamic_learner.gestures:
            self.dynamic_status_var.set(
                f"Dynamic gesture '{name}' already exists."
            )
            return

        self.dynamic_teaching = True
        self.dynamic_demo_recording = False
        self.dynamic_teaching_name = name
        self.dynamic_required_signature = None
        self.dynamic_demo_observations = []
        self.dynamic_templates = []
        self.motion_segmenter.reset()
        self.dynamic_demo_progress_var.set(
            f"Demonstrations: 0/{self.dynamic_target_demos}"
        )

        self.dynamic_name_entry.config(state="disabled")
        self.dynamic_teach_button.config(state="disabled")
        self.dynamic_start_demo_button.config(state="normal")
        self.dynamic_stop_demo_button.config(state="disabled")
        self.dynamic_cancel_button.config(state="normal")
        self.dynamic_status_var.set(
            f"Ready to teach '{name}'. Press Start Demo, perform the entire "
            "movement once, then press Stop Demo."
        )

    def start_dynamic_demo(self):
        if not self.dynamic_teaching or self.dynamic_demo_recording:
            return
        if not self.current_hands:
            self.dynamic_status_var.set(
                "Show the hand(s) you will use before starting the demo."
            )
            return

        now = time.monotonic()
        observation = build_dynamic_observation(self.current_hands, now)
        if observation is None:
            self.dynamic_status_var.set("No stable hand configuration detected.")
            return

        if self.dynamic_required_signature is None:
            self.dynamic_required_signature = observation.hand_signature
        elif observation.hand_signature != self.dynamic_required_signature:
            self.dynamic_status_var.set(
                f"This gesture is locked to {self.dynamic_required_signature} input. "
                f"Currently seeing {observation.hand_signature}."
            )
            return

        self.dynamic_demo_recording = True
        self.dynamic_demo_observations = [observation]
        self.last_dynamic_sample_time = now
        demo_number = len(self.dynamic_templates) + 1
        self.dynamic_start_demo_button.config(state="disabled")
        self.dynamic_stop_demo_button.config(state="normal")
        self.dynamic_status_var.set(
            f"● Recording demo {demo_number}/{self.dynamic_target_demos}. "
            "Perform the complete movement naturally, then press Stop Demo."
        )

    def process_dynamic_training_frame(self, now):
        if not self.dynamic_demo_recording:
            return
        if now - self.last_dynamic_sample_time < self.dynamic_sample_interval:
            return
        self.last_dynamic_sample_time = now

        observation = build_dynamic_observation(self.current_hands, now)
        if observation is None:
            self.dynamic_status_var.set(
                "Recording: keep the hand(s) visible. Missing frames are ignored."
            )
            return
        if observation.hand_signature != self.dynamic_required_signature:
            self.dynamic_status_var.set(
                f"Recording expects {self.dynamic_required_signature} input. "
                "Keep the same hand configuration visible."
            )
            return

        self.dynamic_demo_observations.append(observation)

    def stop_dynamic_demo(self):
        if not self.dynamic_teaching or not self.dynamic_demo_recording:
            return

        self.dynamic_demo_recording = False
        self.dynamic_stop_demo_button.config(state="disabled")
        self.dynamic_start_demo_button.config(state="normal")

        try:
            trajectory = prepare_dynamic_trajectory(
                self.dynamic_demo_observations
            )
        except Exception as error:
            self.dynamic_demo_observations = []
            self.dynamic_status_var.set(
                f"Demo was not accepted: {error} Please record this demo again."
            )
            return

        self.dynamic_templates.append(trajectory)
        self.dynamic_demo_observations = []
        completed = len(self.dynamic_templates)
        self.dynamic_demo_progress_var.set(
            f"Demonstrations: {completed}/{self.dynamic_target_demos}"
        )

        if completed < self.dynamic_target_demos:
            self.dynamic_status_var.set(
                f"✓ Demo {completed} accepted "
                f"({trajectory.duration_seconds:.2f}s, "
                f"motion extent {trajectory.motion_extent:.2f}). "
                "Return to the starting pose and record the next demonstration."
            )
            return

        try:
            gesture = self.dynamic_learner.learn_gesture(
                self.dynamic_teaching_name,
                self.dynamic_templates,
            )
            saved = self.save_dynamic_gesture_memory()
        except Exception as error:
            self.dynamic_status_var.set(f"Dynamic learning failed: {error}")
            return

        message = (
            f"✓ Learned dynamic gesture '{gesture.name}' from "
            f"{gesture.template_count} demonstrations. "
            f"DTW threshold: {gesture.threshold:.4f}."
        )
        if saved:
            message += " Landmark trajectories saved."

        self.finish_dynamic_teaching_state()
        self.refresh_dynamic_gesture_table()
        self.dynamic_status_var.set(message)

    def cancel_dynamic_teaching(self):
        if not self.dynamic_teaching:
            return
        self.finish_dynamic_teaching_state()
        self.dynamic_status_var.set("Dynamic gesture teaching cancelled.")

    def finish_dynamic_teaching_state(self):
        self.dynamic_teaching = False
        self.dynamic_demo_recording = False
        self.dynamic_teaching_name = None
        self.dynamic_required_signature = None
        self.dynamic_demo_observations = []
        self.dynamic_templates = []
        self.motion_segmenter.reset()

        self.dynamic_name_entry.config(state="normal")
        self.dynamic_teach_button.config(state="normal")
        self.dynamic_start_demo_button.config(state="disabled")
        self.dynamic_stop_demo_button.config(state="disabled")
        self.dynamic_cancel_button.config(state="disabled")
        self.dynamic_name_var.set("")
        self.dynamic_demo_progress_var.set(
            f"Demonstrations: 0/{self.dynamic_target_demos}"
        )

    def get_selected_dynamic_gesture_name(self):
        selection = self.dynamic_gesture_table.selection()
        if not selection:
            messagebox.showinfo(
                "Select Dynamic Gesture",
                "Select a gesture in the Dynamic Gestures table first.",
                parent=self.root,
            )
            return None
        values = self.dynamic_gesture_table.item(selection[0], "values")
        return values[0] if values else None

    def rename_selected_dynamic_gesture(self):
        name = self.get_selected_dynamic_gesture_name()
        if not name:
            return
        new_name = simpledialog.askstring(
            "Rename Dynamic Gesture",
            f"New name for '{name}':",
            initialvalue=name,
            parent=self.root,
        )
        if new_name is None:
            return
        try:
            gesture = self.dynamic_learner.rename_gesture(name, new_name)
            self.save_dynamic_gesture_memory()
            self.refresh_dynamic_gesture_table()
            self.dynamic_status_var.set(
                f"✓ Renamed dynamic gesture '{name}' to '{gesture.name}'."
            )
        except Exception as error:
            messagebox.showerror("Rename Failed", str(error), parent=self.root)

    def delete_selected_dynamic_gesture(self):
        name = self.get_selected_dynamic_gesture_name()
        if not name:
            return
        if not messagebox.askyesno(
            "Delete Dynamic Gesture",
            f"Delete '{name}' and all of its stored landmark trajectories?",
            parent=self.root,
        ):
            return
        try:
            self.dynamic_learner.delete_gesture(name)
            self.save_dynamic_gesture_memory()
            self.refresh_dynamic_gesture_table()
            self.motion_segmenter.reset()
            self.dynamic_status_var.set(f"Deleted dynamic gesture '{name}'.")
        except Exception as error:
            messagebox.showerror("Delete Failed", str(error), parent=self.root)

    def clear_all_dynamic_gestures(self):
        if not self.dynamic_learner.gestures:
            self.dynamic_status_var.set("There are no dynamic gestures to clear.")
            return
        if not messagebox.askyesno(
            "Clear Dynamic Gesture Memory",
            "Delete ALL learned dynamic gestures and trajectories?",
            parent=self.root,
        ):
            return
        self.dynamic_learner.clear()
        self.save_dynamic_gesture_memory()
        self.refresh_dynamic_gesture_table()
        self.motion_segmenter.reset()
        self.dynamic_prediction_var.set("—")
        self.dynamic_distance_var.set("Distance: —\nConfidence index: —")
        self.dynamic_status_var.set("All dynamic gesture memory cleared.")

    def process_dynamic_runtime(self, now):
        # Explicit training owns the temporal stream while a demo is recorded.
        if self.dynamic_teaching:
            self.motion_segmenter.reset()
            self.process_dynamic_training_frame(now)
            self.dynamic_runtime_state_var.set(
                "Dynamic: RECORDING DEMO"
                if self.dynamic_demo_recording
                else "Dynamic: TEACHING"
            )
            return

        # Do not let static teaching movements accidentally become dynamic
        # recognition segments.
        if self.teaching:
            self.motion_segmenter.reset()
            self.dynamic_runtime_state_var.set("Dynamic: PAUSED DURING STATIC TEACHING")
            return

        if now - self.last_dynamic_sample_time < self.dynamic_sample_interval:
            if now >= self.dynamic_prediction_hold_until and self.motion_segmenter.state == self.motion_segmenter.IDLE:
                self.dynamic_prediction_var.set("—")
                self.dynamic_distance_var.set("Distance: —\nConfidence index: —")
            return

        self.last_dynamic_sample_time = now

        if not self.dynamic_learner.gestures:
            self.motion_segmenter.reset()
            self.dynamic_runtime_state_var.set("Dynamic: no gestures learned")
            return

        observation = build_dynamic_observation(self.current_hands, now)
        result = self.motion_segmenter.update(observation, now)
        self.dynamic_runtime_state_var.set(f"Dynamic: {result.state}")

        if result.completed is None:
            if now >= self.dynamic_prediction_hold_until and result.state == self.motion_segmenter.IDLE:
                self.dynamic_prediction_var.set("—")
                self.dynamic_distance_var.set("Distance: —\nConfidence index: —")
            return

        try:
            trajectory = prepare_dynamic_trajectory(result.completed)
        except Exception:
            self.motion_segmenter.set_cooldown(now, seconds=0.35)
            return

        prediction = self.dynamic_learner.predict(trajectory)
        self.dynamic_prediction_hold_until = now + (2.0 if prediction.accepted else 1.0)

        if prediction.accepted:
            self.dynamic_prediction_var.set(prediction.label)
            confidence_text = (
                f"Confidence index: {prediction.confidence * 100.0:.0f}% "
                "(not probability)"
                if prediction.confidence is not None
                else "Confidence index: —"
            )
            self.dynamic_distance_var.set(
                f"DTW distance: {prediction.distance:.4f} / "
                f"threshold {prediction.threshold:.4f}\n"
                + confidence_text
            )
            self.dynamic_status_var.set(
                f"✓ Dynamic gesture recognized: '{prediction.label}'."
            )
        else:
            self.dynamic_prediction_var.set("UNKNOWN")
            if prediction.distance is None:
                self.dynamic_distance_var.set("Distance: —\nConfidence index: —")
            else:
                confidence_text = (
                    f"Confidence index: {prediction.confidence * 100.0:.0f}% "
                    "(not probability)"
                    if prediction.confidence is not None
                    else "Confidence index: —"
                )
                self.dynamic_distance_var.set(
                    f"Nearest DTW distance: {prediction.distance:.4f}\n"
                    + confidence_text
                )

        self.motion_segmenter.set_cooldown(now, seconds=0.65)

    def refresh_dynamic_gesture_table(self):
        if not hasattr(self, "dynamic_gesture_table"):
            return
        for item in self.dynamic_gesture_table.get_children():
            self.dynamic_gesture_table.delete(item)
        for name, gesture in self.dynamic_learner.gestures.items():
            self.dynamic_gesture_table.insert(
                "",
                "end",
                values=(
                    name,
                    gesture.template_count,
                    f"{gesture.threshold:.4f}",
                    f"{gesture.median_duration:.2f}",
                    gesture.hand_signature,
                ),
            )

    # ========================================================
    # Prediction
    # ========================================================

    def update_prediction(self):
        if self.teaching:
            self.prediction_stabilizer.reset()
            self.current_prediction = None
            self.current_raw_prediction = None
            self.prediction_var.set(f"Teaching: {self.teaching_name}")
            self.distance_var.set("Distance: —")
            self.threshold_var.set("Threshold: —")
            self.relative_distance_var.set("Relative score: —\nConfidence index: —")
            self.rejection_reason_var.set("")
            return

        if self.current_features is None:
            self.prediction_stabilizer.reset()
            self.current_prediction = None
            self.current_raw_prediction = None
            self.prediction_var.set("NO HAND")
            self.distance_var.set("Distance: —")
            self.threshold_var.set("Threshold: —")
            self.relative_distance_var.set("Relative score: —\nConfidence index: —")
            self.rejection_reason_var.set("")
            return

        raw_prediction = self.learner.predict(
            self.current_features,
            hand_signature=self.current_hand_signature,
        )
        self.current_raw_prediction = raw_prediction

        stabilized = self.prediction_stabilizer.update(raw_prediction)
        prediction = stabilized.prediction
        self.current_prediction = prediction

        # ----------------------------------------------------
        # Initial confirmation period. Do not expose a gesture
        # as confirmed until it has stayed consistent briefly.
        # ----------------------------------------------------
        if prediction is None:
            self.prediction_var.set("STABILIZING")
            self.distance_var.set("Distance: —")
            self.threshold_var.set("Threshold: —")
            self.relative_distance_var.set("Relative score: —\nConfidence index: —")

            candidate = stabilized.candidate_label or "prediction"
            self.rejection_reason_var.set(
                f"Checking '{candidate}' "
                f"({stabilized.candidate_count}/{stabilized.required_count})"
            )
            return

        # The visible label and feedback target are now always the SAME
        # confirmed prediction object. This prevents feedback from being
        # applied to a transient raw label that the user never saw.
        self.prediction_var.set(prediction.label)

        if prediction.distance is None:
            self.distance_var.set("Distance: —")
            self.threshold_var.set("Threshold: —")
            self.relative_distance_var.set("Relative score: —\nConfidence index: —")
        else:
            self.distance_var.set(
                f"Nearest positive distance: {prediction.distance:.4f}"
            )
            self.threshold_var.set(
                f"Acceptance threshold: {prediction.threshold:.4f}"
            )
            relative_text = (
                f"Relative score: {prediction.relative_distance:.2f}x"
                if prediction.relative_distance is not None
                else "Relative score: —"
            )
            confidence_text = (
                f"Confidence index: {prediction.confidence * 100.0:.0f}% "
                "(not probability)"
                if prediction.confidence is not None
                else "Confidence index: —"
            )
            self.relative_distance_var.set(
                relative_text + "\n" + confidence_text
            )

        # A different raw candidate may be present, but the confirmed label is
        # deliberately held until that candidate persists for the required
        # number of recognition updates.
        if stabilized.pending:
            candidate = stabilized.candidate_label or "prediction"
            self.rejection_reason_var.set(
                f"Holding '{prediction.label}' — checking '{candidate}' "
                f"({stabilized.candidate_count}/{stabilized.required_count})"
            )
            return

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

            # Dynamic recognition uses a fixed-rate landmark stream and only
            # runs DTW after a motion segment completes, keeping the preview
            # responsive and leaving static recognition untouched.
            self.process_dynamic_runtime(now)

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
        self.save_dynamic_gesture_memory()
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
