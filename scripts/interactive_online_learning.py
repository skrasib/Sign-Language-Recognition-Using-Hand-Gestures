from pathlib import Path
import logging
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
from adaptive_gesture.utils.logging_config import configure_logging


logger = logging.getLogger(__name__)


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
            logger.exception("Failed to restore static gesture memory")

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
            logger.exception("Failed to restore dynamic gesture memory")

        # Camera.
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.error("Could not open webcam device 0 using CAP_DSHOW")
            raise RuntimeError("Could not open webcam.")
        logger.info("Webcam opened successfully")
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

    def setup_styles(self):
        """Create a restrained dark UI using only built-in ttk/Tkinter."""
        self.colors = {
            "bg": "#0B1120",
            "surface": "#111827",
            "card": "#172033",
            "card_alt": "#1C2940",
            "border": "#2A3954",
            "text": "#F8FAFC",
            "muted": "#9CA9BC",
            "accent": "#3B82F6",
            "accent_hover": "#2563EB",
            "success": "#22C55E",
            "danger": "#EF4444",
            "camera": "#020617",
        }

        self.root.configure(bg=self.colors["bg"])
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=self.colors["bg"])
        style.configure("Surface.TFrame", background=self.colors["surface"])
        style.configure("Card.TFrame", background=self.colors["card"])
        style.configure("CardAlt.TFrame", background=self.colors["card_alt"])

        style.configure(
            "Title.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            font=("Segoe UI", 15, "bold"),
        )
        style.configure(
            "CardTitle.TLabel",
            background=self.colors["card"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "CardText.TLabel",
            background=self.colors["card"],
            foreground=self.colors["text"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "CardMuted.TLabel",
            background=self.colors["card"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "Prediction.TLabel",
            background=self.colors["card"],
            foreground=self.colors["text"],
            font=("Segoe UI", 28, "bold"),
        )
        style.configure(
            "DynamicPrediction.TLabel",
            background=self.colors["card"],
            foreground=self.colors["text"],
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "CameraHeader.TLabel",
            background=self.colors["surface"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "CameraStatus.TLabel",
            background=self.colors["surface"],
            foreground=self.colors["text"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "PageText.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 10),
        )

        style.configure(
            "Primary.TButton",
            background=self.colors["accent"],
            foreground="#FFFFFF",
            borderwidth=0,
            focusthickness=0,
            padding=(14, 9),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.colors["accent_hover"]), ("disabled", "#334155")],
            foreground=[("disabled", "#94A3B8")],
        )
        style.configure(
            "Secondary.TButton",
            background=self.colors["card_alt"],
            foreground=self.colors["text"],
            borderwidth=0,
            padding=(12, 8),
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#263650"), ("disabled", "#1F2937")],
            foreground=[("disabled", "#64748B")],
        )
        style.configure(
            "Danger.TButton",
            background="#3B1C25",
            foreground="#FCA5A5",
            borderwidth=0,
            padding=(12, 8),
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Danger.TButton", background=[("active", "#55232E")])

        style.configure(
            "Nav.TButton",
            background=self.colors["surface"],
            foreground=self.colors["muted"],
            borderwidth=0,
            padding=(12, 9),
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Nav.TButton", background=[("active", self.colors["card_alt"])])
        style.configure(
            "NavActive.TButton",
            background=self.colors["accent"],
            foreground="#FFFFFF",
            borderwidth=0,
            padding=(12, 9),
            font=("Segoe UI", 9, "bold"),
        )

        style.configure(
            "Modern.TEntry",
            fieldbackground=self.colors["card_alt"],
            foreground=self.colors["text"],
            insertcolor=self.colors["text"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
            padding=8,
        )
        style.configure(
            "Modern.TCombobox",
            fieldbackground=self.colors["card_alt"],
            background=self.colors["card_alt"],
            foreground=self.colors["text"],
            arrowcolor=self.colors["text"],
            bordercolor=self.colors["border"],
            padding=6,
        )
        style.map(
            "Modern.TCombobox",
            fieldbackground=[("readonly", self.colors["card_alt"])],
            foreground=[("readonly", self.colors["text"])],
        )

        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=self.colors["card_alt"],
            background=self.colors["accent"],
            bordercolor=self.colors["card_alt"],
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
            thickness=10,
        )

        style.configure(
            "Treeview",
            background=self.colors["card"],
            fieldbackground=self.colors["card"],
            foreground=self.colors["text"],
            rowheight=31,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.map(
            "Treeview",
            background=[("selected", self.colors["accent"])],
            foreground=[("selected", "#FFFFFF")],
        )
        style.configure(
            "Treeview.Heading",
            background=self.colors["card_alt"],
            foreground=self.colors["muted"],
            borderwidth=0,
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padding=(6, 7),
        )
        style.map("Treeview.Heading", background=[("active", "#263650")])

    def build_ui(self):
        self.setup_styles()
        self.root.geometry("1500x900")
        self.root.minsize(1180, 760)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ----------------------------------------------------
        # App header
        # ----------------------------------------------------
        header = ttk.Frame(self.root, style="App.TFrame", padding=(22, 14, 22, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title_block = ttk.Frame(header, style="App.TFrame")
        title_block.grid(row=0, column=0, sticky="w")
        ttk.Label(
            title_block,
            text="Adaptive Sign & Gesture Recognition",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            title_block,
            text="Personalized online few-shot learning • Static + dynamic gestures",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        privacy = ttk.Label(
            header,
            text="●  Landmark-only memory • No video stored",
            style="Subtitle.TLabel",
        )
        privacy.grid(row=0, column=1, rowspan=2, sticky="e")

        # ----------------------------------------------------
        # Main body: persistent camera + navigable workspace
        # ----------------------------------------------------
        body = ttk.Frame(self.root, style="App.TFrame", padding=(20, 6, 20, 18))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=7, uniform="main")
        body.columnconfigure(1, weight=5, uniform="main")
        body.rowconfigure(0, weight=1)

        # Camera surface.
        camera_shell = ttk.Frame(body, style="Surface.TFrame", padding=12)
        camera_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        camera_shell.columnconfigure(0, weight=1)
        camera_shell.rowconfigure(1, weight=1)

        camera_header = ttk.Frame(camera_shell, style="Surface.TFrame")
        camera_header.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        camera_header.columnconfigure(0, weight=1)
        ttk.Label(
            camera_header,
            text="LIVE CAMERA",
            style="CameraHeader.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            camera_header,
            text="MediaPipe • 1–2 hands",
            style="CameraHeader.TLabel",
        ).grid(row=0, column=1, sticky="e")

        camera_canvas = tk.Frame(
            camera_shell,
            bg=self.colors["camera"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        camera_canvas.grid(row=1, column=0, sticky="nsew")
        camera_canvas.rowconfigure(0, weight=1)
        camera_canvas.columnconfigure(0, weight=1)

        self.video_label = tk.Label(
            camera_canvas,
            bg=self.colors["camera"],
            bd=0,
            anchor="center",
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        camera_footer = ttk.Frame(camera_shell, style="Surface.TFrame")
        camera_footer.grid(row=2, column=0, sticky="ew", pady=(9, 0))
        camera_footer.columnconfigure(0, weight=1)
        self.tracking_label = ttk.Label(
            camera_footer,
            textvariable=self.tracking_var,
            style="CameraStatus.TLabel",
        )
        self.tracking_label.grid(row=0, column=0, sticky="w")
        ttk.Label(
            camera_footer,
            text="Camera frames stay in memory only",
            style="CameraHeader.TLabel",
        ).grid(row=0, column=1, sticky="e")

        # Right workspace.
        workspace = ttk.Frame(body, style="Surface.TFrame", padding=12)
        workspace.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(1, weight=1)

        nav = ttk.Frame(workspace, style="Surface.TFrame")
        nav.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for column in range(5):
            nav.columnconfigure(column, weight=1)

        self.nav_buttons = {}
        nav_items = [
            ("Live", "Live"),
            ("Teach", "Teach"),
            ("Library", "Library"),
            ("Dynamic", "Dynamic"),
            ("Settings", "About"),
        ]
        for column, (key, label) in enumerate(nav_items):
            button = ttk.Button(
                nav,
                text=label,
                style="Nav.TButton",
                command=lambda page=key: self.show_page(page),
            )
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 3, 0))
            self.nav_buttons[key] = button

        self.page_container = ttk.Frame(workspace, style="App.TFrame")
        self.page_container.grid(row=1, column=0, sticky="nsew")
        self.page_container.columnconfigure(0, weight=1)
        self.page_container.rowconfigure(0, weight=1)

        self.pages = {}
        for name in ("Live", "Teach", "Library", "Dynamic", "Settings"):
            page = ttk.Frame(self.page_container, style="App.TFrame")
            page.grid(row=0, column=0, sticky="nsew")
            page.columnconfigure(0, weight=1)
            self.pages[name] = page

        self.build_live_tab(self.pages["Live"])
        self.build_teach_tab(self.pages["Teach"])
        self.build_library_tab(self.pages["Library"])
        self.build_dynamic_tab(self.pages["Dynamic"])
        self.build_settings_tab(self.pages["Settings"])
        self.show_page("Live")

    def show_page(self, name):
        page = self.pages.get(name)
        if page is None:
            return
        page.tkraise()
        for key, button in self.nav_buttons.items():
            button.configure(style="NavActive.TButton" if key == name else "Nav.TButton")

    def _card(self, parent, row, pady=(0, 10)):
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.grid(row=row, column=0, sticky="ew", pady=pady)
        card.columnconfigure(0, weight=1)
        return card

    def build_live_tab(self, panel):
        panel.rowconfigure(4, weight=1)

        ttk.Label(panel, text="Live Recognition", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(2, 2)
        )
        ttk.Label(
            panel,
            text="Confirmed predictions are temporally stabilized before they are shown.",
            style="PageText.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        recognition = self._card(panel, 2)
        top = ttk.Frame(recognition, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Label(top, text="STATIC RECOGNITION", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            top,
            text="STABILIZED",
            style="CardTitle.TLabel",
        ).grid(row=0, column=1, sticky="e")

        ttk.Label(
            recognition,
            textvariable=self.prediction_var,
            style="Prediction.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(8, 2))
        ttk.Label(
            recognition,
            textvariable=self.rejection_reason_var,
            style="CardMuted.TLabel",
            wraplength=430,
        ).grid(row=2, column=0, sticky="w", pady=(0, 10))

        # Main users only need the confidence index. Raw metrics remain visible
        # as compact research details underneath for development/evaluation.
        ttk.Label(
            recognition,
            textvariable=self.relative_distance_var,
            style="CardText.TLabel",
        ).grid(row=3, column=0, sticky="w")

        details = ttk.Frame(recognition, style="Card.TFrame")
        details.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        details.columnconfigure(0, weight=1)
        ttk.Label(details, textvariable=self.distance_var, style="CardMuted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(details, textvariable=self.threshold_var, style="CardMuted.TLabel").grid(
            row=1, column=0, sticky="w"
        )

        feedback = self._card(panel, 3)
        ttk.Label(feedback, text="TEACH THROUGH FEEDBACK", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 9)
        )
        feedback_buttons = ttk.Frame(feedback, style="Card.TFrame")
        feedback_buttons.grid(row=1, column=0, sticky="ew")
        feedback_buttons.columnconfigure(0, weight=1)
        feedback_buttons.columnconfigure(1, weight=1)

        self.correct_button = ttk.Button(
            feedback_buttons,
            text="✓  Correct",
            style="Primary.TButton",
            command=self.confirm_prediction,
        )
        self.correct_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.wrong_button = ttk.Button(
            feedback_buttons,
            text="✕  Wrong",
            style="Secondary.TButton",
            command=self.begin_correction,
        )
        self.wrong_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.correction_frame = ttk.Frame(feedback, style="CardAlt.TFrame", padding=12)
        self.correction_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.correction_frame.columnconfigure(0, weight=1)
        ttk.Label(
            self.correction_frame,
            text="What was the actual gesture?",
            background=self.colors["card_alt"],
            foreground=self.colors["text"],
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.actual_gesture_combo = ttk.Combobox(
            self.correction_frame,
            textvariable=self.actual_gesture_var,
            state="readonly",
            style="Modern.TCombobox",
        )
        self.actual_gesture_combo.grid(row=1, column=0, sticky="ew", pady=(7, 8))
        ttk.Button(
            self.correction_frame,
            text="Apply Correction",
            style="Primary.TButton",
            command=self.apply_feedback_correction,
        ).grid(row=2, column=0, sticky="ew")
        ttk.Button(
            self.correction_frame,
            text="Mark as Unknown",
            style="Secondary.TButton",
            command=self.mark_feedback_unknown,
        ).grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(
            self.correction_frame,
            text="Cancel",
            style="Secondary.TButton",
            command=self.cancel_feedback,
        ).grid(row=4, column=0, sticky="ew", pady=(6, 0))
        self.correction_frame.grid_remove()

        dynamic = self._card(panel, 4, pady=(0, 10))
        ttk.Label(dynamic, text="DYNAMIC RECOGNITION", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            dynamic,
            textvariable=self.dynamic_prediction_var,
            style="DynamicPrediction.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(7, 2))
        ttk.Label(
            dynamic,
            textvariable=self.dynamic_distance_var,
            style="CardText.TLabel",
        ).grid(row=2, column=0, sticky="w")
        ttk.Label(
            dynamic,
            textvariable=self.dynamic_runtime_state_var,
            style="CardMuted.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(7, 0))

        status = self._card(panel, 5, pady=(0, 0))
        ttk.Label(status, text="SESSION STATUS", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            status,
            textvariable=self.status_var,
            style="CardText.TLabel",
            wraplength=430,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    def build_teach_tab(self, panel):
        ttk.Label(panel, text="Teach a Static Gesture", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(2, 2)
        )
        ttk.Label(
            panel,
            text="Name the gesture, demonstrate it naturally, and Smart Capture keeps only useful examples.",
            style="PageText.TLabel",
            wraplength=450,
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        teach = self._card(panel, 2)
        ttk.Label(teach, text="GESTURE NAME", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.gesture_entry = ttk.Entry(
            teach,
            textvariable=self.gesture_name_var,
            style="Modern.TEntry",
        )
        self.gesture_entry.grid(row=1, column=0, sticky="ew", pady=(7, 10))
        self.gesture_entry.bind("<Return>", lambda event: self.start_new_teaching())

        teaching_buttons = ttk.Frame(teach, style="Card.TFrame")
        teaching_buttons.grid(row=2, column=0, sticky="ew")
        for column in range(3):
            teaching_buttons.columnconfigure(column, weight=1)
        self.teach_button = ttk.Button(
            teaching_buttons,
            text="Teach Gesture",
            style="Primary.TButton",
            command=self.start_new_teaching,
        )
        self.teach_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.finish_button = ttk.Button(
            teaching_buttons,
            text="Finish",
            style="Secondary.TButton",
            command=self.finish_teaching,
            state="disabled",
        )
        self.finish_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.cancel_button = ttk.Button(
            teaching_buttons,
            text="Cancel",
            style="Secondary.TButton",
            command=self.cancel_teaching,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        self.progress = ttk.Progressbar(
            teach,
            maximum=12,
            value=0,
            style="Accent.Horizontal.TProgressbar",
        )
        self.progress.grid(row=3, column=0, sticky="ew", pady=(14, 7))
        ttk.Label(
            teach,
            textvariable=self.status_var,
            style="CardText.TLabel",
            wraplength=430,
        ).grid(row=4, column=0, sticky="w")

        smart = self._card(panel, 3)
        ttk.Label(smart, text="SMART CAPTURE", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        smart.columnconfigure(0, weight=1)
        smart.columnconfigure(1, weight=1)
        self.add_stat_row(smart, 1, "Frames observed", self.observed_var)
        self.add_stat_row(smart, 2, "Useful samples", self.accepted_var)
        self.add_stat_row(smart, 3, "Duplicates ignored", self.duplicate_var)
        self.add_stat_row(smart, 4, "Unstable ignored", self.unstable_var)
        self.add_stat_row(smart, 5, "Current stability", self.stability_var)
        self.add_stat_row(smart, 6, "Input mode", self.input_mode_var)

        hint = self._card(panel, 4, pady=(0, 0))
        ttk.Label(hint, text="HOW TO DEMONSTRATE", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            hint,
            text=(
                "Hold the same sign naturally. Small wrist/pose variation is useful; "
                "large movement or switching the hand configuration is rejected. "
                "One-hand and two-hand static gestures are both supported."
            ),
            style="CardText.TLabel",
            wraplength=430,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    def build_library_tab(self, panel):
        panel.rowconfigure(3, weight=1)
        ttk.Label(panel, text="Gesture Library", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(2, 2)
        )
        ttk.Label(
            panel,
            text="Manage the personalized static gestures stored on this device.",
            style="PageText.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        table_card = self._card(panel, 2)
        table_card.rowconfigure(1, weight=1)
        ttk.Label(table_card, text="LEARNED STATIC GESTURES", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        table_wrap = ttk.Frame(table_card, style="Card.TFrame")
        table_wrap.grid(row=1, column=0, sticky="nsew")
        table_wrap.columnconfigure(0, weight=1)
        table_wrap.rowconfigure(0, weight=1)
        self.gesture_table = ttk.Treeview(
            table_wrap,
            columns=("gesture", "positive", "negative", "prototypes", "spread", "radius", "input"),
            show="headings",
            height=11,
        )
        headings = {
            "gesture": "Gesture",
            "positive": "+",
            "negative": "−",
            "prototypes": "P",
            "spread": "Spread",
            "radius": "Radius",
            "input": "Input",
        }
        widths = {
            "gesture": 130,
            "positive": 38,
            "negative": 38,
            "prototypes": 38,
            "spread": 65,
            "radius": 65,
            "input": 65,
        }
        for column, heading in headings.items():
            self.gesture_table.heading(column, text=heading)
            self.gesture_table.column(
                column,
                width=widths[column],
                minwidth=widths[column],
                anchor="w" if column == "gesture" else "center",
            )
        scroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.gesture_table.yview)
        self.gesture_table.configure(yscrollcommand=scroll.set)
        self.gesture_table.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        management = self._card(panel, 3, pady=(0, 0))
        ttk.Label(management, text="SELECTED GESTURE", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        controls = ttk.Frame(management, style="Card.TFrame")
        controls.grid(row=1, column=0, sticky="ew")
        for column in range(2):
            controls.columnconfigure(column, weight=1)
        ttk.Button(controls, text="Improve", style="Primary.TButton", command=self.improve_selected_gesture).grid(
            row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 5)
        )
        ttk.Button(controls, text="Retrain", style="Secondary.TButton", command=self.retrain_selected_gesture).grid(
            row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 5)
        )
        ttk.Button(controls, text="Rename", style="Secondary.TButton", command=self.rename_selected_gesture).grid(
            row=1, column=0, sticky="ew", padx=(0, 4), pady=5
        )
        ttk.Button(controls, text="Delete", style="Danger.TButton", command=self.delete_selected_gesture).grid(
            row=1, column=1, sticky="ew", padx=(4, 0), pady=5
        )
        ttk.Button(
            management,
            text="Clear All Static Gesture Memory",
            style="Danger.TButton",
            command=self.clear_all_gestures,
        ).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            management,
            text="P = adaptive prototypes. +/− = positive and hard-negative examples.",
            style="CardMuted.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(8, 0))

    def build_dynamic_tab(self, panel):
        panel.rowconfigure(4, weight=1)
        ttk.Label(panel, text="Dynamic Gestures", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(2, 2)
        )
        ttk.Label(
            panel,
            text="Teach movements such as swipe, wave, or circle from live landmark trajectories.",
            style="PageText.TLabel",
            wraplength=450,
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        teach = self._card(panel, 2)
        ttk.Label(teach, text="NEW DYNAMIC GESTURE", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.dynamic_name_entry = ttk.Entry(
            teach,
            textvariable=self.dynamic_name_var,
            style="Modern.TEntry",
        )
        self.dynamic_name_entry.grid(row=1, column=0, sticky="ew", pady=(7, 8))
        self.dynamic_teach_button = ttk.Button(
            teach,
            text="Teach Dynamic Gesture",
            style="Primary.TButton",
            command=self.start_dynamic_teaching,
        )
        self.dynamic_teach_button.grid(row=2, column=0, sticky="ew")

        demo_controls = ttk.Frame(teach, style="Card.TFrame")
        demo_controls.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        for column in range(3):
            demo_controls.columnconfigure(column, weight=1)
        self.dynamic_start_demo_button = ttk.Button(
            demo_controls,
            text="●  Start Demo",
            style="Primary.TButton",
            command=self.start_dynamic_demo,
            state="disabled",
        )
        self.dynamic_start_demo_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.dynamic_stop_demo_button = ttk.Button(
            demo_controls,
            text="■  Stop",
            style="Secondary.TButton",
            command=self.stop_dynamic_demo,
            state="disabled",
        )
        self.dynamic_stop_demo_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.dynamic_cancel_button = ttk.Button(
            demo_controls,
            text="Cancel",
            style="Secondary.TButton",
            command=self.cancel_dynamic_teaching,
            state="disabled",
        )
        self.dynamic_cancel_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        ttk.Label(
            teach,
            textvariable=self.dynamic_demo_progress_var,
            style="CardText.TLabel",
        ).grid(row=4, column=0, sticky="w", pady=(10, 2))
        ttk.Label(
            teach,
            textvariable=self.dynamic_status_var,
            style="CardMuted.TLabel",
            wraplength=430,
        ).grid(row=5, column=0, sticky="w")

        table_card = self._card(panel, 3)
        ttk.Label(table_card, text="LEARNED DYNAMIC GESTURES", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        wrap = ttk.Frame(table_card, style="Card.TFrame")
        wrap.grid(row=1, column=0, sticky="nsew")
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)
        self.dynamic_gesture_table = ttk.Treeview(
            wrap,
            columns=("gesture", "templates", "threshold", "duration", "input"),
            show="headings",
            height=7,
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
            "templates": 55,
            "threshold": 75,
            "duration": 70,
            "input": 65,
        }
        for column, heading in dynamic_headings.items():
            self.dynamic_gesture_table.heading(column, text=heading)
            self.dynamic_gesture_table.column(
                column,
                width=dynamic_widths[column],
                minwidth=dynamic_widths[column],
                anchor="w" if column == "gesture" else "center",
            )
        dscroll = ttk.Scrollbar(wrap, orient="vertical", command=self.dynamic_gesture_table.yview)
        self.dynamic_gesture_table.configure(yscrollcommand=dscroll.set)
        self.dynamic_gesture_table.grid(row=0, column=0, sticky="nsew")
        dscroll.grid(row=0, column=1, sticky="ns")

        management = self._card(panel, 4, pady=(0, 0))
        ttk.Label(management, text="DYNAMIC GESTURE MANAGEMENT", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        buttons = ttk.Frame(management, style="Card.TFrame")
        buttons.grid(row=1, column=0, sticky="ew")
        for column in range(3):
            buttons.columnconfigure(column, weight=1)
        ttk.Button(buttons, text="Rename", style="Secondary.TButton", command=self.rename_selected_dynamic_gesture).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(buttons, text="Delete", style="Danger.TButton", command=self.delete_selected_dynamic_gesture).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(buttons, text="Clear All", style="Danger.TButton", command=self.clear_all_dynamic_gestures).grid(
            row=0, column=2, sticky="ew", padx=(4, 0)
        )
        ttk.Label(
            management,
            text="Each gesture is learned from three live demonstrations by default; only landmark trajectories are persisted.",
            style="CardMuted.TLabel",
            wraplength=430,
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

    def build_settings_tab(self, panel):
        ttk.Label(panel, text="About & Data", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(2, 2)
        )
        ttk.Label(
            panel,
            text="Current runtime configuration and privacy-oriented storage design.",
            style="PageText.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        engine = self._card(panel, 2)
        ttk.Label(engine, text="RECOGNITION ENGINE", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            engine,
            text=(
                "MediaPipe hand landmarks\n"
                "Adaptive multi-prototype few-shot learner\n"
                "Hard-negative feedback learning\n"
                "Prediction stabilization\n"
                "DTW-based dynamic gesture recognition"
            ),
            style="CardText.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(7, 0))

        storage = self._card(panel, 3)
        ttk.Label(storage, text="LOCAL MEMORY", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            storage,
            text=(
                "Static: data/gesture_memory.json\n"
                "Dynamic: data/dynamic_gesture_memory.json\n\n"
                "The application persists normalized numerical hand-landmark data. "
                "It does not intentionally save camera images or videos."
            ),
            style="CardText.TLabel",
            wraplength=430,
        ).grid(row=1, column=0, sticky="w", pady=(7, 0))

        camera = self._card(panel, 4, pady=(0, 0))
        ttk.Label(camera, text="CAMERA", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            camera,
            text="Capture target: 960 × 540 • UI loop: 15 ms • Up to 2 hands",
            style="CardText.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(7, 0))

    def add_stat_row(self, parent, row, label, variable):
        ttk.Label(parent, text=label, style="CardMuted.TLabel").grid(
            row=row, column=0, sticky="w", pady=3
        )
        ttk.Label(parent, textvariable=variable, style="CardText.TLabel").grid(
            row=row, column=1, sticky="e", pady=3
        )

    # ========================================================
    # Persistence
    # ========================================================

    def save_gesture_memory(self):
        try:
            self.gesture_store.save(self.learner)
            return True
        except Exception as error:
            logger.exception("Failed to save static gesture memory")
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
            logger.exception("Failed to save dynamic gesture memory")
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
        if hasattr(self, "pages"):
            self.show_page("Teach")
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
            logger.debug("Discarded invalid dynamic motion segment", exc_info=True)
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
        logger.info("Application shutdown requested")
        self.save_gesture_memory()
        self.save_dynamic_gesture_memory()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.tracker.close()
        cv2.destroyAllWindows()
        self.root.destroy()


def main():
    log_path = configure_logging(PROJECT_ROOT / "logs")
    logger.info("Starting Adaptive Real-Time Hand Gesture Recognition")
    logger.info("Log file: %s", log_path)

    root = tk.Tk()
    try:
        InteractiveGestureApp(root)
        root.mainloop()
    except Exception:
        logger.exception("Fatal application error")
        try:
            root.destroy()
        except tk.TclError:
            pass
        raise


if __name__ == "__main__":
    main()
