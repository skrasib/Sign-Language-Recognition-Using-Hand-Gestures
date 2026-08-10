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
    observation_motion_score,
    prepare_dynamic_trajectory,
)
from adaptive_gesture.features.hand_features import build_frame_features
from adaptive_gesture.learning.dynamic_learner import DynamicGestureLearner
from adaptive_gesture.learning.motion_segmenter import MotionSegmenter
from adaptive_gesture.learning.evt_open_set import EVTOpenSetGestureLearner
from adaptive_gesture.learning.metric_runtime import (
    load_or_train_metric_bank,
    migrate_source_memory_to_metric,
)
from adaptive_gesture.learning.prediction_stabilizer import PredictionStabilizer
from adaptive_gesture.learning.sample_selector import SmartSampleSelector
from adaptive_gesture.learning.teaching_flow import (
    HandReadinessGate,
    ONE_HAND_MODE,
    TWO_HAND_MODE,
    describe_hand_requirement,
    hand_mode_from_signature,
    required_hand_count,
)
from adaptive_gesture.storage.dynamic_gesture_store import DynamicGestureStore
from adaptive_gesture.storage.gesture_store import GestureStore
from adaptive_gesture.tracking.hand_tracker import HandTracker
from adaptive_gesture.ui.layout import choose_window_layout, fit_size, responsive_profile
from adaptive_gesture.utils.logging_config import configure_logging


logger = logging.getLogger(__name__)


class InteractiveGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive Real-Time Hand Gesture Recognition — V3.6.3 Responsive UI")
        self._resize_after_id = None
        self._last_responsive_profile = None

        # V3.6 tracking backend. MediaPipe Tasks LIVE_STREAM performs landmark
        # inference asynchronously, while the existing temporal stabilization layer
        # still produces the same TrackedHand interface used by V3.1-V3.5.
        self.hand_landmarker_model_path = (
            PROJECT_ROOT / "data" / "v3" / "models" / "hand_landmarker.task"
        )
        self.tracker = HandTracker(
            max_num_hands=2,
            model_path=self.hand_landmarker_model_path,
            auto_download_model=True,
        )
        logger.info("V3.6 tracking backend active: %s", self.tracker.backend_name)

        # V3.3 learned metric stage.  The encoder is trained once from the
        # existing V3.1/V3.2 raw hybrid landmark memory, persisted, and then
        # frozen during normal runtime teaching.  New gesture classes therefore
        # still become usable immediately without retraining the encoder.
        self.metric_source_path = (
            PROJECT_ROOT / "data" / "v3" / "gesture_memory_hybrid.json"
        )
        self.metric_encoder_path = (
            PROJECT_ROOT / "data" / "v3" / "metric_encoder_v33.npz"
        )
        try:
            self.metric_bootstrap = load_or_train_metric_bank(
                source_memory_path=self.metric_source_path,
                encoder_path=self.metric_encoder_path,
                embedding_dim=48,
                hidden_dim=96,
                seed=42,
            )
            self.metric_bank = self.metric_bootstrap.bank
            logger.info(
                "Metric embedding ready: source_gestures=%s dimensions=%s trained_now=%s",
                self.metric_bootstrap.source_gesture_count,
                self.metric_bank.active_dimensions,
                self.metric_bootstrap.trained_now,
            )
            for report in self.metric_bootstrap.reports:
                logger.info(
                    "Metric training d=%s classes=%s samples=%s epochs=%s "
                    "loss=%.5f loo=%.3f separation=%.2fx",
                    report.input_dimension,
                    report.class_count,
                    report.sample_count,
                    report.epochs_ran,
                    report.final_loss,
                    report.leave_one_out_accuracy,
                    report.separation_ratio,
                )
        except Exception:
            logger.exception("Failed to initialise V3.3 metric embedding")
            raise

        self.learner = EVTOpenSetGestureLearner(
            radius_multiplier=2.5,
            minimum_threshold=0.035,
            prototype_multiplier=2.2,
            max_prototypes=3,
            evt_tail_size=10,
            evt_min_negatives=3,
            inclusion_threshold=0.35,
            top_k_inclusion=1,
            exemplar_memory_strategy="diversity",
            hard_negative_memory_strategy="boundary_diversity",
        )
        logger.info(
            "V3.4 exemplar memory active: positives=%s hard_negatives=%s",
            self.learner.exemplar_memory_strategy,
            self.learner.hard_negative_memory_strategy,
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
            temporal_prototype_strategy="dtw_barycenter",
            max_temporal_prototypes=2,
            prototype_iterations=4,
        )
        logger.info(
            "V3.5 temporal prototype engine active: strategy=%s max_prototypes=%s",
            self.dynamic_learner.temporal_prototype_strategy,
            self.dynamic_learner.max_temporal_prototypes,
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

        # V3.3 keeps metric-space runtime memory separate from the V3.1/V3.2
        # raw hybrid source memory.  On first launch we migrate the old classes
        # through the frozen encoder so the user can compare versions directly.
        self.gesture_store = GestureStore(
            PROJECT_ROOT / "data" / "v3" / "gesture_memory_metric_v33.json"
        )
        self.restore_error = None
        self.metric_migrated_gestures = 0
        try:
            self.restored_gestures = self.gesture_store.load_into(self.learner)
            if (
                self.restored_gestures == 0
                and self.metric_bootstrap.source_gesture_count > 0
            ):
                self.metric_migrated_gestures = migrate_source_memory_to_metric(
                    self.metric_bootstrap.source_learner,
                    self.learner,
                    self.metric_bank,
                )
                self.restored_gestures = self.metric_migrated_gestures
                if self.metric_migrated_gestures:
                    self.gesture_store.save(self.learner)
                    logger.info(
                        "Migrated %s hybrid gestures into V3.3 metric memory",
                        self.metric_migrated_gestures,
                    )
        except Exception as error:
            self.restored_gestures = 0
            self.restore_error = str(error)
            logger.exception("Failed to restore V3.3 static gesture memory")

        # Dynamic gesture trajectories are stored separately from the existing
        # static memory. This avoids risky schema changes and stores landmarks
        # only -- never camera frames or videos.
        self.dynamic_gesture_store = DynamicGestureStore(
            PROJECT_ROOT / "data" / "v3" / "dynamic_gesture_memory.json"
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
        self.current_raw_features = None
        self.current_features = None
        self.current_hand_signature = None
        self.current_prediction = None
        self.current_raw_prediction = None
        self.current_hands = []
        # Latest completed HandLandmarker callback timestamp. Dynamic sampling
        # uses this to avoid duplicating the same asynchronous result.
        self.current_tracking_timestamp_ms = None

        # Keep the camera preview smooth. Recognition/UI metrics do not need
        # to be recalculated at camera-frame rate. Updating those values less
        # frequently also prevents Tkinter from redrawing several changing
        # labels on every frame in one-hand mode.
        self.prediction_update_interval = 0.08
        self.last_prediction_update = 0.0

        # Static teaching state.  V3.6.2 makes teaching hand-count aware: the
        # user chooses one or two hands before starting, and Smart Capture stays
        # armed until that exact configuration has been stable for several fresh
        # asynchronous tracker results.
        self.teaching = False
        self.teaching_mode = None  # new | improve | retrain
        self.teaching_name = None
        self.teaching_signature = None
        self.required_teaching_signature = None
        self.teaching_required_hand_count = 1
        self.teaching_phase = None  # WAITING_HANDS | COUNTDOWN | CAPTURING
        self.teaching_readiness_gate = None
        self.teaching_countdown_seconds = 1.5
        self.prepare_until = None

        # Dynamic teaching/runtime state.  Teaching is fully hands-free after
        # the initial button press: required hands -> still start pose -> motion
        # onset -> motion offset -> accepted demo -> automatically arm next demo.
        self.dynamic_teaching = False
        self.dynamic_demo_recording = False
        self.dynamic_teaching_name = None
        self.dynamic_required_signature = None
        self.dynamic_required_hand_count = 1
        self.dynamic_teaching_phase = None
        self.dynamic_readiness_gate = None
        self.dynamic_training_segmenter = MotionSegmenter()
        self.dynamic_previous_ready_observation = None
        self.dynamic_next_arm_time = 0.0
        self.dynamic_inter_demo_delay = 0.75
        self.dynamic_demo_observations = []
        self.dynamic_templates = []
        self.dynamic_target_demos = self.dynamic_learner.minimum_templates
        self.dynamic_sample_interval = 0.04  # ~25 Hz temporal sampling
        self.last_dynamic_sample_time = 0.0
        self.last_dynamic_tracking_timestamp_ms = None
        self.dynamic_prediction_hold_until = 0.0

        # Feedback state.
        self.feedback_features = None
        self.feedback_predicted_label = None
        self.feedback_nearest_label = None
        self.feedback_hand_signature = None

        # Tk variables.
        self.gesture_name_var = tk.StringVar()
        self.static_hand_mode_var = tk.StringVar(value=ONE_HAND_MODE)
        self.teaching_state_var = tk.StringVar(value="● IDLE")
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
        self.dynamic_hand_mode_var = tk.StringVar(value=ONE_HAND_MODE)
        self.dynamic_teaching_state_var = tk.StringVar(value="● IDLE")
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
            metric_text = (
                f"metric encoder active for raw dimensions "
                f"{self.metric_bank.active_dimensions}"
                if self.metric_bank.is_active
                else "metric encoder unavailable — using hybrid fallback"
            )
            migration_text = (
                " Migrated from V3.2 source memory."
                if self.metric_migrated_gestures
                else ""
            )
            self.status_var.set(
                f"✓ Restored {self.restored_gestures} learned gesture{suffix}; "
                f"{metric_text}." + migration_text
            )
        else:
            if self.metric_bank.is_active:
                self.status_var.set(
                    "Metric encoder ready. No V3.3 gesture memory yet; teach a "
                    "new gesture to begin."
                )
            else:
                self.status_var.set(
                    "No metric encoder could be trained from V3.2 memory. "
                    "V3.3 is using the hybrid descriptor fallback."
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
            "HandMode.TRadiobutton",
            background=self.colors["card"],
            foreground=self.colors["text"],
            font=("Segoe UI", 9, "bold"),
            padding=(4, 2),
        )
        style.map(
            "HandMode.TRadiobutton",
            background=[("active", self.colors["card"])],
            foreground=[("disabled", self.colors["muted"])],
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
        screen_layout = choose_window_layout(
            self.root.winfo_screenwidth(),
            self.root.winfo_screenheight(),
        )
        self.root.geometry(screen_layout.geometry)
        self.root.minsize(screen_layout.min_width, screen_layout.min_height)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ----------------------------------------------------
        # App header
        # ----------------------------------------------------
        header = ttk.Frame(self.root, style="App.TFrame", padding=(22, 14, 22, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.header = header

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
        self.privacy_label = privacy

        # ----------------------------------------------------
        # Main body: persistent camera + navigable workspace
        # ----------------------------------------------------
        body = ttk.Frame(self.root, style="App.TFrame", padding=(20, 6, 20, 18))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=7, uniform="main")
        body.columnconfigure(1, weight=5, uniform="main")
        body.rowconfigure(0, weight=1)
        self.body = body

        # Camera surface.
        camera_shell = ttk.Frame(body, style="Surface.TFrame", padding=12)
        camera_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        camera_shell.columnconfigure(0, weight=1)
        camera_shell.rowconfigure(1, weight=1)
        self.camera_shell = camera_shell

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

        # A Canvas is used instead of an image Label so the PhotoImage does not
        # dictate the requested widget size. This lets the camera region shrink
        # and grow naturally when the window is restored, resized, or maximized.
        self.camera_canvas = tk.Canvas(
            camera_shell,
            bg=self.colors["camera"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            bd=0,
        )
        self.camera_canvas.grid(row=1, column=0, sticky="nsew")
        self.camera_image_item = self.camera_canvas.create_image(
            0,
            0,
            anchor="center",
        )
        self.camera_photo = None

        camera_footer = ttk.Frame(camera_shell, style="Surface.TFrame")
        camera_footer.grid(row=2, column=0, sticky="ew", pady=(9, 0))
        camera_footer.columnconfigure(0, weight=1)
        self.tracking_label = ttk.Label(
            camera_footer,
            textvariable=self.tracking_var,
            style="CameraStatus.TLabel",
        )
        self.tracking_label.grid(row=0, column=0, sticky="w")
        self.camera_footer_note = ttk.Label(
            camera_footer,
            text="Camera frames stay in memory only",
            style="CameraHeader.TLabel",
        )
        self.camera_footer_note.grid(row=0, column=1, sticky="e")

        # Right workspace.
        workspace = ttk.Frame(body, style="Surface.TFrame", padding=12)
        workspace.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(1, weight=1)
        self.workspace = workspace

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

        # Each page is hosted inside a vertically scrollable canvas. On taller
        # windows the inner page expands to fill the available height; on laptop
        # screens only the overflowing page content scrolls instead of being cut
        # off below the window.
        self.pages = {}
        self.page_hosts = {}
        self.page_canvases = {}
        self.page_scrollbars = {}
        self.page_window_items = {}
        self.current_page_name = None
        for name in ("Live", "Teach", "Library", "Dynamic", "Settings"):
            self._create_scrollable_page(name)

        self.build_live_tab(self.pages["Live"])
        self.build_teach_tab(self.pages["Teach"])
        self.build_library_tab(self.pages["Library"])
        self.build_dynamic_tab(self.pages["Dynamic"])
        self.build_settings_tab(self.pages["Settings"])
        self.show_page("Live")

        # Debounced resize handling keeps layout updates cheap while the user
        # drags a window edge or switches between restored and maximized states.
        self.root.bind("<Configure>", self._on_root_configure, add="+")
        self.root.bind_all("<MouseWheel>", self._on_page_mousewheel, add="+")
        self.root.after_idle(self._apply_responsive_layout)

    def _create_scrollable_page(self, name):
        host = ttk.Frame(self.page_container, style="App.TFrame")
        host.grid(row=0, column=0, sticky="nsew")
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)

        canvas = tk.Canvas(
            host,
            bg=self.colors["bg"],
            highlightthickness=0,
            bd=0,
        )
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(host, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        page = ttk.Frame(canvas, style="App.TFrame")
        page.columnconfigure(0, weight=1)
        window_item = canvas.create_window((0, 0), window=page, anchor="nw")

        self.page_hosts[name] = host
        self.page_canvases[name] = canvas
        self.page_scrollbars[name] = scrollbar
        self.page_window_items[name] = window_item
        self.pages[name] = page

        page.bind(
            "<Configure>",
            lambda _event, page_name=name: self._schedule_page_geometry_sync(page_name),
            add="+",
        )
        canvas.bind(
            "<Configure>",
            lambda event, page_name=name: self._resize_page_window(page_name, event),
            add="+",
        )

    def _schedule_page_geometry_sync(self, name):
        self.root.after_idle(lambda page_name=name: self._sync_page_geometry(page_name))

    def _resize_page_window(self, name, event):
        page = self.pages[name]
        canvas = self.page_canvases[name]
        item = self.page_window_items[name]
        requested_height = max(1, page.winfo_reqheight())
        target_height = max(int(event.height), requested_height)
        canvas.itemconfigure(
            item,
            width=max(1, int(event.width)),
            height=target_height,
        )
        self._schedule_page_geometry_sync(name)

    def _sync_page_geometry(self, name):
        canvas = self.page_canvases.get(name)
        page = self.pages.get(name)
        scrollbar = self.page_scrollbars.get(name)
        item = self.page_window_items.get(name)
        if canvas is None or page is None or scrollbar is None or item is None:
            return

        canvas_width = max(1, canvas.winfo_width())
        canvas_height = max(1, canvas.winfo_height())
        requested_height = max(1, page.winfo_reqheight())
        target_height = max(canvas_height, requested_height)
        canvas.itemconfigure(item, width=canvas_width, height=target_height)
        canvas.configure(scrollregion=(0, 0, canvas_width, target_height))

        if requested_height > canvas_height + 2:
            scrollbar.grid()
        else:
            scrollbar.grid_remove()
            canvas.yview_moveto(0.0)

    def _on_page_mousewheel(self, event):
        if not self.current_page_name:
            return None

        widget_class = ""
        try:
            widget_class = event.widget.winfo_class()
        except Exception:
            pass
        # Let controls with their own scrolling consume the wheel normally.
        if widget_class in {"Treeview", "TCombobox", "Listbox", "Text"}:
            return None

        try:
            pointer_x = self.root.winfo_pointerx()
            pointer_y = self.root.winfo_pointery()
            wx = self.workspace.winfo_rootx()
            wy = self.workspace.winfo_rooty()
            ww = self.workspace.winfo_width()
            wh = self.workspace.winfo_height()
            if not (wx <= pointer_x <= wx + ww and wy <= pointer_y <= wy + wh):
                return None
        except tk.TclError:
            return None

        canvas = self.page_canvases.get(self.current_page_name)
        if canvas is None:
            return None
        bbox = canvas.cget("scrollregion")
        if not bbox:
            return None
        units = int(-event.delta / 120) if event.delta else 0
        if units:
            canvas.yview_scroll(units, "units")
            return "break"
        return None

    def show_page(self, name):
        host = self.page_hosts.get(name)
        if host is None:
            return
        self.current_page_name = name
        host.tkraise()
        self._schedule_page_geometry_sync(name)
        for key, button in self.nav_buttons.items():
            button.configure(style="NavActive.TButton" if key == name else "Nav.TButton")

    def _on_root_configure(self, event):
        if event.widget is not self.root:
            return
        if event.width < 200 or event.height < 200:
            return
        if self._resize_after_id is not None:
            try:
                self.root.after_cancel(self._resize_after_id)
            except tk.TclError:
                pass
        self._resize_after_id = self.root.after(90, self._apply_responsive_layout)

    def _apply_responsive_layout(self):
        self._resize_after_id = None
        width = max(1, self.root.winfo_width())
        profile = responsive_profile(width)

        self.body.configure(padding=profile.outer_padding)
        self.body.columnconfigure(0, weight=profile.camera_weight, uniform="main")
        self.body.columnconfigure(1, weight=profile.workspace_weight, uniform="main")
        self.camera_shell.configure(padding=profile.inner_padding)
        self.workspace.configure(padding=profile.inner_padding)
        self.camera_shell.grid_configure(padx=(0, profile.column_gap))
        self.workspace.grid_configure(padx=(profile.column_gap, 0))

        if profile.hide_secondary_header:
            self.privacy_label.grid_remove()
            self.camera_footer_note.grid_remove()
        else:
            self.privacy_label.grid()
            self.camera_footer_note.grid()

        workspace_width = max(320, self.workspace.winfo_width())
        wraplength = max(250, workspace_width - 70)
        self._update_wrapped_labels(self.workspace, wraplength)

        for name in self.pages:
            self._sync_page_geometry(name)
        self._last_responsive_profile = profile

    def _update_wrapped_labels(self, parent, wraplength):
        for widget in parent.winfo_children():
            if isinstance(widget, ttk.Label):
                try:
                    current = int(float(widget.cget("wraplength")))
                except (tk.TclError, TypeError, ValueError):
                    current = 0
                if current > 0:
                    try:
                        widget.configure(wraplength=wraplength)
                    except tk.TclError:
                        pass
            self._update_wrapped_labels(widget, wraplength)

    def _render_camera_frame(self, image):
        canvas_width = max(1, self.camera_canvas.winfo_width() - 4)
        canvas_height = max(1, self.camera_canvas.winfo_height() - 4)

        # During the very first Tk layout pass the canvas can temporarily report
        # 1x1. Use a modest fallback for that single frame; subsequent frames
        # immediately adopt the true resized canvas dimensions.
        if canvas_width < 20 or canvas_height < 20:
            canvas_width, canvas_height = 640, 360

        target_width, target_height = fit_size(
            image.width,
            image.height,
            canvas_width,
            canvas_height,
        )
        if (target_width, target_height) != image.size:
            image = image.resize(
                (target_width, target_height),
                Image.Resampling.BILINEAR,
            )

        photo = ImageTk.PhotoImage(image=image)
        self.camera_photo = photo
        center_x = max(1, self.camera_canvas.winfo_width()) / 2
        center_y = max(1, self.camera_canvas.winfo_height()) / 2
        self.camera_canvas.itemconfigure(self.camera_image_item, image=photo)
        self.camera_canvas.coords(self.camera_image_item, center_x, center_y)

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

        hand_mode = ttk.Frame(teach, style="Card.TFrame")
        hand_mode.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        hand_mode.columnconfigure(1, weight=1)
        hand_mode.columnconfigure(2, weight=1)
        ttk.Label(hand_mode, text="HANDS", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        self.static_one_hand_radio = ttk.Radiobutton(
            hand_mode,
            text="One hand",
            variable=self.static_hand_mode_var,
            value=ONE_HAND_MODE,
            style="HandMode.TRadiobutton",
        )
        self.static_one_hand_radio.grid(row=0, column=1, sticky="w")
        self.static_two_hand_radio = ttk.Radiobutton(
            hand_mode,
            text="Two hands",
            variable=self.static_hand_mode_var,
            value=TWO_HAND_MODE,
            style="HandMode.TRadiobutton",
        )
        self.static_two_hand_radio.grid(row=0, column=2, sticky="w")

        teaching_buttons = ttk.Frame(teach, style="Card.TFrame")
        teaching_buttons.grid(row=3, column=0, sticky="ew")
        for column in range(3):
            teaching_buttons.columnconfigure(column, weight=1)
        self.teach_button = ttk.Button(
            teaching_buttons,
            text="Start Teaching",
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
        self.progress.grid(row=4, column=0, sticky="ew", pady=(14, 7))
        ttk.Label(
            teach,
            textvariable=self.teaching_state_var,
            style="CardTitle.TLabel",
        ).grid(row=5, column=0, sticky="w", pady=(0, 4))
        ttk.Label(
            teach,
            textvariable=self.status_var,
            style="CardText.TLabel",
            wraplength=430,
        ).grid(row=6, column=0, sticky="w")

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
                "Choose One hand or Two hands before Start Teaching. The session "
                "waits until that exact hand count is stable, then Smart Capture "
                "starts automatically. Small wrist/pose variation is useful."
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
            text=(
                "P = adaptive prototypes. +/− = positive and hard-negative examples. "
                "V3.4 keeps a diversity-aware bounded exemplar memory."
            ),
            style="CardMuted.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(8, 0))

    def build_dynamic_tab(self, panel):
        panel.rowconfigure(4, weight=1)
        ttk.Label(panel, text="Dynamic Gestures", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(2, 2)
        )
        ttk.Label(
            panel,
            text=(
                "Choose one or two hands, press Start Hands-Free Teaching once, "
                "then perform each movement without touching the mouse."
            ),
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

        hand_mode = ttk.Frame(teach, style="Card.TFrame")
        hand_mode.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        hand_mode.columnconfigure(1, weight=1)
        hand_mode.columnconfigure(2, weight=1)
        ttk.Label(hand_mode, text="HANDS", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        self.dynamic_one_hand_radio = ttk.Radiobutton(
            hand_mode,
            text="One hand",
            variable=self.dynamic_hand_mode_var,
            value=ONE_HAND_MODE,
            style="HandMode.TRadiobutton",
        )
        self.dynamic_one_hand_radio.grid(row=0, column=1, sticky="w")
        self.dynamic_two_hand_radio = ttk.Radiobutton(
            hand_mode,
            text="Two hands",
            variable=self.dynamic_hand_mode_var,
            value=TWO_HAND_MODE,
            style="HandMode.TRadiobutton",
        )
        self.dynamic_two_hand_radio.grid(row=0, column=2, sticky="w")

        actions = ttk.Frame(teach, style="Card.TFrame")
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure(0, weight=2)
        actions.columnconfigure(1, weight=1)
        self.dynamic_teach_button = ttk.Button(
            actions,
            text="Start Hands-Free Teaching",
            style="Primary.TButton",
            command=self.start_dynamic_teaching,
        )
        self.dynamic_teach_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.dynamic_cancel_button = ttk.Button(
            actions,
            text="Cancel",
            style="Secondary.TButton",
            command=self.cancel_dynamic_teaching,
            state="disabled",
        )
        self.dynamic_cancel_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(
            teach,
            textvariable=self.dynamic_demo_progress_var,
            style="CardText.TLabel",
        ).grid(row=4, column=0, sticky="w", pady=(10, 2))
        ttk.Label(
            teach,
            textvariable=self.dynamic_teaching_state_var,
            style="CardTitle.TLabel",
        ).grid(row=5, column=0, sticky="w", pady=(2, 4))
        ttk.Label(
            teach,
            textvariable=self.dynamic_status_var,
            style="CardMuted.TLabel",
            wraplength=430,
        ).grid(row=6, column=0, sticky="w")

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
            columns=("gesture", "templates", "prototypes", "threshold", "duration", "input"),
            show="headings",
            height=7,
        )
        dynamic_headings = {
            "gesture": "Gesture",
            "templates": "Demos",
            "prototypes": "Protos",
            "threshold": "Threshold",
            "duration": "Median s",
            "input": "Input",
        }
        dynamic_widths = {
            "gesture": 130,
            "templates": 55,
            "prototypes": 55,
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
            text="Each gesture is learned from three live demonstrations by default. V3.5 derives DTW-aligned temporal prototype(s); only landmark trajectories are persisted.",
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
                "V3 hybrid geometry descriptor (XYZ + joint angles)\n"
                "Frozen learned metric embedding (when source memory supports it)\n"
                "EVT open-set rejection + adaptive multi-prototypes\n"
                "Hard-negative feedback learning\n"
                "Prediction stabilization\n"
                "DTW-aligned temporal-prototype dynamic gesture recognition"
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
                "Static: data/v3/gesture_memory_hybrid.json\n"
                "Dynamic: data/v3/dynamic_gesture_memory.json\n\n"
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

        if required_signature is not None:
            hand_mode = hand_mode_from_signature(required_signature)
            self.static_hand_mode_var.set(hand_mode)
            required_count = required_hand_count(hand_mode)
            exact_signature = required_signature
        else:
            hand_mode = self.static_hand_mode_var.get()
            required_count = required_hand_count(hand_mode)
            # Two-hand input always has the learner signature Both. For a new
            # one-hand gesture, Left/Right is locked only after the readiness
            # gate has observed a stable consistent hand.
            exact_signature = "Both" if required_count == 2 else None

        self.teaching = True
        self.teaching_mode = mode
        self.teaching_name = name
        self.teaching_signature = None
        self.required_teaching_signature = exact_signature
        self.teaching_required_hand_count = required_count
        self.teaching_phase = "WAITING_HANDS"
        self.teaching_readiness_gate = HandReadinessGate(
            required_count,
            expected_signature=exact_signature,
            confirm_results=5,
            hold_seconds=0.30,
        )
        self.prepare_until = None

        self.progress["value"] = 0
        self.reset_stats_display()
        self.teach_button.config(state="disabled")
        self.finish_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.gesture_entry.config(state="disabled")
        self.static_one_hand_radio.config(state="disabled")
        self.static_two_hand_radio.config(state="disabled")
        self.set_feedback_buttons_enabled(False)

        action = {
            "new": "teach",
            "improve": "improve",
            "retrain": "retrain",
        }[mode]
        requirement = describe_hand_requirement(
            required_count,
            exact_signature if mode != "new" else None,
        )
        self.teaching_state_var.set(
            f"● WAITING FOR {requirement.upper()}"
        )
        self.status_var.set(
            f"Ready to {action} '{name}'. Show {requirement}; capture will "
            "start automatically only after the required hand configuration "
            "is stable."
        )

    def _static_teaching_configuration_matches(self) -> bool:
        if self.current_feature_set is None or self.teaching_signature is None:
            return False
        return (
            self.current_feature_set.hand_count == self.teaching_required_hand_count
            and self.current_feature_set.hand_signature == self.teaching_signature
        )

    def process_teaching_frame(self):
        if not self.teaching or self.teaching_readiness_gate is None:
            return

        now = time.monotonic()
        feature_set = self.current_feature_set
        hand_count = feature_set.hand_count if feature_set is not None else 0
        hand_signature = (
            feature_set.hand_signature if feature_set is not None else None
        )
        requirement = describe_hand_requirement(
            self.teaching_required_hand_count,
            self.required_teaching_signature,
        )

        if self.teaching_phase == "WAITING_HANDS":
            readiness = self.teaching_readiness_gate.update(
                hand_count=hand_count,
                hand_signature=hand_signature,
                result_token=self.current_tracking_timestamp_ms,
                now=now,
            )
            self.teaching_state_var.set(
                f"● WAITING FOR {requirement.upper()}"
            )

            if not readiness.ready:
                if hand_count == 0:
                    detail = "No hand detected yet."
                elif hand_count != self.teaching_required_hand_count:
                    detail = (
                        f"Currently detecting {hand_count}; "
                        f"this gesture requires {self.teaching_required_hand_count}."
                    )
                else:
                    detail = "Hold that configuration steady for a moment."
                self.status_var.set(
                    f"Teaching is armed but not capturing. {detail}"
                )
                return

            self.teaching_signature = (
                self.required_teaching_signature or readiness.hand_signature
            )
            if self.teaching_signature == "Both":
                # Two-hand vectors include relative geometry and naturally vary
                # more than one-hand vectors.
                self.selector.stability_threshold = 0.050
                self.selector.duplicate_threshold = 0.025

            self.teaching_phase = "COUNTDOWN"
            self.prepare_until = now + self.teaching_countdown_seconds
            self.teaching_state_var.set("● HANDS READY")
            self.status_var.set(
                f"{requirement.capitalize()} detected and stable. Keep the "
                "gesture visible; Smart Capture will start automatically."
            )
            return

        if self.teaching_phase == "COUNTDOWN":
            if not self._static_teaching_configuration_matches():
                self.teaching_phase = "WAITING_HANDS"
                self.prepare_until = None
                self.teaching_signature = (
                    self.required_teaching_signature
                    if self.required_teaching_signature is not None
                    else None
                )
                self.teaching_readiness_gate.reset()
                self.teaching_state_var.set(
                    f"● WAITING FOR {requirement.upper()}"
                )
                self.status_var.set(
                    f"Hand configuration changed before capture. Show "
                    f"{requirement} again."
                )
                return

            remaining = max(0.0, (self.prepare_until or now) - now)
            if remaining > 0.0:
                self.teaching_state_var.set("● HOLD STEADY")
                self.status_var.set(
                    f"Hands ready. Capture starts automatically in {remaining:.1f}s..."
                )
                return

            self.teaching_phase = "CAPTURING"
            self.prepare_until = None
            self.teaching_state_var.set("● CAPTURING")

        if self.teaching_phase != "CAPTURING":
            return

        if not self._static_teaching_configuration_matches():
            self.teaching_state_var.set("● CAPTURE PAUSED")
            self.status_var.set(
                f"Capture paused — keep {requirement} visible using the same "
                "hand configuration."
            )
            return

        self.teaching_state_var.set("● CAPTURING")

        # Smart Capture continues to judge stability/diversity in the raw hybrid
        # geometry space. The learned encoder may intentionally compress same-
        # class variation, which would otherwise make useful live samples look
        # like duplicates before they reach the learner.
        self.selector.consider(self.current_raw_features)
        self.update_stats_display()
        self.progress["value"] = len(self.selector.samples)

        if self.selector.ready:
            self.finish_button.config(state="normal")
            if not self.selector.complete:
                self.status_var.set(
                    "Enough useful samples to learn. Keep the selected hand "
                    "configuration visible; the session will finish automatically "
                    "at the target."
                )
        else:
            self.status_var.set(
                "Smart Capture is active. Hold the gesture steady; small natural "
                "variations are useful."
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
        raw_samples = [sample.copy() for sample in self.selector.samples]
        samples = [self.metric_bank.encode(sample) for sample in raw_samples]
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

            # Keep a raw hybrid research archive alongside the frozen metric
            # runtime memory.  This does NOT retrain the encoder automatically;
            # it simply preserves future source material for controlled V3.3
            # experiments or an explicit encoder retraining run.
            source = self.metric_bootstrap.source_learner
            try:
                if mode == "new":
                    if name not in source.gestures:
                        source.learn_gesture(
                            name=name,
                            samples=raw_samples,
                            hand_signature=signature,
                        )
                elif mode == "improve" and name in source.gestures:
                    source.add_samples_to_gesture(
                        name=name,
                        samples=raw_samples,
                        hand_signature=signature,
                    )
                elif mode == "retrain":
                    if name in source.gestures:
                        source.replace_gesture_samples(
                            name=name,
                            samples=raw_samples,
                            hand_signature=signature,
                            keep_negatives=True,
                        )
                    else:
                        source.learn_gesture(
                            name=name,
                            samples=raw_samples,
                            hand_signature=signature,
                        )
                GestureStore(self.metric_source_path).save(source)
            except Exception:
                # The runtime metric memory is authoritative for V3.3.  A failure
                # to update the raw research archive must not discard a gesture
                # that was successfully learned in the live app.
                logger.exception("Could not update raw V3 metric source archive")

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
        self.teaching_phase = None
        self.teaching_readiness_gate = None
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
        self.static_one_hand_radio.config(state="normal")
        self.static_two_hand_radio.config(state="normal")
        self.teaching_state_var.set("● IDLE")
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

        hand_mode = self.dynamic_hand_mode_var.get()
        required_count = required_hand_count(hand_mode)
        exact_signature = "Both" if required_count == 2 else None

        self.dynamic_teaching = True
        self.dynamic_demo_recording = False
        self.dynamic_teaching_name = name
        self.dynamic_required_signature = exact_signature
        self.dynamic_required_hand_count = required_count
        self.dynamic_teaching_phase = "WAITING_HANDS"
        self.dynamic_demo_observations = []
        self.dynamic_templates = []
        self.dynamic_previous_ready_observation = None
        self.dynamic_next_arm_time = 0.0
        self.last_dynamic_tracking_timestamp_ms = None
        self.motion_segmenter.reset()
        self.dynamic_training_segmenter.reset()
        self._reset_dynamic_readiness_gate()
        self.dynamic_demo_progress_var.set(
            f"Demonstrations: 0/{self.dynamic_target_demos}"
        )

        self.dynamic_name_entry.config(state="disabled")
        self.dynamic_one_hand_radio.config(state="disabled")
        self.dynamic_two_hand_radio.config(state="disabled")
        self.dynamic_teach_button.config(state="disabled")
        self.dynamic_cancel_button.config(state="normal")

        requirement = describe_hand_requirement(required_count, exact_signature)
        self.dynamic_teaching_state_var.set(
            f"● WAITING FOR {requirement.upper()}"
        )
        self.dynamic_status_var.set(
            f"Teaching '{name}' is armed. Show {requirement} and hold the "
            "starting pose still. You will not need the mouse again: movement "
            "start and stop are detected automatically."
        )

    def _reset_dynamic_readiness_gate(self):
        self.dynamic_readiness_gate = HandReadinessGate(
            self.dynamic_required_hand_count,
            expected_signature=self.dynamic_required_signature,
            confirm_results=5,
            hold_seconds=0.30,
            max_motion_score=0.030,
        )
        self.dynamic_previous_ready_observation = None

    def _dynamic_requirement_text(self):
        return describe_hand_requirement(
            self.dynamic_required_hand_count,
            self.dynamic_required_signature,
        )

    def _restart_dynamic_demo_arming(self, now, message, *, delay=0.50):
        self.dynamic_demo_recording = False
        self.dynamic_teaching_phase = "BETWEEN_DEMOS"
        self.dynamic_next_arm_time = float(now) + max(0.0, float(delay))
        self.dynamic_training_segmenter.reset()
        self._reset_dynamic_readiness_gate()
        self.dynamic_teaching_state_var.set("● RETURN TO START")
        self.dynamic_status_var.set(message)

    def _accept_dynamic_demo(self, observations, now):
        self.dynamic_demo_recording = False
        try:
            trajectory = prepare_dynamic_trajectory(observations)
        except Exception as error:
            self._restart_dynamic_demo_arming(
                now,
                f"Demo was not accepted: {error} Return to the starting pose; "
                "the same demo will re-arm automatically.",
            )
            return

        self.dynamic_templates.append(trajectory)
        completed = len(self.dynamic_templates)
        self.dynamic_demo_progress_var.set(
            f"Demonstrations: {completed}/{self.dynamic_target_demos}"
        )

        if completed < self.dynamic_target_demos:
            self._restart_dynamic_demo_arming(
                now,
                f"✓ Demo {completed} accepted "
                f"({trajectory.duration_seconds:.2f}s, motion extent "
                f"{trajectory.motion_extent:.2f}). Return to the starting pose "
                "and hold still; the next demo will arm automatically.",
                delay=self.dynamic_inter_demo_delay,
            )
            return

        try:
            gesture = self.dynamic_learner.learn_gesture(
                self.dynamic_teaching_name,
                self.dynamic_templates,
            )
            saved = self.save_dynamic_gesture_memory()
        except Exception as error:
            self.dynamic_teaching_phase = "WAITING_HANDS"
            self.dynamic_training_segmenter.reset()
            self._reset_dynamic_readiness_gate()
            self.dynamic_teaching_state_var.set("● LEARNING ERROR")
            self.dynamic_status_var.set(f"Dynamic learning failed: {error}")
            return

        message = (
            f"✓ Learned dynamic gesture '{gesture.name}' from "
            f"{gesture.template_count} demonstrations and "
            f"{gesture.prototype_count} temporal prototype(s). "
            f"DTW threshold: {gesture.threshold:.4f}."
        )
        if saved:
            message += " Landmark trajectories saved."

        self.finish_dynamic_teaching_state()
        self.refresh_dynamic_gesture_table()
        self.dynamic_status_var.set(message)

    def process_dynamic_training_frame(self, now):
        if not self.dynamic_teaching:
            return
        if now - self.last_dynamic_sample_time < self.dynamic_sample_interval:
            return
        if (
            self.current_tracking_timestamp_ms is None
            or self.current_tracking_timestamp_ms
            == self.last_dynamic_tracking_timestamp_ms
        ):
            return

        self.last_dynamic_sample_time = now
        self.last_dynamic_tracking_timestamp_ms = (
            self.current_tracking_timestamp_ms
        )
        observation = build_dynamic_observation(self.current_hands, now)
        hand_count = len(self.current_hands[:2]) if observation is not None else 0
        hand_signature = observation.hand_signature if observation is not None else None
        requirement = self._dynamic_requirement_text()

        if self.dynamic_teaching_phase == "BETWEEN_DEMOS":
            if now < self.dynamic_next_arm_time:
                self.dynamic_teaching_state_var.set("● RETURN TO START")
                return
            self.dynamic_teaching_phase = "WAITING_HANDS"
            self._reset_dynamic_readiness_gate()

        if self.dynamic_teaching_phase == "WAITING_HANDS":
            motion_score = 0.0
            if observation is not None:
                previous = self.dynamic_previous_ready_observation
                if (
                    previous is not None
                    and previous.hand_signature == observation.hand_signature
                ):
                    motion_score = observation_motion_score(previous, observation)
                self.dynamic_previous_ready_observation = observation
            else:
                self.dynamic_previous_ready_observation = None

            readiness = self.dynamic_readiness_gate.update(
                hand_count=hand_count,
                hand_signature=hand_signature,
                result_token=self.current_tracking_timestamp_ms,
                now=now,
                motion_score=motion_score if observation is not None else None,
            )

            requirement = describe_hand_requirement(
                self.dynamic_required_hand_count,
                self.dynamic_required_signature,
            )
            self.dynamic_teaching_state_var.set(
                f"● WAITING FOR {requirement.upper()}"
            )
            if not readiness.ready:
                if hand_count == 0:
                    detail = "No hand detected."
                elif hand_count != self.dynamic_required_hand_count:
                    detail = (
                        f"Currently detecting {hand_count}; "
                        f"{self.dynamic_required_hand_count} required."
                    )
                elif motion_score > 0.030:
                    detail = "Return to the starting pose and hold still."
                else:
                    detail = "Hold the starting pose still for a moment."
                self.dynamic_status_var.set(
                    f"Demo {len(self.dynamic_templates) + 1}/"
                    f"{self.dynamic_target_demos} is waiting. {detail}"
                )
                return

            if self.dynamic_required_signature is None:
                self.dynamic_required_signature = readiness.hand_signature
                # All later demonstrations use the exact same left/right hand.
                self._reset_dynamic_readiness_gate()
                requirement = self._dynamic_requirement_text()

            self.dynamic_training_segmenter.reset()
            if observation is not None:
                self.dynamic_training_segmenter.update(observation, now)
            self.dynamic_teaching_phase = "READY"
            self.dynamic_teaching_state_var.set("● READY — START MOVING")
            self.dynamic_status_var.set(
                f"Demo {len(self.dynamic_templates) + 1}/"
                f"{self.dynamic_target_demos} ready. Begin the gesture whenever "
                "you are ready; recording will start automatically on movement."
            )
            return

        # READY and RECORDING both require the exact locked configuration.
        if (
            observation is None
            or observation.hand_signature != self.dynamic_required_signature
            or hand_count != self.dynamic_required_hand_count
        ):
            if self.dynamic_teaching_phase == "RECORDING":
                message = (
                    f"Demo interrupted because {requirement} was lost. Return "
                    "to the starting pose; this demo will retry automatically."
                )
            else:
                message = (
                    f"Lost the required {requirement} before movement started. "
                    "Show the same configuration again."
                )
            self._restart_dynamic_demo_arming(now, message)
            return

        result = self.dynamic_training_segmenter.update(observation, now)

        if result.state == self.dynamic_training_segmenter.MOTION:
            self.dynamic_teaching_phase = "RECORDING"
            self.dynamic_demo_recording = True
            self.dynamic_teaching_state_var.set("● RECORDING MOVEMENT")
            self.dynamic_status_var.set(
                f"Recording demo {len(self.dynamic_templates) + 1}/"
                f"{self.dynamic_target_demos}. Finish the movement naturally; "
                "recording will stop automatically when motion settles."
            )

        if result.completed:
            self._accept_dynamic_demo(result.completed, now)
            return

        if self.dynamic_teaching_phase == "READY":
            self.dynamic_teaching_state_var.set("● READY — START MOVING")

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
        self.dynamic_required_hand_count = 1
        self.dynamic_teaching_phase = None
        self.dynamic_readiness_gate = None
        self.dynamic_previous_ready_observation = None
        self.dynamic_next_arm_time = 0.0
        self.dynamic_demo_observations = []
        self.dynamic_templates = []
        self.last_dynamic_tracking_timestamp_ms = None
        self.motion_segmenter.reset()
        self.dynamic_training_segmenter.reset()

        self.dynamic_name_entry.config(state="normal")
        self.dynamic_one_hand_radio.config(state="normal")
        self.dynamic_two_hand_radio.config(state="normal")
        self.dynamic_teach_button.config(state="normal")
        self.dynamic_cancel_button.config(state="disabled")
        self.dynamic_name_var.set("")
        self.dynamic_teaching_state_var.set("● IDLE")
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
        # Hands-free teaching owns the temporal stream. Runtime recognition is
        # paused while the training state machine waits, arms, records, and
        # automatically accepts demonstrations.
        if self.dynamic_teaching:
            self.motion_segmenter.reset()
            self.process_dynamic_training_frame(now)
            phase = self.dynamic_teaching_phase or "TEACHING"
            self.dynamic_runtime_state_var.set(
                "Dynamic teaching: " + phase.replace("_", " ")
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

        if (
            self.current_tracking_timestamp_ms is None
            or self.current_tracking_timestamp_ms == self.last_dynamic_tracking_timestamp_ms
        ):
            return
        self.last_dynamic_sample_time = now
        self.last_dynamic_tracking_timestamp_ms = self.current_tracking_timestamp_ms

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
                f"Temporal-prototype DTW: {prediction.distance:.4f} / "
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
                    f"Nearest temporal-prototype DTW: {prediction.distance:.4f}\n"
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
                    gesture.prototype_count,
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

            open_set_method = getattr(prediction, "open_set_method", None)
            open_set_score = getattr(prediction, "open_set_score", None)
            open_set_threshold = getattr(prediction, "open_set_threshold", None)
            extreme_vector_count = getattr(prediction, "extreme_vector_count", 0)

            if (
                open_set_method == "evt_evm"
                and open_set_score is not None
                and open_set_threshold is not None
            ):
                self.threshold_var.set(
                    f"EVT inclusion: {open_set_score:.3f} / "
                    f"threshold {open_set_threshold:.2f}"
                )
                relative_text = (
                    f"Open-set support: {open_set_score * 100.0:.0f}% "
                    f"• EV models: {extreme_vector_count}"
                )
            else:
                self.threshold_var.set(
                    f"Acceptance threshold: {prediction.threshold:.4f}"
                    if prediction.threshold is not None
                    else "Acceptance threshold: —"
                )
                relative_text = (
                    f"Relative score: {prediction.relative_distance:.2f}x"
                    if prediction.relative_distance is not None
                    else "Relative score: —"
                )
                if open_set_method == "radius_fallback":
                    relative_text += " • open-set fallback"

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
        elif prediction.rejection_reason == "evt_open_set":
            nearest = prediction.nearest_label or "known gesture"
            self.rejection_reason_var.set(
                f"UNKNOWN — EVT inclusion below open-set threshold for '{nearest}'"
            )
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

            # V3.6 submits the mirrored camera frame to MediaPipe Tasks
            # asynchronously. process() immediately returns the latest completed,
            # temporally stabilized result instead of blocking the Tk camera loop.
            hands = self.tracker.process(frame)
            task_diag = self.tracker.diagnostics()
            self.current_tracking_timestamp_ms = task_diag.latest_timestamp_ms
            self.tracker.draw(frame, hands)
            self.current_hands = hands

            self.current_feature_set = build_frame_features(hands, representation="hybrid")
            if self.current_feature_set is None:
                self.current_raw_features = None
                self.current_features = None
                self.current_hand_signature = None
                self._set_stringvar_if_changed(
                    self.tracking_var,
                    "No hand detected",
                )
            else:
                self.current_raw_features = self.current_feature_set.vector.copy()
                self.current_features = self.metric_bank.encode(
                    self.current_raw_features
                )
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

                raw_dim = int(self.current_raw_features.shape[0])
                if self.metric_bank.can_encode_dimension(raw_dim):
                    tracking_text += (
                        f" • metric {raw_dim}D→{self.current_features.shape[0]}D"
                    )
                else:
                    tracking_text += f" • hybrid {raw_dim}D fallback"

                if task_diag.latest_latency_ms is not None:
                    tracking_text += f" • async {task_diag.latest_latency_ms:.0f} ms"
                else:
                    tracking_text += " • async Tasks"

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
            self._render_camera_frame(image)

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
    logger.info("Starting Adaptive Real-Time Hand Gesture Recognition V3.6.3")
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
