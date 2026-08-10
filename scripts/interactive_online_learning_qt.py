from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
import time

import cv2

try:
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QImage, QPixmap, QFont
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "PySide6 is not installed. Activate your project .venv and run:\n"
        "    uv pip install PySide6\n"
        "Then launch this preview again."
    ) from exc


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


# ============================================================
# Small reusable Qt widgets
# ============================================================


class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")


class StatusPill(QLabel):
    def __init__(self, text="", tone="neutral", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setProperty("tone", tone)
        self.setObjectName("StatusPill")

    def set_tone(self, tone: str):
        self.setProperty("tone", tone)
        self.style().unpolish(self)
        self.style().polish(self)


class CorrectionDialog(QDialog):
    """Small correction sheet used by the live feedback flow."""

    def __init__(self, prediction_label: str, compatible: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Correct recognition")
        self.setModal(True)
        self.resize(440, 260)
        self.action = None
        self.actual_label = None

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 22, 22, 22)
        root.setSpacing(14)

        title = QLabel("What was the correct gesture?")
        title.setObjectName("DialogTitle")
        root.addWidget(title)

        subtitle_text = (
            f"The captured prediction was “{prediction_label}”."
            if prediction_label
            else "The captured pose was classified as UNKNOWN."
        )
        subtitle = QLabel(subtitle_text)
        subtitle.setObjectName("MutedLabel")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        self.combo = QComboBox()
        self.combo.addItem("Select the actual gesture…", None)
        for name in compatible:
            self.combo.addItem(name, name)
        root.addWidget(self.combo)

        apply_button = QPushButton("Apply correction")
        apply_button.setProperty("role", "primary")
        apply_button.clicked.connect(self._apply)
        root.addWidget(apply_button)

        unknown_button = QPushButton("This gesture is unknown")
        unknown_button.setProperty("role", "secondary")
        unknown_button.clicked.connect(self._unknown)
        root.addWidget(unknown_button)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _apply(self):
        label = self.combo.currentData()
        if not label:
            QMessageBox.information(self, "Select gesture", "Select the actual gesture first.")
            return
        self.action = "correct"
        self.actual_label = str(label)
        self.accept()

    def _unknown(self):
        self.action = "unknown"
        self.accept()


# ============================================================
# Main Qt application
# ============================================================


class AdaptiveGestureQtApp(QMainWindow):
    """
    PySide6/Qt preview UI for the existing V2 recognition engine.

    Important: this file is intentionally parallel to the Tkinter application.
    It imports the exact same tracking, static learner, dynamic learner,
    stabilizer and persistence modules. No backend/data schema changes are made.
    """

    PAGE_LIVE = "Live"
    PAGE_TEACH = "Teach"
    PAGE_LIBRARY = "Library"
    PAGE_DYNAMIC = "Dynamic"
    PAGE_ABOUT = "About"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptive Gesture AI — Qt Preview")
        self.resize(1500, 940)
        self.setMinimumSize(1180, 760)

        # ----------------------------------------------------
        # Core engine — same parameters as the current V2 app.
        # ----------------------------------------------------
        self.tracker = HandTracker(max_num_hands=2)
        self.learner = OnlineGestureLearner(
            radius_multiplier=2.5,
            minimum_threshold=0.035,
            prototype_multiplier=2.2,
            max_prototypes=3,
        )
        self.dynamic_learner = DynamicGestureLearner(
            minimum_templates=3,
            threshold_multiplier=1.60,
            minimum_threshold=0.045,
            ambiguity_ratio=1.12,
            max_templates=6,
        )
        self.motion_segmenter = MotionSegmenter()
        self.prediction_stabilizer = PredictionStabilizer(
            confirm_frames=3,
            unknown_confirm_frames=2,
            unknown_label=self.learner.UNKNOWN_LABEL,
        )

        # ----------------------------------------------------
        # Persistence — exact same JSON stores as Tkinter.
        # ----------------------------------------------------
        self.gesture_store = GestureStore(
            PROJECT_ROOT / "data" / "gesture_memory.json"
        )
        self.dynamic_gesture_store = DynamicGestureStore(
            PROJECT_ROOT / "data" / "dynamic_gesture_memory.json"
        )

        restore_messages = []
        try:
            restored_static = self.gesture_store.load_into(self.learner)
            if restored_static:
                restore_messages.append(f"{restored_static} static")
        except Exception as error:
            restore_messages.append(f"static restore failed: {error}")

        try:
            restored_dynamic = self.dynamic_gesture_store.load_into(self.dynamic_learner)
            if restored_dynamic:
                restore_messages.append(f"{restored_dynamic} dynamic")
        except Exception as error:
            restore_messages.append(f"dynamic restore failed: {error}")

        # ----------------------------------------------------
        # Camera.
        # ----------------------------------------------------
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        # ----------------------------------------------------
        # Runtime state.
        # ----------------------------------------------------
        self.current_hands = []
        self.current_feature_set = None
        self.current_features = None
        self.current_hand_signature = None
        self.current_prediction = None
        self.current_raw_prediction = None

        self.prediction_update_interval = 0.08
        self.last_prediction_update = 0.0

        # Static teaching.
        self.selector = None
        self.teaching = False
        self.teaching_mode = None
        self.teaching_name = None
        self.teaching_signature = None
        self.required_teaching_signature = None
        self.prepare_until = None

        # Dynamic teaching/runtime.
        self.dynamic_teaching = False
        self.dynamic_demo_recording = False
        self.dynamic_teaching_name = None
        self.dynamic_required_signature = None
        self.dynamic_demo_observations = []
        self.dynamic_templates = []
        self.dynamic_target_demos = self.dynamic_learner.minimum_templates
        self.dynamic_sample_interval = 0.04
        self.last_dynamic_sample_time = 0.0
        self.dynamic_prediction_hold_until = 0.0

        # Feedback frozen sample.
        self.feedback_features = None
        self.feedback_predicted_label = None
        self.feedback_hand_signature = None

        # Recent successful recognition history for demo usability.
        self.recent_history = deque(maxlen=8)
        self.last_history_label = None
        self.last_history_time = 0.0

        # ----------------------------------------------------
        # UI construction.
        # ----------------------------------------------------
        self._apply_style()
        self._build_ui()
        self.refresh_gesture_table()
        self.refresh_dynamic_gesture_table()

        if restore_messages:
            self.set_global_status("Loaded: " + " • ".join(restore_messages))
        else:
            self.set_global_status("Ready — teach a gesture or start recognizing.")

        # ----------------------------------------------------
        # Camera timer. 15 ms keeps the preview responsive;
        # recognition itself remains throttled separately.
        # ----------------------------------------------------
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera)
        self.camera_timer.start(15)

    # ========================================================
    # Styling / layout helpers
    # ========================================================

    def _apply_style(self):
        QApplication.setStyle("Fusion")
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0A1020;
                color: #F7FAFC;
                font-family: "Segoe UI";
                font-size: 14px;
            }
            QFrame#Sidebar {
                background: #0D1528;
                border-right: 1px solid #22304A;
            }
            QFrame#Card {
                background: #121C31;
                border: 1px solid #263653;
                border-radius: 16px;
            }
            QLabel#AppTitle {
                font-size: 21px;
                font-weight: 700;
                color: #FFFFFF;
            }
            QLabel#AppSubtitle, QLabel#MutedLabel {
                color: #91A0B8;
            }
            QLabel#PageTitle {
                font-size: 25px;
                font-weight: 700;
                color: #FFFFFF;
            }
            QLabel#SectionTitle {
                font-size: 16px;
                font-weight: 650;
                color: #F7FAFC;
            }
            QLabel#PredictionLabel {
                font-size: 38px;
                font-weight: 750;
                color: #FFFFFF;
            }
            QLabel#ConfidenceLabel {
                font-size: 28px;
                font-weight: 700;
                color: #67E8F9;
            }
            QLabel#MetricValue {
                font-size: 18px;
                font-weight: 650;
                color: #FFFFFF;
            }
            QLabel#DialogTitle {
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#Camera {
                background: #020617;
                border: 1px solid #263653;
                border-radius: 18px;
            }
            QLabel#StatusPill {
                padding: 6px 11px;
                border-radius: 11px;
                font-weight: 650;
            }
            QLabel#StatusPill[tone="neutral"] {
                background: #1C2942;
                color: #B6C3D6;
            }
            QLabel#StatusPill[tone="success"] {
                background: #123525;
                color: #6EE7A0;
            }
            QLabel#StatusPill[tone="warning"] {
                background: #3B2A12;
                color: #F5C76B;
            }
            QLabel#StatusPill[tone="danger"] {
                background: #3A1820;
                color: #FDA4AF;
            }
            QPushButton {
                background: #18243B;
                border: 1px solid #2C3E5F;
                border-radius: 10px;
                padding: 10px 14px;
                color: #E8EEF8;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #20304E;
                border-color: #3A527B;
            }
            QPushButton:pressed {
                background: #142039;
            }
            QPushButton:disabled {
                color: #66758D;
                background: #111A2C;
                border-color: #202D45;
            }
            QPushButton[role="primary"] {
                background: #2563EB;
                border-color: #3B82F6;
                color: white;
            }
            QPushButton[role="primary"]:hover {
                background: #2F6FF2;
            }
            QPushButton[role="success"] {
                background: #137A47;
                border-color: #1A9B5B;
                color: white;
            }
            QPushButton[role="danger"] {
                background: #3A1820;
                border-color: #7F1D2D;
                color: #FFE4E6;
            }
            QPushButton[nav="true"] {
                text-align: left;
                padding: 11px 14px;
                background: transparent;
                border: 1px solid transparent;
                color: #9EACC1;
            }
            QPushButton[nav="true"]:hover {
                background: #131F34;
                color: #FFFFFF;
            }
            QPushButton[navActive="true"] {
                background: #182845;
                border-color: #2F4C79;
                color: #FFFFFF;
            }
            QLineEdit, QComboBox {
                background: #0E1729;
                border: 1px solid #2A3A58;
                border-radius: 10px;
                padding: 10px 12px;
                color: #F8FAFC;
                selection-background-color: #2563EB;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #3B82F6;
            }
            QProgressBar {
                border: 1px solid #2A3A58;
                border-radius: 7px;
                background: #0D1628;
                text-align: center;
                color: #E7EDF7;
                min-height: 14px;
            }
            QProgressBar::chunk {
                background: #3B82F6;
                border-radius: 6px;
            }
            QTableWidget {
                background: #0D1628;
                alternate-background-color: #101B2F;
                border: 1px solid #263653;
                border-radius: 12px;
                gridline-color: #263653;
                selection-background-color: #244675;
                selection-color: #FFFFFF;
            }
            QHeaderView::section {
                background: #17243B;
                color: #B8C4D6;
                border: none;
                border-bottom: 1px solid #2A3A58;
                padding: 9px;
                font-weight: 650;
            }
            QScrollArea {
                border: none;
            }
            """
        )

    def _build_ui(self):
        shell = QWidget()
        self.setCentralWidget(shell)
        outer = QHBoxLayout(shell)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ----------------------------------------------------
        # Sidebar
        # ----------------------------------------------------
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(230)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 24, 18, 20)
        sidebar_layout.setSpacing(8)

        app_title = QLabel("Gesture AI")
        app_title.setObjectName("AppTitle")
        sidebar_layout.addWidget(app_title)

        app_subtitle = QLabel("Adaptive few-shot recognition")
        app_subtitle.setObjectName("AppSubtitle")
        app_subtitle.setWordWrap(True)
        sidebar_layout.addWidget(app_subtitle)
        sidebar_layout.addSpacing(18)

        self.nav_buttons = {}
        for name, icon in [
            (self.PAGE_LIVE, "●"),
            (self.PAGE_TEACH, "+"),
            (self.PAGE_LIBRARY, "▦"),
            (self.PAGE_DYNAMIC, "↝"),
            (self.PAGE_ABOUT, "i"),
        ]:
            button = QPushButton(f"{icon}   {name}")
            button.setProperty("nav", True)
            button.clicked.connect(lambda checked=False, page=name: self.show_page(page))
            sidebar_layout.addWidget(button)
            self.nav_buttons[name] = button

        sidebar_layout.addStretch(1)

        self.camera_state_pill = StatusPill("Camera ready", "success")
        sidebar_layout.addWidget(self.camera_state_pill)
        self.sidebar_tracking = QLabel("No hand detected")
        self.sidebar_tracking.setObjectName("MutedLabel")
        self.sidebar_tracking.setWordWrap(True)
        sidebar_layout.addWidget(self.sidebar_tracking)

        outer.addWidget(sidebar)

        # ----------------------------------------------------
        # Main content + persistent status line
        # ----------------------------------------------------
        content_shell = QWidget()
        content_layout = QVBoxLayout(content_shell)
        content_layout.setContentsMargins(26, 22, 26, 18)
        content_layout.setSpacing(14)

        self.stack = QStackedWidget()
        content_layout.addWidget(self.stack, 1)

        status_card = QFrame()
        status_card.setObjectName("Card")
        status_layout = QHBoxLayout(status_card)
        status_layout.setContentsMargins(14, 9, 14, 9)
        self.global_status = QLabel("Ready")
        self.global_status.setObjectName("MutedLabel")
        self.global_status.setWordWrap(True)
        status_layout.addWidget(self.global_status, 1)
        self.memory_pill = StatusPill("Landmark memory", "neutral")
        status_layout.addWidget(self.memory_pill)
        content_layout.addWidget(status_card)

        outer.addWidget(content_shell, 1)

        # Create pages.
        self.pages = {}
        self.pages[self.PAGE_LIVE] = self._build_live_page()
        self.pages[self.PAGE_TEACH] = self._build_teach_page()
        self.pages[self.PAGE_LIBRARY] = self._build_library_page()
        self.pages[self.PAGE_DYNAMIC] = self._build_dynamic_page()
        self.pages[self.PAGE_ABOUT] = self._build_about_page()

        for page in self.pages.values():
            self.stack.addWidget(page)

        self.current_page_name = self.PAGE_LIVE
        self.show_page(self.PAGE_LIVE)

    def _page_shell(self, title: str, subtitle: str):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        root.addWidget(title_label)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("MutedLabel")
        subtitle_label.setWordWrap(True)
        root.addWidget(subtitle_label)

        return page, root

    def _new_camera_label(self):
        label = QLabel("Starting camera…")
        label.setObjectName("Camera")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(600, 360)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return label

    # ========================================================
    # Page builders
    # ========================================================

    def _build_live_page(self):
        page, root = self._page_shell(
            "Live recognition",
            "Recognize static and dynamic gestures, then correct the learner when needed.",
        )

        grid = QGridLayout()
        grid.setSpacing(14)
        grid.setColumnStretch(0, 7)
        grid.setColumnStretch(1, 4)
        root.addLayout(grid, 1)

        # Camera card.
        camera_card = Card()
        camera_layout = QVBoxLayout(camera_card)
        camera_layout.setContentsMargins(12, 12, 12, 12)
        self.live_camera = self._new_camera_label()
        camera_layout.addWidget(self.live_camera, 1)

        camera_footer = QHBoxLayout()
        self.live_tracking = QLabel("No hand detected")
        self.live_tracking.setObjectName("MutedLabel")
        camera_footer.addWidget(self.live_tracking, 1)
        self.input_pill = StatusPill("No input", "neutral")
        camera_footer.addWidget(self.input_pill)
        camera_layout.addLayout(camera_footer)
        grid.addWidget(camera_card, 0, 0, 3, 1)

        # Recognition card.
        recog_card = Card()
        recog = QVBoxLayout(recog_card)
        recog.setContentsMargins(20, 18, 20, 18)
        recog.setSpacing(10)

        section = QLabel("CURRENT RECOGNITION")
        section.setObjectName("MutedLabel")
        recog.addWidget(section)

        self.prediction_label = QLabel("UNKNOWN")
        self.prediction_label.setObjectName("PredictionLabel")
        self.prediction_label.setWordWrap(True)
        recog.addWidget(self.prediction_label)

        self.confidence_label = QLabel("—")
        self.confidence_label.setObjectName("ConfidenceLabel")
        recog.addWidget(self.confidence_label)

        self.stable_pill = StatusPill("Waiting", "neutral")
        self.stable_pill.setMaximumWidth(180)
        recog.addWidget(self.stable_pill)

        self.reason_label = QLabel("")
        self.reason_label.setObjectName("MutedLabel")
        self.reason_label.setWordWrap(True)
        recog.addWidget(self.reason_label)

        buttons = QHBoxLayout()
        self.correct_button = QPushButton("✓  Correct")
        self.correct_button.setProperty("role", "success")
        self.correct_button.clicked.connect(self.confirm_prediction)
        buttons.addWidget(self.correct_button)
        self.wrong_button = QPushButton("✕  Wrong")
        self.wrong_button.clicked.connect(self.begin_correction)
        buttons.addWidget(self.wrong_button)
        recog.addLayout(buttons)

        # Advanced metrics are visible but intentionally quiet.
        self.static_metrics = QLabel("Distance: —\nThreshold: —\nRelative score: —")
        self.static_metrics.setObjectName("MutedLabel")
        self.static_metrics.setWordWrap(True)
        recog.addWidget(self.static_metrics)
        grid.addWidget(recog_card, 0, 1)

        # Dynamic card.
        dynamic_card = Card()
        dyn = QVBoxLayout(dynamic_card)
        dyn.setContentsMargins(18, 16, 18, 16)
        title = QLabel("Dynamic recognition")
        title.setObjectName("SectionTitle")
        dyn.addWidget(title)
        self.dynamic_live_prediction = QLabel("—")
        self.dynamic_live_prediction.setObjectName("MetricValue")
        dyn.addWidget(self.dynamic_live_prediction)
        self.dynamic_live_metrics = QLabel("Waiting for motion")
        self.dynamic_live_metrics.setObjectName("MutedLabel")
        self.dynamic_live_metrics.setWordWrap(True)
        dyn.addWidget(self.dynamic_live_metrics)
        self.dynamic_state_pill = StatusPill("IDLE", "neutral")
        self.dynamic_state_pill.setMaximumWidth(210)
        dyn.addWidget(self.dynamic_state_pill)
        grid.addWidget(dynamic_card, 1, 1)

        # History card.
        history_card = Card()
        hist = QVBoxLayout(history_card)
        hist.setContentsMargins(18, 16, 18, 16)
        title = QLabel("Recent recognition")
        title.setObjectName("SectionTitle")
        hist.addWidget(title)
        self.history_labels = []
        for _ in range(4):
            row = QLabel("—")
            row.setObjectName("MutedLabel")
            self.history_labels.append(row)
            hist.addWidget(row)
        grid.addWidget(history_card, 2, 1)

        return page

    def _build_teach_page(self):
        page, root = self._page_shell(
            "Teach a static gesture",
            "Give the recognizer a few stable, informative examples. One- and two-hand gestures are supported.",
        )

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        camera_card = Card()
        cam_layout = QVBoxLayout(camera_card)
        cam_layout.setContentsMargins(12, 12, 12, 12)
        self.teach_camera = self._new_camera_label()
        cam_layout.addWidget(self.teach_camera, 1)
        self.teach_tracking = QLabel("No hand detected")
        self.teach_tracking.setObjectName("MutedLabel")
        cam_layout.addWidget(self.teach_tracking)
        body.addWidget(camera_card, 7)

        control_card = Card()
        controls = QVBoxLayout(control_card)
        controls.setContentsMargins(20, 20, 20, 20)
        controls.setSpacing(12)

        heading = QLabel("New gesture")
        heading.setObjectName("SectionTitle")
        controls.addWidget(heading)

        label = QLabel("Gesture name")
        label.setObjectName("MutedLabel")
        controls.addWidget(label)
        self.gesture_name_entry = QLineEdit()
        self.gesture_name_entry.setPlaceholderText("e.g. Victory")
        self.gesture_name_entry.returnPressed.connect(self.start_new_teaching)
        controls.addWidget(self.gesture_name_entry)

        self.teach_start_button = QPushButton("Teach gesture")
        self.teach_start_button.setProperty("role", "primary")
        self.teach_start_button.clicked.connect(self.start_new_teaching)
        controls.addWidget(self.teach_start_button)

        self.teaching_mode_label = QLabel("Ready for a new gesture")
        self.teaching_mode_label.setObjectName("MutedLabel")
        self.teaching_mode_label.setWordWrap(True)
        controls.addWidget(self.teaching_mode_label)

        self.teach_progress = QProgressBar()
        self.teach_progress.setRange(0, 12)
        self.teach_progress.setValue(0)
        controls.addWidget(self.teach_progress)

        stats_grid = QGridLayout()
        self.stat_labels = {}
        for row, key in enumerate(["Observed", "Useful", "Duplicates", "Unstable", "Stability", "Input"]):
            name = QLabel(key)
            name.setObjectName("MutedLabel")
            value = QLabel("0" if key not in {"Stability", "Input"} else "—")
            value.setObjectName("MetricValue")
            stats_grid.addWidget(name, row, 0)
            stats_grid.addWidget(value, row, 1)
            self.stat_labels[key] = value
        controls.addLayout(stats_grid)

        self.teach_status = QLabel("Enter a gesture name to begin.")
        self.teach_status.setObjectName("MutedLabel")
        self.teach_status.setWordWrap(True)
        controls.addWidget(self.teach_status)

        teach_buttons = QHBoxLayout()
        self.finish_teach_button = QPushButton("Finish")
        self.finish_teach_button.setProperty("role", "primary")
        self.finish_teach_button.setEnabled(False)
        self.finish_teach_button.clicked.connect(self.finish_teaching)
        teach_buttons.addWidget(self.finish_teach_button)
        self.cancel_teach_button = QPushButton("Cancel")
        self.cancel_teach_button.setEnabled(False)
        self.cancel_teach_button.clicked.connect(self.cancel_teaching)
        teach_buttons.addWidget(self.cancel_teach_button)
        controls.addLayout(teach_buttons)
        controls.addStretch(1)

        body.addWidget(control_card, 4)
        return page

    def _build_library_page(self):
        page, root = self._page_shell(
            "Gesture library",
            "Manage persistent static gestures and improve them without retraining unrelated classes.",
        )

        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        root.addWidget(card, 1)

        self.gesture_table = QTableWidget(0, 7)
        self.gesture_table.setHorizontalHeaderLabels(
            ["Gesture", "+", "−", "Prototypes", "Spread", "Radius", "Input"]
        )
        self.gesture_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.gesture_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.gesture_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.gesture_table.setAlternatingRowColors(True)
        self.gesture_table.verticalHeader().setVisible(False)
        self.gesture_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.gesture_table, 1)

        buttons = QHBoxLayout()
        improve = QPushButton("Improve")
        improve.clicked.connect(self.improve_selected_gesture)
        buttons.addWidget(improve)
        retrain = QPushButton("Retrain")
        retrain.clicked.connect(self.retrain_selected_gesture)
        buttons.addWidget(retrain)
        rename = QPushButton("Rename")
        rename.clicked.connect(self.rename_selected_gesture)
        buttons.addWidget(rename)
        delete = QPushButton("Delete")
        delete.setProperty("role", "danger")
        delete.clicked.connect(self.delete_selected_gesture)
        buttons.addWidget(delete)
        buttons.addStretch(1)
        clear = QPushButton("Clear all static gestures")
        clear.setProperty("role", "danger")
        clear.clicked.connect(self.clear_all_gestures)
        buttons.addWidget(clear)
        layout.addLayout(buttons)

        self.library_status = QLabel("Select a gesture to manage it.")
        self.library_status.setObjectName("MutedLabel")
        root.addWidget(self.library_status)
        return page

    def _build_dynamic_page(self):
        page, root = self._page_shell(
            "Dynamic gesture learning",
            "Teach movement-based gestures from live landmark trajectories. No videos are stored.",
        )

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        left_card = Card()
        left = QVBoxLayout(left_card)
        left.setContentsMargins(12, 12, 12, 12)
        self.dynamic_camera = self._new_camera_label()
        left.addWidget(self.dynamic_camera, 1)
        self.dynamic_tracking = QLabel("No hand detected")
        self.dynamic_tracking.setObjectName("MutedLabel")
        left.addWidget(self.dynamic_tracking)
        body.addWidget(left_card, 7)

        right = QVBoxLayout()
        right.setSpacing(14)
        body.addLayout(right, 5)

        teach_card = Card()
        teach = QVBoxLayout(teach_card)
        teach.setContentsMargins(18, 18, 18, 18)
        title = QLabel("Teach movement")
        title.setObjectName("SectionTitle")
        teach.addWidget(title)

        self.dynamic_name_entry = QLineEdit()
        self.dynamic_name_entry.setPlaceholderText("e.g. Swipe Right")
        teach.addWidget(self.dynamic_name_entry)

        self.dynamic_teach_button = QPushButton("Teach dynamic gesture")
        self.dynamic_teach_button.setProperty("role", "primary")
        self.dynamic_teach_button.clicked.connect(self.start_dynamic_teaching)
        teach.addWidget(self.dynamic_teach_button)

        demo_buttons = QHBoxLayout()
        self.dynamic_start_demo_button = QPushButton("Start demo")
        self.dynamic_start_demo_button.setEnabled(False)
        self.dynamic_start_demo_button.clicked.connect(self.start_dynamic_demo)
        demo_buttons.addWidget(self.dynamic_start_demo_button)
        self.dynamic_stop_demo_button = QPushButton("Stop demo")
        self.dynamic_stop_demo_button.setEnabled(False)
        self.dynamic_stop_demo_button.clicked.connect(self.stop_dynamic_demo)
        demo_buttons.addWidget(self.dynamic_stop_demo_button)
        teach.addLayout(demo_buttons)

        self.dynamic_demo_progress = QLabel(
            f"Demonstrations: 0/{self.dynamic_target_demos}"
        )
        self.dynamic_demo_progress.setObjectName("MetricValue")
        teach.addWidget(self.dynamic_demo_progress)

        self.dynamic_teach_status = QLabel(
            "Enter a name, then record three complete demonstrations."
        )
        self.dynamic_teach_status.setObjectName("MutedLabel")
        self.dynamic_teach_status.setWordWrap(True)
        teach.addWidget(self.dynamic_teach_status)

        self.dynamic_cancel_button = QPushButton("Cancel teaching")
        self.dynamic_cancel_button.setEnabled(False)
        self.dynamic_cancel_button.clicked.connect(self.cancel_dynamic_teaching)
        teach.addWidget(self.dynamic_cancel_button)
        right.addWidget(teach_card)

        library_card = Card()
        lib = QVBoxLayout(library_card)
        lib.setContentsMargins(14, 14, 14, 14)
        title = QLabel("Dynamic library")
        title.setObjectName("SectionTitle")
        lib.addWidget(title)
        self.dynamic_table = QTableWidget(0, 5)
        self.dynamic_table.setHorizontalHeaderLabels(
            ["Gesture", "Demos", "Threshold", "Duration", "Input"]
        )
        self.dynamic_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.dynamic_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.dynamic_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.dynamic_table.verticalHeader().setVisible(False)
        self.dynamic_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lib.addWidget(self.dynamic_table)

        management = QHBoxLayout()
        rename = QPushButton("Rename")
        rename.clicked.connect(self.rename_selected_dynamic_gesture)
        management.addWidget(rename)
        delete = QPushButton("Delete")
        delete.setProperty("role", "danger")
        delete.clicked.connect(self.delete_selected_dynamic_gesture)
        management.addWidget(delete)
        clear = QPushButton("Clear all")
        clear.setProperty("role", "danger")
        clear.clicked.connect(self.clear_all_dynamic_gestures)
        management.addWidget(clear)
        lib.addLayout(management)
        right.addWidget(library_card, 1)

        return page

    def _build_about_page(self):
        page, root = self._page_shell(
            "About this prototype",
            "The Qt preview uses the same recognition engine as the Tkinter application; only the presentation layer is different.",
        )

        grid = QGridLayout()
        grid.setSpacing(14)
        root.addLayout(grid)

        cards = [
            (
                "Runtime few-shot learning",
                "New user-defined static gestures are learned while the application is running without retraining all previous classes.",
            ),
            (
                "Interactive adaptation",
                "Correct and Wrong feedback updates positive examples and hard-negative memory immediately.",
            ),
            (
                "Dynamic landmark trajectories",
                "Movement gestures use normalized landmark trajectories and DTW-style sequence matching rather than stored videos.",
            ),
            (
                "Privacy-friendly persistence",
                "The application persists normalized numerical hand representations. Camera images and videos are not required for gesture memory.",
            ),
        ]
        for i, (title_text, body_text) in enumerate(cards):
            card = Card()
            layout = QVBoxLayout(card)
            layout.setContentsMargins(20, 18, 20, 18)
            title = QLabel(title_text)
            title.setObjectName("SectionTitle")
            layout.addWidget(title)
            body = QLabel(body_text)
            body.setObjectName("MutedLabel")
            body.setWordWrap(True)
            layout.addWidget(body)
            grid.addWidget(card, i // 2, i % 2)

        warning = Card()
        w = QVBoxLayout(warning)
        w.setContentsMargins(20, 18, 20, 18)
        title = QLabel("Qt preview safety")
        title.setObjectName("SectionTitle")
        w.addWidget(title)
        text = QLabel(
            "This script is separate from scripts/interactive_online_learning.py. "
            "You can compare both frontends without replacing the existing Tkinter code. "
            "Both applications intentionally use the same gesture-memory files so your learned gestures appear in either UI."
        )
        text.setObjectName("MutedLabel")
        text.setWordWrap(True)
        w.addWidget(text)
        root.addWidget(warning)
        root.addStretch(1)
        return page

    # ========================================================
    # Navigation / general UI
    # ========================================================

    def show_page(self, name: str):
        page = self.pages[name]
        self.stack.setCurrentWidget(page)
        self.current_page_name = name
        for page_name, button in self.nav_buttons.items():
            button.setProperty("navActive", page_name == name)
            button.style().unpolish(button)
            button.style().polish(button)

    def set_global_status(self, text: str):
        self.global_status.setText(text)

    def set_static_teach_status(self, text: str):
        self.teach_status.setText(text)
        self.set_global_status(text)

    def set_dynamic_status(self, text: str):
        self.dynamic_teach_status.setText(text)
        self.set_global_status(text)

    # ========================================================
    # Persistence
    # ========================================================

    def save_gesture_memory(self):
        try:
            self.gesture_store.save(self.learner)
            return True
        except Exception as error:
            self.set_global_status(f"Static gesture memory could not be saved: {error}")
            return False

    def save_dynamic_gesture_memory(self):
        try:
            self.dynamic_gesture_store.save(self.dynamic_learner)
            return True
        except Exception as error:
            self.set_global_status(f"Dynamic gesture memory could not be saved: {error}")
            return False

    # ========================================================
    # Static gesture management
    # ========================================================

    def selected_static_name(self):
        row = self.gesture_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Select gesture", "Select a gesture in the library first.")
            return None
        item = self.gesture_table.item(row, 0)
        return item.text() if item else None

    def rename_selected_gesture(self):
        name = self.selected_static_name()
        if not name:
            return
        new_name, ok = QInputDialog.getText(self, "Rename gesture", f"New name for '{name}':", text=name)
        if not ok:
            return
        try:
            gesture = self.learner.rename_gesture(name, new_name)
            self.prediction_stabilizer.reset()
            self.current_prediction = None
            self.current_raw_prediction = None
            self.save_gesture_memory()
            self.refresh_gesture_table()
            self.library_status.setText(f"Renamed '{name}' to '{gesture.name}'.")
        except Exception as error:
            QMessageBox.critical(self, "Rename failed", str(error))

    def delete_selected_gesture(self):
        name = self.selected_static_name()
        if not name:
            return
        answer = QMessageBox.question(
            self,
            "Delete gesture",
            f"Delete '{name}' and all of its learned positive/negative memory?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.learner.delete_gesture(name)
        self.prediction_stabilizer.reset()
        self.current_prediction = None
        self.current_raw_prediction = None
        self.save_gesture_memory()
        self.refresh_gesture_table()
        self.library_status.setText(f"Deleted '{name}'.")

    def clear_all_gestures(self):
        if not self.learner.gestures:
            self.library_status.setText("Static gesture memory is already empty.")
            return
        answer = QMessageBox.question(
            self,
            "Clear static gesture memory",
            "Delete ALL learned static gestures and feedback memory?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.learner.clear()
        self.prediction_stabilizer.reset()
        self.current_prediction = None
        self.current_raw_prediction = None
        self.save_gesture_memory()
        self.refresh_gesture_table()
        self.library_status.setText("All static gesture memory cleared.")

    def improve_selected_gesture(self):
        name = self.selected_static_name()
        if not name:
            return
        gesture = self.learner.gestures[name]
        self.begin_teaching(name, "improve", gesture.hand_signature)

    def retrain_selected_gesture(self):
        name = self.selected_static_name()
        if not name:
            return
        gesture = self.learner.gestures[name]
        answer = QMessageBox.question(
            self,
            "Retrain gesture",
            f"Replace the positive examples for '{name}' with a new live demonstration? Compatible negative feedback will be preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.begin_teaching(name, "retrain", gesture.hand_signature)

    # ========================================================
    # Static teaching
    # ========================================================

    def start_new_teaching(self):
        name = self.gesture_name_entry.text().strip()
        if not name:
            self.set_static_teach_status("Enter a gesture name first.")
            self.gesture_name_entry.setFocus()
            return
        if name in self.learner.gestures:
            self.set_static_teach_status(
                f"'{name}' already exists. Use Improve or Retrain from the Library page."
            )
            return
        self.begin_teaching(name, "new", None)

    def begin_teaching(self, name: str, mode: str, required_signature: str | None):
        if self.teaching:
            return
        if self.dynamic_teaching:
            self.set_static_teach_status("Finish or cancel dynamic-gesture teaching first.")
            return

        self.show_page(self.PAGE_TEACH)
        self.prediction_stabilizer.reset()
        self.current_prediction = None
        self.current_raw_prediction = None
        self.selector = SmartSampleSelector()

        self.teaching = True
        self.teaching_mode = mode
        self.teaching_name = name
        self.teaching_signature = None
        self.required_teaching_signature = required_signature
        self.prepare_until = time.monotonic() + 2.0

        self.teach_progress.setRange(0, self.selector.target_samples)
        self.teach_progress.setValue(0)
        self._reset_teach_stats()
        self.teach_start_button.setEnabled(False)
        self.finish_teach_button.setEnabled(False)
        self.cancel_teach_button.setEnabled(True)
        self.gesture_name_entry.setEnabled(False)
        self.correct_button.setEnabled(False)
        self.wrong_button.setEnabled(False)

        action = {"new": "Teach", "improve": "Improve", "retrain": "Retrain"}[mode]
        self.teaching_mode_label.setText(f"{action}: {name}")
        expected = (
            f" Use {required_signature} input."
            if required_signature is not None
            else " One or two hands are supported."
        )
        self.set_static_teach_status(
            f"Get ready to {action.lower()} '{name}'. Hold the gesture naturally.{expected}"
        )

    def process_teaching_frame(self):
        if self.current_feature_set is None:
            self.set_static_teach_status("Waiting for a hand…")
            return

        now = time.monotonic()
        if self.prepare_until is not None and now < self.prepare_until:
            self.set_static_teach_status(
                f"Teaching '{self.teaching_name}' starts in {self.prepare_until - now:.1f}s…"
            )
            return

        signature = self.current_feature_set.hand_signature
        if self.required_teaching_signature is not None:
            if signature != self.required_teaching_signature:
                self.set_static_teach_status(
                    f"'{self.teaching_name}' expects {self.required_teaching_signature} input. Current input is {signature}."
                )
                return
            self.teaching_signature = self.required_teaching_signature
        elif self.teaching_signature is None:
            self.teaching_signature = signature
            if signature == "Both":
                self.selector.stability_threshold = 0.050
                self.selector.duplicate_threshold = 0.025
        elif signature != self.teaching_signature:
            self.set_static_teach_status(
                f"Continue using {self.teaching_signature} input for this gesture."
            )
            return

        self.selector.consider(self.current_features)
        self._update_teach_stats()
        self.teach_progress.setValue(len(self.selector.samples))

        if self.selector.ready:
            self.finish_teach_button.setEnabled(True)
            if not self.selector.complete:
                self.set_static_teach_status(
                    "Enough useful samples to learn. You may finish now, or make small natural variations of the SAME gesture."
                )
        else:
            self.set_static_teach_status(
                "Keep the gesture steady. Small natural variations are useful."
            )

        if self.selector.complete:
            self.finish_teaching()

    def finish_teaching(self):
        if not self.teaching:
            return
        if self.selector is None or not self.selector.ready:
            self.set_static_teach_status("Not enough useful samples yet.")
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
                gesture = self.learner.learn_gesture(name, samples, hand_signature=signature)
                action_text = "Learned"
            elif mode == "improve":
                gesture = self.learner.add_samples_to_gesture(
                    name, samples, hand_signature=signature
                )
                action_text = "Improved"
            elif mode == "retrain":
                gesture = self.learner.replace_gesture_samples(
                    name,
                    samples,
                    hand_signature=signature,
                    keep_negatives=True,
                )
                action_text = "Retrained"
            else:
                raise ValueError("Unknown teaching mode.")
            saved = self.save_gesture_memory()
        except Exception as error:
            self.set_static_teach_status(f"Learning failed: {error}")
            return

        self.finish_teaching_state()
        self.refresh_gesture_table()
        message = (
            f"✓ {action_text} '{gesture.name}' from {accepted} useful samples. "
            f"{observed} frames observed; {duplicates} duplicates and {unstable} unstable frames ignored. "
            f"{gesture.prototype_count} adaptive prototype(s) active."
        )
        if saved:
            message += " Gesture memory saved."
        self.set_static_teach_status(message)

    def cancel_teaching(self):
        if not self.teaching:
            return
        self.finish_teaching_state()
        self.set_static_teach_status("Teaching cancelled.")

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

        self.teach_progress.setValue(0)
        self.teach_start_button.setEnabled(True)
        self.finish_teach_button.setEnabled(False)
        self.cancel_teach_button.setEnabled(False)
        self.gesture_name_entry.setEnabled(True)
        self.gesture_name_entry.clear()
        self.correct_button.setEnabled(True)
        self.wrong_button.setEnabled(True)
        self.teaching_mode_label.setText("Ready for a new gesture")

    def _reset_teach_stats(self):
        for key, label in self.stat_labels.items():
            label.setText("—" if key in {"Stability", "Input"} else "0")

    def _update_teach_stats(self):
        if self.selector is None:
            return
        stats = self.selector.stats
        self.stat_labels["Observed"].setText(str(stats.observed))
        self.stat_labels["Useful"].setText(str(stats.accepted))
        self.stat_labels["Duplicates"].setText(str(stats.duplicates))
        self.stat_labels["Unstable"].setText(str(stats.unstable))
        self.stat_labels["Stability"].setText(
            "—" if self.selector.last_stability is None else f"{self.selector.last_stability:.4f}"
        )
        self.stat_labels["Input"].setText(self.teaching_signature or "—")

    # ========================================================
    # Dynamic teaching / management
    # ========================================================

    def start_dynamic_teaching(self):
        if self.teaching:
            self.set_dynamic_status("Finish or cancel static-gesture teaching first.")
            return
        if self.dynamic_teaching:
            return

        name = self.dynamic_name_entry.text().strip()
        if not name:
            self.set_dynamic_status("Enter a dynamic gesture name first.")
            self.dynamic_name_entry.setFocus()
            return
        if name in self.dynamic_learner.gestures:
            self.set_dynamic_status(f"Dynamic gesture '{name}' already exists.")
            return

        self.dynamic_teaching = True
        self.dynamic_demo_recording = False
        self.dynamic_teaching_name = name
        self.dynamic_required_signature = None
        self.dynamic_demo_observations = []
        self.dynamic_templates = []
        self.motion_segmenter.reset()

        self.dynamic_demo_progress.setText(
            f"Demonstrations: 0/{self.dynamic_target_demos}"
        )
        self.dynamic_name_entry.setEnabled(False)
        self.dynamic_teach_button.setEnabled(False)
        self.dynamic_start_demo_button.setEnabled(True)
        self.dynamic_stop_demo_button.setEnabled(False)
        self.dynamic_cancel_button.setEnabled(True)
        self.set_dynamic_status(
            f"Ready to teach '{name}'. Press Start demo, perform the movement once, then press Stop demo."
        )

    def start_dynamic_demo(self):
        if not self.dynamic_teaching or self.dynamic_demo_recording:
            return
        if not self.current_hands:
            self.set_dynamic_status("Show the hand(s) you will use before starting the demo.")
            return

        now = time.monotonic()
        observation = build_dynamic_observation(self.current_hands, now)
        if observation is None:
            self.set_dynamic_status("No stable hand configuration detected.")
            return

        if self.dynamic_required_signature is None:
            self.dynamic_required_signature = observation.hand_signature
        elif observation.hand_signature != self.dynamic_required_signature:
            self.set_dynamic_status(
                f"This gesture is locked to {self.dynamic_required_signature} input. Currently seeing {observation.hand_signature}."
            )
            return

        self.dynamic_demo_recording = True
        self.dynamic_demo_observations = [observation]
        self.last_dynamic_sample_time = now
        demo_number = len(self.dynamic_templates) + 1
        self.dynamic_start_demo_button.setEnabled(False)
        self.dynamic_stop_demo_button.setEnabled(True)
        self.set_dynamic_status(
            f"● Recording demo {demo_number}/{self.dynamic_target_demos}. Perform the complete movement, then press Stop demo."
        )

    def process_dynamic_training_frame(self, now: float):
        if not self.dynamic_demo_recording:
            return
        if now - self.last_dynamic_sample_time < self.dynamic_sample_interval:
            return
        self.last_dynamic_sample_time = now

        observation = build_dynamic_observation(self.current_hands, now)
        if observation is None:
            self.set_dynamic_status("Recording: keep the hand(s) visible. Missing frames are ignored.")
            return
        if observation.hand_signature != self.dynamic_required_signature:
            self.set_dynamic_status(
                f"Recording expects {self.dynamic_required_signature} input. Keep the same hand configuration visible."
            )
            return
        self.dynamic_demo_observations.append(observation)

    def stop_dynamic_demo(self):
        if not self.dynamic_teaching or not self.dynamic_demo_recording:
            return

        self.dynamic_demo_recording = False
        self.dynamic_stop_demo_button.setEnabled(False)
        self.dynamic_start_demo_button.setEnabled(True)

        try:
            trajectory = prepare_dynamic_trajectory(self.dynamic_demo_observations)
        except Exception as error:
            self.dynamic_demo_observations = []
            self.set_dynamic_status(f"Demo not accepted: {error} Record this demo again.")
            return

        self.dynamic_templates.append(trajectory)
        self.dynamic_demo_observations = []
        completed = len(self.dynamic_templates)
        self.dynamic_demo_progress.setText(
            f"Demonstrations: {completed}/{self.dynamic_target_demos}"
        )

        if completed < self.dynamic_target_demos:
            self.set_dynamic_status(
                f"✓ Demo {completed} accepted ({trajectory.duration_seconds:.2f}s, motion extent {trajectory.motion_extent:.2f}). Return to the starting pose and record the next demonstration."
            )
            return

        try:
            gesture = self.dynamic_learner.learn_gesture(
                self.dynamic_teaching_name, self.dynamic_templates
            )
            saved = self.save_dynamic_gesture_memory()
        except Exception as error:
            self.set_dynamic_status(f"Dynamic learning failed: {error}")
            return

        message = (
            f"✓ Learned dynamic gesture '{gesture.name}' from {gesture.template_count} demonstrations. DTW threshold: {gesture.threshold:.4f}."
        )
        if saved:
            message += " Landmark trajectories saved."
        self.finish_dynamic_teaching_state()
        self.refresh_dynamic_gesture_table()
        self.set_dynamic_status(message)

    def cancel_dynamic_teaching(self):
        if not self.dynamic_teaching:
            return
        self.finish_dynamic_teaching_state()
        self.set_dynamic_status("Dynamic gesture teaching cancelled.")

    def finish_dynamic_teaching_state(self):
        self.dynamic_teaching = False
        self.dynamic_demo_recording = False
        self.dynamic_teaching_name = None
        self.dynamic_required_signature = None
        self.dynamic_demo_observations = []
        self.dynamic_templates = []
        self.motion_segmenter.reset()

        self.dynamic_name_entry.setEnabled(True)
        self.dynamic_teach_button.setEnabled(True)
        self.dynamic_start_demo_button.setEnabled(False)
        self.dynamic_stop_demo_button.setEnabled(False)
        self.dynamic_cancel_button.setEnabled(False)
        self.dynamic_name_entry.clear()
        self.dynamic_demo_progress.setText(
            f"Demonstrations: 0/{self.dynamic_target_demos}"
        )

    def selected_dynamic_name(self):
        row = self.dynamic_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Select dynamic gesture", "Select a dynamic gesture first.")
            return None
        item = self.dynamic_table.item(row, 0)
        return item.text() if item else None

    def rename_selected_dynamic_gesture(self):
        name = self.selected_dynamic_name()
        if not name:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename dynamic gesture", f"New name for '{name}':", text=name
        )
        if not ok:
            return
        try:
            gesture = self.dynamic_learner.rename_gesture(name, new_name)
            self.save_dynamic_gesture_memory()
            self.refresh_dynamic_gesture_table()
            self.set_dynamic_status(f"Renamed '{name}' to '{gesture.name}'.")
        except Exception as error:
            QMessageBox.critical(self, "Rename failed", str(error))

    def delete_selected_dynamic_gesture(self):
        name = self.selected_dynamic_name()
        if not name:
            return
        answer = QMessageBox.question(
            self,
            "Delete dynamic gesture",
            f"Delete '{name}' and all stored landmark trajectories?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.dynamic_learner.delete_gesture(name)
            self.save_dynamic_gesture_memory()
            self.refresh_dynamic_gesture_table()
            self.motion_segmenter.reset()
            self.set_dynamic_status(f"Deleted dynamic gesture '{name}'.")
        except Exception as error:
            QMessageBox.critical(self, "Delete failed", str(error))

    def clear_all_dynamic_gestures(self):
        if not self.dynamic_learner.gestures:
            self.set_dynamic_status("There are no dynamic gestures to clear.")
            return
        answer = QMessageBox.question(
            self,
            "Clear dynamic gesture memory",
            "Delete ALL learned dynamic gestures and trajectories?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.dynamic_learner.clear()
        self.save_dynamic_gesture_memory()
        self.refresh_dynamic_gesture_table()
        self.motion_segmenter.reset()
        self.dynamic_live_prediction.setText("—")
        self.dynamic_live_metrics.setText("Waiting for motion")
        self.set_dynamic_status("All dynamic gesture memory cleared.")

    # ========================================================
    # Dynamic runtime recognition
    # ========================================================

    def process_dynamic_runtime(self, now: float):
        if self.dynamic_teaching:
            self.motion_segmenter.reset()
            self.process_dynamic_training_frame(now)
            state = "RECORDING DEMO" if self.dynamic_demo_recording else "TEACHING"
            self._set_dynamic_state(state, "warning")
            return

        if self.teaching:
            self.motion_segmenter.reset()
            self._set_dynamic_state("PAUSED", "neutral")
            return

        if now - self.last_dynamic_sample_time < self.dynamic_sample_interval:
            if (
                now >= self.dynamic_prediction_hold_until
                and self.motion_segmenter.state == self.motion_segmenter.IDLE
            ):
                self.dynamic_live_prediction.setText("—")
                self.dynamic_live_metrics.setText("Waiting for motion")
            return

        self.last_dynamic_sample_time = now

        if not self.dynamic_learner.gestures:
            self.motion_segmenter.reset()
            self._set_dynamic_state("NO GESTURES", "neutral")
            return

        observation = build_dynamic_observation(self.current_hands, now)
        result = self.motion_segmenter.update(observation, now)
        state_tone = "warning" if result.state != self.motion_segmenter.IDLE else "neutral"
        self._set_dynamic_state(str(result.state), state_tone)

        if result.completed is None:
            if now >= self.dynamic_prediction_hold_until and result.state == self.motion_segmenter.IDLE:
                self.dynamic_live_prediction.setText("—")
                self.dynamic_live_metrics.setText("Waiting for motion")
            return

        try:
            trajectory = prepare_dynamic_trajectory(result.completed)
        except Exception:
            self.motion_segmenter.set_cooldown(now, seconds=0.35)
            return

        prediction = self.dynamic_learner.predict(trajectory)
        self.dynamic_prediction_hold_until = now + (2.0 if prediction.accepted else 1.0)

        confidence = (
            f"{prediction.confidence * 100.0:.0f}% confidence index"
            if prediction.confidence is not None
            else "confidence —"
        )
        if prediction.accepted:
            self.dynamic_live_prediction.setText(prediction.label)
            self.dynamic_live_metrics.setText(
                f"{confidence}\nDTW {prediction.distance:.4f} / threshold {prediction.threshold:.4f}"
            )
            self._add_history(prediction.label, prediction.confidence, "Dynamic")
            self.set_global_status(f"Dynamic gesture recognized: {prediction.label}")
        else:
            self.dynamic_live_prediction.setText("UNKNOWN")
            if prediction.distance is None:
                self.dynamic_live_metrics.setText(confidence)
            else:
                self.dynamic_live_metrics.setText(
                    f"{confidence}\nNearest DTW {prediction.distance:.4f}"
                )

        self.motion_segmenter.set_cooldown(now, seconds=0.65)

    def _set_dynamic_state(self, state: str, tone: str):
        self.dynamic_state_pill.setText(state)
        self.dynamic_state_pill.set_tone(tone)

    # ========================================================
    # Static recognition / prediction stabilization
    # ========================================================

    def update_prediction(self):
        if self.teaching:
            self.prediction_stabilizer.reset()
            self.current_prediction = None
            self.current_raw_prediction = None
            self.prediction_label.setText(f"Teaching: {self.teaching_name}")
            self.confidence_label.setText("—")
            self.static_metrics.setText("Distance: —\nThreshold: —\nRelative score: —")
            self.reason_label.setText("")
            self.stable_pill.setText("Teaching")
            self.stable_pill.set_tone("warning")
            return

        if self.current_features is None:
            self.prediction_stabilizer.reset()
            self.current_prediction = None
            self.current_raw_prediction = None
            self.prediction_label.setText("NO HAND")
            self.confidence_label.setText("—")
            self.static_metrics.setText("Distance: —\nThreshold: —\nRelative score: —")
            self.reason_label.setText("")
            self.stable_pill.setText("Waiting")
            self.stable_pill.set_tone("neutral")
            return

        raw_prediction = self.learner.predict(
            self.current_features,
            hand_signature=self.current_hand_signature,
        )
        self.current_raw_prediction = raw_prediction
        stabilized = self.prediction_stabilizer.update(raw_prediction)
        prediction = stabilized.prediction
        self.current_prediction = prediction

        if prediction is None:
            candidate = stabilized.candidate_label or "prediction"
            self.prediction_label.setText("STABILIZING")
            self.confidence_label.setText("—")
            self.static_metrics.setText("Distance: —\nThreshold: —\nRelative score: —")
            self.reason_label.setText(
                f"Checking '{candidate}' ({stabilized.candidate_count}/{stabilized.required_count})"
            )
            self.stable_pill.setText("Stabilizing")
            self.stable_pill.set_tone("warning")
            return

        self.prediction_label.setText(prediction.label)
        if prediction.confidence is not None:
            self.confidence_label.setText(f"{prediction.confidence * 100.0:.0f}%")
        else:
            self.confidence_label.setText("—")

        if prediction.distance is None:
            self.static_metrics.setText("Distance: —\nThreshold: —\nRelative score: —")
        else:
            relative = (
                f"{prediction.relative_distance:.2f}x"
                if prediction.relative_distance is not None
                else "—"
            )
            self.static_metrics.setText(
                f"Nearest positive distance: {prediction.distance:.4f}\n"
                f"Acceptance threshold: {prediction.threshold:.4f}\n"
                f"Relative score: {relative}\nConfidence index is not a calibrated probability."
            )

        if stabilized.pending:
            candidate = stabilized.candidate_label or "prediction"
            self.reason_label.setText(
                f"Holding '{prediction.label}' — checking '{candidate}' ({stabilized.candidate_count}/{stabilized.required_count})"
            )
            self.stable_pill.setText("Holding stable")
            self.stable_pill.set_tone("warning")
            return

        if prediction.accepted:
            self.reason_label.setText("")
            self.stable_pill.setText("Stable")
            self.stable_pill.set_tone("success")
            self._add_history(prediction.label, prediction.confidence, "Static")
        else:
            self.stable_pill.setText("Unknown")
            self.stable_pill.set_tone("neutral")
            if prediction.rejection_reason == "outside_positive_region":
                nearest = prediction.nearest_label or "known gesture"
                self.reason_label.setText(f"Outside learned region for '{nearest}'")
            elif prediction.rejection_reason == "hard_negative":
                self.reason_label.setText("Rejected using learned negative feedback")
            elif prediction.rejection_reason == "no_gestures":
                self.reason_label.setText("No static gestures learned yet")
            elif prediction.rejection_reason == "hand_configuration":
                self.reason_label.setText(
                    f"No learned gesture matches {self.current_hand_signature} input"
                )
            else:
                self.reason_label.setText("")

    # ========================================================
    # Interactive feedback
    # ========================================================

    def confirm_prediction(self):
        if self.teaching:
            return
        if self.current_features is None or self.current_prediction is None:
            self.set_global_status("No gesture available to confirm.")
            return
        if not self.current_prediction.accepted:
            self.set_global_status("The current pose is UNKNOWN; there is no known prediction to confirm.")
            return

        before = self.learner.gestures[self.current_prediction.label].sample_count
        try:
            gesture = self.learner.update_gesture(
                self.current_prediction.label,
                self.current_features.copy(),
            )
            saved = self.save_gesture_memory()
        except Exception as error:
            self.set_global_status(f"Feedback failed: {error}")
            return

        added = gesture.sample_count > before
        self.refresh_gesture_table()
        message = (
            f"Confirmed '{gesture.name}'. Useful positive evidence was added."
            if added
            else f"Confirmed '{gesture.name}'. The pose was already represented, so no duplicate was stored."
        )
        if saved:
            message += " Memory saved."
        self.set_global_status(message)

    def begin_correction(self):
        if self.teaching:
            return
        if self.current_features is None or self.current_prediction is None:
            self.set_global_status("No current pose available for correction.")
            return

        self.feedback_features = self.current_features.copy()
        self.feedback_hand_signature = self.current_hand_signature
        prediction = self.current_prediction
        self.feedback_predicted_label = prediction.label if prediction.accepted else None

        compatible = [
            name
            for name, gesture in self.learner.gestures.items()
            if gesture.hand_signature == self.feedback_hand_signature
            and gesture.feature_dimension == self.feedback_features.shape[0]
        ]

        dialog = CorrectionDialog(
            self.feedback_predicted_label,
            compatible,
            self,
        )
        result = dialog.exec()
        if result != QDialog.DialogCode.Accepted:
            self._clear_feedback()
            self.set_global_status("Feedback cancelled.")
            return

        if dialog.action == "unknown":
            self._apply_unknown_feedback()
        elif dialog.action == "correct":
            self._apply_known_correction(dialog.actual_label)

    def _apply_known_correction(self, actual_label: str):
        predicted_label = self.feedback_predicted_label
        try:
            self.learner.apply_correction(
                predicted_label=predicted_label,
                actual_label=actual_label,
                sample=self.feedback_features,
            )
            saved = self.save_gesture_memory()
        except Exception as error:
            self.set_global_status(f"Correction failed: {error}")
            self._clear_feedback()
            return

        if predicted_label and predicted_label != actual_label:
            message = (
                f"Learned correction: this is '{actual_label}', not '{predicted_label}'. Positive and hard-negative memory were updated."
            )
        else:
            message = f"Learned that this previously UNKNOWN example belongs to '{actual_label}'."
        if saved:
            message += " Memory saved."
        self.refresh_gesture_table()
        self._clear_feedback()
        self.set_global_status(message)

    def _apply_unknown_feedback(self):
        predicted_label = self.feedback_predicted_label
        try:
            self.learner.mark_unknown(
                predicted_label=predicted_label,
                sample=self.feedback_features,
            )
            saved = self.save_gesture_memory()
        except Exception as error:
            self.set_global_status(f"Negative feedback failed: {error}")
            self._clear_feedback()
            return

        if predicted_label:
            message = f"Learned that this pose is NOT '{predicted_label}'. A hard-negative example was stored."
        else:
            message = "The captured pose was already UNKNOWN; no accepted class required a negative correction."
        if saved:
            message += " Memory saved."
        self.refresh_gesture_table()
        self._clear_feedback()
        self.set_global_status(message)

    def _clear_feedback(self):
        self.feedback_features = None
        self.feedback_predicted_label = None
        self.feedback_hand_signature = None

    # ========================================================
    # Library refresh
    # ========================================================

    def refresh_gesture_table(self):
        if not hasattr(self, "gesture_table"):
            return
        gestures = list(self.learner.gestures.items())
        self.gesture_table.setRowCount(len(gestures))
        for row, (name, gesture) in enumerate(gestures):
            values = [
                name,
                str(gesture.sample_count),
                str(gesture.negative_count),
                str(gesture.prototype_count),
                f"{gesture.spread:.4f}",
                f"{gesture.sample_radius:.4f}",
                gesture.hand_signature,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column != 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.gesture_table.setItem(row, column, item)
        self.gesture_table.resizeRowsToContents()

    def refresh_dynamic_gesture_table(self):
        if not hasattr(self, "dynamic_table"):
            return
        gestures = list(self.dynamic_learner.gestures.items())
        self.dynamic_table.setRowCount(len(gestures))
        for row, (name, gesture) in enumerate(gestures):
            values = [
                name,
                str(gesture.template_count),
                f"{gesture.threshold:.4f}",
                f"{gesture.median_duration:.2f}s",
                gesture.hand_signature,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column != 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.dynamic_table.setItem(row, column, item)
        self.dynamic_table.resizeRowsToContents()

    # ========================================================
    # Recent history
    # ========================================================

    def _add_history(self, label: str, confidence: float | None, kind: str):
        now = time.monotonic()
        # Avoid inserting the same stable label every prediction update.
        if label == self.last_history_label and now - self.last_history_time < 1.5:
            return
        self.last_history_label = label
        self.last_history_time = now
        stamp = time.strftime("%H:%M:%S")
        conf = "—" if confidence is None else f"{confidence * 100.0:.0f}%"
        self.recent_history.appendleft(f"{stamp}   {label}   {conf}   {kind}")
        for index, widget in enumerate(self.history_labels):
            widget.setText(self.recent_history[index] if index < len(self.recent_history) else "—")

    # ========================================================
    # Camera loop
    # ========================================================

    def update_camera(self):
        success, frame = self.cap.read()
        if not success:
            self.camera_state_pill.setText("Camera error")
            self.camera_state_pill.set_tone("danger")
            return

        frame = cv2.flip(frame, 1)

        hands = self.tracker.process(frame)
        self.tracker.draw(frame, hands)
        self.current_hands = hands

        self.current_feature_set = build_frame_features(hands)
        if self.current_feature_set is None:
            self.current_features = None
            self.current_hand_signature = None
            tracking_text = "No hand detected"
            self.input_pill.setText("No input")
            self.input_pill.set_tone("neutral")
        else:
            self.current_features = self.current_feature_set.vector
            self.current_hand_signature = self.current_feature_set.hand_signature
            if self.current_hand_signature == "Both":
                confidence = min(hand.handedness_score for hand in hands[:2])
                tracking_text = f"Tracking both hands — {confidence:.0%} minimum handedness confidence"
            else:
                confidence = hands[0].handedness_score
                tracking_text = f"Tracking {self.current_hand_signature} hand — {confidence:.0%}"
            self.input_pill.setText(self.current_hand_signature)
            self.input_pill.set_tone("success")

        self.live_tracking.setText(tracking_text)
        self.teach_tracking.setText(tracking_text)
        self.dynamic_tracking.setText(tracking_text)
        self.sidebar_tracking.setText(tracking_text)

        if self.teaching:
            self.process_teaching_frame()

        now = time.monotonic()
        if self.teaching or now - self.last_prediction_update >= self.prediction_update_interval:
            self.update_prediction()
            self.last_prediction_update = now

        self.process_dynamic_runtime(now)
        self._render_frame(frame)

    def _render_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

        if self.current_page_name == self.PAGE_TEACH:
            target = self.teach_camera
        elif self.current_page_name == self.PAGE_DYNAMIC:
            target = self.dynamic_camera
        else:
            target = self.live_camera

        pixmap = QPixmap.fromImage(image)
        target.setPixmap(
            pixmap.scaled(
                target.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # ========================================================
    # Cleanup
    # ========================================================

    def closeEvent(self, event):
        try:
            self.save_gesture_memory()
            self.save_dynamic_gesture_memory()
        finally:
            if self.camera_timer.isActive():
                self.camera_timer.stop()
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.tracker.close()
            cv2.destroyAllWindows()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Adaptive Gesture AI")
    app.setOrganizationName("FYP")
    window = AdaptiveGestureQtApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
