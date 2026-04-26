"""
main_window.py — Fenêtre principale Neural Forge.
Sidebar collapsible avec animation + auto-load du dernier modèle.
"""

import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QStackedWidget, QStatusBar,
    QSizePolicy, QFrame, QScrollArea,
)
from PyQt6.QtCore import (
    Qt, pyqtSlot, QSize, QTimer,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
)

from core.nodes.llm_node import LLMNode
from core.nodes.vision_node import VisionNode
from core.session_manager import load_last_model, save_last_model

from gui.tabs.chat_tab import ChatTab
from gui.tabs.models_tab import ModelsTab
from gui.tabs.training_tab import TrainingTab
from gui.tabs.settings_tab import SettingsTab
from gui.system_monitor import SystemMonitor
from core.utils.hardware import get_profile

_NAV = [
    ("Chat",         "Discuter avec l'IA"),
    ("Bibliothèque", "Gérer les modèles"),
    ("Entraînement", "Spécialiser un modèle"),
    ("Paramètres",   "Configurer l'application"),
]

SIDEBAR_W = 182


class NavBtn(QPushButton):
    def __init__(self, label: str, tip: str, parent=None):
        super().__init__(f"  {label}", parent)
        self.setToolTip(tip)
        self.setObjectName("navBtn")
        self.setFixedHeight(42)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_active(self, on: bool):
        self.setProperty("active", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Forge")
        self.setMinimumSize(QSize(720, 500))
        self.resize(QSize(1060, 680))

        self.llm_node    = LLMNode(name="LLM-Principal")
        self.vision_node = VisionNode(name="Vision-Principal")
        self._sidebar_open = True
        self._animating    = False

        self._build_ui()
        self._switch(0)
        QTimer.singleShot(600, self._auto_load_last_model)

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("contentArea")
        self.setCentralWidget(central)

        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Sidebar ──
        self._sidebar = self._make_sidebar()
        outer.addWidget(self._sidebar)

        self._vsep = QFrame()
        self._vsep.setFrameShape(QFrame.Shape.VLine)
        self._vsep.setStyleSheet(
            "background:rgba(255,255,255,0.06);max-width:1px;border:none;"
        )
        outer.addWidget(self._vsep)

        # ── Stack ──
        self.stack = QStackedWidget()
        self.stack.setObjectName("contentArea")

        self.chat_tab     = ChatTab(self.llm_node, self.vision_node)
        self.models_tab   = ModelsTab()
        self.training_tab = TrainingTab()
        self.settings_tab = SettingsTab(self.llm_node, self.vision_node)

        self.stack.addWidget(self.chat_tab)
        self.stack.addWidget(self._scroll(self.models_tab))
        self.stack.addWidget(self._scroll(self.training_tab))
        self.stack.addWidget(self._scroll(self.settings_tab))

        outer.addWidget(self.stack, stretch=1)

        self.settings_tab.model_loaded.connect(self._on_model_loaded)
        self._build_statusbar()

    def _make_sidebar(self) -> QWidget:
        sb = QWidget()
        sb.setObjectName("sidebar")
        sb.setFixedWidth(SIDEBAR_W)
        sb.setMinimumWidth(0)

        layout = QVBoxLayout(sb)
        layout.setContentsMargins(10, 16, 10, 14)
        layout.setSpacing(0)

        # Logo + toggle
        top_row = QHBoxLayout()
        top_row.setContentsMargins(6, 0, 0, 0)
        top_row.setSpacing(6)

        name_col = QVBoxLayout()
        name_col.setSpacing(1)
        name_lbl = QLabel("Neural Forge")
        name_lbl.setObjectName("appTitle")
        sub_lbl = QLabel("EDGE AI STUDIO")
        sub_lbl.setObjectName("appSubtitle")
        name_col.addWidget(name_lbl)
        name_col.addWidget(sub_lbl)
        top_row.addLayout(name_col)
        top_row.addStretch()

        toggle = QPushButton("☰")
        toggle.setFixedSize(32, 32)
        toggle.setStyleSheet(
            "background:rgba(255,255,255,0.06);color:#636366;"
            "border:none;border-radius:8px;font-size:15px;"
        )
        toggle.setToolTip("Masquer le menu  (cliquez ☰ pour rouvrir)")
        toggle.clicked.connect(self._close_sidebar)
        top_row.addWidget(toggle)
        layout.addLayout(top_row)
        layout.addSpacing(16)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:rgba(255,255,255,0.07);max-height:1px;border:none;")
        layout.addWidget(sep)
        layout.addSpacing(10)

        self._nav_btns: list[NavBtn] = []
        for i, (label, tip) in enumerate(_NAV):
            btn = NavBtn(label, tip)
            btn.clicked.connect(lambda _, idx=i: self._switch(idx))
            layout.addWidget(btn)
            layout.addSpacing(2)
            self._nav_btns.append(btn)

        layout.addStretch()

        self._sidebar_model_lbl = QLabel("Aucun modèle")
        self._sidebar_model_lbl.setStyleSheet(
            "color:#3A3A3C;font-size:10px;background:transparent;padding:0 4px;"
        )
        self._sidebar_model_lbl.setWordWrap(True)
        layout.addWidget(self._sidebar_model_lbl)
        layout.addSpacing(4)

        ver = QLabel("v0.1.0 — alpha")
        ver.setObjectName("versionLabel")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver)

        return sb

    # ── Animation sidebar ──────────────────────────────────────────────

    def _close_sidebar(self):
        if self._animating: return
        self._animating = True

        anim = QPropertyAnimation(self._sidebar, b"maximumWidth")
        anim.setDuration(220)
        anim.setStartValue(SIDEBAR_W)
        anim.setEndValue(0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        anim2 = QPropertyAnimation(self._vsep, b"maximumWidth")
        anim2.setDuration(220)
        anim2.setStartValue(1)
        anim2.setEndValue(0)
        anim2.setEasingCurve(QEasingCurve.Type.OutCubic)

        grp = QParallelAnimationGroup(self)
        grp.addAnimation(anim)
        grp.addAnimation(anim2)
        grp.finished.connect(self._on_sidebar_closed)
        grp.start()
        self._anim_grp = grp

    def _on_sidebar_closed(self):
        self._sidebar.setFixedWidth(0)
        self._vsep.setMaximumWidth(0)
        self._sidebar_open = False
        self._animating = False
        # Bouton flottant dans le header du chat (pas par-dessus tout)
        self._open_btn = QPushButton("☰")
        self._open_btn.setParent(self.stack)
        self._open_btn.setFixedSize(30, 30)
        self._open_btn.setStyleSheet(
            "background:rgba(255,255,255,0.08);color:#8E8E93;"
            "border:none;border-radius:8px;font-size:14px;"
        )
        self._open_btn.move(6, 10)
        self._open_btn.show()
        self._open_btn.raise_()
        self._open_btn.clicked.connect(self._open_sidebar)

    def _open_sidebar(self):
        if self._animating: return
        self._animating = True
        if hasattr(self, "_open_btn"):
            self._open_btn.hide()
            self._open_btn.deleteLater()

        self._sidebar.setFixedWidth(0)
        self._sidebar.setMaximumWidth(SIDEBAR_W)
        self._vsep.setMaximumWidth(1)

        anim = QPropertyAnimation(self._sidebar, b"maximumWidth")
        anim.setDuration(220)
        anim.setStartValue(0)
        anim.setEndValue(SIDEBAR_W)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: (
            self._sidebar.setFixedWidth(SIDEBAR_W),
            setattr(self, "_sidebar_open", True),
            setattr(self, "_animating", False),
        ))
        anim.start()
        self._anim_open = anim

    # ── Scroll wrapper ─────────────────────────────────────────────────

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        area.setStyleSheet(
            "QScrollArea{background:#161618;border:none;}"
            "QScrollArea > QWidget > QWidget{background:#161618;}"
        )
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return area

    # ── Status bar ────────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._sb_dot = QLabel()
        self._sb_dot.setFixedSize(8, 8)
        self._sb_dot_color("#3A3A3C")
        sb.addWidget(self._sb_dot)
        sb.addWidget(QLabel(" "))

        self._sb_lbl = QLabel("Aucun modèle chargé")
        sb.addWidget(self._sb_lbl)

        sep1 = QLabel("  ·  ")
        sep1.setStyleSheet("color:#3A3A3C;")
        sb.addPermanentWidget(sep1)

        self._cpu_lbl = QLabel("CPU —%")
        self._cpu_lbl.setStyleSheet("color:#48484A;font-size:11px;min-width:58px;")
        sb.addPermanentWidget(self._cpu_lbl)

        self._ram_lbl = QLabel("RAM —%")
        self._ram_lbl.setStyleSheet("color:#48484A;font-size:11px;min-width:110px;margin-left:8px;")
        sb.addPermanentWidget(self._ram_lbl)

        self._gpu_lbl = QLabel("")
        self._gpu_lbl.setStyleSheet("color:#48484A;font-size:11px;margin-left:8px;")
        sb.addPermanentWidget(self._gpu_lbl)

        sep2 = QLabel("  ·  ")
        sep2.setStyleSheet("color:#3A3A3C;")
        sb.addPermanentWidget(sep2)
        sb.addPermanentWidget(QLabel("100% local"))

        # Badge performance
        sep3 = QLabel("  ·  ")
        sep3.setStyleSheet("color:#2C2C2E;")
        sb.addPermanentWidget(sep3)

        hw = get_profile()
        self._perf_badge = QLabel(hw.badge_text)
        self._perf_badge.setStyleSheet(
            f"color:{hw.badge_color};font-size:10px;"
            "font-weight:600;letter-spacing:0.5px;"
        )
        self._perf_badge.setToolTip(
            f"RAM : {hw.ram_total_gb} Go  ·  "
            f"CPU : {hw.cpu_cores_phys} cœurs  ·  "
            f"GPU : {hw.gpu.name}\n"
            f"n_threads={hw.n_threads}  n_ctx={hw.n_ctx}  "
            f"n_batch={hw.n_batch}  "
            f"flash_attn={'oui' if hw.flash_attn else 'non'}\n"
            f"CUDA : {'oui ✓' if hw.gpu.cuda_ok else 'non — recompiler llama-cpp'}"
        )
        sb.addPermanentWidget(self._perf_badge)

        self._monitor = SystemMonitor(self)
        self._monitor.stats_updated.connect(self._on_stats)
        self._monitor.start()

    # ── Navigation ────────────────────────────────────────────────────

    def _switch(self, idx: int):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._nav_btns):
            btn.set_active(i == idx)
        if idx == 1:
            from core.model_manager import get_models_dir
            get_models_dir()
            self.models_tab.refresh_installed()

    # ── Auto-load ─────────────────────────────────────────────────────

    def _auto_load_last_model(self):
        model_path, lora_path = load_last_model()
        if not model_path or not os.path.exists(model_path):
            return
        self.settings_tab.model_picker.set_path(model_path)
        if lora_path and os.path.exists(lora_path):
            self.settings_tab.lora_picker.set_path(lora_path)
        self.settings_tab._load()
        name = os.path.basename(model_path)
        lora = f" + {os.path.basename(lora_path)}" if lora_path else ""
        self._sb_lbl.setText(f"Chargement auto : {name}{lora}…")

    # ── Slots ─────────────────────────────────────────────────────────

    @pyqtSlot(str, str)
    def _on_model_loaded(self, model_path: str, lora_path: str):
        name = os.path.basename(model_path)
        lora = f" + {os.path.basename(lora_path)}" if lora_path else ""
        self._sb_dot_color("#30D158")
        self._sb_lbl.setText(f"{name}{lora}")
        self._sidebar_model_lbl.setText(f"● {name}")
        self._sidebar_model_lbl.setStyleSheet(
            "color:#30D158;font-size:10px;background:transparent;padding:0 4px;"
        )
        self.chat_tab.on_model_loaded(model_path, lora_path)
        self.models_tab.refresh_installed()
        save_last_model(model_path, lora_path)

    @pyqtSlot(float, float, float, float, str)
    def _on_stats(self, cpu: float, ram: float, ram_used: float, ram_total: float, gpu: str):
        cpu_c = "#30D158" if cpu < 50 else "#FF9F0A" if cpu < 80 else "#FF453A"
        self._cpu_lbl.setText(f"CPU {cpu:.0f}%")
        self._cpu_lbl.setStyleSheet(f"color:{cpu_c};font-size:11px;min-width:58px;")
        ram_c = "#30D158" if ram < 60 else "#FF9F0A" if ram < 85 else "#FF453A"
        self._ram_lbl.setText(f"RAM {ram:.0f}%  {ram_used:.1f}/{ram_total:.0f}Go")
        self._ram_lbl.setStyleSheet(f"color:{ram_c};font-size:11px;min-width:110px;margin-left:8px;")
        self._gpu_lbl.setText(f"  ·  {gpu}" if gpu else "")

    def _sb_dot_color(self, c: str):
        self._sb_dot.setStyleSheet(
            f"background-color:{c};border-radius:4px;"
            "min-width:8px;max-width:8px;min-height:8px;max-height:8px;"
        )
