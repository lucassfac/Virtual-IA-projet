"""
main_window.py — Fenêtre principale de Neural Forge.

Architecture :
  ┌──────────────┬──────────────────────────────────┐
  │   Sidebar    │         QStackedWidget            │
  │   (180px)    │   Chat / Vision / Train / Params  │
  └──────────────┴──────────────────────────────────┘

La sidebar est inspirée du design macOS (System Settings / Finder).
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QStackedWidget, QStatusBar,
    QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSlot, QSize
from PyQt6.QtGui import QFont, QIcon

from core.nodes.llm_node import LLMNode
from core.nodes.vision_node import VisionNode

from gui.tabs.chat_tab import ChatTab
from gui.tabs.vision_tab import VisionTab
from gui.tabs.training_tab import TrainingTab
from gui.tabs.settings_tab import SettingsTab


# ── Définition des onglets ────────────────────────────────────────────
_NAV_ITEMS = [
    ("💬", "Chat",          "Converser avec l'IA"),
    ("🔍", "Vision",        "Analyser des images"),
    ("⚡", "Entraînement",  "Spécialiser un modèle"),
    ("⚙",  "Paramètres",   "Configurer l'application"),
]


class NavButton(QPushButton):
    """Bouton de navigation latérale."""

    def __init__(self, icon: str, label: str, tooltip: str, parent=None):
        super().__init__(f"  {icon}  {label}", parent)
        self.setObjectName("navBtn")
        self.setCheckable(False)
        self.setToolTip(tooltip)
        self.setFixedHeight(44)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._active = False

    def set_active(self, active: bool):
        self._active = active
        self.setProperty("active", "true" if active else "false")
        # Force le rechargement du style
        self.style().unpolish(self)
        self.style().polish(self)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Forge")
        self.setMinimumSize(QSize(860, 580))
        self.resize(QSize(1080, 680))

        # ── Nœuds partagés ──
        self.llm_node = LLMNode(name="LLM-Principal")
        self.vision_node = VisionNode(name="Vision-Principal")

        self._build_ui()
        self._nav_buttons[0].set_active(True)

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("contentArea")
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Sidebar ──
        sidebar = self._build_sidebar()
        main_layout.addWidget(sidebar)

        # ── Séparateur vertical ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("background-color: #2C2C2E; max-width: 1px;")
        main_layout.addWidget(sep)

        # ── Contenu ──
        self.stack = QStackedWidget()
        self.stack.setObjectName("contentArea")

        # Instanciation des onglets
        self.chat_tab = ChatTab(self.llm_node)
        self.vision_tab = VisionTab(self.vision_node)
        self.training_tab = TrainingTab()
        self.settings_tab = SettingsTab(self.llm_node)

        self.stack.addWidget(self.chat_tab)      # 0
        self.stack.addWidget(self.vision_tab)    # 1
        self.stack.addWidget(self.training_tab)  # 2
        self.stack.addWidget(self.settings_tab)  # 3

        main_layout.addWidget(self.stack, stretch=1)

        # ── Connexion du chargement de modèle ──
        self.settings_tab.model_loaded.connect(self._on_model_loaded)

        # ── Status bar ──
        self._build_status_bar()

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(190)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 20, 12, 16)
        layout.setSpacing(0)

        # ── Logo ──
        logo_area = QVBoxLayout()
        logo_area.setSpacing(2)
        logo_area.setContentsMargins(8, 0, 0, 0)

        app_name = QLabel("Neural Forge")
        app_name.setObjectName("appTitle")
        logo_area.addWidget(app_name)

        app_sub = QLabel("EDGE AI STUDIO")
        app_sub.setObjectName("appSubtitle")
        logo_area.addWidget(app_sub)

        layout.addLayout(logo_area)
        layout.addSpacing(20)

        # ── Séparateur ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(
            "background-color: #2C2C2E; max-height: 1px; border: none;"
        )
        layout.addWidget(sep)
        layout.addSpacing(12)

        # ── Boutons de navigation ──
        self._nav_buttons: list[NavButton] = []
        for i, (icon, label, tooltip) in enumerate(_NAV_ITEMS):
            btn = NavButton(icon, label, tooltip)
            btn.clicked.connect(lambda checked, idx=i: self._switch_tab(idx))
            layout.addWidget(btn)
            layout.addSpacing(2)
            self._nav_buttons.append(btn)

        layout.addStretch()

        # ── Version ──
        version = QLabel("v0.1.0 — alpha")
        version.setObjectName("versionLabel")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        return sidebar

    def _build_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._status_model_dot = QLabel()
        self._status_model_dot.setFixedSize(8, 8)
        self._set_status_dot("#3A3A3C")
        self.status_bar.addWidget(self._status_model_dot)
        self.status_bar.addWidget(QLabel(" "))

        self._status_model_label = QLabel("Aucun modèle chargé")
        self.status_bar.addWidget(self._status_model_label)

        self.status_bar.addPermanentWidget(
            QLabel("Neural Forge  ·  Edge AI  ·  100% local")
        )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _switch_tab(self, index: int):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_buttons):
            btn.set_active(i == index)

    # ------------------------------------------------------------------
    # Slot : modèle chargé
    # ------------------------------------------------------------------

    @pyqtSlot(str, str)
    def _on_model_loaded(self, model_path: str, lora_path: str):
        import os
        name = os.path.basename(model_path)
        lora = f" + {os.path.basename(lora_path)}" if lora_path else ""

        self._set_status_dot("#30D158")
        self._status_model_label.setText(f"{name}{lora}")

        # Notifie l'onglet Chat
        self.chat_tab.on_model_loaded(model_path, lora_path)

    def _set_status_dot(self, color: str):
        self._status_model_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 4px;"
            " min-width: 8px; max-width: 8px;"
            " min-height: 8px; max-height: 8px;"
        )
