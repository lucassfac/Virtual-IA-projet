"""
main_window.py — Fenêtre principale Neural Forge.
4 onglets : Chat, Bibliothèque, Entraînement, Paramètres
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QStackedWidget, QStatusBar,
    QSizePolicy, QFrame, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSlot, QSize

from core.nodes.llm_node import LLMNode
from core.nodes.vision_node import VisionNode

from gui.tabs.chat_tab import ChatTab
from gui.tabs.models_tab import ModelsTab
from gui.tabs.training_tab import TrainingTab
from gui.tabs.settings_tab import SettingsTab


_NAV = [
    ("Chat",          "Discuter avec l'IA"),
    ("Bibliothèque",  "Gérer les modèles"),
    ("Entraînement",  "Spécialiser un modèle"),
    ("Paramètres",    "Configurer l'application"),
]


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

        self._build_ui()
        self._switch(0)

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("contentArea")
        self.setCentralWidget(central)

        main = QHBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        main.addWidget(self._make_sidebar())

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(
            "background:rgba(255,255,255,0.06);max-width:1px;border:none;"
        )
        main.addWidget(sep)

        self.stack = QStackedWidget()
        self.stack.setObjectName("contentArea")

        self.chat_tab     = ChatTab(self.llm_node, self.vision_node)
        self.models_tab   = ModelsTab()
        self.training_tab = TrainingTab()
        self.settings_tab = SettingsTab(self.llm_node)

        self.stack.addWidget(self.chat_tab)                          # 0
        self.stack.addWidget(self._scroll(self.models_tab))          # 1
        self.stack.addWidget(self._scroll(self.training_tab))        # 2
        self.stack.addWidget(self._scroll(self.settings_tab))        # 3

        main.addWidget(self.stack, stretch=1)

        self.settings_tab.model_loaded.connect(self._on_model_loaded)
        self._build_statusbar()

    def _make_sidebar(self) -> QWidget:
        sb = QWidget()
        sb.setObjectName("sidebar")
        sb.setFixedWidth(180)

        layout = QVBoxLayout(sb)
        layout.setContentsMargins(10, 20, 10, 14)
        layout.setSpacing(0)

        name_lbl = QLabel("Neural Forge")
        name_lbl.setObjectName("appTitle")
        name_lbl.setContentsMargins(6, 0, 0, 0)
        layout.addWidget(name_lbl)

        sub_lbl = QLabel("EDGE AI STUDIO")
        sub_lbl.setObjectName("appSubtitle")
        sub_lbl.setContentsMargins(6, 2, 0, 0)
        layout.addWidget(sub_lbl)

        layout.addSpacing(18)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(
            "background:rgba(255,255,255,0.07);max-height:1px;border:none;"
        )
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

        ver = QLabel("v0.1.0 — alpha")
        ver.setObjectName("versionLabel")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver)

        return sb

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
        sb.addPermanentWidget(QLabel("Neural Forge  ·  Edge AI  ·  100% local"))

    def _switch(self, idx: int):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._nav_btns):
            btn.set_active(i == idx)
        # Refresh liste installés à chaque visite de la bibliothèque
        if idx == 1:
            # Forcer une relecture du dossier models/ à chaque visite
            from core.model_manager import get_models_dir, list_local_models
            get_models_dir()  # crée le dossier si absent
            self.models_tab.refresh_installed()

    @pyqtSlot(str, str)
    def _on_model_loaded(self, model_path: str, lora_path: str):
        import os
        name = os.path.basename(model_path)
        lora = f" + {os.path.basename(lora_path)}" if lora_path else ""
        self._sb_dot_color("#30D158")
        self._sb_lbl.setText(f"{name}{lora}")
        self.chat_tab.on_model_loaded(model_path, lora_path)
        # Refresh bibliothèque pour marquer le modèle actif
        self.models_tab.refresh_installed()

    def _sb_dot_color(self, c: str):
        self._sb_dot.setStyleSheet(
            f"background-color:{c};border-radius:4px;"
            "min-width:8px;max-width:8px;min-height:8px;max-height:8px;"
        )
