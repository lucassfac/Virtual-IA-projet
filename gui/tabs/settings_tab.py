"""
settings_tab.py — Onglet Paramètres & Chargement du modèle.

Permet de :
  - Choisir le fichier .gguf du modèle
  - Choisir un adaptateur .lora (optionnel)
  - Ajuster n_ctx et max_tokens
  - Charger le modèle via un worker (non-bloquant)
  - Voir l'état du modèle actif
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFileDialog, QSpinBox,
    QFrame, QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from gui.workers import ModelLoadWorker


class SettingsTab(QWidget):
    """Onglet de configuration et chargement du modèle."""

    # Émis quand le modèle est chargé avec succès → notifie MainWindow
    model_loaded = pyqtSignal(str, str)   # (model_path, lora_path or "")

    def __init__(self, llm_node, parent=None):
        super().__init__(parent)
        self.llm_node = llm_node
        self._worker = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(0)

        # ── Titre ──
        title = QLabel("Paramètres")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addSpacing(6)

        subtitle = QLabel("Configuration du modèle et des paramètres d'inférence")
        subtitle.setObjectName("statusLabel")
        root.addWidget(subtitle)
        root.addSpacing(28)

        # ── Section : Modèle ──
        root.addWidget(self._section_label("MODÈLE"))
        root.addSpacing(10)

        model_card = self._card()
        card_layout = QVBoxLayout(model_card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(14)

        # Champ modèle
        card_layout.addWidget(self._field_label("Fichier modèle (.gguf)"))
        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Sélectionnez un fichier .gguf …")
        model_row.addWidget(self.model_path_edit)
        btn_model = QPushButton("…")
        btn_model.setObjectName("browseBtn")
        btn_model.setToolTip("Parcourir")
        btn_model.clicked.connect(self._browse_model)
        model_row.addWidget(btn_model)
        card_layout.addLayout(model_row)

        # Champ LoRA
        card_layout.addWidget(self._field_label("Adaptateur LoRA (optionnel)"))
        lora_row = QHBoxLayout()
        lora_row.setSpacing(8)
        self.lora_path_edit = QLineEdit()
        self.lora_path_edit.setPlaceholderText("Laissez vide pour le modèle de base …")
        lora_row.addWidget(self.lora_path_edit)
        btn_lora = QPushButton("…")
        btn_lora.setObjectName("browseBtn")
        btn_lora.setToolTip("Parcourir")
        btn_lora.clicked.connect(self._browse_lora)
        lora_row.addWidget(btn_lora)
        card_layout.addLayout(lora_row)

        root.addWidget(model_card)
        root.addSpacing(20)

        # ── Section : Paramètres ──
        root.addWidget(self._section_label("INFÉRENCE"))
        root.addSpacing(10)

        params_card = self._card()
        params_layout = QVBoxLayout(params_card)
        params_layout.setContentsMargins(20, 20, 20, 20)
        params_layout.setSpacing(14)

        row1 = QHBoxLayout()
        row1.setSpacing(20)

        # n_ctx
        col_ctx = QVBoxLayout()
        col_ctx.setSpacing(6)
        col_ctx.addWidget(self._field_label("Fenêtre de contexte"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 32768)
        self.ctx_spin.setValue(2048)
        self.ctx_spin.setSingleStep(512)
        self.ctx_spin.setToolTip("Nombre de tokens mémorisés (RAM)")
        col_ctx.addWidget(self.ctx_spin)
        row1.addLayout(col_ctx)

        # max_tokens
        col_tok = QVBoxLayout()
        col_tok.setSpacing(6)
        col_tok.addWidget(self._field_label("Tokens max en sortie"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(50, 4096)
        self.max_tokens_spin.setValue(200)
        self.max_tokens_spin.setSingleStep(50)
        self.max_tokens_spin.setToolTip("Longueur maximale de la réponse")
        col_tok.addWidget(self.max_tokens_spin)
        row1.addLayout(col_tok)
        row1.addStretch()

        params_layout.addLayout(row1)
        root.addWidget(params_card)
        root.addSpacing(24)

        # ── Bouton charger ──
        self.load_btn = QPushButton("  Charger le modèle")
        self.load_btn.setObjectName("primaryBtn")
        self.load_btn.setFixedHeight(46)
        self.load_btn.clicked.connect(self._load_model)
        root.addWidget(self.load_btn)
        root.addSpacing(20)

        # ── Status du modèle actif ──
        root.addWidget(self._section_label("MODÈLE ACTIF"))
        root.addSpacing(10)

        self.status_card = self._card()
        status_layout = QVBoxLayout(self.status_card)
        status_layout.setContentsMargins(20, 18, 20, 18)
        status_layout.setSpacing(10)

        status_row = QHBoxLayout()
        self.status_dot = QLabel()
        self.status_dot.setObjectName("statusDot")
        self.status_dot.setFixedSize(10, 10)
        self._set_dot_color("#3A3A3C")
        status_row.addWidget(self.status_dot)
        self.status_text = QLabel("Aucun modèle chargé")
        self.status_text.setObjectName("statusLabel")
        status_row.addWidget(self.status_text)
        status_row.addStretch()
        status_layout.addLayout(status_row)

        self.model_info_label = QLabel("")
        self.model_info_label.setObjectName("statusLabel")
        self.model_info_label.setWordWrap(True)
        status_layout.addWidget(self.model_info_label)
        self.model_info_label.hide()

        root.addWidget(self.status_card)
        root.addStretch()

    # ------------------------------------------------------------------
    # Helpers UI
    # ------------------------------------------------------------------

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionLabel")
        return lbl

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("statusLabel")
        font = lbl.font()
        font.setPointSize(12)
        lbl.setFont(font)
        return lbl

    def _card(self) -> QWidget:
        card = QWidget()
        card.setObjectName("card")
        return card

    def _set_dot_color(self, color: str):
        self.status_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 5px;"
            " min-width: 10px; max-width: 10px;"
            " min-height: 10px; max-height: 10px;"
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un modèle", "models/",
            "Modèles GGUF (*.gguf);;Tous les fichiers (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def _browse_lora(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un adaptateur LoRA", "models/",
            "Adaptateurs LoRA (*.lora *.bin);;Tous les fichiers (*)"
        )
        if path:
            self.lora_path_edit.setText(path)

    def _load_model(self):
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self._set_status("error", "Veuillez sélectionner un fichier modèle.")
            return

        lora_path = self.lora_path_edit.text().strip() or None

        self.load_btn.setEnabled(False)
        self.load_btn.setText("  Chargement en cours…")
        self._set_dot_color("#FF9F0A")
        self.status_text.setText("Chargement du modèle…")
        self.model_info_label.hide()

        self._worker = ModelLoadWorker(self.llm_node, model_path, lora_path)
        self._worker.success.connect(self._on_load_success)
        self._worker.error.connect(self._on_load_error)
        self._worker.finished.connect(self._on_load_finished)
        self._worker.start()

    def _on_load_success(self, model_path: str):
        lora_path = self.lora_path_edit.text().strip()
        model_name = os.path.basename(model_path)
        lora_info = f"LoRA : {os.path.basename(lora_path)}" if lora_path else "LoRA : aucun"

        self._set_dot_color("#30D158")
        self.status_text.setText(f"Modèle chargé : {model_name}")
        self.model_info_label.setText(
            f"Contexte : {self.ctx_spin.value()} tokens  ·  "
            f"Max tokens : {self.max_tokens_spin.value()}  ·  {lora_info}"
        )
        self.model_info_label.show()
        self.model_loaded.emit(model_path, lora_path or "")

    def _on_load_error(self, message: str):
        self._set_dot_color("#FF453A")
        self.status_text.setText("Échec du chargement")
        self.model_info_label.setText(message)
        self.model_info_label.show()

    def _on_load_finished(self):
        self.load_btn.setEnabled(True)
        self.load_btn.setText("  Charger le modèle")

    def _set_status(self, level: str, msg: str):
        colors = {"error": "#FF453A", "success": "#30D158", "info": "#0A84FF"}
        self._set_dot_color(colors.get(level, "#8E8E93"))
        self.status_text.setText(msg)
