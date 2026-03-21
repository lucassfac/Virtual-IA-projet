"""
training_tab.py — Onglet Entraînement LoRA.

Fonctionnalités :
  - Sélection dataset (.jsonl) et modèle de base (.gguf)
  - Progression étape par étape + courbe de loss
  - Log en temps réel
  - Chemin de sortie de l'adaptateur
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFileDialog, QProgressBar,
    QPlainTextEdit, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSlot, QDateTime
from PyQt6.QtGui import QFont

from core.types import DataPacket, DataType
from core.nodes.trainer_node import TrainerNode
from gui.workers import TrainingWorker


class TrainingTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trainer_node = None
        self._worker = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(0)

        # ── Titre ──
        title = QLabel("Entraînement LoRA")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addSpacing(6)

        subtitle = QLabel("Créez un expert métier en spécialisant un modèle de base")
        subtitle.setObjectName("statusLabel")
        root.addWidget(subtitle)
        root.addSpacing(28)

        # ── Section : Fichiers ──
        root.addWidget(self._section_label("FICHIERS"))
        root.addSpacing(10)

        files_card = QWidget()
        files_card.setObjectName("card")
        f_layout = QVBoxLayout(files_card)
        f_layout.setContentsMargins(20, 20, 20, 20)
        f_layout.setSpacing(14)

        # Dataset
        f_layout.addWidget(self._field_label("Dataset d'entraînement (.jsonl)"))
        dataset_row = QHBoxLayout()
        dataset_row.setSpacing(8)
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("data/mon_dataset.jsonl")
        dataset_row.addWidget(self.dataset_edit)
        btn_d = QPushButton("…")
        btn_d.setObjectName("browseBtn")
        btn_d.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(btn_d)
        f_layout.addLayout(dataset_row)

        # Modèle de base
        f_layout.addWidget(self._field_label("Modèle de base (.gguf)"))
        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        self.base_model_edit = QLineEdit()
        self.base_model_edit.setPlaceholderText("models/tinyllama.gguf")
        model_row.addWidget(self.base_model_edit)
        btn_m = QPushButton("…")
        btn_m.setObjectName("browseBtn")
        btn_m.clicked.connect(self._browse_model)
        model_row.addWidget(btn_m)
        f_layout.addLayout(model_row)

        # Dossier de sortie
        f_layout.addWidget(self._field_label("Dossier de sortie"))
        self.output_edit = QLineEdit()
        self.output_edit.setText("models/")
        f_layout.addWidget(self.output_edit)

        root.addWidget(files_card)
        root.addSpacing(20)

        # ── Section : Progression ──
        root.addWidget(self._section_label("PROGRESSION"))
        root.addSpacing(10)

        progress_card = QWidget()
        progress_card.setObjectName("card")
        p_layout = QVBoxLayout(progress_card)
        p_layout.setContentsMargins(20, 18, 20, 18)
        p_layout.setSpacing(12)

        # Barre de progression
        progress_header = QHBoxLayout()
        self.step_label = QLabel("En attente…")
        self.step_label.setObjectName("statusLabel")
        progress_header.addWidget(self.step_label)
        progress_header.addStretch()
        self.loss_label = QLabel("")
        self.loss_label.setObjectName("statusLabel")
        self.loss_label.setStyleSheet("color: #30D158; font-weight: 500;")
        progress_header.addWidget(self.loss_label)
        p_layout.addLayout(progress_header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(6)
        p_layout.addWidget(self.progress_bar)

        # Log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(130)
        self.log_view.setPlaceholderText("Les logs d'entraînement apparaîtront ici…")
        self.log_view.setStyleSheet(
            "background-color: #0A0A0A; border: none;"
            " border-radius: 10px; padding: 10px;"
            " font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;"
            " font-size: 11px; color: #8E8E93;"
        )
        p_layout.addWidget(self.log_view)

        root.addWidget(progress_card)
        root.addSpacing(20)

        # ── Bouton lancer ──
        btn_row = QHBoxLayout()

        self.train_btn = QPushButton("  Lancer l'entraînement")
        self.train_btn.setObjectName("primaryBtn")
        self.train_btn.setFixedHeight(46)
        self.train_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self.train_btn, stretch=1)

        self.stop_btn = QPushButton("Arrêter")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setFixedHeight(46)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setFixedWidth(100)
        btn_row.addWidget(self.stop_btn)

        root.addLayout(btn_row)

        # ── Résultat ──
        self.result_label = QLabel("")
        self.result_label.setObjectName("statusLabel")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addSpacing(10)
        root.addWidget(self.result_label)
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
        return lbl

    def _log(self, msg: str):
        ts = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_view.appendPlainText(f"[{ts}] {msg}")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un dataset", "data/",
            "JSONL (*.jsonl);;JSON (*.json);;Texte (*.txt);;Tous (*)"
        )
        if path:
            self.dataset_edit.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un modèle de base", "models/",
            "Modèles GGUF (*.gguf);;Tous les fichiers (*)"
        )
        if path:
            self.base_model_edit.setText(path)

    def _start_training(self):
        dataset_path = self.dataset_edit.text().strip()
        model_path = self.base_model_edit.text().strip()
        output_dir = self.output_edit.text().strip() or "models/"

        if not dataset_path:
            self._log("Erreur : sélectionnez un dataset.")
            return
        if not model_path:
            self._log("Erreur : sélectionnez un modèle de base.")
            return

        # Création d'un nouveau TrainerNode pour chaque run
        self.trainer_node = TrainerNode(
            name="LoRA-Trainer",
            output_dir=output_dir,
        )
        dataset_pkt = DataPacket(DataType.TEXT, dataset_path)
        model_pkt = DataPacket(DataType.TEXT, model_path)
        self.trainer_node.set_inputs(dataset_pkt, model_pkt)

        # Reset UI
        self.progress_bar.setValue(0)
        self.step_label.setText("Initialisation…")
        self.loss_label.setText("")
        self.result_label.setText("")
        self.log_view.clear()
        self._log(f"Démarrage — dataset: {os.path.basename(dataset_path)}")
        self._log(f"Modèle de base: {os.path.basename(model_path)}")

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._worker = TrainingWorker(self.trainer_node)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop_training(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._log("Entraînement interrompu par l'utilisateur.")
            self._reset_buttons()

    @pyqtSlot(int, int, float)
    def _on_progress(self, step: int, total: int, loss: float):
        pct = int((step / total) * 100)
        self.progress_bar.setValue(pct)
        self.step_label.setText(f"Étape {step} / {total}")
        self.loss_label.setText(f"loss = {loss:.4f}")
        self._log(f"Étape {step}/{total}  —  loss = {loss:.4f}")

    @pyqtSlot(str)
    def _on_finished(self, lora_path: str):
        self.progress_bar.setValue(100)
        self.step_label.setText("Entraînement terminé ✓")
        self.result_label.setText(
            f"Adaptateur créé : {lora_path}"
        )
        self.result_label.setStyleSheet("color: #30D158; font-size: 12px;")
        self._log(f"Adaptateur sauvegardé : {lora_path}")
        self._reset_buttons()

    @pyqtSlot(str)
    def _on_error(self, message: str):
        self.step_label.setText("Erreur")
        self.result_label.setText(f"Erreur : {message}")
        self.result_label.setStyleSheet("color: #FF453A; font-size: 12px;")
        self._log(f"ERREUR : {message}")
        self._reset_buttons()

    def _reset_buttons(self):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
