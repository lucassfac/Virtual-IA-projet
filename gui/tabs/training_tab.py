"""
training_tab.py — Onglet Création de Compétence (.skill).
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QProgressBar, QPlainTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSlot, QDateTime
from core.types import DataPacket, DataType
from core.nodes.trainer_node import TrainerNode
from gui.workers import TrainingWorker
from gui.widgets import FilePickerRow

class TrainingTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trainer_node = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 32, 36, 32)
        root.setSpacing(0)

        # ── Titre ──
        title = QLabel("Création de Compétence")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addSpacing(4)
        sub = QLabel("Compilez vos documents en une compétence experte (.skill)")
        sub.setObjectName("pageSubtitle")
        root.addWidget(sub)
        root.addSpacing(32)

        # ── Card fichiers ──
        root.addWidget(self._section("DOCUMENTS D'ENTRÉE"))
        root.addSpacing(10)

        files_card = self._card()
        fl = QVBoxLayout(files_card)
        fl.setContentsMargins(20, 20, 20, 20)
        fl.setSpacing(10)

        fl.addWidget(self._lbl("Base de connaissances  (.txt, .jsonl, .md)"))
        self.dataset_picker = FilePickerRow(
            placeholder="Sélectionnez votre document / dataset…",
            icon="📊",
            filters="Textes (*.txt *.md *.jsonl *.json);;Tous (*)",
            dialog_title="Choisir un dataset",
            start_dir="storage/data/",
        )
        fl.addWidget(self.dataset_picker)

        fl.addSpacing(4)
        fl.addWidget(self._lbl("Dossier de sortie"))
        self.output_edit = QLineEdit()
        self.output_edit.setText("storage/models/")
        self.output_edit.setFixedHeight(40)
        fl.addWidget(self.output_edit)

        root.addWidget(files_card)
        root.addSpacing(20)

        # ── Card progression ──
        root.addWidget(self._section("COMPILATION"))
        root.addSpacing(10)

        prog_card = self._card()
        pl = QVBoxLayout(prog_card)
        pl.setContentsMargins(20, 18, 20, 18)
        pl.setSpacing(12)

        prog_header = QHBoxLayout()
        self.step_label = QLabel("En attente")
        self.step_label.setObjectName("pageSubtitle")
        prog_header.addWidget(self.step_label)
        prog_header.addStretch()
        self.loss_label = QLabel("")
        self.loss_label.setStyleSheet("color: #30D158; font-size: 13px; font-weight: 500; background: transparent;")
        prog_header.addWidget(self.loss_label)
        pl.addLayout(prog_header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(5)
        pl.addWidget(self.progress_bar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(120)
        self.log_view.setPlaceholderText("Les logs apparaîtront ici…")
        self.log_view.setStyleSheet(
            "background-color: rgba(0,0,0,0.30); border: none; border-radius: 10px; padding: 10px;"
            " font-family: 'SF Mono','Menlo','Consolas',monospace; font-size: 11px; color: #636366;"
        )
        pl.addWidget(self.log_view)
        root.addWidget(prog_card)
        root.addSpacing(20)

        # ── Boutons ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.train_btn = QPushButton("  Créer la compétence")
        self.train_btn.setObjectName("primaryBtn")
        self.train_btn.setFixedHeight(46)
        self.train_btn.clicked.connect(self._start)
        btn_row.addWidget(self.train_btn, stretch=1)

        self.stop_btn = QPushButton("Annuler")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setFixedHeight(46)
        self.stop_btn.setFixedWidth(110)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.stop_btn)

        root.addLayout(btn_row)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addSpacing(10)
        root.addWidget(self.result_label)
        root.addStretch()

    def _section(self, t):
        l = QLabel(t); l.setObjectName("sectionLabel"); return l
    def _lbl(self, t):
        l = QLabel(t); l.setStyleSheet("color: #636366; font-size: 12px; background: transparent;"); return l
    def _card(self):
        w = QWidget(); w.setObjectName("card"); return w
    def _log(self, msg):
        ts = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_view.appendPlainText(f"[{ts}]  {msg}")

    def _start(self):
        dataset_path = self.dataset_picker.get_path()
        output_dir = self.output_edit.text().strip() or "models/"

        if not dataset_path:
            self._log("Erreur : sélectionnez un document.")
            return

        self.trainer_node = TrainerNode(name="Skill-Compiler", output_dir=output_dir)
        self.trainer_node.set_inputs(
            DataPacket(DataType.TEXT, dataset_path),
            DataPacket(DataType.TEXT, ""), # Modèle de base ignoré
        )

        self.progress_bar.setValue(0)
        self.step_label.setText("Initialisation…")
        self.loss_label.setText("")
        self.result_label.setText("")
        self.log_view.clear()
        self._log(f"Document    : {os.path.basename(dataset_path)}")
        self._log(f"Sortie      : {output_dir}")

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._worker = TrainingWorker(self.trainer_node)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._log("Interrompu par l'utilisateur.")
            self._reset_btns()

    @pyqtSlot(int, int, float)
    def _on_progress(self, step, total, dummy_loss):
        self.progress_bar.setValue(int(step / total * 100))
        self.step_label.setText(f"Étape  {step} / {total}")
        self._log(f"Compilation en cours... ({step}/{total})")

    @pyqtSlot(str)
    def _on_finished(self, skill_path):
        self.progress_bar.setValue(100)
        self.step_label.setText("Terminé  ✓")
        self.result_label.setText(f"Compétence créée : {skill_path}")
        self.result_label.setStyleSheet("color: #30D158; font-size: 12px; background: transparent;")
        self._log(f"Sauvegardé : {skill_path}")
        self._reset_btns()

    @pyqtSlot(str)
    def _on_error(self, msg):
        self.step_label.setText("Erreur")
        self.result_label.setText(f"Erreur : {msg}")
        self.result_label.setStyleSheet("color: #FF453A; font-size: 12px; background: transparent;")
        self._log(f"ERREUR : {msg}")
        self._reset_btns()

    def _reset_btns(self):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)