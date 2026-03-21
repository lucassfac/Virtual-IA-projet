"""
settings_tab.py — Onglet Paramètres redesigné.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox,
)
from PyQt6.QtCore import pyqtSignal
from gui.workers import ModelLoadWorker
from gui.widgets import FilePickerRow


class SettingsTab(QWidget):

    model_loaded = pyqtSignal(str, str)

    def __init__(self, llm_node, parent=None):
        super().__init__(parent)
        self.llm_node = llm_node
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 32, 36, 32)
        root.setSpacing(0)

        title = QLabel("Paramètres")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addSpacing(4)
        sub = QLabel("Configuration du modèle et de l'inférence")
        sub.setObjectName("pageSubtitle")
        root.addWidget(sub)
        root.addSpacing(32)

        root.addWidget(self._section("MODÈLE"))
        root.addSpacing(10)

        model_card = self._card()
        cl = QVBoxLayout(model_card)
        cl.setContentsMargins(20, 20, 20, 20)
        cl.setSpacing(10)

        cl.addWidget(self._lbl("Fichier modèle  (.gguf)"))
        self.model_picker = FilePickerRow(
            placeholder="Sélectionnez un modèle GGUF…",
            icon="🧠",
            filters="Modèles GGUF (*.gguf);;Tous (*)",
            dialog_title="Choisir un modèle",
            start_dir="models/",
        )
        cl.addWidget(self.model_picker)
        cl.addSpacing(4)
        cl.addWidget(self._lbl("Adaptateur LoRA  (optionnel)"))
        self.lora_picker = FilePickerRow(
            placeholder="Laissez vide pour le modèle de base…",
            icon="🧬",
            filters="Adaptateurs LoRA (*.lora *.bin);;Tous (*)",
            dialog_title="Choisir un adaptateur LoRA",
            start_dir="models/",
        )
        cl.addWidget(self.lora_picker)
        root.addWidget(model_card)
        root.addSpacing(20)

        root.addWidget(self._section("INFÉRENCE"))
        root.addSpacing(10)

        inf_card = self._card()
        il = QHBoxLayout(inf_card)
        il.setContentsMargins(20, 18, 20, 18)
        il.setSpacing(24)

        for label, attr, lo, hi, val, step, tip in [
            ("Contexte (tokens)", "ctx_spin", 512, 32768, 2048, 512,
             "Mémoire à court terme du modèle"),
            ("Réponse max (tokens)", "max_tok_spin", 50, 4096, 200, 50,
             "Longueur maximale de la réponse générée"),
        ]:
            col = QVBoxLayout()
            col.setSpacing(6)
            col.addWidget(self._lbl(label))
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(val)
            spin.setSingleStep(step)
            spin.setToolTip(tip)
            setattr(self, attr, spin)
            col.addWidget(spin)
            il.addLayout(col)
        il.addStretch()
        root.addWidget(inf_card)
        root.addSpacing(28)

        self.load_btn = QPushButton("  Charger le modèle")
        self.load_btn.setObjectName("primaryBtn")
        self.load_btn.setFixedHeight(46)
        self.load_btn.clicked.connect(self._load)
        root.addWidget(self.load_btn)
        root.addSpacing(24)

        root.addWidget(self._section("MODÈLE ACTIF"))
        root.addSpacing(10)

        st_card = self._card()
        sl = QVBoxLayout(st_card)
        sl.setContentsMargins(20, 16, 20, 16)
        sl.setSpacing(8)

        dot_row = QHBoxLayout()
        self._dot = QLabel()
        self._dot.setFixedSize(10, 10)
        self._dot_color("#3A3A3C")
        dot_row.addWidget(self._dot)
        dot_row.addSpacing(8)
        self._st_text = QLabel("Aucun modèle chargé")
        self._st_text.setObjectName("pageSubtitle")
        dot_row.addWidget(self._st_text)
        dot_row.addStretch()
        sl.addLayout(dot_row)

        self._st_detail = QLabel("")
        self._st_detail.setObjectName("sectionLabel")
        self._st_detail.setWordWrap(True)
        sl.addWidget(self._st_detail)
        self._st_detail.hide()
        root.addWidget(st_card)
        root.addStretch()

    def _section(self, t):
        l = QLabel(t); l.setObjectName("sectionLabel"); return l

    def _lbl(self, t):
        l = QLabel(t)
        l.setStyleSheet("color: #636366; font-size: 12px; background: transparent;")
        return l

    def _card(self):
        w = QWidget(); w.setObjectName("card"); return w

    def _dot_color(self, c):
        self._dot.setStyleSheet(
            f"background-color:{c}; border-radius:5px;"
            " min-width:10px; max-width:10px;"
            " min-height:10px; max-height:10px;"
        )

    def _load(self):
        model_path = self.model_picker.get_path()
        if not model_path:
            self._dot_color("#FF453A")
            self._st_text.setText("Sélectionnez un fichier modèle")
            return
        lora_path = self.lora_picker.get_path() or None
        self.load_btn.setEnabled(False)
        self.load_btn.setText("  Chargement…")
        self._dot_color("#FF9F0A")
        self._st_text.setText("Chargement en cours…")
        self._st_detail.hide()

        self._worker = ModelLoadWorker(self.llm_node, model_path, lora_path)
        self._worker.success.connect(self._ok)
        self._worker.error.connect(self._err)
        self._worker.finished.connect(lambda: (
            self.load_btn.setEnabled(True),
            self.load_btn.setText("  Charger le modèle"),
        ))
        self._worker.start()

    def _ok(self, model_path):
        lora = self.lora_picker.get_path()
        name = os.path.basename(model_path)
        lora_info = f"LoRA : {os.path.basename(lora)}" if lora else "Sans LoRA"
        self._dot_color("#30D158")
        self._st_text.setText(name)
        self._st_detail.setText(
            f"Contexte : {self.ctx_spin.value()} t  ·  "
            f"Max : {self.max_tok_spin.value()} t  ·  {lora_info}"
        )
        self._st_detail.show()
        self.model_loaded.emit(model_path, lora or "")

    def _err(self, msg):
        self._dot_color("#FF453A")
        self._st_text.setText("Échec du chargement")
        self._st_detail.setText(msg)
        self._st_detail.show()
