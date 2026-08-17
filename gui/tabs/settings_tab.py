"""
settings_tab.py — Onglet Paramètres.
Architecture révisée : Import matériel corrigé, encapsulation OOP, délégation des Threads.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QLineEdit, QFrame, QButtonGroup, 
    QRadioButton, QCheckBox, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from gui.widgets import FilePickerRow
from core.session_manager import save_hf_token, load_hf_token, clear_hf_token
from core.utils.hardware import get_profile  # <-- CORRECTION DE L'IMPORT MATÉRIEL
from core.session_manager import save_vision_model, save_last_model
from gui.workers import ModelLoadWorker      # <-- UTILISATION DU WORKER CENTRALISÉ


class SettingsTab(QWidget):

    model_loaded        = pyqtSignal(str, str)
    vision_model_loaded = pyqtSignal(str, str)

    def __init__(self, llm_node, vision_node=None, parent=None):
        super().__init__(parent)
        self.llm_node    = llm_node
        self.vision_node = vision_node
        self._worker     = None
        self._build_ui()

    def _check_hardware_limits(self, model_path: str, mmproj_path: str) -> bool:
        """Vérifie si le modèle sélectionné risque de faire crasher la carte graphique (VRAM)."""
        hw = get_profile()
        vram_dispo = getattr(hw, "gpu_vram_gb", 0)

        # Si pas de GPU détecté, on laisse passer (fallback CPU assuré par llama-cpp)
        if vram_dispo <= 0:
            return True

        taille_modele = os.path.getsize(model_path) / (1024**3) if model_path and os.path.exists(model_path) else 0
        taille_proj = os.path.getsize(mmproj_path) / (1024**3) if mmproj_path and os.path.exists(mmproj_path) else 0

        # Marge pour le KV Cache
        besoin_estime = taille_modele + taille_proj + 1.5

        if besoin_estime > vram_dispo:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("⚠️ Risque de saturation VRAM")
            msg.setText("Ce modèle semble trop lourd pour votre carte graphique.")
            msg.setInformativeText(
                f"• Besoin estimé : ~<b>{besoin_estime:.1f} Go</b> (Modèle + Contexte)<br>"
                f"• VRAM disponible : <b>{vram_dispo:.1f} Go</b><br><br>"
                "Si vous continuez, l'application risque de crasher (Out of Memory) "
                "lors du traitement d'une image.<br><br>"
                "Forcer le chargement à vos risques et périls ?"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Abort)
            msg.setDefaultButton(QMessageBox.StandardButton.Abort)
            return msg.exec() == QMessageBox.StandardButton.Yes
            
        return True

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 32, 36, 32)
        root.setSpacing(0)

        # ── Titre ──
        title = QLabel("Paramètres")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addSpacing(4)
        sub = QLabel("Configuration de l'Architecture (Composite AI)")
        sub.setObjectName("pageSubtitle")
        root.addWidget(sub)
        root.addSpacing(32)

        # ── Card : Compte HuggingFace ──
        root.addWidget(self._section("COMPTE HUGGINGFACE"))
        root.addSpacing(10)
        hf_card = self._card()
        hl = QVBoxLayout(hf_card)
        hl.setContentsMargins(20, 20, 20, 20)
        hl.setSpacing(10)
        hl.addWidget(self._lbl("Token d'accès (requis pour Gemma, LLaMA officiel…)"))

        token_row = QHBoxLayout()
        token_row.setSpacing(8)
        self._token_input = QLineEdit()
        self._token_input.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        self._token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_input.setFixedHeight(42)
        saved = load_hf_token()
        if saved:
            self._token_input.setText(saved)
        token_row.addWidget(self._token_input)

        self._show_btn = QPushButton("👁")
        self._show_btn.setFixedSize(42, 42)
        self._show_btn.setStyleSheet("background:rgba(255,255,255,0.07);border:none;border-radius:10px;color:#8E8E93;")
        self._show_btn.setCheckable(True)
        self._show_btn.toggled.connect(self._toggle_visibility)
        token_row.addWidget(self._show_btn)
        hl.addLayout(token_row)

        token_btns = QHBoxLayout()
        token_btns.setSpacing(8)
        save_btn = QPushButton("Sauvegarder")
        save_btn.setObjectName("primaryBtn")
        save_btn.setFixedHeight(36)
        save_btn.clicked.connect(self._save_token)
        token_btns.addWidget(save_btn)

        clear_btn = QPushButton("Effacer")
        clear_btn.setObjectName("dangerBtn")
        clear_btn.setFixedHeight(36)
        clear_btn.clicked.connect(self._clear_token)
        token_btns.addWidget(clear_btn)
        token_btns.addStretch()
        hl.addLayout(token_btns)

        self._token_status = QLabel("")
        self._token_status.setStyleSheet("color:#636366;font-size:11px;background:transparent;")
        hl.addWidget(self._token_status)
        root.addWidget(hf_card)
        root.addSpacing(20)

        # ── Card : Modèle Principal ──
        root.addWidget(self._section("MOTEUR PRINCIPAL (Texte ou Multimodal)"))
        root.addSpacing(10)
        model_card = self._card()
        cl = QVBoxLayout(model_card)
        cl.setContentsMargins(20, 20, 20, 20)
        cl.setSpacing(10)

        cl.addWidget(self._lbl("Fichier modèle (.gguf)"))
        self.model_picker = FilePickerRow(
            placeholder="Sélectionnez un modèle GGUF (ex: Gemma, Mistral, Qwen-VL)…",
            icon="🧠",
            filters="Modèles GGUF (*.gguf);;Tous (*)",
            dialog_title="Choisir un modèle principal",
        )
        cl.addWidget(self.model_picker)

        cl.addSpacing(4)
        cl.addWidget(self._lbl("Projecteur multimodal principal (.mmproj — Optionnel)"))
        self.main_mmproj_picker = FilePickerRow(
            placeholder="Laissez vide si le modèle principal est purement textuel…",
            icon="🔮",
            filters="Projecteur mmproj (*.gguf);;Tous (*)",
            dialog_title="Choisir le projecteur principal",
        )
        cl.addWidget(self.main_mmproj_picker)

        cl.addSpacing(4)
        cl.addWidget(self._lbl("Équiper une Compétence (.skill — Optionnel)"))
        self.skill_picker = FilePickerRow(
            placeholder="Laissez vide pour le comportement par défaut…",
            icon="📚",
            filters="Compétences (*.skill);;Tous (*)",
            dialog_title="Choisir une Compétence",
        )
        cl.addWidget(self.skill_picker)
        root.addWidget(model_card)
        root.addSpacing(20)

        # ── Card : Orchestrateur Visuel ──
        root.addWidget(self._section("ORCHESTRATEUR VISUEL (Expert Dédié)"))
        root.addSpacing(10)
        vision_card = self._card()
        vl = QVBoxLayout(vision_card)
        vl.setContentsMargins(20, 20, 20, 20)
        vl.setSpacing(10)

        vl.addWidget(self._lbl("Modèle Vision Poids-Plume (.gguf)"))
        self.vision_model_picker = FilePickerRow(
            placeholder="Ex: moondream2-q8_0.gguf",
            icon="👁",
            filters="Modèles GGUF (*.gguf);;Tous (*)",
            dialog_title="Choisir le modèle Expert Vision",
        )
        vl.addWidget(self.vision_model_picker)

        vl.addSpacing(4)
        vl.addWidget(self._lbl("Projecteur Vision (.mmproj)"))
        self.vision_mmproj_picker = FilePickerRow(
            placeholder="Ex: moondream2-mmproj-f16.gguf",
            icon="🔮",
            filters="Projecteur mmproj (*.gguf);;Tous (*)",
            dialog_title="Choisir le projecteur Expert Vision",
        )
        vl.addWidget(self.vision_mmproj_picker)

        vl.addSpacing(10)
        self.checkbox_force_orch = QCheckBox(" Forcer l'Orchestration Composite (OCR + VLM + LLM)")
        self.checkbox_force_orch.setStyleSheet("color:#F2F2F7; font-size:12px; background:transparent;")
        vl.addWidget(self.checkbox_force_orch)

        root.addWidget(vision_card)
        root.addSpacing(20)

        # ── Card : Mode d'inférence ──
        root.addWidget(self._section("MODE D'INFÉRENCE & MATÉRIEL"))
        root.addSpacing(10)
        mode_card = self._card()
        ml = QVBoxLayout(mode_card)
        ml.setContentsMargins(20, 18, 20, 18)
        ml.setSpacing(12)

        hw = get_profile()
        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)

        self._mode_group = QButtonGroup(self)
        self._radio_standard = QRadioButton("  Standard")
        self._radio_standard.setChecked(True)
        self._radio_standard.setStyleSheet("QRadioButton{color:#F2F2F7;font-size:13px;background:transparent;}")
        self._mode_group.addButton(self._radio_standard, 0)
        mode_row.addWidget(self._radio_standard)

        self._radio_turbo = QRadioButton("  Turbo  ⚡")
        self._radio_turbo.setEnabled(hw.turbo_eligible)
        self._radio_turbo.setStyleSheet(f"QRadioButton{{color:{'#F2F2F7' if hw.turbo_eligible else '#3A3A3C'};font-size:13px;background:transparent;}}")
        self._mode_group.addButton(self._radio_turbo, 1)
        mode_row.addWidget(self._radio_turbo)
        mode_row.addStretch()
        ml.addLayout(mode_row)

        hw_info = QLabel(f"RAM : {hw.ram_total_gb} Go  ·  GPU : {hw.gpu_name} ({hw.gpu_vram_gb} Go)  ·  CPU : {hw.cpu_cores_phys} cœurs")
        hw_info.setStyleSheet(f"color:{'#30D158' if hw.turbo_eligible else '#FF9F0A'};font-size:11px;background:transparent;")
        ml.addWidget(hw_info)

        if hw.turbo_eligible:
            ml.addWidget(self._lbl("Modèle draft pour Turbo (.gguf)"))
            self.draft_picker = FilePickerRow(
                placeholder="Ex : tinyllama.gguf...",
                icon="⚡",
                filters="Modèles GGUF (*.gguf);;Tous (*)",
            )
            ml.addWidget(self.draft_picker)
        else:
            self.draft_picker = None

        root.addWidget(mode_card)
        root.addSpacing(20)

        # ── Card : Inférence (Contexte/Tokens) ──
        root.addWidget(self._section("PARAMÈTRES D'INFÉRENCE"))
        root.addSpacing(10)
        inf_card = self._card()
        il = QHBoxLayout(inf_card)
        il.setContentsMargins(20, 18, 20, 18)
        il.setSpacing(24)

        for label, attr, lo, hi, val, step in [
            ("Contexte (tokens)", "ctx_spin", 512, 32768, 4096, 512),
            ("Réponse max (tokens)", "max_tok_spin", 50, 4096, 1024, 50),
        ]:
            col = QVBoxLayout()
            col.addWidget(self._lbl(label))
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(val)
            spin.setSingleStep(step)
            setattr(self, attr, spin)
            col.addWidget(spin)
            il.addLayout(col)
        il.addStretch()
        root.addWidget(inf_card)
        root.addSpacing(28)

        # ── BOUTONS D'ACTION ──
        text_btn_row = QHBoxLayout()
        text_btn_row.setSpacing(12)

        self.btn_load_architecture = QPushButton("🚀 CHARGER L'ARCHITECTURE")
        self.btn_load_architecture.setObjectName("primaryBtn")
        self.btn_load_architecture.setFixedHeight(50)
        self.btn_load_architecture.setStyleSheet("""
            QPushButton#primaryBtn {
                font-size: 14px; font-weight: bold; border-radius: 8px;
                background-color: #0D6EFD; color: white;
            }
            QPushButton#primaryBtn:hover { background-color: #0B5ED7; }
        """)
        self.btn_load_architecture.clicked.connect(self._on_load_architecture_clicked)
        text_btn_row.addWidget(self.btn_load_architecture, stretch=1)

        self.unload_btn = QPushButton("Décharger")
        self.unload_btn.setObjectName("dangerBtn")
        self.unload_btn.setFixedHeight(50)
        self.unload_btn.setFixedWidth(120)
        self.unload_btn.clicked.connect(self._unload)
        text_btn_row.addWidget(self.unload_btn)

        root.addLayout(text_btn_row)
        root.addSpacing(24)

        # ── Statut Modèle Actif ──
        root.addWidget(self._section("STATUT DU SYSTÈME"))
        root.addSpacing(10)
        st_card = self._card()
        sl = QVBoxLayout(st_card)
        sl.setContentsMargins(20, 16, 20, 16)
        
        dot_row = QHBoxLayout()
        self._dot = QLabel()
        self._dot.setFixedSize(10, 10)
        self._dot_color("#3A3A3C")
        dot_row.addWidget(self._dot)
        
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

    # ── LOGIQUE MÉTIER ──

    def _on_load_architecture_clicked(self):
        vision_model = self.vision_model_picker.get_path()
        vision_proj  = self.vision_mmproj_picker.get_path()
        
        if vision_model and vision_proj:
            save_vision_model(vision_model, vision_proj)
            self.vision_model_loaded.emit(vision_model, vision_proj)
        else:
            save_vision_model("", "")
            self.vision_model_loaded.emit("", "")

        main_model = self.model_picker.get_path()
        main_proj  = self.main_mmproj_picker.get_path()
        skill_path = self.skill_picker.get_path()
        draft_path = self.draft_picker.get_path() if self.draft_picker else None
        turbo      = self._radio_turbo.isChecked()

        if not main_model:
            self._dot_color("#FF453A")
            self._st_text.setText("Sélectionnez au moins un Fichier Modèle Principal.")
            return

        # Appel interne au Garde-Fou matériel
        if not self._check_hardware_limits(main_model, main_proj):
            self._dot_color("#FF9F0A")
            self._st_text.setText("Chargement annulé par sécurité matérielle.")
            return

        self.btn_load_architecture.setEnabled(False)
        self.btn_load_architecture.setText("🚀 Chargement en cours...")
        self._dot_color("#FF9F0A")
        self._st_text.setText("Démarrage des moteurs IA...")
        self._st_detail.hide()

        # Démarrage du thread centralisé
        self._worker = ModelLoadWorker(self.llm_node, main_model, main_proj, skill_path, draft_path, turbo)
        self._worker.success.connect(self._on_load_success)
        self._worker.error.connect(self._on_load_error)
        self._worker.finished.connect(lambda: (
            self.btn_load_architecture.setEnabled(True),
            self.btn_load_architecture.setText("🚀 CHARGER L'ARCHITECTURE"),
        ))
        self._worker.start()

    def _on_load_success(self, model_path):
        skill = self.skill_picker.get_path()
        name = os.path.basename(model_path)
        skill_info = f"Skill : {os.path.basename(skill)}" if skill else "Aucun Skill actif"
        
        self._dot_color("#30D158")
        self._st_text.setText(f"Moteur Actif : {name}")
        self._st_detail.setText(
            f"Contexte : {self.ctx_spin.value()} tokens  ·  "
            f"Orchestration Forcée : {'Oui' if self.checkbox_force_orch.isChecked() else 'Non'}  ·  "
            f"{skill_info}"
        )
        self._st_detail.show()
        self.model_loaded.emit(model_path, skill or "")

    def _on_load_error(self, msg):
        self._dot_color("#FF453A")
        self._st_text.setText("Échec du chargement")
        self._st_detail.setText(msg)
        self._st_detail.show()

    def _unload(self):
        self.llm_node.unload_model()
        save_last_model("", "")
        self._dot_color("#FF453A")
        self._st_text.setText("Système hors ligne")
        self._st_detail.hide()
        self.model_loaded.emit("", "")

    def _toggle_visibility(self, checked: bool):
        self._token_input.setEchoMode(QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password)

    def _save_token(self):
        token = self._token_input.text().strip()
        if not token:
            self._token_status.setText("⚠ Token vide")
            self._token_status.setStyleSheet("color:#FF9F0A;")
            return
        save_hf_token(token)
        self._token_status.setText("✓ Token sauvegardé")
        self._token_status.setStyleSheet("color:#30D158;")

    def _clear_token(self):
        clear_hf_token()
        self._token_input.clear()
        self._token_status.setText("Token supprimé")
        self._token_status.setStyleSheet("color:#636366;")

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionLabel")
        return lbl

    def _lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#636366;font-size:12px;background:transparent;")
        return lbl

    def _card(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("card")
        return widget

    def _dot_color(self, color: str):
        self._dot.setStyleSheet(f"background-color:{color}; border-radius:5px; min-width:10px; max-width:10px; min-height:10px; max-height:10px;")