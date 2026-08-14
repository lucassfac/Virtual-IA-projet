"""
settings_tab.py — Onglet Paramètres.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QLineEdit, QFrame, QButtonGroup, QRadioButton,
)
from PyQt6.QtCore import pyqtSignal, Qt
from gui.widgets import FilePickerRow
from core.model_manager import save_hf_token, load_hf_token, clear_hf_token
from core.utils.hardware_check import get_hw

def _open_url(url: str) -> None:
    import subprocess, sys, os
    try:
        if sys.platform == "win32":
            os.startfile(url)
        elif "microsoft" in open("/proc/version").read().lower():
            subprocess.Popen(["explorer.exe", url])
        else:
            subprocess.Popen(["xdg-open", url])
    except Exception:
        pass

class SettingsTab(QWidget):

    model_loaded        = pyqtSignal(str, str)   # (model_path, skill_path)
    vision_model_loaded = pyqtSignal(str, str)

    def __init__(self, llm_node, vision_node=None, parent=None):
        super().__init__(parent)
        self.llm_node    = llm_node
        self.vision_node = vision_node
        self._worker     = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 32, 36, 32)
        root.setSpacing(0)

        # ── Titre ──
        title = QLabel("Paramètres")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addSpacing(4)
        sub = QLabel("Configuration du modèle et de l'inférence")
        sub.setObjectName("pageSubtitle")
        root.addWidget(sub)
        root.addSpacing(32)

        # ── Card : Token HuggingFace ──
        root.addWidget(self._section("COMPTE HUGGINGFACE"))
        root.addSpacing(10)

        hf_card = self._card()
        hl = QVBoxLayout(hf_card)
        hl.setContentsMargins(20, 20, 20, 20)
        hl.setSpacing(10)
        hl.addWidget(self._lbl("Token d'accès  (requis pour Gemma, LLaMA officiel…)"))

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
        self._show_btn.setStyleSheet(
            "background:rgba(255,255,255,0.07);border:none;"
            "border-radius:10px;font-size:15px;color:#8E8E93;"
        )
        self._show_btn.setCheckable(True)
        self._show_btn.toggled.connect(self._toggle_visibility)
        token_row.addWidget(self._show_btn)
        hl.addLayout(token_row)

        token_btns = QHBoxLayout()
        token_btns.setSpacing(8)

        save_btn = QPushButton("Sauvegarder")
        save_btn.setObjectName("primaryBtn")
        save_btn.setFixedHeight(36)
        save_btn.setStyleSheet("QPushButton#primaryBtn{font-size:12px;min-height:36px;border-radius:9px;}")
        save_btn.clicked.connect(self._save_token)
        token_btns.addWidget(save_btn)

        clear_btn = QPushButton("Effacer")
        clear_btn.setObjectName("dangerBtn")
        clear_btn.setFixedHeight(36)
        clear_btn.setStyleSheet("QPushButton#dangerBtn{font-size:12px;min-height:36px;border-radius:9px;}")
        clear_btn.clicked.connect(self._clear_token)
        token_btns.addWidget(clear_btn)
        token_btns.addStretch()
        hl.addLayout(token_btns)

        self._token_status = QLabel("")
        self._token_status.setStyleSheet("color:#636366;font-size:11px;background:transparent;")
        hl.addWidget(self._token_status)

        help_btn = QPushButton("Créer un token sur huggingface.co →")
        help_btn.setStyleSheet("background:transparent;border:none;color:#409CFF;font-size:11px;text-align:left;padding:0;")
        help_btn.setCursor(__import__('PyQt6.QtGui', fromlist=['QCursor']).QCursor(__import__('PyQt6.QtCore', fromlist=['Qt']).Qt.CursorShape.PointingHandCursor))
        help_btn.clicked.connect(lambda: _open_url("https://huggingface.co/settings/tokens"))
        hl.addWidget(help_btn)

        root.addWidget(hf_card)
        root.addSpacing(20)

        # ── Séparateur ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:rgba(255,255,255,0.06);max-height:1px;border:none;")
        root.addWidget(sep)
        root.addSpacing(20)

        # ── Card : Modèle ──
        root.addWidget(self._section("MODÈLE ET COMPÉTENCE"))
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
            start_dir="storage/models/",
        )
        cl.addWidget(self.model_picker)

        cl.addSpacing(4)
        cl.addWidget(self._lbl("Équiper une Compétence  (.skill — optionnel)"))
        self.skill_picker = FilePickerRow(
            placeholder="Laissez vide pour le comportement par défaut…",
            icon="📚",
            filters="Compétences Neural Forge (*.skill);;Tous (*)",
            dialog_title="Choisir une Compétence",
            start_dir="storage/models/",
        )
        cl.addWidget(self.skill_picker)
        root.addWidget(model_card)
        root.addSpacing(20)

        # ── Card : Mode d'inférence ──
        root.addWidget(self._section("MODE D'INFÉRENCE"))
        root.addSpacing(10)

        mode_card = self._card()
        ml = QVBoxLayout(mode_card)
        ml.setContentsMargins(20, 18, 20, 18)
        ml.setSpacing(12)

        hw = get_hw()
        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)

        self._mode_group = QButtonGroup(self)
        self._radio_standard = QRadioButton("  Standard")
        self._radio_standard.setChecked(True)
        self._radio_standard.setStyleSheet(
            "QRadioButton{color:#F2F2F7;font-size:13px;background:transparent;}"
            "QRadioButton::indicator{width:16px;height:16px;}"
            "QRadioButton::indicator:checked{background:#0A84FF;border-radius:8px;border:2px solid #0A84FF;}"
            "QRadioButton::indicator:unchecked{background:transparent;border-radius:8px;border:2px solid #3A3A3C;}"
        )
        self._mode_group.addButton(self._radio_standard, 0)
        mode_row.addWidget(self._radio_standard)

        self._radio_turbo = QRadioButton("  Turbo  ⚡")
        self._radio_turbo.setEnabled(hw.turbo_eligible)
        self._radio_turbo.setToolTip(hw.turbo_reason)
        turbo_color = "#F2F2F7" if hw.turbo_eligible else "#3A3A3C"
        self._radio_turbo.setStyleSheet(
            f"QRadioButton{{color:{turbo_color};font-size:13px;background:transparent;}}"
            "QRadioButton::indicator{width:16px;height:16px;}"
            "QRadioButton::indicator:checked{background:#FF9F0A;border-radius:8px;border:2px solid #FF9F0A;}"
            "QRadioButton::indicator:unchecked{background:transparent;border-radius:8px;border:2px solid #3A3A3C;}"
        )
        self._mode_group.addButton(self._radio_turbo, 1)
        mode_row.addWidget(self._radio_turbo)
        mode_row.addStretch()
        ml.addLayout(mode_row)

        ram_color = "#30D158" if hw.turbo_eligible else "#FF9F0A"
        hw_info = QLabel(
            f"RAM : {hw.ram_total_gb} Go  ·  "
            f"GPU : {hw.gpu_name}  ({hw.gpu_vram_gb} Go VRAM)  ·  "
            f"CPU : {hw.cpu_cores_phys} cœurs physiques"
        )
        hw_info.setStyleSheet(f"color:{ram_color};font-size:11px;background:transparent;")
        ml.addWidget(hw_info)

        if hw.turbo_eligible:
            ml.addWidget(self._lbl("Modèle draft pour Turbo  (.gguf — modèle plus petit)"))
            self.draft_picker = FilePickerRow(
                placeholder="Ex : tinyllama.gguf, gemma-3-1b.gguf…",
                icon="⚡",
                filters="Modèles GGUF (*.gguf);;Tous (*)",
                dialog_title="Choisir un modèle draft (petit)",
                start_dir="storage/models/",
            )
            ml.addWidget(self.draft_picker)
        else:
            self.draft_picker = None
            locked_lbl = QLabel(f"Turbo verrouillé — {12 - hw.ram_total_gb:.1f} Go RAM supplémentaires requis.")
            locked_lbl.setStyleSheet("color:#3A3A3C;font-size:11px;background:transparent;")
            ml.addWidget(locked_lbl)

        root.addWidget(mode_card)
        root.addSpacing(20)

        # ── Card : Vision (LLaVA) ──
        root.addWidget(self._section("VISION  (MULTIMODAL)"))
        root.addSpacing(10)

        vision_card = self._card()
        vl = QVBoxLayout(vision_card)
        vl.setContentsMargins(20, 20, 20, 20)
        vl.setSpacing(10)

        vl.addWidget(self._lbl("Modèle LLaVA  (.gguf)"))
        self.llava_picker = FilePickerRow(
            placeholder="Sélectionnez un modèle LLaVA…",
            icon="👁",
            filters="Modèles GGUF (*.gguf);;Tous (*)",
            dialog_title="Choisir un modèle LLaVA",
            start_dir="storage/models/",
        )
        vl.addWidget(self.llava_picker)

        vl.addSpacing(4)
        vl.addWidget(self._lbl("Projecteur multimodal  (mmproj .gguf)"))
        self.mmproj_picker = FilePickerRow(
            placeholder="Sélectionnez le fichier mmproj…",
            icon="🔮",
            filters="Projecteur mmproj (*.gguf);;Tous (*)",
            dialog_title="Choisir le fichier mmproj",
            start_dir="storage/models/",
        )
        vl.addWidget(self.mmproj_picker)

        self._load_vision_btn = QPushButton("  Charger le modèle Vision")
        self._load_vision_btn.setObjectName("primaryBtn")
        self._load_vision_btn.setFixedHeight(40)
        self._load_vision_btn.setStyleSheet("QPushButton#primaryBtn{font-size:13px;min-height:40px;border-radius:10px;}")
        self._load_vision_btn.clicked.connect(self._load_vision)
        vl.addWidget(self._load_vision_btn)

        self._vision_status = QLabel("")
        self._vision_status.setObjectName("sectionLabel")
        self._vision_status.setWordWrap(True)
        vl.addWidget(self._vision_status)

        root.addWidget(vision_card)
        root.addSpacing(20)

        # ── Card : Inférence ──
        root.addWidget(self._section("INFÉRENCE"))
        root.addSpacing(10)

        inf_card = self._card()
        il = QHBoxLayout(inf_card)
        il.setContentsMargins(20, 18, 20, 18)
        il.setSpacing(24)

        for label, attr, lo, hi, val, step, tip in [
            ("Contexte (tokens)", "ctx_spin", 512, 32768, 4096, 512, "Mémoire à court terme du modèle"),
            ("Réponse max (tokens)", "max_tok_spin", 50, 4096, 1024, 50, "Longueur maximale de la réponse"),
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

        # ── Boutons ──
        self.load_btn = QPushButton("  Charger le modèle")
        self.load_btn.setObjectName("primaryBtn")
        self.load_btn.setFixedHeight(46)
        self.load_btn.clicked.connect(self._load)
        root.addWidget(self.load_btn)
        root.addSpacing(24)

        # ── Statut modèle actif ──
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

    def _toggle_visibility(self, checked: bool):
        self._token_input.setEchoMode(QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password)

    def _save_token(self):
        token = self._token_input.text().strip()
        if not token:
            self._token_status.setText("⚠ Token vide")
            self._token_status.setStyleSheet("color:#FF9F0A;font-size:11px;background:transparent;")
            return
        save_hf_token(token)
        self._token_status.setText("✓ Token sauvegardé")
        self._token_status.setStyleSheet("color:#30D158;font-size:11px;background:transparent;")

    def _clear_token(self):
        clear_hf_token()
        self._token_input.clear()
        self._token_status.setText("Token supprimé")
        self._token_status.setStyleSheet("color:#636366;font-size:11px;background:transparent;")

    def _load(self):
        model_path = self.model_picker.get_path()
        if not model_path:
            self._dot_color("#FF453A")
            self._st_text.setText("Sélectionnez un fichier modèle")
            return
        skill_path  = self.skill_picker.get_path() or None
        turbo      = self._radio_turbo.isChecked()
        draft_path = (self.draft_picker.get_path() or None) if self.draft_picker else None

        self.load_btn.setEnabled(False)
        mode_label = "TURBO ⚡" if turbo else "Standard"
        self.load_btn.setText(f"  Chargement {mode_label}…")
        self._dot_color("#FF9F0A")
        self._st_text.setText(f"Chargement {mode_label}…")
        self._st_detail.hide()

        from PyQt6.QtCore import QThread
        from PyQt6.QtCore import pyqtSignal as pS

        class TurboLoadWorker(QThread):
            success  = pS(str)
            error    = pS(str)
            finished = pS()
            def __init__(self_, node, mp, sp, dp, t):
                super().__init__()
                self_.node=node; self_.mp=mp; self_.sp=sp
                self_.dp=dp; self_.t=t
            def run(self_):
                try:
                    self_.node.load_model(
                        self_.mp, skill_path=self_.sp,
                        draft_model_path=self_.dp, turbo=self_.t
                    )
                    self_.success.emit(self_.mp)
                except Exception as e:
                    self_.error.emit(str(e))
                finally:
                    self_.finished.emit()

        self._worker = TurboLoadWorker(self.llm_node, model_path, skill_path, draft_path, turbo)
        self._worker.success.connect(self._ok)
        self._worker.error.connect(self._err)
        self._worker.finished.connect(lambda: (
            self.load_btn.setEnabled(True),
            self.load_btn.setText("  Charger le modèle"),
        ))
        self._worker.start()

    def _load_vision(self):
        """Valide et sauvegarde les chemins Vision pour l'Orchestrateur sans saturer la VRAM."""
        llava_path  = self.llava_picker.get_path()
        mmproj_path = self.mmproj_picker.get_path()

        if not llava_path or not mmproj_path:
            self._vision_status.setText("⚠ Sélectionnez le modèle et le mmproj.")
            self._vision_status.setStyleSheet("color:#FF9F0A;font-size:11px;background:transparent;")
            return

        import os
        if not os.path.exists(llava_path) or not os.path.exists(mmproj_path):
            self._vision_status.setText("⚠ Fichier introuvable sur le disque.")
            self._vision_status.setStyleSheet("color:#FF453A;font-size:11px;background:transparent;")
            return

        # VRAIE MAGIE : Plus aucune trace de QThread ou de load_model() ici !
        from core.session_manager import save_vision_model
        save_vision_model(llava_path, mmproj_path)

        name = os.path.basename(llava_path)
        self._vision_status.setText(f"✓ Configuration prête pour l'Orchestrateur : {name}")
        self._vision_status.setStyleSheet("color:#30D158;font-size:11px;background:transparent;")
        self.vision_model_loaded.emit(llava_path, mmproj_path)

    def _ok(self, model_path):
        skill = self.skill_picker.get_path()
        name = os.path.basename(model_path)
        skill_info = f"Skill : {os.path.basename(skill)}" if skill else "Aucun Skill"
        self._dot_color("#30D158")
        self._st_text.setText(name)
        self._st_detail.setText(f"Contexte : {self.ctx_spin.value()} t  ·  Max : {self.max_tok_spin.value()} t  ·  {skill_info}")
        self._st_detail.show()
        self.model_loaded.emit(model_path, skill or "")

    def _err(self, msg):
        self._dot_color("#FF453A")
        self._st_text.setText("Échec du chargement")
        self._st_detail.setText(msg)
        self._st_detail.show()

    def _section(self, t):
        l = QLabel(t); l.setObjectName("sectionLabel"); return l
    def _lbl(self, t):
        l = QLabel(t); l.setStyleSheet("color:#636366;font-size:12px;background:transparent;"); return l
    def _card(self):
        w = QWidget(); w.setObjectName("card"); return w
    def _dot_color(self, c):
        self._dot.setStyleSheet(f"background-color:{c};border-radius:5px;min-width:10px;max-width:10px;min-height:10px;max-height:10px;")