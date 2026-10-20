"""
training_tab.py — Onglet d'Entraînement Avancé (SFT & Model Soup).
"""

import os
import shutil
from PyQt6.QtCore import QDateTime, Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.nodes.trainer_node import TrainerNode
from core.types import DataPacket, DataType
from gui.widgets import FilePickerRow
from gui.workers import TrainingWorker


class TrainingTab(QWidget):
    """Interface graphique unifiée pour l'entraînement LoRA (SFT) et la fusion (Model Soup)."""

    # Boutique de modèles complets Hugging Face adaptés aux configurations locales
    SFT_CATALOG = {
        "🌟 Qwen 2.5 - 1.5B (Recommandé — ~3.0 Go)": "Qwen/Qwen2.5-1.5B-Instruct",
        "⚡ TinyLlama - 1.1B (Ultra Léger — ~2.2 Go)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "🧠 Qwen 2.5 - 3B (Performant — ~6.0 Go)": "Qwen/Qwen2.5-3B-Instruct",
        "🤖 DeepSeek R1 Distill - 1.5B (~3.5 Go)": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    }

    MODELS_ROOT = "storage/models"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trainer_node = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 32, 36, 32)
        root.setSpacing(0)

        # ── En-tête ──
        header_row = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Forge Neurale (Entraînement)")
        title.setObjectName("pageTitle")
        title_box.addWidget(title)
        title_box.addSpacing(4)
        sub = QLabel("Entraînez des adaptateurs (LoRA) et fusionnez-les (Model Soup)")
        sub.setObjectName("pageSubtitle")
        title_box.addWidget(sub)
        header_row.addLayout(title_box)
        header_row.addStretch()

        # Bouton de maintenance du disque (Purge base_hf)
        self.purge_btn = QPushButton("🗑️ Purger le cache HF")
        self.purge_btn.setFixedHeight(32)
        self.purge_btn.setToolTip("Supprime les modèles complets téléchargés dans storage/models/base_hf/")
        self.purge_btn.setStyleSheet(
            "background: #2C2C2E; color: #FF453A; border-radius: 6px; padding: 0 12px; font-size: 11px;"
        )
        self.purge_btn.clicked.connect(self._purge_hf_cache)
        header_row.addWidget(self.purge_btn)

        root.addLayout(header_row)
        root.addSpacing(24)

        # ── Onglets Internes (SFT vs SOUP) ──
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; background: #151517; }
            QTabBar::tab { background: #0D0D0F; color: #8E8E93; padding: 10px 20px; border-top-left-radius: 8px; border-top-right-radius: 8px; }
            QTabBar::tab:selected { background: #151517; color: #F2F2F7; font-weight: bold; border-bottom: 2px solid #0A84FF; }
        """)

        self.sft_tab = QWidget()
        self._build_sft_tab()
        self.tabs.addTab(self.sft_tab, "1. Création d'Adaptateur (SFT)")

        self.soup_tab = QWidget()
        self._build_soup_tab()
        self.tabs.addTab(self.soup_tab, "2. Fusion Model Soup")

        root.addWidget(self.tabs)
        root.addSpacing(20)

        # ── Card Progression Globale ──
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
        self.log_view.setPlaceholderText("Les logs du compilateur apparaîtront ici…")
        self.log_view.setStyleSheet(
            "background-color: rgba(0,0,0,0.30); border: none; border-radius: 10px; padding: 10px;"
            " font-family: 'SF Mono','Menlo','Consolas',monospace; font-size: 11px; color: #636366;"
        )
        pl.addWidget(self.log_view)
        root.addWidget(prog_card)
        root.addSpacing(10)

        self.stop_btn = QPushButton("Interrompre le processus")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        root.addWidget(self.stop_btn)

    def _build_sft_tab(self):
        layout = QVBoxLayout(self.sft_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # 1. Boutique / Sélecteur de modèle complet Hugging Face
        layout.addWidget(self._lbl("Modèle d'entraînement (Catalogue Hugging Face ou ID personnalisé)"))
        self.hf_model_combo = QComboBox()
        self.hf_model_combo.setEditable(True)
        for label, repo_id in self.SFT_CATALOG.items():
            self.hf_model_combo.addItem(label, repo_id)
        self.hf_model_combo.setFixedHeight(38)
        layout.addWidget(self.hf_model_combo)

        info_lbl = QLabel(
            "💡 L'entraînement nécessite un modèle non-compressé (stocké dans 'storage/models/base_hf/').\n"
            "Les adaptateurs générés (.safetensors) iront dans 'storage/models/adapters/'."
        )
        info_lbl.setStyleSheet("color: #8E8E93; font-size: 11px; font-style: italic; background: transparent;")
        layout.addWidget(info_lbl)

        layout.addSpacing(4)

        # 2. Dataset
        layout.addWidget(self._lbl("Dataset d'entraînement (.jsonl)"))
        self.dataset_picker = FilePickerRow(
            "Sélectionnez le dataset...", "📊", "Datasets (*.jsonl)", start_dir="storage/data/"
        )
        layout.addWidget(self.dataset_picker)

        # 3. Hyperparamètres
        hp_layout = QHBoxLayout()
        self.lr_combo = QComboBox()
        self.lr_combo.addItems([
            "Taux d'apprentissage: Rapide (2e-4)",
            "Taux d'apprentissage: Normal (5e-5)",
            "Taux d'apprentissage: Précis (1e-5)",
        ])
        self.lr_combo.setFixedHeight(35)

        self.epochs_combo = QComboBox()
        self.epochs_combo.addItems(["1 Époque", "3 Époques", "5 Époques"])
        self.epochs_combo.setFixedHeight(35)

        hp_layout.addWidget(self.lr_combo)
        hp_layout.addWidget(self.epochs_combo)
        layout.addLayout(hp_layout)

        self.train_btn = QPushButton("Lancer l'entraînement SFT")
        self.train_btn.setObjectName("primaryBtn")
        self.train_btn.setFixedHeight(46)
        self.train_btn.clicked.connect(self._start_sft)
        layout.addWidget(self.train_btn)
        layout.addStretch()

    def _build_soup_tab(self):
        layout = QVBoxLayout(self.soup_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # 1. Nom du fichier de sortie (plus intuitif qu'un sélecteur .gguf inutile ici)
        layout.addWidget(self._lbl("Nom de l'adaptateur fusionné (sera stocké dans 'storage/models/soups/')"))
        self.soup_name_edit = QLineEdit()
        self.soup_name_edit.setPlaceholderText("Ex: expert_juridique_v1")
        self.soup_name_edit.setText("neural_soup_v1")
        self.soup_name_edit.setFixedHeight(38)
        layout.addWidget(self.soup_name_edit)

        # 2. Liste des adaptateurs
        layout.addWidget(self._lbl("Adaptateurs LoRA à fusionner (Minimum 2 requis)"))
        self.adapter_list = QListWidget()
        self.adapter_list.setFixedHeight(85)
        self.adapter_list.setStyleSheet(
            "background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);"
            " border-radius: 5px; color: #F2F2F7; padding: 5px;"
        )
        layout.addWidget(self.adapter_list)

        list_btns = QHBoxLayout()
        add_adapter_btn = QPushButton("➕ Ajouter des adaptateurs (.safetensors)")
        add_adapter_btn.setFixedHeight(32)
        add_adapter_btn.setStyleSheet("background: #2C2C2E; color: #AEAEB2; border-radius: 5px;")
        add_adapter_btn.clicked.connect(self._add_adapters)

        clear_adapter_btn = QPushButton("Vider la liste")
        clear_adapter_btn.setFixedHeight(32)
        clear_adapter_btn.setFixedWidth(110)
        clear_adapter_btn.setStyleSheet("background: #1C1C1E; color: #8E8E93; border-radius: 5px;")
        clear_adapter_btn.clicked.connect(self.adapter_list.clear)

        list_btns.addWidget(add_adapter_btn, stretch=1)
        list_btns.addWidget(clear_adapter_btn)
        layout.addLayout(list_btns)

        self.soup_btn = QPushButton("Lancer le Model Soup")
        self.soup_btn.setObjectName("primaryBtn")
        self.soup_btn.setFixedHeight(46)
        self.soup_btn.setStyleSheet("background: #BF5AF2; color: white; font-weight: bold; border-radius: 8px;")
        self.soup_btn.clicked.connect(self._start_soup)
        layout.addWidget(self.soup_btn)
        layout.addStretch()

    # --- Utilitaires UI ---
    def _lbl(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #636366; font-size: 12px; background: transparent; font-weight: bold;")
        return label

    def _card(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("card")
        return widget

    def _log(self, msg: str) -> None:
        ts = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_view.appendPlainText(f"[{ts}]  {msg}")

    # --- Logique d'actions ---
    def _add_adapters(self):
        start_folder = os.path.join(self.MODELS_ROOT, "adapters")
        os.makedirs(start_folder, exist_ok=True)
        files, _ = QFileDialog.getOpenFileNames(
            self, "Sélectionner des adaptateurs", start_folder, "Adaptateurs (*.safetensors *.bin)"
        )
        for file_path in files:
            if not self.adapter_list.findItems(file_path, Qt.MatchFlag.MatchExactly):
                self.adapter_list.addItem(file_path)

    def _purge_hf_cache(self):
        base_hf_path = os.path.join(self.MODELS_ROOT, "base_hf")
        if not os.path.exists(base_hf_path) or not os.listdir(base_hf_path):
            self._log("Info : Le dossier 'storage/models/base_hf/' est déjà vide.")
            return

        reply = QMessageBox.question(
            self,
            "Purger les modèles d'entraînement",
            "Voulez-vous supprimer les modèles complets téléchargés dans 'storage/models/base_hf/' ?\n\n"
            "Vos adaptateurs (.safetensors) et vos modèles de chat (.gguf) seront conservés.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            shutil.rmtree(base_hf_path, ignore_errors=True)
            os.makedirs(base_hf_path, exist_ok=True)
            self._log("🗑️ Cache 'storage/models/base_hf/' purgé avec succès.")

    def _start_sft(self):
        # Si l'utilisateur a choisi un élément de la liste, on prend son ID Hugging Face (currentData)
        # S'il a tapé un ID manuellement, on prend le texte (currentText)
        idx = self.hf_model_combo.currentIndex()
        if idx >= 0 and self.hf_model_combo.itemText(idx) == self.hf_model_combo.currentText():
            base_repo = self.hf_model_combo.itemData(idx)
        else:
            base_repo = self.hf_model_combo.currentText().strip()

        dataset_path = self.dataset_picker.get_path()

        if not base_repo or not dataset_path:
            self._log("Erreur : Sélectionnez un modèle dans la liste et un fichier dataset (.jsonl).")
            return

        self._log("--- DÉMARRAGE ENTRAÎNEMENT SFT ---")
        self._log(f"Modèle HF : {base_repo}")
        self._log(f"Dataset   : {os.path.basename(dataset_path)}")
        self._lock_ui()

        config = {
            "mode": "sft",
            "base_model": base_repo,
            "dataset": dataset_path,
            "lr": self.lr_combo.currentText(),
            "epochs": self.epochs_combo.currentText(),
        }

        self._run_worker("SFT-Compiler", config)

    def _start_soup(self):
        soup_name = self.soup_name_edit.text().strip() or "neural_soup"
        adapters = [self.adapter_list.item(i).text() for i in range(self.adapter_list.count())]

        if len(adapters) < 2:
            self._log("Erreur : Ajoutez au moins DEUX adaptateurs (.safetensors) pour lancer une fusion.")
            return

        self._log("--- DÉMARRAGE FUSION MODEL SOUP ---")
        self._log(f"Sortie     : {soup_name}_soup.safetensors")
        self._log(f"Ingrédients: {len(adapters)} adaptateurs sélectionnés")
        self._lock_ui()

        config = {
            "mode": "soup",
            "soup_name": soup_name,
            "adapters": adapters,
        }

        self._run_worker("SOUP-Compiler", config)

    def _run_worker(self, node_name: str, config: dict):
        """Factorisation DRY pour éviter de dupliquer le lancement du QThread."""
        self.trainer_node = TrainerNode(name=node_name, output_dir=self.MODELS_ROOT)
        self.trainer_node.set_input(DataPacket(DataType.ANY, config))

        self._worker = TrainingWorker(self.trainer_node)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop(self):
        if self.trainer_node and hasattr(self.trainer_node, "request_abort"):
            self.trainer_node.request_abort()
        if self._worker and self._worker.isRunning():
            self._log("Arrêt coopératif demandé…")
            self.stop_btn.setEnabled(False)

    def _lock_ui(self):
        self.train_btn.setEnabled(False)
        self.soup_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.step_label.setText("Initialisation...")
        self.loss_label.setText("")

    def _unlock_ui(self):
        self.train_btn.setEnabled(True)
        self.soup_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # --- Slots de progression (Travailleur asynchrone) ---
    @pyqtSlot(int, int, float)
    def _on_progress(self, step: int, total: int, loss: float):
        self.progress_bar.setValue(int(step / total * 100))
        self.step_label.setText(f"Étape {step} / {total}")
        self.loss_label.setText(f"Loss: {loss:.4f}" if loss > 0 else "")

    @pyqtSlot(str)
    def _on_finished(self, output_path: str):
        self.progress_bar.setValue(100)
        self.step_label.setText("Terminé ✓")
        self.loss_label.setText("")
        self._log(f"Succès ! Fichier généré : {output_path}")
        self._unlock_ui()

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self.step_label.setText("Erreur")
        self.loss_label.setText("")
        self._log(f"ERREUR : {msg}")
        self._unlock_ui()