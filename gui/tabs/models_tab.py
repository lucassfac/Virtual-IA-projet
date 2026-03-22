"""
models_tab.py — Onglet Bibliothèque de modèles.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QScrollArea, QProgressBar, QFrame,
    QSizePolicy, QListWidget, QListWidgetItem, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSlot, QSize, QTimer
from PyQt6.QtGui import QFont

from core.model_manager import (
    FEATURED_MODELS, list_local_models, delete_model,
    get_models_dir_size, search_models, load_hf_token, get_models_dir,
)
from gui.workers import SearchWorker, DownloadWorker
from gui.download_dialog import FileSelectorDialog, DownloadProgressDialog


# ── Suggestions d'autocomplétion ──────────────────────────────────────
AUTOCOMPLETE_SUGGESTIONS = [
    "gemma", "gemma 3", "mistral", "mistral 7b", "llama", "llama 3",
    "phi", "phi-3", "phi-4", "qwen", "qwen 2.5", "deepseek", "deepseek r1",
    "tinyllama", "wizardlm", "codellama", "neural-chat", "openhermes",
    "stablelm", "zephyr", "vicuna", "orca", "falcon", "mpt",
    "llava", "bakllava", "moondream",
]


def _sep():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("background:rgba(255,255,255,0.06);max-height:1px;border:none;")
    return f


def _tag(text):
    l = QLabel(text)
    l.setStyleSheet(
        "background:rgba(10,132,255,0.15);color:#409CFF;"
        "font-size:10px;font-weight:500;padding:2px 7px;"
        "border-radius:5px;border:none;"
    )
    l.setFixedHeight(18)
    return l


# ── Barre de recherche avec autocomplétion ────────────────────────────

class SearchBar(QWidget):
    """
    Barre de recherche avec dropdown d'autocomplétion style Google.
    Émet search_requested(query) via callback.
    """

    def __init__(self, on_search, parent=None):
        super().__init__(parent)
        self._on_search = on_search
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._update_suggestions)
        self._build()

    def _build(self):
        self.setStyleSheet("background:transparent;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Ligne input + bouton
        row = QHBoxLayout()
        row.setSpacing(8)

        self._input = QLineEdit()
        self._input.setPlaceholderText(
            "Rechercher un modèle  (ex: mistral, phi, llama…)"
        )
        self._input.setFixedHeight(44)
        self._input.textChanged.connect(self._on_text_changed)
        self._input.returnPressed.connect(self._submit)
        self._input.installEventFilter(self)
        row.addWidget(self._input)

        self._btn = QPushButton("Rechercher")
        self._btn.setObjectName("primaryBtn")
        self._btn.setFixedHeight(44)
        self._btn.setFixedWidth(125)
        self._btn.clicked.connect(self._submit)
        row.addWidget(self._btn)

        layout.addLayout(row)

        # Dropdown
        self._dropdown = QListWidget()
        self._dropdown.setStyleSheet(
            "QListWidget{"
            "  background:#242426;"
            "  border:1px solid rgba(255,255,255,0.12);"
            "  border-top:none;"
            "  border-bottom-left-radius:10px;"
            "  border-bottom-right-radius:10px;"
            "  outline:none;"
            "  padding:4px 0;"
            "}"
            "QListWidget::item{"
            "  color:#AEAEB2;"
            "  font-size:13px;"
            "  padding:9px 16px;"
            "  border:none;"
            "}"
            "QListWidget::item:selected, QListWidget::item:hover{"
            "  background:rgba(10,132,255,0.16);"
            "  color:#FFFFFF;"
            "  border-radius:6px;"
            "  margin:0 4px;"
            "}"
        )
        self._dropdown.hide()
        self._dropdown.itemClicked.connect(self._pick_suggestion)
        self._dropdown.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.addWidget(self._dropdown)

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent, Qt as QtKey
        if obj == self._input:
            if event.type() == QEvent.Type.FocusOut:
                QTimer.singleShot(150, self._dropdown.hide)
            elif event.type() == QEvent.Type.KeyPress:
                key = event.key()
                if key == QtKey.Key.Key_Down and not self._dropdown.isHidden():
                    self._dropdown.setFocus()
                    self._dropdown.setCurrentRow(0)
                    return True
                elif key == QtKey.Key.Key_Escape:
                    self._dropdown.hide()
                    return True
        return super().eventFilter(obj, event)

    def _on_text_changed(self, text: str):
        self._debounce.start(120)

    def _update_suggestions(self):
        text = self._input.text().strip().lower()
        self._dropdown.clear()

        if len(text) < 1:
            self._dropdown.hide()
            return

        matches = [s for s in AUTOCOMPLETE_SUGGESTIONS if text in s.lower()][:8]

        if not matches:
            self._dropdown.hide()
            return

        for m in matches:
            item = QListWidgetItem()
            # Met en gras la partie tapée
            item.setText(m)
            item.setData(Qt.ItemDataRole.UserRole, m)
            self._dropdown.addItem(item)

        rows = min(len(matches), 7)
        self._dropdown.setFixedHeight(rows * 38 + 8)
        self._dropdown.show()

    def _pick_suggestion(self, item: QListWidgetItem):
        text = item.data(Qt.ItemDataRole.UserRole)
        self._input.setText(text)
        self._dropdown.hide()
        self._submit()

    def _submit(self):
        query = self._input.text().strip()
        self._dropdown.hide()
        if query:
            self._on_search(query)

    def set_loading(self, loading: bool):
        self._btn.setEnabled(not loading)
        self._btn.setText("…" if loading else "Rechercher")

    def clear(self):
        self._input.clear()
        self._dropdown.hide()


# ── Carte modèle ──────────────────────────────────────────────────────

class ModelCard(QWidget):

    def __init__(self, info: dict, mode: str = "catalog", parent=None):
        super().__init__(parent)
        self.info = info
        self.mode = mode
        self._dl_worker = None
        self._build()

    def _build(self):
        self.setObjectName("card")
        self.setStyleSheet(
            "#card{background:rgba(255,255,255,0.03);"
            "border:1px solid rgba(255,255,255,0.07);border-radius:12px;}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(8)

        # Nom + taille
        top = QHBoxLayout()
        top.setSpacing(8)
        name_lbl = QLabel(self.info.get("name", self.info.get("filename", "?")))
        name_lbl.setStyleSheet(
            "color:#F2F2F7;font-size:13px;font-weight:500;background:transparent;"
        )
        top.addWidget(name_lbl, stretch=1)
        size = self.info.get("size_gb", 0)
        if size:
            size_lbl = QLabel(f"{size} Go")
            size_lbl.setStyleSheet("color:#48484A;font-size:11px;background:transparent;")
            top.addWidget(size_lbl)
        root.addLayout(top)

        # Description
        desc = self.info.get("description", "")
        if desc and desc != "Pas de description":
            d = QLabel(desc[:120] + ("…" if len(desc) > 120 else ""))
            d.setStyleSheet("color:#636366;font-size:12px;background:transparent;")
            d.setWordWrap(True)
            root.addWidget(d)

        # Tags + action
        bot = QHBoxLayout()
        bot.setSpacing(6)
        for t in self.info.get("tags", [])[:3]:
            if t not in ("gguf", "transformers", "pytorch"):
                bot.addWidget(_tag(t))
        bot.addStretch()

        if self.mode == "catalog":
            self._dl_btn = QPushButton("Télécharger")
            self._dl_btn.setObjectName("primaryBtn")
            self._dl_btn.setFixedHeight(30)
            self._dl_btn.setFixedWidth(115)
            self._dl_btn.setStyleSheet(
                "QPushButton#primaryBtn{font-size:12px;border-radius:8px;"
                "min-height:30px;padding:0 12px;}"
            )
            self._dl_btn.clicked.connect(self._start_download)
            bot.addWidget(self._dl_btn)

            root.addLayout(bot)

            self._status = QLabel("")
            self._status.setStyleSheet("color:#636366;font-size:11px;background:transparent;")
            self._status.hide()
            root.addWidget(self._status)

        elif self.mode == "installed":
            path = self.info.get("path", "")
            path_lbl = QLabel(path)
            path_lbl.setStyleSheet("color:#3A3A3C;font-size:10px;background:transparent;")
            path_lbl.setWordWrap(True)
            root.addWidget(path_lbl)

            del_btn = QPushButton("Supprimer")
            del_btn.setObjectName("dangerBtn")
            del_btn.setFixedHeight(30)
            del_btn.setFixedWidth(100)
            del_btn.setStyleSheet(
                "QPushButton#dangerBtn{font-size:12px;border-radius:8px;"
                "min-height:30px;padding:0 12px;}"
            )
            del_btn.clicked.connect(self._confirm_delete)
            bot.addWidget(del_btn)
            root.addLayout(bot)

    def _start_download(self):
        repo_id  = self.info.get("repo_id", "")
        filename = self.info.get("filename", "")

        if not repo_id:
            return

        # Si filename déjà connu (modèles recommandés) → direct
        # Si pas de filename (résultats de recherche) → sélecteur de fichier
        if not filename:
            dlg = FileSelectorDialog(repo_id, parent=self.window())
            if dlg.exec() != FileSelectorDialog.DialogCode.Accepted:
                return
            if not dlg.selected_file:
                return
            file_info = dlg.selected_file
        else:
            file_info = {
                "filename": filename,
                "size_gb": self.info.get("size_gb", 0),
            }

        # Lancer le dialog de progression style Steam
        prog_dlg = DownloadProgressDialog(
            repo_id, file_info, "models/", parent=self.window()
        )
        result = prog_dlg.exec()

        if result == DownloadProgressDialog.DialogCode.Accepted and prog_dlg.result_path:
            self._dl_btn.setText("Installé")
            self._dl_btn.setEnabled(False)
            self._status.setText(f"✓ {os.path.basename(prog_dlg.result_path)}")
            self._status.setStyleSheet("color:#30D158;font-size:11px;background:transparent;")
            self._status.show()
            # Refresh bibliothèque
            p = self.parent()
            while p:
                if hasattr(p, "refresh_installed"):
                    p.refresh_installed()
                    break
                p = p.parent()

    def _confirm_delete(self):
        path = self.info.get("path", "")
        name = self.info.get("filename", path)
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Supprimer le modèle")
        dlg.setText(f"Supprimer « {name} » ?")
        dlg.setInformativeText(f"Taille : {self.info.get('size_gb','?')} Go\n{path}")
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok
        )
        dlg.button(QMessageBox.StandardButton.Ok).setText("Supprimer")
        dlg.setStyleSheet(
            "QMessageBox{background:#1C1C1E;color:#F2F2F7;}"
            "QLabel{color:#F2F2F7;background:transparent;}"
            "QPushButton{background:#2C2C2E;color:#F2F2F7;"
            "border:1px solid rgba(255,255,255,0.10);border-radius:8px;padding:6px 16px;}"
            "QPushButton:hover{background:#3A3A3C;}"
        )
        if dlg.exec() == QMessageBox.StandardButton.Ok:
            try:
                delete_model(path)
                self.setVisible(False)
                self.deleteLater()
                p = self.parent()
                while p:
                    if hasattr(p, "refresh_installed"):
                        p.refresh_installed()
                        break
                    p = p.parent()
            except Exception as e:
                QMessageBox.critical(self, "Erreur", str(e))


# ── Onglet principal ──────────────────────────────────────────────────

class ModelsTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._search_worker = None
        self._build_ui()
        self.refresh_installed()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 32, 36, 24)
        root.setSpacing(0)

        # Titre
        title = QLabel("Bibliothèque de modèles")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addSpacing(4)
        sub = QLabel("Recherchez, téléchargez et gérez vos modèles locaux")
        sub.setObjectName("pageSubtitle")
        root.addWidget(sub)
        root.addSpacing(20)

        # Barre de recherche avec autocomplétion
        self._search_bar = SearchBar(on_search=self._do_search)
        root.addWidget(self._search_bar)
        root.addSpacing(20)

        # Zone scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")

        self._scroll_content = QWidget()
        self._scroll_content.setStyleSheet("background:transparent;")
        self._content_layout = QVBoxLayout(self._scroll_content)
        self._content_layout.setContentsMargins(0, 0, 6, 0)
        self._content_layout.setSpacing(0)

        scroll.setWidget(self._scroll_content)
        root.addWidget(scroll, stretch=1)

        self._show_catalog(FEATURED_MODELS, title="MODÈLES RECOMMANDÉS")

    def _do_search(self, query: str):
        self._search_bar.set_loading(True)
        self._clear_content()
        loading = QLabel(f"Recherche « {query} » sur HuggingFace…")
        loading.setStyleSheet(
            "color:#636366;font-size:13px;padding:20px 0;background:transparent;"
        )
        loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._content_layout.addWidget(loading)

        self._search_worker = SearchWorker(query)
        self._search_worker.results.connect(self._on_results)
        self._search_worker.error.connect(self._on_error)
        self._search_worker.finished.connect(
            lambda: self._search_bar.set_loading(False)
        )
        self._search_worker.start()

    @pyqtSlot(list)
    def _on_results(self, results: list):
        self._clear_content()

        back_btn = QPushButton("← Retour aux suggestions")
        back_btn.setObjectName("secondaryBtn")
        back_btn.setFixedHeight(32)
        back_btn.setStyleSheet(
            "QPushButton#secondaryBtn{font-size:12px;border-radius:8px;"
            "min-height:32px;margin-bottom:12px;}"
        )
        back_btn.clicked.connect(self._show_featured)
        self._content_layout.addWidget(back_btn)
        self._content_layout.addSpacing(8)

        if results:
            self._show_catalog(results, title=f"RÉSULTATS ({len(results)})")
        else:
            lbl = QLabel("Aucun résultat — essayez un autre terme.")
            lbl.setStyleSheet(
                "color:#636366;font-size:13px;padding:20px;background:transparent;"
            )
            self._content_layout.addWidget(lbl)

        self.refresh_installed()

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._clear_content()
        err = QLabel(f"Erreur réseau : {msg}")
        err.setStyleSheet(
            "color:#FF453A;font-size:13px;padding:20px;background:transparent;"
        )
        self._content_layout.addWidget(err)
        self.refresh_installed()

    def _show_featured(self):
        self._clear_content()
        self._search_bar.clear()
        self._show_catalog(FEATURED_MODELS, title="MODÈLES RECOMMANDÉS")
        self.refresh_installed()

    def _show_catalog(self, models: list, title: str = ""):
        if title:
            self._content_layout.addWidget(self._section(title))
            self._content_layout.addSpacing(10)
        for m in models:
            card = ModelCard(m, mode="catalog")
            self._content_layout.addWidget(card)
            self._content_layout.addSpacing(8)
        self._content_layout.addSpacing(4)

    def refresh_installed(self):
        if hasattr(self, "_installed_container"):
            self._installed_container.deleteLater()
            del self._installed_container

        models_dir = get_models_dir()
        installed  = list_local_models(models_dir)
        total_gb   = get_models_dir_size(models_dir)

        self._installed_container = QWidget()
        self._installed_container.setStyleSheet("background:transparent;")
        ic = QVBoxLayout(self._installed_container)
        ic.setContentsMargins(0, 0, 0, 0)
        ic.setSpacing(8)

        ic.addWidget(_sep())
        ic.addSpacing(12)

        hdr = QHBoxLayout()
        hdr.addWidget(self._section("MODÈLES INSTALLÉS"))
        hdr.addStretch()
        hdr.addWidget(QLabel(
            f"{len(installed)} modèle(s)  ·  {total_gb} Go"
        ))

        ic.addLayout(hdr)
        ic.addSpacing(10)

        if not installed:
            none_lbl = QLabel("Aucun modèle dans le dossier models/")
            none_lbl.setStyleSheet(
                "color:#3A3A3C;font-size:12px;font-style:italic;"
                "padding:8px 0;background:transparent;"
            )
            ic.addWidget(none_lbl)
        else:
            for m in installed:
                ic.addWidget(ModelCard(m, mode="installed"))

        ic.addSpacing(20)
        self._content_layout.addWidget(self._installed_container)

    def _section(self, text):
        l = QLabel(text); l.setObjectName("sectionLabel"); return l

    def _clear_content(self):
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if hasattr(self, "_installed_container"):
            del self._installed_container
