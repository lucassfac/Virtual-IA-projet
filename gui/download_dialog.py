"""
download_dialog.py — Dialog de sélection de fichier + progression style Steam.
Utilise les Threads de gui.workers pour éviter de figer l'interface.
"""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem,
    QProgressBar, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QThread, pyqtSignal
from core.model_manager import get_repo_gguf_files, load_hf_token, get_models_dir
from gui.workers import DownloadWorker

# ── Worker de chargement de liste ─────────────────────────────────────
class FileListWorker(QThread):
    done = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, repo_id):
        super().__init__()
        self.repo_id = repo_id
        
    def run(self):
        try:
            self.done.emit(get_repo_gguf_files(self.repo_id))
        except Exception as e:
            self.error.emit(str(e))

# ── Dialog sélection du fichier GGUF ─────────────────────────────────

class FileSelectorDialog(QDialog):
    def __init__(self, repo_id: str, parent=None):
        super().__init__(parent)
        self.repo_id = repo_id
        self.selected_file = None
        self.setWindowTitle(f"Choisir un fichier — {repo_id.split('/')[-1]}")
        self.setFixedWidth(560)
        self.setStyleSheet(
            "QDialog{background:#1C1C1E;border-radius:14px;}"
            "QLabel{background:transparent;color:#F2F2F7;}"
        )
        self._build()
        self._load_files()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 20)
        root.setSpacing(14)

        title = QLabel(f"Fichiers disponibles")
        title.setStyleSheet("font-size:16px;font-weight:600;color:#F2F2F7;")
        root.addWidget(title)

        repo_lbl = QLabel(self.repo_id)
        repo_lbl.setStyleSheet("font-size:11px;color:#48484A;")
        root.addWidget(repo_lbl)

        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget{background:#111113;border:1px solid rgba(255,255,255,0.08);"
            "border-radius:10px;padding:4px;outline:none;}"
            "QListWidget::item{color:#AEAEB2;font-size:13px;"
            "padding:10px 14px;border-radius:8px;border:none;}"
            "QListWidget::item:selected{background:rgba(10,132,255,0.20);color:#fff;}"
            "QListWidget::item:hover{background:rgba(255,255,255,0.05);}"
        )
        self._list.setMinimumHeight(200)
        self._list.itemDoubleClicked.connect(self._confirm)
        root.addWidget(self._list)

        self._loading = QLabel("Chargement de la liste…")
        self._loading.setStyleSheet("color:#636366;font-size:12px;")
        self._loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._loading)

        self._size_lbl = QLabel("")
        self._size_lbl.setStyleSheet("color:#636366;font-size:11px;")
        root.addWidget(self._size_lbl)

        btns = QHBoxLayout()
        btns.addStretch()

        cancel_btn = QPushButton("Annuler")
        cancel_btn.setObjectName("secondaryBtn")
        cancel_btn.setFixedHeight(38)
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)

        self._ok_btn = QPushButton("Télécharger")
        self._ok_btn.setObjectName("primaryBtn")
        self._ok_btn.setFixedHeight(38)
        self._ok_btn.setEnabled(False)
        self._ok_btn.clicked.connect(self._confirm)
        btns.addWidget(self._ok_btn)

        root.addLayout(btns)

    def _load_files(self):
        self._loader = FileListWorker(self.repo_id)
        self._loader.done.connect(self._on_files)
        self._loader.error.connect(self._on_error)
        self._loader.start()

    @pyqtSlot(list)
    def _on_files(self, files: list):
        self._loading.hide()
        self._files = files
        if not files:
            self._loading.setText("Aucun fichier .gguf trouvé dans ce dépôt.")
            self._loading.show()
            return

        for f in files:
            item = QListWidgetItem()
            name = f["filename"]
            size = f"  {f['size_gb']} Go" if f.get("size_gb") else ""
            badge = "  ★ Recommandé" if "Q4_K_M" in name else ""
            item.setText(f"{name}{size}{badge}")
            item.setData(Qt.ItemDataRole.UserRole, f)
            if "Q4_K_M" in name:
                item.setForeground(__import__('PyQt6.QtGui', fromlist=['QColor']).QColor("#F2F2F7"))
            self._list.addItem(item)

        for i in range(self._list.count()):
            if "Q4_K_M" in self._list.item(i).text():
                self._list.setCurrentRow(i)
                break
        else:
            self._list.setCurrentRow(0)

        self._list.currentItemChanged.connect(self._on_select)
        self._ok_btn.setEnabled(True)
        self._on_select(self._list.currentItem(), None)

    def _on_select(self, current, previous):
        if not current: return
        f = current.data(Qt.ItemDataRole.UserRole)
        if f:
            self._size_lbl.setText(
                f"Taille : {f.get('size_gb', '?')} Go  "
                f"({f.get('size_mb', '?'):.0f} Mo)" if f.get("size_mb") else ""
            )

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._loading.setText(f"Erreur : {msg[:80]}")
        self._loading.setStyleSheet("color:#FF453A;font-size:12px;")

    def _confirm(self, *_):
        item = self._list.currentItem()
        if item:
            self.selected_file = item.data(Qt.ItemDataRole.UserRole)
            self.accept()

# ── Fenêtre de progression style Steam ───────────────────────────────

class DownloadProgressDialog(QDialog):
    def __init__(self, repo_id: str, file_info: dict, dest_dir: str, parent=None):
        super().__init__(parent)
        self.repo_id = repo_id
        self.file_info = file_info
        self.dest_dir = dest_dir
        self._worker = None
        self._start_time = None
        self._last_bytes = 0
        self._last_time = 0
        self._speed_samples = []
        self.result_path = ""

        self.setWindowTitle("Téléchargement")
        self.setFixedWidth(520)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint | Qt.WindowType.CustomizeWindowHint)
        self.setStyleSheet("QDialog{background:#1C1C1E;}")
        self._build()
        self._start()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 24)
        root.setSpacing(14)

        title = QLabel(self.file_info.get("filename", ""))
        title.setStyleSheet("font-size:14px;font-weight:600;color:#F2F2F7;background:transparent;")
        title.setWordWrap(True)
        root.addWidget(title)

        repo_lbl = QLabel(self.repo_id)
        repo_lbl.setStyleSheet("font-size:11px;color:#48484A;background:transparent;")
        root.addWidget(repo_lbl)
        root.addSpacing(4)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setFixedHeight(10)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            "QProgressBar{background:rgba(255,255,255,0.08);border:none;border-radius:5px;}"
            "QProgressBar::chunk{background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1A6FFF, stop:0.5 #0A84FF, stop:1 #30D158);border-radius:5px;}"
        )
        root.addWidget(self._bar)

        stats = QHBoxLayout()
        stats.setSpacing(0)

        self._pct_lbl = QLabel("0%")
        self._pct_lbl.setStyleSheet("font-size:22px;font-weight:700;color:#F2F2F7;background:transparent;")
        stats.addWidget(self._pct_lbl)
        stats.addStretch()

        right_stats = QVBoxLayout()
        right_stats.setSpacing(2)
        right_stats.setAlignment(Qt.AlignmentFlag.AlignRight)

        self._speed_lbl = QLabel("— Mo/s")
        self._speed_lbl.setStyleSheet("font-size:13px;color:#0A84FF;font-weight:500;background:transparent;")
        self._speed_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_stats.addWidget(self._speed_lbl)

        self._eta_lbl = QLabel("Calcul en cours…")
        self._eta_lbl.setStyleSheet("font-size:11px;color:#636366;background:transparent;")
        self._eta_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_stats.addWidget(self._eta_lbl)

        stats.addLayout(right_stats)
        root.addLayout(stats)

        self._detail_bar = QProgressBar()
        self._detail_bar.setRange(0, 0)
        self._detail_bar.setFixedHeight(3)
        self._detail_bar.setTextVisible(False)
        self._detail_bar.setStyleSheet(
            "QProgressBar{background:rgba(255,255,255,0.04);border:none;border-radius:1px;}"
            "QProgressBar::chunk{background:#48484A;border-radius:1px;}"
        )
        root.addWidget(self._detail_bar)

        self._size_lbl = QLabel("Connexion à HuggingFace…")
        self._size_lbl.setStyleSheet("font-size:12px;color:#636366;background:transparent;")
        root.addWidget(self._size_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:rgba(255,255,255,0.06);max-height:1px;border:none;")
        root.addWidget(sep)

        btns = QHBoxLayout()
        btns.addStretch()
        self._cancel_btn = QPushButton("Annuler")
        self._cancel_btn.setObjectName("dangerBtn")
        self._cancel_btn.setFixedHeight(36)
        self._cancel_btn.setFixedWidth(100)
        self._cancel_btn.setStyleSheet("QPushButton#dangerBtn{font-size:12px;min-height:36px;border-radius:9px;}")
        self._cancel_btn.clicked.connect(self._cancel)
        btns.addWidget(self._cancel_btn)
        root.addLayout(btns)

    def _start(self):
        import time
        self._start_time = time.time()
        self._last_time = self._start_time
        self._max_total = 0

        self._worker = DownloadWorker(self.repo_id, self.file_info["filename"], self.dest_dir, token=load_hf_token())
        self._worker.progress.connect(self._on_progress)
        self._worker.success.connect(self._on_success)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @pyqtSlot(int, int)
    def _on_progress(self, done: int, total: int):
        import time
        now = time.time()

        if not hasattr(self, "_max_total"): self._max_total = total
        if total > self._max_total: self._max_total = total
        real_total = max(self._max_total, done)

        if real_total <= 0: return

        pct = min(done / real_total, 1.0)
        self._bar.setValue(int(pct * 1000))
        self._detail_bar.setRange(0, 1000)
        self._detail_bar.setValue(int(pct * 1000))
        self._pct_lbl.setText(f"{pct*100:.1f}%")
        self._size_lbl.setText(f"{done / 1024**2:.1f} Mo  /  {real_total / 1024**2:.1f} Mo")

        elapsed = now - self._last_time
        if elapsed >= 0.3:
            delta_bytes = done - self._last_bytes
            if delta_bytes > 0:
                speed = delta_bytes / elapsed
                self._speed_samples.append(speed)
                if len(self._speed_samples) > 6: self._speed_samples.pop(0)
                avg_speed = sum(self._speed_samples) / len(self._speed_samples)
                self._speed_lbl.setText(f"{avg_speed/1024**2:.1f} Mo/s")

                remaining = real_total - done
                if avg_speed > 0 and remaining > 0:
                    eta_s = remaining / avg_speed
                    if eta_s < 60: self._eta_lbl.setText(f"{eta_s:.0f}s restantes")
                    elif eta_s < 3600: self._eta_lbl.setText(f"{eta_s/60:.0f} min restantes")
                    else: self._eta_lbl.setText(f"{eta_s/3600:.1f}h restantes")

            self._last_bytes = done
            self._last_time = now

    @pyqtSlot(str)
    def _on_success(self, path: str):
        self.result_path = path
        self._bar.setValue(1000)
        self._pct_lbl.setText("100%")
        self._speed_lbl.setText("Installé ✓")
        self._speed_lbl.setStyleSheet("font-size:13px;color:#30D158;font-weight:500;background:transparent;")
        self._eta_lbl.setText(os.path.basename(path))
        self._size_lbl.setText("Téléchargement terminé")

        try: self._cancel_btn.clicked.disconnect()
        except Exception: pass
        self._cancel_btn.setText("Fermer")
        self._cancel_btn.setStyleSheet(
            "background:rgba(255,255,255,0.08);color:#EBEBF5;"
            "border:1px solid rgba(255,255,255,0.10);border-radius:9px;"
            "font-size:12px;min-height:36px;padding:0 20px;"
        )
        self._cancel_btn.clicked.connect(self.accept)
        QTimer.singleShot(1500, self.accept)

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._pct_lbl.setText("Erreur")
        self._pct_lbl.setStyleSheet("font-size:22px;font-weight:700;color:#FF453A;background:transparent;")
        self._speed_lbl.setText("")
        self._eta_lbl.setText(msg[:80])
        self._eta_lbl.setStyleSheet("font-size:11px;color:#FF453A;background:transparent;")
        self._cancel_btn.setText("Fermer")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.reject)
        self._detail_bar.setRange(0, 1)
        self._detail_bar.setValue(0)

    def _cancel(self):
        if self._worker and self._worker.isRunning(): self._worker.cancel()
        self.reject()

    def closeEvent(self, event):
        self._cancel()
        super().closeEvent(event)
