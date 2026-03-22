"""
widgets.py — Composants réutilisables pour Neural Forge.

FilePickerRow : sélecteur de fichier style macOS — une seule zone cliquable,
pas de double déclenchement, feedback visuel clair.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor


class FilePickerRow(QWidget):
    """
    Sélecteur de fichier compact façon macOS.

    ┌──────────────────────────────────────────────┬──────────┐
    │  icon   parent/fichier.ext                   │ Parcourir│
    └──────────────────────────────────────────────┴──────────┘

    Émet file_selected(str) quand l'utilisateur choisit un fichier.
    """

    file_selected = pyqtSignal(str)

    _STYLE_IDLE = (
        "background-color: rgba(255,255,255,0.04);"
        "border: 1px solid rgba(255,255,255,0.08);"
        "border-radius: 10px;"
    )
    _STYLE_HOVER = (
        "background-color: rgba(255,255,255,0.07);"
        "border: 1px solid rgba(255,255,255,0.14);"
        "border-radius: 10px;"
    )

    def __init__(
        self,
        placeholder: str = "Aucun fichier sélectionné",
        icon: str = "📄",
        filters: str = "Tous les fichiers (*)",
        dialog_title: str = "Sélectionner un fichier",
        start_dir: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._placeholder = placeholder
        self._icon_char = icon
        self._filters = filters
        self._dialog_title = dialog_title
        self._start_dir = start_dir
        self._path = ""
        self._hovering = False
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self):
        self.setFixedHeight(48)
        self.setStyleSheet(self._STYLE_IDLE)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Icône ──
        self._icon_lbl = QLabel(self._icon_char)
        self._icon_lbl.setFixedWidth(46)
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setStyleSheet(
            "color: #48484A; font-size: 16px;"
            " background: transparent; border: none;"
        )
        # Bloque les événements souris sur l'icône → évite le double déclenchement
        self._icon_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self._icon_lbl)

        # ── Séparateur ──
        sep = QWidget()
        sep.setFixedSize(1, 24)
        sep.setStyleSheet("background-color: rgba(255,255,255,0.07);")
        sep.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(sep)
        layout.addSpacing(14)

        # ── Label chemin ──
        self._path_lbl = QLabel(self._placeholder)
        self._path_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._path_lbl.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        self._path_lbl.setStyleSheet(
            "color: #48484A; font-size: 13px;"
            " font-style: italic; background: transparent; border: none;"
        )
        self._path_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self._path_lbl)

        # ── Bouton Parcourir ──
        self._btn = QPushButton("Parcourir")
        self._btn.setObjectName("filePickBtn")
        self._btn.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        layout.addWidget(self._btn)

        # ── Bouton poubelle (visible seulement quand fichier sélectionné) ──
        self._clear_btn = QPushButton("🗑")
        self._clear_btn.setFixedSize(46, 46)
        self._clear_btn.setStyleSheet(
            "background:rgba(255,69,58,0.12);color:#FF453A;"
            "border:none;border-radius:0px;"
            "border-top-right-radius:10px;border-bottom-right-radius:10px;"
            "font-size:14px;"
        )
        self._clear_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._clear_btn.setToolTip("Effacer le chemin")
        self._clear_btn.hide()
        # Ce bouton gère son propre clic — pas transparent
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        layout.addWidget(self._clear_btn)

    # ------------------------------------------------------------------
    # Événements souris (hover + clic)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._open_dialog()

    def enterEvent(self, event):
        self.setStyleSheet(self._STYLE_HOVER)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self._STYLE_IDLE)
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # Dialogue de sélection
    # ------------------------------------------------------------------

    def _open_dialog(self):
        start = self._start_dir
        # Si le chemin de départ est relatif, le résoudre depuis le CWD
        if start and not os.path.isabs(start):
            cwd_start = os.path.join(os.getcwd(), start)
            start = cwd_start if os.path.exists(cwd_start) else os.path.expanduser("~")
        elif not start:
            start = os.path.expanduser("~")

        path, _ = QFileDialog.getOpenFileName(
            self, self._dialog_title, start, self._filters
        )
        if path:
            self.set_path(path)
            self.file_selected.emit(path)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def _on_clear_clicked(self):
        """Clic sur la poubelle — efface sans ouvrir le dialog."""
        self.clear()

    def set_path(self, path: str):
        self._path = path
        parent = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)
        display = f"{parent}/{name}" if parent else name
        self._path_lbl.setText(display)
        self._path_lbl.setStyleSheet(
            "color:#AEAEB2;font-size:13px;"
            "font-style:normal;background:transparent;border:none;"
        )
        self._icon_lbl.setStyleSheet(
            "color:#0A84FF;font-size:16px;"
            "background:transparent;border:none;"
        )
        # Ajuster le rayon du bouton Parcourir pour faire place à la poubelle
        self._btn.setStyleSheet(
            "QPushButton#filePickBtn{"
            "border-top-right-radius:0px;border-bottom-right-radius:0px;"
            "border-right:1px solid rgba(255,255,255,0.05);}"
        )
        self._clear_btn.show()
        self.setToolTip(path)

    def get_path(self) -> str:
        return self._path

    def clear(self):
        self._path = ""
        self._path_lbl.setText(self._placeholder)
        self._path_lbl.setStyleSheet(
            "color:#48484A;font-size:13px;"
            "font-style:italic;background:transparent;border:none;"
        )
        self._icon_lbl.setStyleSheet(
            "color:#48484A;font-size:16px;"
            "background:transparent;border:none;"
        )
        self._clear_btn.hide()
        self._btn.setStyleSheet("")  # reset radius
        self.setToolTip("")
