"""
vision_tab.py — Onglet analyse d'image (LLaVA multimodal).

Fonctionnalités :
  - Glisser-déposer ou sélection d'image
  - Prévisualisation
  - Zone de question personnalisable
  - Résultat en streaming
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextBrowser, QFileDialog,
    QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSlot, QMimeData, QSize
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFont

from core.types import DataPacket, DataType
from gui.workers import VisionWorker


class ImageDropZone(QLabel):
    """Zone de drop pour les images avec retour visuel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path = None          # ← doit être avant _reset_style
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self._reset_style(hover=False)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._reset_style(hover=True)

    def dragLeaveEvent(self, event):
        self._reset_style(hover=False)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                self.set_image(path)
        self._reset_style(hover=False)

    def set_image(self, path: str):
        self._image_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                QSize(300, 280),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
            self.setStyleSheet(
                "background-color: #1C1C1E;"
                " border: 1px solid #3A3A3C;"
                " border-radius: 14px;"
                " padding: 8px;"
            )

    def get_image_path(self) -> str:
        return self._image_path or ""

    def clear_image(self):
        self._image_path = None
        self.setPixmap(QPixmap())
        self._reset_style(hover=False)

    def _reset_style(self, hover: bool):
        border_color = "#0A84FF" if hover else "#3A3A3C"
        self.setStyleSheet(
            f"background-color: #1C1C1E;"
            f" border: 2px dashed {border_color};"
            f" border-radius: 14px;"
            f" color: #636366;"
            f" font-size: 13px;"
            f" padding: 20px;"
        )
        if not self._image_path:
            self.setText("Glissez une image ici\nou cliquez pour parcourir")


class VisionTab(QWidget):

    def __init__(self, vision_node, parent=None):
        super().__init__(parent)
        self.vision_node = vision_node
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
        title = QLabel("Vision")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addSpacing(6)

        subtitle = QLabel("Analysez des images avec un modèle LLaVA")
        subtitle.setObjectName("statusLabel")
        root.addWidget(subtitle)
        root.addSpacing(24)

        # ── Corps : image | résultat ──
        body = QHBoxLayout()
        body.setSpacing(20)

        # ── Colonne gauche : image ──
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        self.drop_zone = ImageDropZone()
        self.drop_zone.setFixedSize(320, 290)
        self.drop_zone.setCursor(Qt.CursorShape.PointingHandCursor)
        self.drop_zone.mousePressEvent = lambda e: self._browse_image()
        left_col.addWidget(self.drop_zone)

        img_btn_row = QHBoxLayout()
        img_btn_row.setSpacing(8)

        browse_btn = QPushButton("Ouvrir une image")
        browse_btn.setObjectName("secondaryBtn")
        browse_btn.clicked.connect(self._browse_image)
        img_btn_row.addWidget(browse_btn)

        clear_img_btn = QPushButton("Effacer")
        clear_img_btn.setObjectName("dangerBtn")
        clear_img_btn.clicked.connect(self._clear_image)
        img_btn_row.addWidget(clear_img_btn)
        img_btn_row.addStretch()

        left_col.addLayout(img_btn_row)
        left_col.addStretch()

        body.addLayout(left_col)

        # ── Colonne droite : résultat ──
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        result_card = QWidget()
        result_card.setObjectName("card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(16, 16, 16, 16)
        result_layout.setSpacing(10)

        result_header = QLabel("Résultat de l'analyse")
        result_header.setObjectName("sectionLabel")
        result_layout.addWidget(result_header)

        self.result_view = QTextBrowser()
        self.result_view.setPlaceholderText(
            "Le résultat de l'analyse apparaîtra ici…"
        )
        self.result_view.setStyleSheet(
            "background-color: #0A0A0A; border: none;"
            " border-radius: 10px; padding: 12px;"
            " color: #EBEBF5; font-size: 14px;"
        )
        result_layout.addWidget(self.result_view, stretch=1)

        right_col.addWidget(result_card, stretch=1)

        body.addLayout(right_col, stretch=1)
        root.addLayout(body, stretch=1)
        root.addSpacing(16)

        # ── Barre de question + bouton ──
        question_bar = QWidget()
        question_bar.setObjectName("card")
        q_layout = QHBoxLayout(question_bar)
        q_layout.setContentsMargins(14, 12, 14, 12)
        q_layout.setSpacing(10)

        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText(
            "Question (ex: Décris cette image en détail…)"
        )
        self.question_input.setText("Décris cette image en détail.")
        self.question_input.setFixedHeight(40)
        self.question_input.returnPressed.connect(self._analyze)
        q_layout.addWidget(self.question_input)

        self.analyze_btn = QPushButton("  Analyser")
        self.analyze_btn.setObjectName("primaryBtn")
        self.analyze_btn.setFixedHeight(40)
        self.analyze_btn.clicked.connect(self._analyze)
        q_layout.addWidget(self.analyze_btn)

        root.addWidget(question_bar)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner une image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;Tous les fichiers (*)"
        )
        if path:
            self.drop_zone.set_image(path)

    def _clear_image(self):
        self.drop_zone.clear_image()
        self.result_view.clear()

    def _analyze(self):
        image_path = self.drop_zone.get_image_path()
        if not image_path:
            self.result_view.setPlainText("Veuillez d'abord sélectionner une image.")
            return
        if not self.vision_node.model_loaded:
            self.result_view.setPlainText(
                "Aucun modèle LLaVA chargé.\n"
                "Chargez un modèle compatible dans Paramètres."
            )
            return
        if self._worker and self._worker.isRunning():
            return

        question = self.question_input.text().strip()
        if question:
            self.vision_node.set_prompt(question)

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("  Analyse…")
        self.result_view.setPlainText("Analyse en cours…")

        packet = DataPacket(DataType.IMAGE_PATH, image_path, source="user")
        self._worker = VisionWorker(self.vision_node, packet)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @pyqtSlot(str)
    def _on_result(self, text: str):
        self.result_view.setPlainText(text)

    @pyqtSlot(str)
    def _on_error(self, message: str):
        self.result_view.setPlainText(f"Erreur : {message}")

    @pyqtSlot()
    def _on_finished(self):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("  Analyser")
