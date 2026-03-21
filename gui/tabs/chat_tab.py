"""
chat_tab.py — Onglet Chat multimodal (texte + image optionnelle).

Fusionne Chat et Vision : l'utilisateur peut joindre une image à n'importe
quel message. Si une image est jointe ET un modèle LLaVA est chargé, on
utilise VisionNode ; sinon on utilise LLMNode en texte seul.
"""

import html
import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextBrowser, QFileDialog,
    QSizePolicy, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSlot, QSize
from PyQt6.QtGui import QPixmap, QKeyEvent

from core.types import DataPacket, DataType
from gui.workers import StreamWorker, VisionWorker


# ── Templates HTML ────────────────────────────────────────────────────

_USER_TEXT = """
<div style="margin:8px 0 8px 80px;text-align:right;">
  <div style="display:inline-block;background:#0A84FF;color:#fff;
    padding:10px 14px;border-radius:18px 18px 4px 18px;
    font-size:13px;line-height:1.6;text-align:left;
    max-width:100%;word-wrap:break-word;">{text}</div>
</div>"""

_USER_IMG = """
<div style="margin:8px 0 4px 80px;text-align:right;">
  <img src="{src}" style="max-width:220px;max-height:160px;
    border-radius:12px;display:inline-block;"/>
</div>"""

_AI_OPEN = """
<div style="margin:8px 80px 8px 0;text-align:left;">
  <div style="display:inline-block;background:#2C2C2E;color:#F2F2F7;
    padding:10px 14px;border-radius:18px 18px 18px 4px;
    font-size:13px;line-height:1.6;max-width:100%;word-wrap:break-word;">"""

_AI_CLOSE = "</div></div>"

_THINKING = """
<div style="margin:8px 80px 8px 0;">
  <div style="display:inline-block;background:#2C2C2E;
    padding:12px 18px;border-radius:18px 18px 18px 4px;">
    <span style="color:#636366;font-size:20px;letter-spacing:5px;">···</span>
  </div>
</div>"""

_SYS = """
<div style="margin:10px 0;text-align:center;">
  <span style="color:#48484A;font-size:11px;background:#1C1C1E;
    padding:3px 12px;border-radius:10px;">{text}</span>
</div>"""


class ChatInput(QLineEdit):
    def keyPressEvent(self, event: QKeyEvent):
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.returnPressed.emit()
        else:
            super().keyPressEvent(event)


class ChatTab(QWidget):

    def __init__(self, llm_node, vision_node, parent=None):
        super().__init__(parent)
        self.llm_node = llm_node
        self.vision_node = vision_node
        self._worker = None
        self._stream_buf = ""
        self._history = ""
        self._attached_image: str = ""
        self._build_ui()
        self._sys("Chargez un modèle dans Paramètres pour commencer.")

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──
        header = QWidget()
        header.setStyleSheet(
            "background-color:#111113;"
            "border-bottom:1px solid rgba(255,255,255,0.07);"
        )
        header.setFixedHeight(52)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(18, 0, 18, 0)
        hl.setSpacing(10)

        self._model_lbl = QLabel("Aucun modèle chargé")
        self._model_lbl.setStyleSheet("color:#636366;font-size:12px;background:transparent;")
        hl.addWidget(self._model_lbl)
        hl.addStretch()

        # Dot statut
        self._dot = QLabel()
        self._dot.setFixedSize(8, 8)
        self._set_dot("#3A3A3C")
        hl.addWidget(self._dot)

        hl.addSpacing(10)
        clear_btn = QPushButton("Effacer")
        clear_btn.setObjectName("secondaryBtn")
        clear_btn.setFixedHeight(28)
        clear_btn.setFixedWidth(70)
        clear_btn.clicked.connect(self._clear)
        hl.addWidget(clear_btn)
        root.addWidget(header)

        # ── Zone chat ──
        self._chat = QTextBrowser()
        self._chat.setOpenExternalLinks(False)
        self._chat.setStyleSheet(
            "background-color:#111113;border:none;padding:8px 16px;"
        )
        root.addWidget(self._chat, stretch=1)

        # ── Preview image attachée ──
        self._preview_bar = QWidget()
        self._preview_bar.setFixedHeight(64)
        self._preview_bar.setStyleSheet(
            "background-color:#1A1A1C;"
            "border-top:1px solid rgba(255,255,255,0.06);"
        )
        self._preview_bar.hide()
        pb_layout = QHBoxLayout(self._preview_bar)
        pb_layout.setContentsMargins(14, 8, 14, 8)
        pb_layout.setSpacing(10)

        self._preview_img = QLabel()
        self._preview_img.setFixedSize(48, 48)
        self._preview_img.setStyleSheet("border-radius:8px;")
        pb_layout.addWidget(self._preview_img)

        self._preview_name = QLabel()
        self._preview_name.setStyleSheet("color:#8E8E93;font-size:12px;background:transparent;")
        pb_layout.addWidget(self._preview_name)
        pb_layout.addStretch()

        rm_btn = QPushButton("✕")
        rm_btn.setFixedSize(24, 24)
        rm_btn.setStyleSheet(
            "background:rgba(255,255,255,0.08);color:#8E8E93;"
            "border:none;border-radius:12px;font-size:11px;"
        )
        rm_btn.clicked.connect(self._remove_image)
        pb_layout.addWidget(rm_btn)
        root.addWidget(self._preview_bar)

        # ── Barre de saisie ──
        input_bar = QWidget()
        input_bar.setStyleSheet(
            "background-color:#111113;"
            "border-top:1px solid rgba(255,255,255,0.07);"
        )
        input_bar.setFixedHeight(66)
        il = QHBoxLayout(input_bar)
        il.setContentsMargins(14, 12, 14, 12)
        il.setSpacing(8)

        # Bouton attacher image
        self._attach_btn = QPushButton("+")
        self._attach_btn.setFixedSize(42, 42)
        self._attach_btn.setStyleSheet(
            "background:rgba(255,255,255,0.08);color:#8E8E93;"
            "border:none;border-radius:10px;font-size:20px;font-weight:300;"
        )
        self._attach_btn.setToolTip("Joindre une image")
        self._attach_btn.clicked.connect(self._attach_image)
        il.addWidget(self._attach_btn)

        self._input = ChatInput()
        self._input.setPlaceholderText("Écrivez un message…")
        self._input.setFixedHeight(42)
        self._input.returnPressed.connect(self._send)
        il.addWidget(self._input)

        self._send_btn = QPushButton("↑")
        self._send_btn.setObjectName("iconBtn")
        self._send_btn.setFixedSize(42, 42)
        self._send_btn.setToolTip("Envoyer")
        self._send_btn.clicked.connect(self._send)
        il.addWidget(self._send_btn)

        root.addWidget(input_bar)

    # ------------------------------------------------------------------
    # API publique
    def on_model_loaded(self, model_path: str, lora_path: str):
        name = os.path.basename(model_path)
        lora = f" + {os.path.basename(lora_path)}" if lora_path else ""
        self._model_lbl.setText(f"{name}{lora}")
        self._set_dot("#30D158")
        self._sys(f"Modèle prêt : {name}{lora}")

    # ------------------------------------------------------------------
    # Gestion image
    def _attach_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Joindre une image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if path:
            self._attached_image = path
            px = QPixmap(path).scaled(
                QSize(48, 48), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._preview_img.setPixmap(px)
            self._preview_name.setText(os.path.basename(path))
            self._preview_bar.show()
            self._attach_btn.setStyleSheet(
                "background:#0A84FF;color:#fff;"
                "border:none;border-radius:10px;font-size:20px;font-weight:300;"
            )

    def _remove_image(self):
        self._attached_image = ""
        self._preview_bar.hide()
        self._preview_img.clear()
        self._attach_btn.setStyleSheet(
            "background:rgba(255,255,255,0.08);color:#8E8E93;"
            "border:none;border-radius:10px;font-size:20px;font-weight:300;"
        )

    # ------------------------------------------------------------------
    # Envoi
    def _send(self):
        text = self._input.text().strip()
        if not text and not self._attached_image:
            return
        if not self.llm_node.model_loaded:
            self._sys("Aucun modèle chargé — rendez-vous dans Paramètres.")
            return
        if self._worker and self._worker.isRunning():
            return

        self._input.clear()
        self._send_btn.setEnabled(False)
        self._input.setEnabled(False)

        # Affiche image en bulle utilisateur
        if self._attached_image:
            self._append(_USER_IMG.format(src=self._attached_image))

        # Affiche texte en bulle utilisateur
        if text:
            self._append(_USER_TEXT.format(text=html.escape(text)))

        self._append(_THINKING)

        # Choix du worker
        if self._attached_image and self.vision_node.model_loaded:
            # Vision multimodal
            if text:
                self.vision_node.set_prompt(text)
            packet = DataPacket(DataType.IMAGE_PATH, self._attached_image, source="user")
            self._worker = VisionWorker(self.vision_node, packet)
            self._worker.result.connect(self._on_result)
            self._worker.error.connect(self._on_error)
            self._worker.finished.connect(self._on_done)
        else:
            # Texte seul via streaming
            prompt = text or "Décris cette image."
            packet = DataPacket(DataType.TEXT, prompt, source="user")
            self._stream_buf = ""
            self._worker = StreamWorker(self.llm_node, packet)
            self._worker.token.connect(self._on_token)
            self._worker.error.connect(self._on_error)
            self._worker.finished.connect(self._on_stream_done)

        self._remove_image()
        self._worker.start()

    # ------------------------------------------------------------------
    # Slots streaming
    @pyqtSlot(str)
    def _on_token(self, token: str):
        self._stream_buf += html.escape(token)
        full = self._history + _AI_OPEN + self._stream_buf + _AI_CLOSE
        self._chat.setHtml(self._wrap(full))
        self._scroll_bottom()

    @pyqtSlot()
    def _on_stream_done(self):
        if self._stream_buf:
            self._history += _AI_OPEN + self._stream_buf + _AI_CLOSE
        self._stream_buf = ""
        self._unlock()

    # Slots vision (résultat complet)
    @pyqtSlot(str)
    def _on_result(self, text: str):
        self._history = self._history.replace(_THINKING, "")
        self._history += _AI_OPEN + html.escape(text) + _AI_CLOSE
        self._chat.setHtml(self._wrap(self._history))
        self._scroll_bottom()

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._history = self._history.replace(_THINKING, "")
        self._sys(f"Erreur : {html.escape(msg)}")

    @pyqtSlot()
    def _on_done(self):
        self._unlock()

    # ------------------------------------------------------------------
    def _append(self, block: str):
        self._history += block
        self._chat.setHtml(self._wrap(self._history))
        self._scroll_bottom()

    def _sys(self, text: str):
        self._append(_SYS.format(text=text))

    def _clear(self):
        self._history = ""
        self._chat.setHtml("")
        self._sys("Conversation effacée.")

    def _unlock(self):
        self._send_btn.setEnabled(True)
        self._input.setEnabled(True)
        self._input.setFocus()

    def _scroll_bottom(self):
        sb = self._chat.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_dot(self, c: str):
        self._dot.setStyleSheet(
            f"background-color:{c};border-radius:4px;"
            "min-width:8px;max-width:8px;min-height:8px;max-height:8px;"
        )

    @staticmethod
    def _wrap(body: str) -> str:
        return (
            "<html><head><style>"
            "body{background:#111113;color:#F2F2F7;"
            "font-family:-apple-system,'Helvetica Neue',Arial,sans-serif;"
            "font-size:13px;margin:0;padding:4px 8px;}"
            "</style></head><body>" + body + "</body></html>"
        )
