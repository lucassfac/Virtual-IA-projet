"""
chat_tab.py — Onglet Chatbot avec bulles iMessage et streaming token par token.
"""

import html
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextBrowser, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QKeyEvent

from core.types import DataPacket, DataType
from gui.workers import StreamWorker


# ── Templates HTML pour les bulles de chat ──────────────────────────────
_USER_BUBBLE = """
<div style="margin: 6px 0 6px 60px; text-align: right;">
  <div style="
    display: inline-block;
    background: #0A84FF;
    color: #FFFFFF;
    padding: 10px 14px;
    border-radius: 18px 18px 4px 18px;
    font-size: 14px;
    line-height: 1.5;
    max-width: 100%;
    word-wrap: break-word;
    text-align: left;
  ">{text}</div>
</div>
"""

_AI_BUBBLE_OPEN = """
<div style="margin: 6px 60px 6px 0; text-align: left;">
  <div style="
    display: inline-block;
    background: #2C2C2E;
    color: #FFFFFF;
    padding: 10px 14px;
    border-radius: 18px 18px 18px 4px;
    font-size: 14px;
    line-height: 1.5;
    max-width: 100%;
    word-wrap: break-word;
  " id="streaming">{text}"""

_AI_BUBBLE_CLOSE = "</div></div>"

_SYSTEM_MSG = """
<div style="margin: 12px 0; text-align: center;">
  <span style="
    color: #636366;
    font-size: 11px;
    background: #1C1C1E;
    padding: 3px 10px;
    border-radius: 10px;
  ">{text}</span>
</div>
"""

_THINKING = """
<div style="margin: 6px 60px 6px 0;">
  <div style="
    display: inline-block;
    background: #2C2C2E;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
  ">
    <span style="color: #636366; font-size: 18px; letter-spacing: 4px;">···</span>
  </div>
</div>
"""


class ChatInput(QLineEdit):
    """QLineEdit qui envoie sur Entrée et ignore Shift+Entrée."""

    def keyPressEvent(self, event: QKeyEvent):
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.returnPressed.emit()
        else:
            super().keyPressEvent(event)


class ChatTab(QWidget):

    def __init__(self, llm_node, parent=None):
        super().__init__(parent)
        self.llm_node = llm_node
        self._worker = None
        self._streaming_html = ""   # Accumule le HTML de la réponse en cours
        self._chat_html = ""        # Historique complet des messages
        self._build_ui()
        self._add_system("Chargez un modèle dans Paramètres pour commencer.")

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──
        header = QWidget()
        header.setObjectName("card")
        header.setStyleSheet(
            "border-radius: 0px; border-left: none; border-right: none;"
            " border-top: none; border-bottom: 1px solid #2C2C2E;"
            " background-color: #0A0A0A;"
        )
        header.setFixedHeight(56)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(20, 0, 20, 0)

        self.model_label = QLabel("Aucun modèle chargé")
        self.model_label.setObjectName("statusLabel")
        h_layout.addWidget(self.model_label)
        h_layout.addStretch()

        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self._set_dot("#3A3A3C")
        h_layout.addWidget(self.status_dot)

        clear_btn = QPushButton("Effacer")
        clear_btn.setObjectName("secondaryBtn")
        clear_btn.setFixedHeight(30)
        clear_btn.clicked.connect(self._clear_chat)
        h_layout.addSpacing(12)
        h_layout.addWidget(clear_btn)

        root.addWidget(header)

        # ── Zone de chat ──
        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setStyleSheet(
            "background-color: #000000; padding: 16px 20px;"
        )
        root.addWidget(self.chat_view, stretch=1)

        # ── Zone de saisie ──
        input_bar = QWidget()
        input_bar.setStyleSheet(
            "background-color: #0A0A0A;"
            " border-top: 1px solid #2C2C2E;"
        )
        input_bar.setFixedHeight(70)
        i_layout = QHBoxLayout(input_bar)
        i_layout.setContentsMargins(16, 14, 16, 14)
        i_layout.setSpacing(10)

        self.input_field = ChatInput()
        self.input_field.setPlaceholderText("Écrivez un message…")
        self.input_field.setFixedHeight(42)
        self.input_field.returnPressed.connect(self._send_message)
        i_layout.addWidget(self.input_field)

        self.send_btn = QPushButton("↑")
        self.send_btn.setObjectName("iconBtn")
        self.send_btn.setFixedSize(42, 42)
        self.send_btn.setToolTip("Envoyer (Entrée)")
        self.send_btn.clicked.connect(self._send_message)
        i_layout.addWidget(self.send_btn)

        root.addWidget(input_bar)

    # ------------------------------------------------------------------
    # Slots publics (appelés par MainWindow)
    # ------------------------------------------------------------------

    def on_model_loaded(self, model_path: str, lora_path: str):
        import os
        name = os.path.basename(model_path)
        lora = f" + {os.path.basename(lora_path)}" if lora_path else ""
        self.model_label.setText(f"{name}{lora}")
        self._set_dot("#30D158")
        self._add_system(f"Modèle prêt : {name}{lora}")

    # ------------------------------------------------------------------
    # Envoi de message
    # ------------------------------------------------------------------

    def _send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return
        if not self.llm_node.model_loaded:
            self._add_system("Aucun modèle chargé — rendez-vous dans Paramètres.")
            return
        if self._worker and self._worker.isRunning():
            return

        self.input_field.clear()
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)

        # Bulle utilisateur
        self._append_html(_USER_BUBBLE.format(text=html.escape(text)))

        # Indicateur "en train d'écrire"
        self._append_html(_THINKING)

        packet = DataPacket(DataType.TEXT, text, source="user")
        self._worker = StreamWorker(self.llm_node, packet)
        self._worker.token.connect(self._on_token)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._streaming_html = ""
        self._worker.start()

    # ------------------------------------------------------------------
    # Slots workers
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_token(self, token: str):
        """Reçoit un token, reconstruit la bulle IA en temps réel."""
        self._streaming_html += html.escape(token)

        # Remplace le contenu complet (historique + bulle en cours)
        full = (
            self._chat_html
            + _AI_BUBBLE_OPEN.format(text=self._streaming_html)
            + _AI_BUBBLE_CLOSE
        )
        self.chat_view.setHtml(self._wrap_html(full))
        # Scroll vers le bas
        sb = self.chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    @pyqtSlot(str)
    def _on_error(self, message: str):
        self._remove_thinking()
        self._append_html(
            _SYSTEM_MSG.format(text=f"Erreur : {html.escape(message)}")
        )

    @pyqtSlot()
    def _on_finished(self):
        if self._streaming_html:
            # Finalise la bulle dans l'historique
            self._chat_html += (
                _AI_BUBBLE_OPEN.format(text=self._streaming_html)
                + _AI_BUBBLE_CLOSE
            )
        self._streaming_html = ""
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append_html(self, block: str):
        self._chat_html += block
        self.chat_view.setHtml(self._wrap_html(self._chat_html))
        sb = self.chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _remove_thinking(self):
        """Supprime l'indicateur de chargement de l'historique."""
        self._chat_html = self._chat_html.replace(_THINKING, "")

    def _add_system(self, text: str):
        self._append_html(_SYSTEM_MSG.format(text=text))

    def _clear_chat(self):
        self._chat_html = ""
        self.chat_view.setHtml("")
        self._add_system("Conversation effacée.")

    def _set_dot(self, color: str):
        self.status_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 4px;"
            " min-width: 8px; max-width: 8px;"
            " min-height: 8px; max-height: 8px;"
        )

    @staticmethod
    def _wrap_html(body: str) -> str:
        return f"""
        <html><head><style>
        body {{
            background-color: #000000;
            color: #FFFFFF;
            font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
            font-size: 14px;
            margin: 0; padding: 8px;
        }}
        </style></head><body>{body}</body></html>
        """
