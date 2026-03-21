"""
chat_tab.py — Onglet Chat multimodal (texte + image + document).

Trois modes d'entrée :
  1. Texte seul           → LLMNode streaming
  2. Image jointe         → VisionNode (si LLaVA chargé) ou LLMNode
  3. Document joint       → contenu injecté dans le prompt → LLMNode streaming

Le bouton "+" ouvre un menu pour choisir image ou document.
"""

import html
import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextBrowser, QFileDialog,
    QSizePolicy, QMenu,
)
from PyQt6.QtCore import Qt, pyqtSlot, QSize, QPoint
from PyQt6.QtGui import QPixmap, QKeyEvent, QAction

from core.types import DataPacket, DataType
from core.document_reader import read_document, get_file_info, DocumentReadError
from gui.workers import StreamWorker, VisionWorker


# ── Templates HTML ────────────────────────────────────────────────────

_USER_TEXT = (
    '<div style="margin:8px 0 8px 80px;text-align:right;">'
    '<div style="display:inline-block;background:#0A84FF;color:#fff;'
    'padding:10px 14px;border-radius:18px 18px 4px 18px;'
    'font-size:13px;line-height:1.6;text-align:left;'
    'max-width:100%;word-wrap:break-word;">{text}</div></div>'
)

_USER_IMG = (
    '<div style="margin:8px 0 4px 80px;text-align:right;">'
    '<img src="{src}" style="max-width:220px;max-height:160px;'
    'border-radius:12px;display:inline-block;"/></div>'
)

_USER_DOC = (
    '<div style="margin:8px 0 4px 80px;text-align:right;">'
    '<div style="display:inline-block;background:#2C2C2E;color:#AEAEB2;'
    'padding:8px 14px;border-radius:12px;font-size:12px;">'
    '<span style="color:#0A84FF;margin-right:6px;">{icon}</span>'
    '{name} <span style="color:#636366;font-size:11px;">({size})</span>'
    '</div></div>'
)

_AI_OPEN = (
    '<div style="margin:8px 80px 8px 0;text-align:left;">'
    '<div style="display:inline-block;background:#2C2C2E;color:#F2F2F7;'
    'padding:10px 14px;border-radius:18px 18px 18px 4px;'
    'font-size:13px;line-height:1.6;max-width:100%;word-wrap:break-word;">'
)
_AI_CLOSE = "</div></div>"

_THINKING = (
    '<div style="margin:8px 80px 8px 0;">'
    '<div style="display:inline-block;background:#2C2C2E;'
    'padding:12px 18px;border-radius:18px 18px 18px 4px;">'
    '<span style="color:#636366;font-size:20px;letter-spacing:5px;">···</span>'
    '</div></div>'
)

_SYS = (
    '<div style="margin:10px 0;text-align:center;">'
    '<span style="color:#48484A;font-size:11px;background:#1C1C1E;'
    'padding:3px 12px;border-radius:10px;">{text}</span></div>'
)

_DOC_ICONS = {
    ".pdf": "PDF", ".docx": "DOC", ".txt": "TXT",
    ".md": "MD", ".json": "JSON", ".jsonl": "JSONL",
}


class ChatInput(QLineEdit):
    def keyPressEvent(self, e: QKeyEvent):
        if (e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.returnPressed.emit()
        else:
            super().keyPressEvent(e)


class ChatTab(QWidget):

    def __init__(self, llm_node, vision_node, parent=None):
        super().__init__(parent)
        self.llm_node = llm_node
        self.vision_node = vision_node
        self._worker = None
        self._stream_buf = ""
        self._history = ""
        self._attached_image: str = ""
        self._attached_doc: str = ""
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
        self._model_lbl.setStyleSheet(
            "color:#636366;font-size:12px;background:transparent;"
        )
        hl.addWidget(self._model_lbl)
        hl.addStretch()

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

        # ── Barre de pièce jointe ──
        self._attach_bar = QWidget()
        self._attach_bar.setFixedHeight(58)
        self._attach_bar.setStyleSheet(
            "background:#1A1A1C;"
            "border-top:1px solid rgba(255,255,255,0.06);"
        )
        self._attach_bar.hide()
        ab = QHBoxLayout(self._attach_bar)
        ab.setContentsMargins(14, 8, 14, 8)
        ab.setSpacing(10)

        self._attach_icon = QLabel()
        self._attach_icon.setFixedSize(42, 42)
        self._attach_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._attach_icon.setStyleSheet(
            "background:#2C2C2E;border-radius:8px;"
            "color:#0A84FF;font-size:11px;font-weight:600;"
        )
        ab.addWidget(self._attach_icon)

        attach_info = QVBoxLayout()
        attach_info.setSpacing(1)
        self._attach_name = QLabel()
        self._attach_name.setStyleSheet(
            "color:#AEAEB2;font-size:12px;font-weight:500;background:transparent;"
        )
        self._attach_meta = QLabel()
        self._attach_meta.setStyleSheet(
            "color:#636366;font-size:11px;background:transparent;"
        )
        attach_info.addWidget(self._attach_name)
        attach_info.addWidget(self._attach_meta)
        ab.addLayout(attach_info)
        ab.addStretch()

        rm_btn = QPushButton("✕")
        rm_btn.setFixedSize(26, 26)
        rm_btn.setStyleSheet(
            "background:rgba(255,255,255,0.08);color:#8E8E93;"
            "border:none;border-radius:13px;font-size:11px;"
        )
        rm_btn.clicked.connect(self._remove_attachment)
        ab.addWidget(rm_btn)
        root.addWidget(self._attach_bar)

        # ── Barre de saisie ──
        input_bar = QWidget()
        input_bar.setStyleSheet(
            "background:#111113;"
            "border-top:1px solid rgba(255,255,255,0.07);"
        )
        input_bar.setFixedHeight(66)
        il = QHBoxLayout(input_bar)
        il.setContentsMargins(14, 12, 14, 12)
        il.setSpacing(8)

        self._attach_btn = QPushButton("+")
        self._attach_btn.setFixedSize(42, 42)
        self._attach_btn.setStyleSheet(
            "background:rgba(255,255,255,0.08);color:#8E8E93;"
            "border:none;border-radius:10px;font-size:22px;font-weight:300;"
        )
        self._attach_btn.setToolTip("Joindre une image ou un document")
        self._attach_btn.clicked.connect(self._show_attach_menu)
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
    # Menu d'attachement
    def _show_attach_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#2C2C2E;border:1px solid rgba(255,255,255,0.10);"
            "border-radius:12px;padding:6px;color:#F2F2F7;font-size:13px;}"
            "QMenu::item{padding:8px 16px;border-radius:8px;}"
            "QMenu::item:selected{background:rgba(10,132,255,0.20);color:#fff;}"
            "QMenu::separator{background:rgba(255,255,255,0.08);height:1px;margin:4px 8px;}"
        )

        act_img = QAction("  Image  (.png, .jpg, .webp…)", self)
        act_img.triggered.connect(self._attach_image)

        act_doc = QAction("  Document  (.txt, .pdf, .docx, .md…)", self)
        act_doc.triggered.connect(self._attach_document)

        menu.addAction(act_img)
        menu.addSeparator()
        menu.addAction(act_doc)

        # Affiche le menu au-dessus du bouton
        btn_pos = self._attach_btn.mapToGlobal(QPoint(0, 0))
        menu.exec(QPoint(btn_pos.x(), btn_pos.y() - menu.sizeHint().height() - 4))

    # ------------------------------------------------------------------
    # Attachements

    def _attach_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Joindre une image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path:
            return
        self._attached_image = path
        self._attached_doc = ""
        px = QPixmap(path).scaled(
            QSize(42, 42), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._attach_icon.setPixmap(px)
        self._attach_name.setText(os.path.basename(path))
        self._attach_meta.setText("Image")
        self._attach_bar.show()
        self._set_attach_btn_active(True)

    def _attach_document(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Joindre un document", "",
            "Documents (*.txt *.md *.pdf *.docx *.json *.jsonl *.rst *.csv)"
            ";;Tous les fichiers (*)"
        )
        if not path:
            return
        self._attached_doc = path
        self._attached_image = ""
        info = get_file_info(path)
        ext = info["ext"]
        icon_text = _DOC_ICONS.get(ext, "DOC")
        self._attach_icon.setText(icon_text)
        self._attach_icon.setPixmap(QPixmap())  # reset image
        self._attach_name.setText(info["name"])
        self._attach_meta.setText(f"{info['label']} · {info['size_kb']} Ko")
        self._attach_bar.show()
        self._set_attach_btn_active(True)

    def _remove_attachment(self):
        self._attached_image = ""
        self._attached_doc = ""
        self._attach_bar.hide()
        self._attach_icon.clear()
        self._set_attach_btn_active(False)

    def _set_attach_btn_active(self, on: bool):
        if on:
            self._attach_btn.setStyleSheet(
                "background:#0A84FF;color:#fff;"
                "border:none;border-radius:10px;font-size:22px;font-weight:300;"
            )
        else:
            self._attach_btn.setStyleSheet(
                "background:rgba(255,255,255,0.08);color:#8E8E93;"
                "border:none;border-radius:10px;font-size:22px;font-weight:300;"
            )

    # ------------------------------------------------------------------
    # Envoi
    def _send(self):
        text = self._input.text().strip()
        has_img = bool(self._attached_image)
        has_doc = bool(self._attached_doc)

        if not text and not has_img and not has_doc:
            return
        if not self.llm_node.model_loaded:
            self._sys("Aucun modèle chargé — rendez-vous dans Paramètres.")
            return
        if self._worker and self._worker.isRunning():
            return

        self._input.clear()
        self._lock()

        # ── Affichage des pièces jointes en bulle ──
        if has_img:
            self._append(_USER_IMG.format(src=self._attached_image))

        if has_doc:
            info = get_file_info(self._attached_doc)
            icon = _DOC_ICONS.get(info["ext"], "DOC")
            self._append(_USER_DOC.format(
                icon=icon,
                name=html.escape(info["name"]),
                size=f"{info['size_kb']} Ko",
            ))

        if text:
            self._append(_USER_TEXT.format(text=html.escape(text)))

        self._append(_THINKING)
        img_path = self._attached_image
        doc_path = self._attached_doc
        self._remove_attachment()

        # ── Choix du mode d'inférence ──
        if has_img and self.vision_node.model_loaded:
            # Vision multimodal
            if text:
                self.vision_node.set_prompt(text)
            packet = DataPacket(DataType.IMAGE_PATH, img_path, source="user")
            self._worker = VisionWorker(self.vision_node, packet)
            self._worker.result.connect(self._on_result)
            self._worker.error.connect(self._on_error)
            self._worker.finished.connect(self._on_done)
            self._worker.start()

        elif has_doc:
            # Document → injection directe ou résumé auto selon la taille
            try:
                result = read_document(doc_path, llm_node=self.llm_node)
                prompt = result.build_prompt(text)
                if result.was_summarized:
                    self._sys(
                        f"Document long ({result.original_chars:,} car.) "
                        "— résumé automatique activé."
                    )
            except DocumentReadError as e:
                self._history = self._history.replace(_THINKING, "")
                self._sys(f"Erreur lecture : {html.escape(str(e))}")
                self._unlock()
                return


            packet = DataPacket(DataType.TEXT, prompt, source="user")
            self._stream_buf = ""
            self._worker = StreamWorker(self.llm_node, packet)
            self._worker.token.connect(self._on_token)
            self._worker.error.connect(self._on_error)
            self._worker.finished.connect(self._on_stream_done)
            self._worker.start()

        else:
            # Texte seul
            prompt = text or "Bonjour !"
            packet = DataPacket(DataType.TEXT, prompt, source="user")
            self._stream_buf = ""
            self._worker = StreamWorker(self.llm_node, packet)
            self._worker.token.connect(self._on_token)
            self._worker.error.connect(self._on_error)
            self._worker.finished.connect(self._on_stream_done)
            self._worker.start()

    # ------------------------------------------------------------------
    # Slots
    @pyqtSlot(str)
    def _on_token(self, token: str):
        # Retire le _THINKING au premier token
        if not self._stream_buf:
            self._history = self._history.replace(_THINKING, "", 1)
        self._stream_buf += html.escape(token)
        full = self._history + _AI_OPEN + self._stream_buf + _AI_CLOSE
        self._chat.setHtml(self._wrap(full))
        self._scroll()

    @pyqtSlot()
    def _on_stream_done(self):
        if self._stream_buf:
            self._history += _AI_OPEN + self._stream_buf + _AI_CLOSE
        self._stream_buf = ""
        self._unlock()

    @pyqtSlot(str)
    def _on_result(self, text: str):
        self._history = self._history.replace(_THINKING, "", 1)
        self._history += _AI_OPEN + html.escape(text) + _AI_CLOSE
        self._chat.setHtml(self._wrap(self._history))
        self._scroll()

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._history = self._history.replace(_THINKING, "", 1)
        self._sys(f"Erreur : {html.escape(msg)}")
        self._unlock()

    @pyqtSlot()
    def _on_done(self):
        self._unlock()

    # ------------------------------------------------------------------
    def _append(self, block: str):
        self._history += block
        self._chat.setHtml(self._wrap(self._history))
        self._scroll()

    def _sys(self, text: str):
        self._append(_SYS.format(text=text))

    def _clear(self):
        self._history = ""
        self._chat.setHtml("")
        self._sys("Conversation effacée.")

    def _lock(self):
        self._send_btn.setEnabled(False)
        self._input.setEnabled(False)

    def _unlock(self):
        self._send_btn.setEnabled(True)
        self._input.setEnabled(True)
        self._input.setFocus()

    def _scroll(self):
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
