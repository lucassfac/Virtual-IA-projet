"""
chat_tab.py — Onglet Chat multimodal.
Bulles via <table> (seule méthode supportée par QTextBrowser).
Le bouton ☰ sidebar est RETIRÉ du chat — géré uniquement dans main_window.
"""

import html
import os
import re
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextBrowser, QFileDialog,
    QSizePolicy, QMenu,
)
from PyQt6.QtCore import Qt, pyqtSlot, QSize, QPoint
from PyQt6.QtGui import QPixmap, QKeyEvent, QAction

from core.types import DataPacket, DataType
from core.document_reader import read_document, get_file_info, DocumentReadError
from core.session_manager import save_conversation
from gui.workers import StreamWorker, VisionWorker


# ── Markdown simple ───────────────────────────────────────────────────

def _md(text: str) -> str:
    t = (text.replace("&","&amp;")
             .replace("<","&lt;")
             .replace(">","&gt;"))
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', t)
    t = re.sub(r'`(.+?)`',
        r'<font face="Courier New" color="#AEAEB2">\1</font>', t)
    t = re.sub(r'^### (.+)$', r'<b>\1</b>', t, flags=re.M)
    t = re.sub(r'^## (.+)$',  r'<b>\1</b>', t, flags=re.M)
    t = re.sub(r'^# (.+)$',   r'<b>\1</b>', t, flags=re.M)
    t = re.sub(r'^[-\*] (.+)$', r'&nbsp;&nbsp;• \1', t, flags=re.M)
    t = t.replace('\n', '<br/>')
    return t


# ── Bulles via table (compatible QTextBrowser) ────────────────────────

def _bubble_user(text: str) -> str:
    """Bulle utilisateur — alignée à droite, fond bleu."""
    return (
        '<table width="100%" cellpadding="0" cellspacing="0" border="0">'
        '<tr><td width="25%"></td>'
        '<td align="right">'
        '<table cellpadding="12" cellspacing="0" border="0">'
        '<tr><td bgcolor="#0A84FF" style="border-radius:18px;">'
        f'<font color="#FFFFFF" size="3">{text}</font>'
        '</td></tr></table>'
        '</td></tr></table>'
    )


def _bubble_ai(text: str) -> str:
    """Bulle IA — alignée à gauche, fond sombre."""
    return (
        '<table width="100%" cellpadding="0" cellspacing="0" border="0">'
        '<tr>'
        '<td align="left">'
        '<table cellpadding="12" cellspacing="0" border="0" width="75%">'
        '<tr><td bgcolor="#252527" style="border-radius:18px;">'
        f'<font color="#F2F2F7" size="3">{text}</font>'
        '</td></tr></table>'
        '</td>'
        '<td width="25%"></td>'
        '</tr></table>'
    )


def _bubble_img(path: str) -> str:
    return (
        '<table width="100%" cellpadding="0" cellspacing="4" border="0">'
        '<tr><td align="right">'
        f'<img src="{path}" width="200"/>'
        '</td></tr></table>'
    )


def _bubble_doc(icon: str, name: str, size: str) -> str:
    return (
        '<table width="100%" cellpadding="0" cellspacing="4" border="0">'
        '<tr><td width="25%"></td><td align="right">'
        '<table cellpadding="8" cellspacing="0" border="0">'
        '<tr><td bgcolor="#1A2A3A" style="border-radius:10px;">'
        f'<font color="#409CFF" size="2"><b>{icon}</b> {html.escape(name)}'
        f' <font color="#48484A">({size})</font></font>'
        '</td></tr></table></td></tr></table>'
    )


def _sys_msg(text: str) -> str:
    return (
        '<table width="100%" cellpadding="4" cellspacing="0" border="0">'
        '<tr><td align="center">'
        f'<font color="#3A3A3C" size="2">{text}</font>'
        '</td></tr></table>'
    )


def _thinking_bubble() -> str:
    return (
        '<table width="100%" cellpadding="0" cellspacing="0" border="0">'
        '<tr><td align="left">'
        '<table cellpadding="12" cellspacing="0" border="0">'
        '<tr><td bgcolor="#252527">'
        '<font color="#48484A" size="4">&nbsp;·&nbsp;·&nbsp;·&nbsp;</font>'
        '</td></tr></table>'
        '</td><td width="25%"></td></tr></table>'
    )


_THINKING = _thinking_bubble()
_DOC_ICONS = {".pdf":"PDF",".docx":"DOC",".txt":"TXT",".md":"MD",".json":"JSON",".jsonl":"JSONL"}


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
        self.llm_node    = llm_node
        self.vision_node = vision_node
        self._worker     = None
        self._stream_buf = ""
        self._history    = ""
        self._attached_image = ""
        self._attached_doc   = ""
        self._current_title  = ""
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
            "background:#0D0D0F;"
            "border-bottom:1px solid rgba(255,255,255,0.06);"
        )
        header.setFixedHeight(50)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.setSpacing(8)

        hist_btn = QPushButton("Historique")
        hist_btn.setObjectName("secondaryBtn")
        hist_btn.setFixedHeight(28)
        hist_btn.clicked.connect(self._show_history_menu)
        hl.addWidget(hist_btn)

        hl.addStretch()

        self._model_lbl = QLabel("Aucun modèle")
        self._model_lbl.setStyleSheet(
            "color:#48484A;font-size:12px;background:transparent;"
        )
        hl.addWidget(self._model_lbl)

        self._dot = QLabel()
        self._dot.setFixedSize(8, 8)
        self._set_dot("#3A3A3C")
        hl.addWidget(self._dot)

        hl.addStretch()

        save_btn = QPushButton("Sauvegarder")
        save_btn.setObjectName("secondaryBtn")
        save_btn.setFixedHeight(28)
        save_btn.clicked.connect(self._save_conv)
        hl.addWidget(save_btn)

        clear_btn = QPushButton("Effacer")
        clear_btn.setObjectName("secondaryBtn")
        clear_btn.setFixedHeight(28)
        clear_btn.clicked.connect(self._clear)
        hl.addWidget(clear_btn)

        root.addWidget(header)

        # ── Zone chat ──
        self._chat = QTextBrowser()
        self._chat.setOpenExternalLinks(True)
        self._chat.setStyleSheet(
            "background:#0D0D0F;border:none;padding:8px 12px;"
        )
        root.addWidget(self._chat, stretch=1)

        # ── Barre pièce jointe ──
        self._attach_bar = QWidget()
        self._attach_bar.setFixedHeight(56)
        self._attach_bar.setStyleSheet(
            "background:#151517;"
            "border-top:1px solid rgba(255,255,255,0.05);"
        )
        self._attach_bar.hide()
        ab = QHBoxLayout(self._attach_bar)
        ab.setContentsMargins(14, 8, 14, 8)
        ab.setSpacing(10)

        self._attach_icon = QLabel()
        self._attach_icon.setFixedSize(40, 40)
        self._attach_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._attach_icon.setStyleSheet(
            "background:#2C2C2E;border-radius:10px;"
            "color:#0A84FF;font-size:11px;font-weight:600;"
        )
        ab.addWidget(self._attach_icon)

        info_col = QVBoxLayout()
        info_col.setSpacing(1)
        self._attach_name = QLabel()
        self._attach_name.setStyleSheet(
            "color:#AEAEB2;font-size:12px;font-weight:500;background:transparent;"
        )
        self._attach_meta = QLabel()
        self._attach_meta.setStyleSheet(
            "color:#48484A;font-size:11px;background:transparent;"
        )
        info_col.addWidget(self._attach_name)
        info_col.addWidget(self._attach_meta)
        ab.addLayout(info_col)
        ab.addStretch()

        rm = QPushButton("✕")
        rm.setFixedSize(24, 24)
        rm.setStyleSheet(
            "background:rgba(255,255,255,0.07);color:#636366;"
            "border:none;border-radius:12px;font-size:11px;"
        )
        rm.clicked.connect(self._remove_attachment)
        ab.addWidget(rm)
        root.addWidget(self._attach_bar)

        # ── Barre de saisie ──
        input_bar = QWidget()
        input_bar.setStyleSheet(
            "background:#0D0D0F;"
            "border-top:1px solid rgba(255,255,255,0.06);"
        )
        input_bar.setFixedHeight(66)
        il = QHBoxLayout(input_bar)
        il.setContentsMargins(12, 12, 12, 12)
        il.setSpacing(8)

        self._attach_btn = QPushButton("+")
        self._attach_btn.setFixedSize(42, 42)
        self._attach_btn.setStyleSheet(
            "background:rgba(255,255,255,0.07);color:#8E8E93;"
            "border:none;border-radius:21px;font-size:22px;font-weight:300;"
        )
        self._attach_btn.setToolTip("Joindre image ou document")
        self._attach_btn.clicked.connect(self._show_attach_menu)
        il.addWidget(self._attach_btn)

        self._input = ChatInput()
        self._input.setPlaceholderText("Écrivez un message…")
        self._input.setFixedHeight(42)
        self._input.setStyleSheet(
            "QLineEdit{"
            "background:rgba(255,255,255,0.06);"
            "border:1px solid rgba(255,255,255,0.09);"
            "border-radius:21px;padding:0 16px;"
            "font-size:13px;color:#F2F2F7;}"
            "QLineEdit:focus{"
            "border-color:rgba(10,132,255,0.50);"
            "background:rgba(10,132,255,0.04);}"
        )
        self._input.returnPressed.connect(self._send)
        il.addWidget(self._input)

        self._send_btn = QPushButton("↑")
        self._send_btn.setFixedSize(42, 42)
        self._send_btn.setStyleSheet(
            "background:#0A84FF;color:#fff;"
            "border:none;border-radius:21px;"
            "font-size:18px;font-weight:600;"
        )
        self._send_btn.clicked.connect(self._send)
        il.addWidget(self._send_btn)

        # Bouton Stop — visible seulement pendant la génération
        self._stop_btn = QPushButton("■")
        self._stop_btn.setFixedSize(42, 42)
        self._stop_btn.setStyleSheet(
            "background:rgba(255,69,58,0.18);color:#FF453A;"
            "border:1px solid rgba(255,69,58,0.35);"
            "border-radius:21px;font-size:16px;font-weight:700;"
        )
        self._stop_btn.setToolTip("Arrêter la génération")
        self._stop_btn.clicked.connect(self._stop_generation)
        self._stop_btn.hide()
        il.addWidget(self._stop_btn)

        root.addWidget(input_bar)

    # ── API publique ──────────────────────────────────────────────────

    def on_model_loaded(self, model_path: str, skill_path: str):
        name = os.path.basename(model_path)
        skill = f" + {os.path.basename(skill_path)}" if skill_path else ""
        self._model_lbl.setText(f"{name}{skill}")
        self._set_dot("#30D158")
        self._sys(f"Modèle prêt : {name}{skill}")

    # ── Attachements ──────────────────────────────────────────────────

    def _show_attach_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#242426;border:1px solid rgba(255,255,255,0.10);"
            "border-radius:12px;padding:6px;color:#F2F2F7;font-size:13px;}"
            "QMenu::item{padding:9px 18px;border-radius:8px;}"
            "QMenu::item:selected{background:rgba(10,132,255,0.18);}"
            "QMenu::separator{background:rgba(255,255,255,0.07);height:1px;margin:4px 8px;}"
        )
        a1 = QAction("  Image  (.png, .jpg…)", self)
        a1.triggered.connect(self._attach_image)
        a2 = QAction("  Document  (.txt, .pdf, .docx…)", self)
        a2.triggered.connect(self._attach_document)
        menu.addAction(a1)
        menu.addSeparator()
        menu.addAction(a2)
        pos = self._attach_btn.mapToGlobal(QPoint(0, 0))
        menu.exec(QPoint(pos.x(), pos.y() - menu.sizeHint().height() - 4))

    def _attach_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Joindre une image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path: return
        self._attached_image = path
        self._attached_doc   = ""
        px = QPixmap(path).scaled(
            QSize(40,40), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._attach_icon.setPixmap(px)
        self._attach_name.setText(os.path.basename(path))
        self._attach_meta.setText("Image")
        self._attach_bar.show()
        self._set_attach_active(True)

    def _attach_document(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Joindre un document", "",
            "Documents (*.txt *.md *.pdf *.docx *.json *.jsonl *.csv)"
        )
        if not path: return
        self._attached_doc   = path
        self._attached_image = ""
        info = get_file_info(path)
        self._attach_icon.setText(_DOC_ICONS.get(info["ext"], "DOC"))
        self._attach_icon.setPixmap(QPixmap())
        self._attach_name.setText(info["name"])
        self._attach_meta.setText(f"{info['label']} · {info['size_kb']} Ko")
        self._attach_bar.show()
        self._set_attach_active(True)

    def _remove_attachment(self):
        self._attached_image = ""
        self._attached_doc   = ""
        self._attach_bar.hide()
        self._attach_icon.clear()
        self._set_attach_active(False)

    def _set_attach_active(self, on: bool):
        self._attach_btn.setStyleSheet(
            f"background:{'#0A84FF' if on else 'rgba(255,255,255,0.07)'};"
            f"color:{'#fff' if on else '#8E8E93'};"
            "border:none;border-radius:21px;font-size:22px;font-weight:300;"
        )

    # ── Envoi ─────────────────────────────────────────────────────────

    def _send(self):
        text    = self._input.text().strip()
        has_img = bool(self._attached_image)
        has_doc = bool(self._attached_doc)
        if not text and not has_img and not has_doc: return
        if not self.llm_node.model_loaded:
            self._sys("Aucun modèle chargé — rendez-vous dans Paramètres.")
            return
        if self._worker and self._worker.isRunning(): return

        self._input.clear()
        self._lock()

        if has_img:
            self._append(_bubble_img(self._attached_image))
        if has_doc:
            info = get_file_info(self._attached_doc)
            icon = _DOC_ICONS.get(info["ext"], "DOC")
            self._append(_bubble_doc(icon, info["name"], f"{info['size_kb']} Ko"))
        if text:
            self._append(_bubble_user(_md(text)))
            if not self._current_title:
                self._current_title = text[:50]

        self._append(_THINKING)
        img_path = self._attached_image
        doc_path = self._attached_doc
        self._remove_attachment()

        if has_img:
            from core.session_manager import load_last_model, load_vision_model
            llm_paths = load_last_model()
            vis_paths = load_vision_model()

            # Vérification : A-t-on les chemins de LLaVA ?
            if not vis_paths[0] or not vis_paths[1]:
                self._history = self._history.replace(_THINKING, "", 1)
                self._sys("⚠ Impossible d'analyser l'image : Modèle Vision ou mmproj introuvable. Allez dans Paramètres et chargez LLaVA au moins une fois.")
                self._unlock()
                return

            # Si le modèle texte est actif, on déclenche le "Model Swapping"
            if self.llm_node.model_loaded:
                self._sys("⚙️ <b>Orchestrateur activé :</b> Model Swapping pour préserver la VRAM (4Go max).")
                from gui.workers import SwapOrchestratorWorker
                self._stream_buf = ""
                self._worker = SwapOrchestratorWorker(
                    self.llm_node, self.vision_node,
                    img_path, text or "Que vois-tu sur cette image ?",
                    llm_paths, vis_paths
                )
                # Le statut s'affichera directement dans la fenêtre de chat !
                self._worker.status.connect(lambda s: self._sys(f"<font color='#0A84FF'>{s}</font>"))
                self._worker.token.connect(self._on_token)
                self._worker.error.connect(self._on_error)
                self._worker.finished.connect(self._on_stream_done)
                self._worker.start()
            else:
                # Comportement classique si aucun LLM n'est chargé
                if text: self.vision_node.set_prompt(text)
                packet = DataPacket(DataType.IMAGE_PATH, img_path, source="user")
                from gui.workers import VisionWorker
                self._worker = VisionWorker(self.vision_node, packet)
                self._worker.result.connect(self._on_result)
                self._worker.error.connect(self._on_error)
                self._worker.finished.connect(self._on_done)
                self._worker.start()
        elif has_doc:
            try:
                result = read_document(doc_path, llm_node=self.llm_node)
                prompt = result.build_prompt(text)
                if result.was_summarized:
                    self._sys("Document long — résumé auto activé.")
            except DocumentReadError as e:
                self._history = self._history.replace(_THINKING, "", 1)
                self._sys(f"Erreur lecture : {html.escape(str(e))}")
                self._unlock()
                return
            self._start_stream(DataPacket(DataType.TEXT, prompt, source="user"))
        else:
            self._start_stream(DataPacket(DataType.TEXT, text or "Bonjour !", source="user"))

    def _start_stream(self, packet):
        self._stream_buf = ""
        self._worker = StreamWorker(self.llm_node, packet)
        self._worker.token.connect(self._on_token)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_stream_done)
        self._worker.start()

    # ── Slots ─────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_token(self, token: str):
        if not self._stream_buf:
            # Retire le thinking au premier token
            self._history = self._history.replace(_THINKING, "", 1)
        self._stream_buf += token
        rendered = _md(self._stream_buf)
        self._chat.setHtml(self._wrap(
            self._history + _bubble_ai(rendered)
        ))
        self._scroll()

    @pyqtSlot()
    def _on_stream_done(self):
        if self._stream_buf:
            self._history += _bubble_ai(_md(self._stream_buf))
        self._stream_buf = ""
        self._unlock()

    @pyqtSlot(str)
    def _on_result(self, text: str):
        self._history = self._history.replace(_THINKING, "", 1)
        self._history += _bubble_ai(_md(text))
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

    # ── Historique ────────────────────────────────────────────────────

    def _save_conv(self):
        if not self._history.strip():
            self._sys("Rien à sauvegarder.")
            return
        title = self._current_title or "Conversation"
        save_conversation(title, self._history)
        self._sys(f"Sauvegardé : « {title} »")

    def _show_history_menu(self):
        from core.session_manager import list_conversations, get_conversation, delete_all_conversations
        convs = list_conversations()
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#242426;border:1px solid rgba(255,255,255,0.10);"
            "border-radius:12px;padding:6px;color:#F2F2F7;font-size:12px;min-width:260px;}"
            "QMenu::item{padding:8px 16px;border-radius:7px;}"
            "QMenu::item:selected{background:rgba(10,132,255,0.18);}"
            "QMenu::separator{background:rgba(255,255,255,0.07);height:1px;margin:4px 8px;}"
        )
        if not convs:
            a = QAction("Aucune conversation sauvegardée", self)
            a.setEnabled(False)
            menu.addAction(a)
        else:
            for c in convs[:15]:
                a = QAction(f"{c['date']}  —  {c['title']}", self)
                a.triggered.connect(lambda _, cid=c["id"]: self._load_conv(cid))
                menu.addAction(a)
            menu.addSeparator()
            ca = QAction("🗑  Supprimer tout l'historique", self)
            ca.triggered.connect(lambda: (delete_all_conversations(), self._sys("Historique effacé.")))
            menu.addAction(ca)

        btn = self.sender()
        pos = btn.mapToGlobal(QPoint(0, btn.height()))
        menu.exec(pos)

    def _load_conv(self, conv_id: str):
        from core.session_manager import get_conversation
        h = get_conversation(conv_id)
        if h:
            self._history = h
            self._current_title = ""
            self._chat.setHtml(self._wrap(self._history))
            self._scroll()
            self._sys("Conversation restaurée.")

    # ── Helpers ───────────────────────────────────────────────────────

    def _append(self, block: str):
        self._history += block
        self._chat.setHtml(self._wrap(self._history))
        self._scroll()

    def _sys(self, text: str):
        self._append(_sys_msg(text))

    def _clear(self):
        self._history = ""
        self._current_title = ""
        self._chat.setHtml("")
        self._sys("Conversation effacée.")

    def _lock(self):
        self._send_btn.hide()
        self._stop_btn.show()
        self._input.setEnabled(False)

    def _unlock(self):
        self._stop_btn.hide()
        self._send_btn.show()
        self._send_btn.setEnabled(True)
        self._input.setEnabled(True)
        self._input.setFocus()

    def _stop_generation(self):
        """Annule la génération en cours proprement."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            # Finalise la bulle avec ce qui a été généré
            if self._stream_buf:
                self._history = self._history.replace(_THINKING, "", 1)
                self._history += _bubble_ai(_md(self._stream_buf) + " <font color='#FF453A'>[arrêté]</font>")
            self._stream_buf = ""
        self._unlock()

    def _scroll(self):
        self._chat.verticalScrollBar().setValue(
            self._chat.verticalScrollBar().maximum()
        )

    def _set_dot(self, c: str):
        self._dot.setStyleSheet(
            f"background-color:{c};border-radius:4px;"
            "min-width:8px;max-width:8px;min-height:8px;max-height:8px;"
        )

    @staticmethod
    def _wrap(body: str) -> str:
        return (
            "<html><head><meta charset='utf-8'/><style>"
            "body{background:#0D0D0F;color:#F2F2F7;"
            "font-family:-apple-system,'Helvetica Neue',Arial,sans-serif;"
            "font-size:13px;margin:0;padding:10px 6px;}"
            "table{border-collapse:collapse;}"
            "td{vertical-align:top;}"
            "</style></head><body>"
            + body + "</body></html>"
        )
