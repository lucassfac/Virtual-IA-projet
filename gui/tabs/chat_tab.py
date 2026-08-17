"""
chat_tab.py — Onglet Chat multimodal.
Architecture révisée : Clean Code (Templates isolés) et Routage Intelligent.
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
from core.session_manager import save_conversation, load_vision_model, load_last_model
from core.router import PipelineRouter, ExecutionMode
from gui.workers import StreamWorker, SwapOrchestratorWorker


# ── Templates HTML / CSS (Isolés pour le Clean Code) ──────────────────

class ChatTemplates:
    """Regroupe tous les rendus HTML pour ne pas polluer la logique métier."""
    
    DOC_ICONS = {".pdf": "PDF", ".docx": "DOC", ".txt": "TXT", ".md": "MD", ".json": "JSON", ".jsonl": "JSONL"}
    THINKING = (
        '<table width="100%" cellpadding="0" cellspacing="0" border="0">'
        '<tr><td align="left">'
        '<table cellpadding="12" cellspacing="0" border="0">'
        '<tr><td bgcolor="#252527">'
        '<font color="#48484A" size="4">&nbsp;·&nbsp;·&nbsp;·&nbsp;</font>'
        '</td></tr></table>'
        '</td><td width="25%"></td></tr></table>'
    )

    @staticmethod
    def markdown_to_html(text: str) -> str:
        t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
        t = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', t)
        t = re.sub(r'`(.+?)`', r'<font face="Courier New" color="#AEAEB2">\1</font>', t)
        t = re.sub(r'^### (.+)$', r'<b>\1</b>', t, flags=re.M)
        t = re.sub(r'^## (.+)$',  r'<b>\1</b>', t, flags=re.M)
        t = re.sub(r'^# (.+)$',   r'<b>\1</b>', t, flags=re.M)
        t = re.sub(r'^[-\*] (.+)$', r'&nbsp;&nbsp;• \1', t, flags=re.M)
        return t.replace('\n', '<br/>')

    @staticmethod
    def bubble_user(text: str) -> str:
        return (
            '<table width="100%" cellpadding="0" cellspacing="0" border="0">'
            '<tr><td width="25%"></td><td align="right">'
            '<table cellpadding="12" cellspacing="0" border="0">'
            '<tr><td bgcolor="#0A84FF" style="border-radius:18px;">'
            f'<font color="#FFFFFF" size="3">{text}</font>'
            '</td></tr></table></td></tr></table>'
        )

    @staticmethod
    def bubble_ai(text: str) -> str:
        return (
            '<table width="100%" cellpadding="0" cellspacing="0" border="0">'
            '<tr><td align="left">'
            '<table cellpadding="12" cellspacing="0" border="0" width="75%">'
            '<tr><td bgcolor="#252527" style="border-radius:18px;">'
            f'<font color="#F2F2F7" size="3">{text}</font>'
            '</td></tr></table></td><td width="25%"></td></tr></table>'
        )

    @staticmethod
    def bubble_img(path: str) -> str:
        return (
            '<table width="100%" cellpadding="0" cellspacing="4" border="0">'
            '<tr><td align="right">'
            f'<img src="{path}" width="200"/>'
            '</td></tr></table>'
        )

    @staticmethod
    def bubble_doc(icon: str, name: str, size: str) -> str:
        return (
            '<table width="100%" cellpadding="0" cellspacing="4" border="0">'
            '<tr><td width="25%"></td><td align="right">'
            '<table cellpadding="8" cellspacing="0" border="0">'
            '<tr><td bgcolor="#1A2A3A" style="border-radius:10px;">'
            f'<font color="#409CFF" size="2"><b>{icon}</b> {html.escape(name)}'
            f' <font color="#48484A">({size})</font></font>'
            '</td></tr></table></td></tr></table>'
        )

    @staticmethod
    def sys_msg(text: str) -> str:
        return (
            '<table width="100%" cellpadding="4" cellspacing="0" border="0">'
            '<tr><td align="center">'
            f'<font color="#3A3A3C" size="2">{text}</font>'
            '</td></tr></table>'
        )

    @staticmethod
    def wrap_html(body: str) -> str:
        return (
            "<html><head><meta charset='utf-8'/><style>"
            "body{background:#0D0D0F;color:#F2F2F7;font-family:-apple-system,sans-serif;font-size:13px;margin:0;padding:10px 6px;}"
            "table{border-collapse:collapse;} td{vertical-align:top;}"
            "</style></head><body>" + body + "</body></html>"
        )


class ChatInput(QLineEdit):
    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
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
        self._attached_image = ""
        self._attached_doc = ""
        self._current_title = ""
        self._build_ui()
        self._sys("Chargez un modèle dans Paramètres pour commencer.")

    # ── UI Construction (Inchangée, respect du layout existant) ────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet("background:#0D0D0F;border-bottom:1px solid rgba(255,255,255,0.06);")
        header.setFixedHeight(50)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.addWidget(self._create_btn("Historique", self._show_history_menu))
        hl.addStretch()
        self._model_lbl = QLabel("Aucun modèle")
        self._model_lbl.setStyleSheet("color:#48484A;font-size:12px;background:transparent;")
        hl.addWidget(self._model_lbl)
        self._dot = QLabel()
        self._set_dot("#3A3A3C")
        hl.addWidget(self._dot)
        hl.addStretch()
        hl.addWidget(self._create_btn("Sauvegarder", self._save_conv))
        hl.addWidget(self._create_btn("Effacer", self._clear))
        root.addWidget(header)

        # Zone Chat
        self._chat = QTextBrowser()
        self._chat.setOpenExternalLinks(True)
        self._chat.setStyleSheet("background:#0D0D0F;border:none;padding:8px 12px;")
        root.addWidget(self._chat, stretch=1)

        # Attachments Bar
        self._attach_bar = QWidget()
        self._attach_bar.setFixedHeight(56)
        self._attach_bar.setStyleSheet("background:#151517;border-top:1px solid rgba(255,255,255,0.05);")
        self._attach_bar.hide()
        ab = QHBoxLayout(self._attach_bar)
        self._attach_icon = QLabel()
        self._attach_icon.setFixedSize(40, 40)
        self._attach_icon.setStyleSheet("background:#2C2C2E;border-radius:10px;color:#0A84FF;font-size:11px;font-weight:600;")
        self._attach_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ab.addWidget(self._attach_icon)
        info_col = QVBoxLayout()
        self._attach_name = QLabel()
        self._attach_name.setStyleSheet("color:#AEAEB2;font-size:12px;font-weight:500;background:transparent;")
        self._attach_meta = QLabel()
        self._attach_meta.setStyleSheet("color:#48484A;font-size:11px;background:transparent;")
        info_col.addWidget(self._attach_name)
        info_col.addWidget(self._attach_meta)
        ab.addLayout(info_col)
        ab.addStretch()
        rm = QPushButton("✕")
        rm.setFixedSize(24, 24)
        rm.setStyleSheet("background:rgba(255,255,255,0.07);color:#636366;border:none;border-radius:12px;font-size:11px;")
        rm.clicked.connect(self._remove_attachment)
        ab.addWidget(rm)
        root.addWidget(self._attach_bar)

        # Input Bar
        input_bar = QWidget()
        input_bar.setFixedHeight(66)
        input_bar.setStyleSheet("background:#0D0D0F;border-top:1px solid rgba(255,255,255,0.06);")
        il = QHBoxLayout(input_bar)
        self._attach_btn = QPushButton("+")
        self._attach_btn.setFixedSize(42, 42)
        self._attach_btn.setStyleSheet("background:rgba(255,255,255,0.07);color:#8E8E93;border:none;border-radius:21px;font-size:22px;font-weight:300;")
        self._attach_btn.clicked.connect(self._show_attach_menu)
        il.addWidget(self._attach_btn)
        self._input = ChatInput()
        self._input.setFixedHeight(42)
        self._input.setStyleSheet("QLineEdit{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.09);border-radius:21px;padding:0 16px;font-size:13px;color:#F2F2F7;}")
        self._input.returnPressed.connect(self._send)
        il.addWidget(self._input)
        self._send_btn = QPushButton("↑")
        self._send_btn.setFixedSize(42, 42)
        self._send_btn.setStyleSheet("background:#0A84FF;color:#fff;border:none;border-radius:21px;font-size:18px;font-weight:600;")
        self._send_btn.clicked.connect(self._send)
        il.addWidget(self._send_btn)
        self._stop_btn = QPushButton("■")
        self._stop_btn.setFixedSize(42, 42)
        self._stop_btn.setStyleSheet("background:rgba(255,69,58,0.18);color:#FF453A;border:1px solid rgba(255,69,58,0.35);border-radius:21px;font-size:16px;font-weight:700;")
        self._stop_btn.clicked.connect(self._stop_generation)
        self._stop_btn.hide()
        il.addWidget(self._stop_btn)
        root.addWidget(input_bar)

    def _create_btn(self, text, callback):
        btn = QPushButton(text)
        btn.setObjectName("secondaryBtn")
        btn.setFixedHeight(28)
        btn.clicked.connect(callback)
        return btn

    # ── API publique ──────────────────────────────────────────────────
    def on_model_loaded(self, model_path: str, skill_path: str):
        name = os.path.basename(model_path)
        skill = f" + {os.path.basename(skill_path)}" if skill_path else ""
        self._model_lbl.setText(f"{name}{skill}")
        self._set_dot("#30D158")
        self._sys(f"Modèle prêt : {name}{skill}")

    # ── Logique de Routage (Nouveau Clean Code) ───────────────────────

    def _send(self):
        text = self._input.text().strip()
        has_img = bool(self._attached_image)
        has_doc = bool(self._attached_doc)

        if not text and not has_img and not has_doc: return
        if self._worker and self._worker.isRunning(): return

        # 1. Utilisation de l'arbre de décision (Le Routeur)
        if has_img:
            self._handle_image_routing(text)
        elif has_doc:
            self._handle_document_routing(text)
        else:
            self._handle_text_routing(text)

    def _handle_image_routing(self, text: str):
        main_is_multimodal = getattr(self.llm_node, "is_multimodal", False)
        vis_paths = load_vision_model()
        vision_expert_loaded = bool(vis_paths[0] and vis_paths[1])

        mode, status_msg = PipelineRouter.determine_pipeline(
            has_image=True,
            main_is_multimodal=main_is_multimodal,
            vision_expert_loaded=vision_expert_loaded,
            force_orchestration=vision_expert_loaded # Si un expert est chargé, on privilégie l'orchestration par défaut
        )

        if mode == ExecutionMode.BLOCKED_ERROR:
            self._sys(f"⚠ {status_msg}")
            return

        self._sys(f"⚙️ {status_msg}")
        self._prepare_ui_for_send(text, has_img=True, has_doc=False)
        img_path = self._attached_image
        self._remove_attachment()

        if mode == ExecutionMode.ORCHESTRATED:
            llm_paths = load_last_model()
            self._stream_buf = ""
            self._worker = SwapOrchestratorWorker(
                self.llm_node, self.vision_node,
                img_path, text or "Que vois-tu sur cette image ?",
                llm_paths, vis_paths
            )
            self._worker.status.connect(lambda s: self._sys(f"<font color='#0A84FF'>{s}</font>"))

        elif mode == ExecutionMode.MULTIMODAL_DIRECT:
            # Traitement natif : l'image est encapsulée directement pour le modèle principal
            packet = DataPacket(DataType.IMAGE_PATH, img_path, metadata={"prompt": text})
            self._stream_buf = ""
            self._worker = StreamWorker(self.llm_node, packet)

        self._start_worker_connections()

    def _handle_document_routing(self, text: str):
        self._prepare_ui_for_send(text, has_img=False, has_doc=True)
        doc_path = self._attached_doc
        self._remove_attachment()
        try:
            result = read_document(doc_path, llm_node=self.llm_node)
            prompt = result.build_prompt(text)
            if result.was_summarized:
                self._sys("Document long — résumé auto activé.")
            self._start_stream(DataPacket(DataType.TEXT, prompt, source="user"))
        except DocumentReadError as e:
            self._remove_thinking()
            self._sys(f"Erreur lecture : {html.escape(str(e))}")
            self._unlock()

    def _handle_text_routing(self, text: str):
        if not getattr(self.llm_node, "model_loaded", False):
            self._sys("Aucun modèle texte chargé — rendez-vous dans Paramètres.")
            return
        self._prepare_ui_for_send(text, has_img=False, has_doc=False)
        self._start_stream(DataPacket(DataType.TEXT, text, source="user"))

    # ── Helpers UI Factorisés ─────────────────────────────────────────

    def _prepare_ui_for_send(self, text: str, has_img: bool, has_doc: bool):
        self._input.clear()
        self._lock()
        
        if has_img:
            self._append(ChatTemplates.bubble_img(self._attached_image))
        if has_doc:
            info = get_file_info(self._attached_doc)
            icon = ChatTemplates.DOC_ICONS.get(info["ext"], "DOC")
            self._append(ChatTemplates.bubble_doc(icon, info["name"], f"{info['size_kb']} Ko"))
        if text:
            self._append(ChatTemplates.bubble_user(ChatTemplates.markdown_to_html(text)))
            if not self._current_title:
                self._current_title = text[:50]
                
        self._append(ChatTemplates.THINKING)

    def _start_worker_connections(self):
        self._worker.token.connect(self._on_token)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_stream_done)
        self._worker.start()

    def _start_stream(self, packet):
        self._stream_buf = ""
        self._worker = StreamWorker(self.llm_node, packet)
        self._start_worker_connections()

    # ── Slots de retour du modèle ─────────────────────────────────────

    @pyqtSlot(str)
    def _on_token(self, token: str):
        if not self._stream_buf:
            self._remove_thinking()
        self._stream_buf += token
        rendered = ChatTemplates.markdown_to_html(self._stream_buf)
        self._chat.setHtml(ChatTemplates.wrap_html(self._history + ChatTemplates.bubble_ai(rendered)))
        self._scroll()

    @pyqtSlot()
    def _on_stream_done(self):
        if self._stream_buf:
            self._history += ChatTemplates.bubble_ai(ChatTemplates.markdown_to_html(self._stream_buf))
        self._stream_buf = ""
        self._unlock()

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._remove_thinking()
        self._sys(f"Erreur : {html.escape(msg)}")
        self._unlock()

    def _remove_thinking(self):
        self._history = self._history.replace(ChatTemplates.THINKING, "", 1)

    # ── Gestion Pièces jointes & Historique (Inchangé) ────────────────

    def _show_attach_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#242426;border:1px solid rgba(255,255,255,0.10);border-radius:12px;padding:6px;color:#F2F2F7;font-size:13px;} QMenu::item{padding:9px 18px;border-radius:8px;} QMenu::item:selected{background:rgba(10,132,255,0.18);} QMenu::separator{background:rgba(255,255,255,0.07);height:1px;margin:4px 8px;}")
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
        path, _ = QFileDialog.getOpenFileName(self, "Joindre une image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not path: return
        self._attached_image, self._attached_doc = path, ""
        self._attach_icon.setPixmap(QPixmap(path).scaled(QSize(40,40), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self._attach_name.setText(os.path.basename(path))
        self._attach_meta.setText("Image")
        self._attach_bar.show()
        self._set_attach_active(True)

    def _attach_document(self):
        path, _ = QFileDialog.getOpenFileName(self, "Joindre un document", "", "Documents (*.txt *.md *.pdf *.docx *.json *.jsonl *.csv)")
        if not path: return
        self._attached_doc, self._attached_image = path, ""
        info = get_file_info(path)
        self._attach_icon.setText(ChatTemplates.DOC_ICONS.get(info["ext"], "DOC"))
        self._attach_icon.setPixmap(QPixmap())
        self._attach_name.setText(info["name"])
        self._attach_meta.setText(f"{info['label']} · {info['size_kb']} Ko")
        self._attach_bar.show()
        self._set_attach_active(True)

    def _remove_attachment(self):
        self._attached_image = ""
        self._attached_doc = ""
        self._attach_bar.hide()
        self._attach_icon.clear()
        self._set_attach_active(False)

    def _set_attach_active(self, on: bool):
        self._attach_btn.setStyleSheet(f"background:{'#0A84FF' if on else 'rgba(255,255,255,0.07)'};color:{'#fff' if on else '#8E8E93'};border:none;border-radius:21px;font-size:22px;font-weight:300;")

    def _save_conv(self):
        if not self._history.strip():
            self._sys("Rien à sauvegarder.")
            return
        title = self._current_title or "Conversation"
        save_conversation(title, self._history)
        self._sys(f"Sauvegardé : « {title} »")

    def _show_history_menu(self):
        from core.session_manager import list_conversations, delete_all_conversations
        convs = list_conversations()
        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#242426;border:1px solid rgba(255,255,255,0.10);border-radius:12px;padding:6px;color:#F2F2F7;font-size:12px;min-width:260px;} QMenu::item{padding:8px 16px;border-radius:7px;} QMenu::item:selected{background:rgba(10,132,255,0.18);} QMenu::separator{background:rgba(255,255,255,0.07);height:1px;margin:4px 8px;}")
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
        menu.exec(btn.mapToGlobal(QPoint(0, btn.height())))

    def _load_conv(self, conv_id: str):
        from core.session_manager import get_conversation
        h = get_conversation(conv_id)
        if h:
            self._history = h
            self._current_title = ""
            self._chat.setHtml(ChatTemplates.wrap_html(self._history))
            self._scroll()
            self._sys("Conversation restaurée.")

    def _append(self, block: str):
        self._history += block
        self._chat.setHtml(ChatTemplates.wrap_html(self._history))
        self._scroll()

    def _sys(self, text: str):
        self._append(ChatTemplates.sys_msg(text))

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
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            if self._stream_buf:
                self._remove_thinking()
                self._history += ChatTemplates.bubble_ai(ChatTemplates.markdown_to_html(self._stream_buf) + " <font color='#FF453A'>[arrêté]</font>")
            self._stream_buf = ""
        self._unlock()

    def _scroll(self):
        self._chat.verticalScrollBar().setValue(self._chat.verticalScrollBar().maximum())

    def _set_dot(self, c: str):
        self._dot.setStyleSheet(f"background-color:{c};border-radius:4px;min-width:8px;max-width:8px;min-height:8px;max-height:8px;")