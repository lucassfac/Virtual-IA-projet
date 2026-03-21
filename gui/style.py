"""
style.py — Thème Apple Dark pour Neural Forge.

Palette inspirée de macOS Ventura / Sonoma dark mode :
  - Background :  #000000 (noir pur) / #1C1C1E (elevated)
  - Surface :     #2C2C2E (cards, inputs)
  - Border :      #3A3A3C (séparateurs)
  - Accent :      #0A84FF (bleu Apple)
  - Success :     #30D158
  - Warning :     #FF9F0A
  - Danger :      #FF453A
  - Text :        #FFFFFF / #8E8E93 (secondaire)
"""

QSS = """

/* ══════════════════════════════════════════════════
   BASE
══════════════════════════════════════════════════ */

QMainWindow {
    background-color: #000000;
}

QWidget {
    background-color: transparent;
    color: #FFFFFF;
    font-family: "SF Pro Display", "Helvetica Neue", "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

/* ══════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════ */

#sidebar {
    background-color: #1C1C1E;
    border-right: 1px solid #2C2C2E;
}

#appTitle {
    color: #FFFFFF;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.3px;
}

#appSubtitle {
    color: #636366;
    font-size: 10px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

#versionLabel {
    color: #48484A;
    font-size: 10px;
}

/* ══════════════════════════════════════════════════
   NAV BUTTONS
══════════════════════════════════════════════════ */

#navBtn {
    background-color: transparent;
    color: #8E8E93;
    border: none;
    border-radius: 10px;
    padding: 11px 14px;
    text-align: left;
    font-size: 13px;
    font-weight: 400;
}

#navBtn:hover {
    background-color: #2C2C2E;
    color: #EBEBF5;
}

#navBtn[active="true"] {
    background-color: #2C2C2E;
    color: #FFFFFF;
    font-weight: 500;
}

/* ══════════════════════════════════════════════════
   CONTENT AREA
══════════════════════════════════════════════════ */

#contentArea {
    background-color: #000000;
}

#pageTitle {
    color: #FFFFFF;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.4px;
}

#sectionLabel {
    color: #636366;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

/* ══════════════════════════════════════════════════
   CARDS
══════════════════════════════════════════════════ */

#card {
    background-color: #1C1C1E;
    border: 1px solid #2C2C2E;
    border-radius: 14px;
}

#cardDark {
    background-color: #0A0A0A;
    border: 1px solid #2C2C2E;
    border-radius: 14px;
}

/* ══════════════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════════ */

QPushButton#primaryBtn {
    background-color: #0A84FF;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 11px 22px;
    font-size: 14px;
    font-weight: 500;
    min-height: 22px;
}

QPushButton#primaryBtn:hover {
    background-color: #409CFF;
}

QPushButton#primaryBtn:pressed {
    background-color: #0060DF;
}

QPushButton#primaryBtn:disabled {
    background-color: #1C1C1E;
    color: #3A3A3C;
    border: 1px solid #2C2C2E;
}

QPushButton#secondaryBtn {
    background-color: #2C2C2E;
    color: #EBEBF5;
    border: 1px solid #3A3A3C;
    border-radius: 10px;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 400;
}

QPushButton#secondaryBtn:hover {
    background-color: #3A3A3C;
    border-color: #48484A;
}

QPushButton#secondaryBtn:pressed {
    background-color: #1C1C1E;
}

QPushButton#iconBtn {
    background-color: #2C2C2E;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 10px;
    font-size: 16px;
    min-width: 40px;
    max-width: 40px;
    min-height: 40px;
    max-height: 40px;
}

QPushButton#iconBtn:hover {
    background-color: #0A84FF;
}

QPushButton#iconBtn:pressed {
    background-color: #0060DF;
}

QPushButton#dangerBtn {
    background-color: transparent;
    color: #FF453A;
    border: 1px solid #3A3A3C;
    border-radius: 10px;
    padding: 9px 18px;
    font-size: 13px;
}

QPushButton#dangerBtn:hover {
    background-color: rgba(255, 69, 58, 0.12);
    border-color: #FF453A;
}

QPushButton#browseBtn {
    background-color: #2C2C2E;
    color: #8E8E93;
    border: 1px solid #3A3A3C;
    border-radius: 8px;
    padding: 9px 14px;
    font-size: 13px;
    min-width: 36px;
    max-width: 36px;
}

QPushButton#browseBtn:hover {
    background-color: #3A3A3C;
    color: #FFFFFF;
}

/* ══════════════════════════════════════════════════
   INPUTS
══════════════════════════════════════════════════ */

QLineEdit {
    background-color: #1C1C1E;
    color: #FFFFFF;
    border: 1px solid #3A3A3C;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
    selection-background-color: #0A84FF;
}

QLineEdit:focus {
    border-color: #0A84FF;
    background-color: #1C1C1E;
}

QLineEdit:disabled {
    color: #48484A;
    background-color: #1C1C1E;
    border-color: #2C2C2E;
}

QTextEdit, QPlainTextEdit {
    background-color: #1C1C1E;
    color: #FFFFFF;
    border: 1px solid #3A3A3C;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
    selection-background-color: #0A84FF;
}

QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #0A84FF;
}

QTextBrowser {
    background-color: transparent;
    border: none;
    color: #FFFFFF;
    font-size: 14px;
    padding: 0px;
    selection-background-color: #0A84FF;
}

/* ══════════════════════════════════════════════════
   PROGRESS BAR
══════════════════════════════════════════════════ */

QProgressBar {
    background-color: #2C2C2E;
    border: none;
    border-radius: 4px;
    max-height: 6px;
    text-align: center;
    color: transparent;
    font-size: 0px;
}

QProgressBar::chunk {
    background-color: #0A84FF;
    border-radius: 4px;
}

/* ══════════════════════════════════════════════════
   SCROLLBARS (macOS style)
══════════════════════════════════════════════════ */

QScrollBar:vertical {
    background-color: transparent;
    width: 6px;
    margin: 4px 2px;
}

QScrollBar::handle:vertical {
    background-color: #3A3A3C;
    border-radius: 3px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background-color: #636366;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}

QScrollBar:horizontal {
    background-color: transparent;
    height: 6px;
    margin: 2px 4px;
}

QScrollBar::handle:horizontal {
    background-color: #3A3A3C;
    border-radius: 3px;
    min-width: 24px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #636366;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: none;
    width: 0px;
}

/* ══════════════════════════════════════════════════
   SPIN BOX
══════════════════════════════════════════════════ */

QSpinBox {
    background-color: #1C1C1E;
    color: #FFFFFF;
    border: 1px solid #3A3A3C;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 13px;
}

QSpinBox:focus {
    border-color: #0A84FF;
}

QSpinBox::up-button, QSpinBox::down-button {
    background: transparent;
    border: none;
    width: 18px;
}

/* ══════════════════════════════════════════════════
   STATUS BAR
══════════════════════════════════════════════════ */

QStatusBar {
    background-color: #1C1C1E;
    color: #636366;
    font-size: 11px;
    border-top: 1px solid #2C2C2E;
    padding: 2px 8px;
}

QStatusBar::item {
    border: none;
}

/* ══════════════════════════════════════════════════
   SEPARATOR
══════════════════════════════════════════════════ */

QFrame[frameShape="4"],
QFrame[frameShape="5"] {
    background-color: #2C2C2E;
    border: none;
    max-height: 1px;
    max-width: 1px;
}

/* ══════════════════════════════════════════════════
   TOOLTIP
══════════════════════════════════════════════════ */

QToolTip {
    background-color: #2C2C2E;
    color: #FFFFFF;
    border: 1px solid #3A3A3C;
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 12px;
}

/* ══════════════════════════════════════════════════
   BADGES / STATUS DOTS
══════════════════════════════════════════════════ */

#statusDot {
    min-width: 8px;
    max-width: 8px;
    min-height: 8px;
    max-height: 8px;
    border-radius: 4px;
}

#statusLabel {
    color: #8E8E93;
    font-size: 12px;
}

"""
