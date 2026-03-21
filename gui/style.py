"""
style.py — Thème Apple Sonoma pour Neural Forge.
"""

QSS = """

QMainWindow, QDialog {
    background-color: #161618;
}

QWidget {
    background-color: transparent;
    color: #F2F2F7;
    font-family: "SF Pro Text", "Helvetica Neue", "Segoe UI", Arial, sans-serif;
    font-size: 13px;
    font-weight: 400;
}

#sidebar {
    background-color: #1C1C1E;
    border-right: 1px solid rgba(255,255,255,0.06);
}

#appTitle {
    color: #F2F2F7;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.2px;
}

#appSubtitle {
    color: #48484A;
    font-size: 10px;
    letter-spacing: 1.4px;
}

#versionLabel {
    color: #3A3A3C;
    font-size: 10px;
}

#navBtn {
    background-color: transparent;
    color: #8E8E93;
    border: none;
    border-radius: 9px;
    padding: 9px 12px;
    text-align: left;
    font-size: 13px;
    font-weight: 400;
}

#navBtn:hover {
    background-color: rgba(255,255,255,0.06);
    color: #F2F2F7;
}

#navBtn[active="true"] {
    background-color: rgba(255,255,255,0.10);
    color: #FFFFFF;
    font-weight: 500;
}

#contentArea {
    background-color: #161618;
}

#pageTitle {
    color: #FFFFFF;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

#pageSubtitle {
    color: #636366;
    font-size: 13px;
}

#sectionLabel {
    color: #48484A;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.0px;
}

#card {
    background-color: #1C1C1E;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
}

#cardDark {
    background-color: #0D0D0F;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
}

QPushButton#primaryBtn {
    background-color: #0A84FF;
    color: #FFFFFF;
    border: none;
    border-radius: 11px;
    padding: 0 24px;
    font-size: 14px;
    font-weight: 500;
    min-height: 44px;
}

QPushButton#primaryBtn:hover {
    background-color: #3D9BFF;
}

QPushButton#primaryBtn:pressed {
    background-color: #0063D1;
}

QPushButton#primaryBtn:disabled {
    background-color: rgba(10,132,255,0.18);
    color: rgba(255,255,255,0.22);
}

QPushButton#secondaryBtn {
    background-color: rgba(255,255,255,0.08);
    color: #EBEBF5;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 9px;
    padding: 0 16px;
    font-size: 13px;
    min-height: 32px;
}

QPushButton#secondaryBtn:hover {
    background-color: rgba(255,255,255,0.12);
}

QPushButton#iconBtn {
    background-color: rgba(255,255,255,0.08);
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-size: 18px;
    min-width: 42px;
    max-width: 42px;
    min-height: 42px;
    max-height: 42px;
}

QPushButton#iconBtn:hover {
    background-color: #0A84FF;
}

QPushButton#iconBtn:pressed {
    background-color: #0063D1;
}

QPushButton#dangerBtn {
    background-color: transparent;
    color: #FF453A;
    border: 1px solid rgba(255,69,58,0.30);
    border-radius: 11px;
    padding: 0 20px;
    font-size: 13px;
    min-height: 44px;
}

QPushButton#dangerBtn:hover {
    background-color: rgba(255,69,58,0.10);
    border-color: rgba(255,69,58,0.55);
}

QPushButton#dangerBtn:disabled {
    color: #3A3A3C;
    border-color: #2C2C2E;
}

QPushButton#filePickBtn {
    background-color: rgba(255,255,255,0.06);
    color: #636366;
    border: none;
    border-left: 1px solid rgba(255,255,255,0.07);
    border-radius: 0px;
    border-top-right-radius: 10px;
    border-bottom-right-radius: 10px;
    padding: 0 14px;
    font-size: 12px;
    font-weight: 500;
    min-height: 46px;
    min-width: 76px;
    max-width: 76px;
}

QPushButton#filePickBtn:hover {
    background-color: rgba(10,132,255,0.14);
    color: #409CFF;
    border-left-color: rgba(10,132,255,0.25);
}

QLineEdit {
    background-color: rgba(255,255,255,0.05);
    color: #F2F2F7;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
    selection-background-color: #0A84FF;
}

QLineEdit:focus {
    border-color: rgba(10,132,255,0.65);
    background-color: rgba(10,132,255,0.05);
}

QLineEdit:disabled {
    color: #3A3A3C;
    background-color: rgba(255,255,255,0.02);
    border-color: rgba(255,255,255,0.04);
}

QLineEdit:read-only {
    color: #8E8E93;
    background-color: rgba(255,255,255,0.03);
}

QTextEdit, QPlainTextEdit {
    background-color: rgba(255,255,255,0.04);
    color: #F2F2F7;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 12px 14px;
    font-size: 13px;
    selection-background-color: #0A84FF;
}

QTextEdit:focus, QPlainTextEdit:focus {
    border-color: rgba(10,132,255,0.55);
}

QTextBrowser {
    background-color: transparent;
    border: none;
    color: #F2F2F7;
    font-size: 14px;
    padding: 0px;
    selection-background-color: #0A84FF;
}

QProgressBar {
    background-color: rgba(255,255,255,0.08);
    border: none;
    border-radius: 3px;
    max-height: 5px;
    color: transparent;
    font-size: 0px;
}

QProgressBar::chunk {
    background-color: #0A84FF;
    border-radius: 3px;
}

QSpinBox {
    background-color: rgba(255,255,255,0.05);
    color: #F2F2F7;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 13px;
}

QSpinBox:focus { border-color: rgba(10,132,255,0.65); }
QSpinBox::up-button, QSpinBox::down-button { background: transparent; border: none; width: 20px; }

QScrollBar:vertical {
    background-color: transparent;
    width: 5px;
    margin: 4px 1px;
}

QScrollBar::handle:vertical {
    background-color: rgba(255,255,255,0.16);
    border-radius: 2px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover { background-color: rgba(255,255,255,0.28); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; height: 0px; }

QScrollBar:horizontal { background-color: transparent; height: 5px; margin: 1px 4px; }
QScrollBar::handle:horizontal { background-color: rgba(255,255,255,0.16); border-radius: 2px; min-width: 20px; }
QScrollBar::handle:horizontal:hover { background-color: rgba(255,255,255,0.28); }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; width: 0px; }

QStatusBar {
    background-color: #1C1C1E;
    color: #48484A;
    font-size: 11px;
    border-top: 1px solid rgba(255,255,255,0.05);
    min-height: 24px;
}

QStatusBar::item { border: none; }

QFrame[frameShape="4"],
QFrame[frameShape="5"] {
    background-color: rgba(255,255,255,0.06);
    border: none;
    max-height: 1px;
    max-width: 1px;
}

QToolTip {
    background-color: #2C2C2E;
    color: #F2F2F7;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 12px;
}

"""
