"""
app.py — Point d'entrée de Neural Forge.

Lance l'application PyQt6 avec le thème Apple Dark.

Usage :
    python app.py
"""

import sys
import os

# Garantit que Python trouve le package 'core' et 'gui'
sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from gui.main_window import MainWindow
from gui.style import QSS


def main():
    # ── Haute résolution (écrans HiDPI / Retina) ──
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Neural Forge")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("Neural Forge")

    # ── Thème Apple Dark ──
    app.setStyleSheet(QSS)

    # ── Police par défaut ──
    font = QFont()
    font.setFamily("SF Pro Display")
    font.setPointSize(13)
    app.setFont(font)

    # ── Fenêtre principale ──
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
