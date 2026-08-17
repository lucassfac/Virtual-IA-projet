"""
app.py — Point d'entrée de Neural Forge.
Lance l'application PyQt6
"""

import sys
import os

# Garantit que Python trouve les packages 'core' et 'gui'
sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from gui.main_window import MainWindow
from gui.style import QSS


def configure_application() -> QApplication:
    """Configure et retourne l'instance principale de l'application PyQt."""
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Neural Forge")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("Neural Forge")

    # Thème et typographie
    app.setStyleSheet(QSS)
    
    font = QFont()
    font.setFamily("SF Pro Display")
    font.setPointSize(13)
    app.setFont(font)

    return app


def main():
    app = configure_application()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
