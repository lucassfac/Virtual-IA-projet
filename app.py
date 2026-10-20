"""
app.py — Point d'entrée de Neural Forge.
Configure l'environnement PyQt6, le thème visuel et le filet de sécurité global.
"""

import os
import sys
import traceback

# Garantit que Python trouve les packages 'core' et 'gui' quel que soit le dossier d'exécution
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from core.logger import forge_logger
from gui.main_window import MainWindow
from gui.style import QSS


def _global_exception_hook(exc_type, exc_value, exc_traceback):
    """
    Filet de sécurité global : empêche PyQt6 d'avorter brutalement (core dump)
    en cas d'exception Python non gérée dans un slot, et journalise la trace.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    forge_logger.error(f"[UNHANDLED EXCEPTION]\n{tb_str}")


def configure_application() -> QApplication:
    """Configure et retourne l'instance principale de l'application PyQt."""
    sys.excepthook = _global_exception_hook

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