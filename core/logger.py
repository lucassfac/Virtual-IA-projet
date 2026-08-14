"""
logger.py — Système de logs centralisé pour Neural Forge.
Singleton : une seule instance partagée par tous les nœuds.
Écrit simultanément en console (INFO) et dans un fichier horodaté (DEBUG).
"""

import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import DataPacket


class NeuralForgeLogger:
    """Logger singleton pour l'ensemble du pipeline Neural Forge."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Modification vers le nouveau dossier de stockage
        os.makedirs("storage/logs", exist_ok=True)
        log_filename = f"storage/logs/neural_forge_{time.strftime('%Y%m%d_%H%M%S')}.log"

        self.logger = logging.getLogger("NeuralForge")
        self.logger.setLevel(logging.DEBUG)

        # Console : INFO uniquement
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s — %(message)s")
        )

        # Fichier : DEBUG complet
        fh = logging.FileHandler(log_filename, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
            )
        )

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    # ------------------------------------------------------------------
    # Méthodes de log génériques
    # ------------------------------------------------------------------

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    # ------------------------------------------------------------------
    # Méthodes spécifiques au pipeline
    # ------------------------------------------------------------------

    def log_packet(self, packet: "DataPacket", stage: str) -> None:
        self.logger.debug(
            f"[PACKET] {stage} | id={packet.id[:8]} "
            f"| type={packet.data_type.value} "
            f"| content={str(packet.content)[:40]}"
        )

    def log_node_event(self, node_name: str, event: str, detail: str = "") -> None:
        msg = f"[NODE:{node_name}] {event}"
        if detail:
            msg += f" — {detail}"
        self.logger.info(msg)


# Instance globale (importée par tous les nœuds)
forge_logger = NeuralForgeLogger()
