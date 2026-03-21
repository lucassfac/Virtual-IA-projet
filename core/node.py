"""
node.py — Classe mère abstraite pour tous les nœuds du pipeline Neural Forge.

Responsabilités :
  - Appliquer les règles de sécurité (modèle chargé ? input présent ?)
  - Valider le type de DataPacket en entrée
  - Alimenter le logger à chaque étape
  - Exposer get_status() et reset() pour l'IHM
"""

from typing import Optional, Tuple

from core.logger import forge_logger
from core.types import DataPacket, DataType


# ---------------------------------------------------------------------------
# Exceptions métier
# ---------------------------------------------------------------------------


class AIModelMissingError(Exception):
    """Levée quand on lance process() sans avoir chargé de modèle."""


class InputMissingError(Exception):
    """Levée quand on lance process() sans DataPacket en entrée."""


class InvalidDataTypeError(Exception):
    """Levée quand le type du DataPacket reçu n'est pas supporté par ce nœud."""


class ModelLoadError(Exception):
    """Levée quand le chargement du modèle échoue (fichier manquant, corrompu…)."""


# ---------------------------------------------------------------------------
# Classe mère
# ---------------------------------------------------------------------------


class BaseNode:
    """
    Socle commun de tous les blocs de traitement.
    Les nœuds enfants doivent :
      - Déclarer ACCEPTED_TYPES (tuple de DataType)
      - Implémenter _run_inference() → DataPacket
    """

    ACCEPTED_TYPES: Tuple[DataType, ...] = (
        DataType.TEXT,
        DataType.IMAGE_PATH,
        DataType.AUDIO_PATH,
        DataType.ANY,
    )

    def __init__(self, name: str):
        self.name = name
        self.model_loaded: bool = False
        self.input_packet: Optional[DataPacket] = None
        self.output_packet: Optional[DataPacket] = None
        forge_logger.log_node_event(self.name, "INIT", "Nœud créé")

    # ------------------------------------------------------------------
    # Chargement du modèle
    # ------------------------------------------------------------------

    def load_model(self, model_path: str) -> None:
        """
        Valide et enregistre le chargement d'un modèle.
        Les classes enfants appellent super().load_model() après leur propre init.
        """
        if not model_path or not isinstance(model_path, str):
            raise ModelLoadError(
                f"[{self.name}] Chemin de modèle invalide : '{model_path}'"
            )
        forge_logger.log_node_event(self.name, "LOAD_MODEL_OK", model_path)
        self.model_loaded = True

    # ------------------------------------------------------------------
    # Entrée de données
    # ------------------------------------------------------------------

    def set_input(self, packet: DataPacket) -> None:
        """
        Connecte un DataPacket à ce nœud.
        Vérifie le type et enregistre la réception dans le log de traçabilité.
        """
        if not isinstance(packet, DataPacket):
            raise TypeError(
                f"[{self.name}] Attendu DataPacket, reçu {type(packet).__name__}"
            )

        if (
            DataType.ANY not in self.ACCEPTED_TYPES
            and packet.data_type not in self.ACCEPTED_TYPES
        ):
            raise InvalidDataTypeError(
                f"[{self.name}] Type '{packet.data_type.value}' non supporté. "
                f"Acceptés : {[t.value for t in self.ACCEPTED_TYPES]}"
            )

        packet.log_step(self.name, "reçu_en_entrée")
        forge_logger.log_packet(packet, f"{self.name}.set_input")
        self.input_packet = packet

    # ------------------------------------------------------------------
    # Traitement principal
    # ------------------------------------------------------------------

    def process(self) -> DataPacket:
        """
        Point d'entrée public du nœud.
        Applique les règles de sécurité puis délègue à _run_inference().
        """
        if not self.model_loaded:
            forge_logger.error(f"[{self.name}] AIModelMissingError")
            raise AIModelMissingError(
                f"Le nœud '{self.name}' n'a pas de modèle chargé !"
            )

        if self.input_packet is None:
            forge_logger.error(f"[{self.name}] InputMissingError")
            raise InputMissingError(
                f"Le nœud '{self.name}' n'a reçu aucune donnée en entrée !"
            )

        forge_logger.log_node_event(self.name, "PROCESS_START")
        result = self._run_inference()

        if result is not None:
            result.log_step(self.name, "produit_en_sortie")
            forge_logger.log_packet(result, f"{self.name}.output")

        self.output_packet = result
        forge_logger.log_node_event(self.name, "PROCESS_END")
        return result

    # ------------------------------------------------------------------
    # Méthode abstraite
    # ------------------------------------------------------------------

    def _run_inference(self) -> DataPacket:
        """À surcharger obligatoirement dans chaque nœud enfant."""
        raise NotImplementedError(
            f"[{self.name}] _run_inference() doit être implémenté dans les sous-classes."
        )

    # ------------------------------------------------------------------
    # Utilitaires pour l'IHM
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Instantané de l'état du nœud (utile pour l'IHM)."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "model_loaded": self.model_loaded,
            "has_input": self.input_packet is not None,
            "has_output": self.output_packet is not None,
            "accepted_types": [t.value for t in self.ACCEPTED_TYPES],
        }

    def reset(self) -> None:
        """
        Vide les buffers d'entrée/sortie sans décharger le modèle.
        Permet de réutiliser le nœud pour une nouvelle inférence.
        """
        self.input_packet = None
        self.output_packet = None
        forge_logger.log_node_event(self.name, "RESET")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name='{self.name}' "
            f"model_loaded={self.model_loaded}>"
        )
