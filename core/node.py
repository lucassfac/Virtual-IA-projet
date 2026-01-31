from typing import Optional, Any
from core.types import DataPacket

class AIModelMissingError(Exception):
    """Erreur levée quand on essaie de lancer un nœud sans modèle chargé."""
    pass

class InputMissingError(Exception):
    """Erreur levée quand on essaie de lancer un nœud sans données en entrée."""
    pass

class BaseNode:
    """
    La classe mère de tous les blocs.
    Elle gère la sécurité et l'état du nœud.
    """
    def __init__(self, name: str):
        self.name = name
        self.model_loaded = False  # État du modèle (chargé ou non)
        self.input_packet: Optional[DataPacket] = None # Zone tampon d'entrée
        self.output_packet: Optional[DataPacket] = None # Zone tampon de sortie

    def load_model(self, model_path: str):
        """Simule le chargement d'un modèle."""
        if not model_path:
            raise ValueError("Le chemin du modèle ne peut pas être vide.")
        # Ici on mettrait le code pour charger Llama/Whisper
        print(f"[{self.name}] Modèle chargé depuis {model_path}")
        self.model_loaded = True

    def set_input(self, packet: DataPacket):
        """Connecte une donnée à ce nœud."""
        self.input_packet = packet

    def process(self):
        """
        L'action principale. C'est ici qu'on vérifie tes règles de sécurité.
        """
        # Règle 1 : Pas de modèle ? Erreur.
        if not self.model_loaded:
            raise AIModelMissingError(f"Le nœud '{self.name}' n'a pas de modèle chargé !")

        # Règle 2 : Pas d'input ? Erreur.
        if self.input_packet is None:
            raise InputMissingError(f"Le nœud '{self.name}' n'a reçu aucune donnée en entrée !")

        # Si tout est bon, on exécute la logique spécifique (à définir par les enfants)
        return self._run_inference()

    def _run_inference(self):
        """Méthode abstraite que les enfants (LlamaNode, etc.) devront remplir."""
        raise NotImplementedError("Cette méthode doit être codée dans les nœuds enfants.")