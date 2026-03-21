"""
trainer_node.py — Nœud de spécialisation LoRA.

Différences vs BaseNode :
  - Accepte deux entrées (dataset + modèle de base) via set_inputs()
  - Surcharge process() avec ses propres règles de sécurité
  - Expose un callback de progression pour la barre de l'IHM PyQt6
  - Génère un nom d'adaptateur dérivé du dataset
"""

import os
import time
from typing import Callable, Optional

from core.logger import forge_logger
from core.node import BaseNode
from core.types import DataPacket, DataType


class TrainingError(Exception):
    """Levée si l'entraînement rencontre une erreur irrécupérable."""


class TrainerNode(BaseNode):
    """
    Usine de spécialisation : dataset + modèle de base → adaptateur .lora
    """

    def __init__(self, name: str = "LoRA Trainer", output_dir: str = "models"):
        super().__init__(name)
        self.dataset_path: Optional[str] = None
        self.base_model_path: Optional[str] = None
        self.output_dir = output_dir
        self.output_adapter_path: Optional[str] = None

        # Callback optionnel : fn(step: int, total: int, loss: float)
        self._progress_callback: Optional[Callable] = None

        # Le Trainer n'a pas besoin d'un modèle d'inférence chargé
        self.model_loaded = True

    # ------------------------------------------------------------------
    # Entrées spécifiques au training
    # ------------------------------------------------------------------

    def set_inputs(
        self, dataset_packet: DataPacket, model_packet: DataPacket
    ) -> None:
        """
        Fournit les deux ingrédients nécessaires à l'entraînement.

        :param dataset_packet: DataPacket TEXT contenant le chemin du .jsonl
        :param model_packet:   DataPacket TEXT contenant le chemin du .gguf
        """
        if dataset_packet.data_type != DataType.TEXT:
            raise TypeError(
                f"[{self.name}] Le dataset doit être de type TEXT "
                f"(chemin vers un .jsonl), reçu : {dataset_packet.data_type.value}"
            )
        if model_packet.data_type != DataType.TEXT:
            raise TypeError(
                f"[{self.name}] Le modèle de base doit être de type TEXT "
                f"(chemin vers un .gguf), reçu : {model_packet.data_type.value}"
            )

        self.dataset_path = dataset_packet.content
        self.base_model_path = model_packet.content

        dataset_packet.log_step(self.name, "reçu_comme_dataset")
        model_packet.log_step(self.name, "reçu_comme_modèle_de_base")

        forge_logger.log_node_event(
            self.name, "INPUTS_SET",
            f"dataset={self.dataset_path}, model={self.base_model_path}"
        )

    def set_progress_callback(self, callback: Callable) -> None:
        """
        Enregistre un callback appelé à chaque étape d'entraînement.
        Signature : callback(step: int, total: int, loss: float)
        Utilisé pour mettre à jour la QProgressBar dans l'IHM.
        """
        self._progress_callback = callback

    # ------------------------------------------------------------------
    # Surcharge de process()
    # ------------------------------------------------------------------

    def process(self) -> DataPacket:
        """
        Règles de sécurité spécifiques au training, puis délégation.
        """
        if not self.dataset_path:
            raise ValueError(
                f"[{self.name}] Aucun dataset fourni. "
                "Appelez set_inputs() avant process()."
            )
        if not self.base_model_path:
            raise ValueError(
                f"[{self.name}] Aucun modèle de base fourni. "
                "Appelez set_inputs() avant process()."
            )

        forge_logger.log_node_event(self.name, "TRAINING_START")
        result = self._run_inference()
        forge_logger.log_node_event(
            self.name, "TRAINING_END", result.content
        )
        return result

    # ------------------------------------------------------------------
    # Entraînement simulé
    # ------------------------------------------------------------------

    def _run_inference(self) -> DataPacket:
        """
        Simule l'entraînement LoRA avec une courbe de loss décroissante.
        Dans la v2, cette méthode appellera llama.cpp ou PEFT.
        """
        forge_logger.info(f"[{self.name}] --- DÉBUT DE L'ENTRAÎNEMENT ---")
        forge_logger.info(f"[{self.name}] Modèle de base : {self.base_model_path}")
        forge_logger.info(f"[{self.name}] Dataset        : {self.dataset_path}")

        os.makedirs(self.output_dir, exist_ok=True)

        # Nom de l'adaptateur dérivé du dataset
        dataset_stem = os.path.splitext(
            os.path.basename(self.dataset_path)
        )[0]
        self.output_adapter_path = os.path.join(
            self.output_dir, f"{dataset_stem}_adapter.lora"
        )

        total_steps = 5
        simulated_losses = [2.45, 1.87, 1.32, 0.94, 0.71]

        for i in range(total_steps):
            time.sleep(0.4)
            loss = simulated_losses[i]
            forge_logger.info(
                f"[{self.name}] Étape {i + 1}/{total_steps} — loss={loss:.4f}"
            )
            if self._progress_callback:
                self._progress_callback(i + 1, total_steps, loss)

        forge_logger.info(
            f"[{self.name}] ✅ Adaptateur sauvegardé : {self.output_adapter_path}"
        )

        return DataPacket(
            DataType.TEXT,
            self.output_adapter_path,
            metadata={
                "base_model": self.base_model_path,
                "dataset": self.dataset_path,
                "steps": total_steps,
                "final_loss": simulated_losses[-1],
            },
            source=self.name,
        )
