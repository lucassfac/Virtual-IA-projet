"""
trainer_node.py — Moteur d'entraînement (SFT) et de Fusion (Model Soup).
"""

import os
import re
import time
from typing import Any, Callable, Dict, Optional

from core.logger import forge_logger
from core.node import BaseNode
from core.soup_manager import SoupManager
from core.types import DataPacket, DataType

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None


class TrainerNode(BaseNode):
    """Nœud orchestrateur pour l'entraînement d'adaptateurs (SFT) et la fusion (SOUP)."""

    ACCEPTED_TYPES = (DataType.ANY,)

    def __init__(self, name: str = "Model Trainer", output_dir: str = "storage/models"):
        super().__init__(name)
        self.output_dir = output_dir
        self._progress_callback: Optional[Callable[[int, int, float], None]] = None
        self.model_loaded = True  # L'entraînement gère sa propre mémoire hors inférence

        # Initialisation de l'arborescence propre dès l'instanciation
        self.base_hf_dir = os.path.join(self.output_dir, "base_hf")
        self.adapters_dir = os.path.join(self.output_dir, "adapters")
        self.soups_dir = os.path.join(self.output_dir, "soups")
        for folder in (self.base_hf_dir, self.adapters_dir, self.soups_dir):
            os.makedirs(folder, exist_ok=True)

    def request_abort(self) -> None:
        """Demande l'arrêt coopératif de l'entraînement en cours."""
        self._abort_requested = True

    def set_progress_callback(self, callback: Callable[[int, int, float], None]) -> None:
        self._progress_callback = callback

    def _report_progress(self, step: int, total: int, loss: float = 0.0) -> None:
        if self._progress_callback:
            self._progress_callback(step, total, loss)

    def _run_inference(self) -> DataPacket:
        """Point d'entrée principal appelé par process()."""
        config = self.input_packet.content
        if not isinstance(config, dict):
            raise TypeError("Le TrainerNode attend un dictionnaire de configuration.")

        mode = config.get("mode")
        if mode == "soup":
            return self._run_soup(config)
        if mode == "sft":
            return self._run_sft(config)

        raise ValueError(f"Mode d'entraînement inconnu : {mode}")

    # ── LOGIQUE SOUP (FUSION MATHÉMATIQUE) ──

    def _run_soup(self, config: Dict[str, Any]) -> DataPacket:
        forge_logger.info(f"[{self.name}] --- DÉBUT DE LA FUSION MODEL SOUP ---")
        adapters = config.get("adapters", [])
        soup_name = config.get("soup_name", "custom_soup").strip()

        if len(adapters) < 2:
            raise ValueError("La fusion SOUP nécessite au moins 2 adaptateurs (.safetensors).")

        # Nettoyage du nom de fichier (KISS & Sécurité OS)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", soup_name) or "model_soup"
        if not safe_name.endswith("_soup"):
            safe_name = f"{safe_name}_soup"
        output_path = os.path.join(self.soups_dir, f"{safe_name}.safetensors")

        self._report_progress(1, 2, 0.0)

        try:
            final_path = SoupManager.make_soup(adapters, output_path)
        except Exception as exc:
            raise RuntimeError(f"Échec de la fusion SOUP : {exc}") from exc

        self._report_progress(2, 2, 0.0)
        forge_logger.info(f"[{self.name}] Fusion terminée : {final_path}")

        return DataPacket(
            DataType.TEXT,
            final_path,
            metadata={"type": "soup_adapter", "ingredients": len(adapters)},
            source=self.name,
        )

    # ── LOGIQUE SFT (ENTRAÎNEMENT D'ADAPTATEUR) ──

    def _run_sft(self, config: Dict[str, Any]) -> DataPacket:
        forge_logger.info(f"[{self.name}] --- DÉBUT DE L'ENTRAÎNEMENT SFT ---")
        dataset = config.get("dataset")
        base_model = config.get("base_model")
        lr_str = config.get("lr", "5e-5")
        epochs_str = config.get("epochs", "1")

        if not dataset or not base_model:
            raise ValueError("L'entraînement SFT requiert un dataset et un modèle Hugging Face.")

        # Extraction propre des tags pour éviter qu'un 2e entraînement n'écrase le 1er
        lr_tag = "2e-4" if "2e-4" in lr_str else ("1e-5" if "1e-5" in lr_str else "5e-5")
        ep_tag = "ep3" if "3" in epochs_str else ("ep5" if "5" in epochs_str else "ep1")

        # SIMULATION TEMPORAIRE (Fake Soup / UI Validation)
        self._abort_requested = False
        total_steps = 100
        for step in range(1, total_steps + 1):
            if self._abort_requested:
                raise RuntimeError("Entraînement interrompu proprement par l'utilisateur.")
            time.sleep(0.04)
            loss = max(0.1, 2.0 - (step * 0.015))
            if step % 10 == 0:
                self._report_progress(step, total_steps, round(loss, 4))

        dataset_name = os.path.splitext(os.path.basename(dataset))[0]
        filename = f"{dataset_name}_{lr_tag}_{ep_tag}_adapter.safetensors"
        output_path = os.path.join(self.adapters_dir, filename)

        if torch is not None:
            # Génération de vrais tenseurs pour tester la fusion SoupManager derrière
            dummy_weights = {
                "lora_layer_1": torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32),
                "lora_layer_2": torch.tensor([0.8, 0.9], dtype=torch.float32),
            }
            save_file(dummy_weights, output_path)
        else:
            with open(output_path, "w", encoding="utf-8") as file_out:
                file_out.write("Simulated Weights")

        forge_logger.info(f"[{self.name}] Entraînement SFT terminé : {output_path}")

        return DataPacket(
            DataType.TEXT,
            output_path,
            metadata={
                "type": "lora_adapter",
                "dataset": dataset,
                "base_model": base_model,
            },
            source=self.name,
        )