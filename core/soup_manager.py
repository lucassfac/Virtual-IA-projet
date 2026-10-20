"""
soup_manager.py — Moteur mathématique de fusion d'adaptateurs (Model Soup).
"""

import os
from typing import List

from core.logger import forge_logger

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None


class SoupManager:
    """Utilitaire mathématique pur pour interpoler les poids de multiples adaptateurs LoRA."""

    @staticmethod
    def make_soup(adapter_paths: List[str], output_path: str) -> str:
        """Additionne les tenseurs de plusieurs fichiers .safetensors et les divise par N."""
        if torch is None:
            raise RuntimeError("Les bibliothèques 'torch' et 'safetensors' sont requises pour le Model Soup.")

        if len(adapter_paths) < 2:
            raise ValueError("Il faut au moins 2 adaptateurs pour réaliser une fusion Model Soup.")

        for path in adapter_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Adaptateur introuvable : {path}")

        forge_logger.info(f"[SoupManager] Début de la fusion de {len(adapter_paths)} adaptateurs...")

        # 1. Chargement de l'adaptateur de référence en Float32 (évite tout Overflow)
        base_state = load_file(adapter_paths[0])
        if not base_state:
            raise ValueError(f"Le fichier {adapter_paths[0]} ne contient aucun tenseur.")

        soup_state = {}
        for key, tensor in base_state.items():
            soup_state[key] = tensor.to(torch.float32) if torch.is_floating_point(tensor) else tensor.clone()

        # 2. Addition contrôlée des adaptateurs suivants
        for path in adapter_paths[1:]:
            forge_logger.info(f"[SoupManager] Ingrédient ajouté : {os.path.basename(path)}")
            current_state = load_file(path)

            if set(soup_state.keys()) != set(current_state.keys()):
                raise KeyError(
                    f"Architecture incompatible dans '{os.path.basename(path)}' : "
                    "les adaptateurs n'ont pas été entraînés sur le même modèle de base."
                )

            for key in soup_state.keys():
                if soup_state[key].shape != current_state[key].shape:
                    raise ValueError(
                        f"Dimensions incompatibles pour '{key}' dans '{os.path.basename(path)}' "
                        f"({soup_state[key].shape} vs {current_state[key].shape})."
                    )
                if torch.is_floating_point(soup_state[key]):
                    soup_state[key] += current_state[key].to(torch.float32)

        # 3. Moyenne arithmétique et re-quantification en Float16
        n_adapters = len(adapter_paths)
        for key, tensor in soup_state.items():
            if torch.is_floating_point(tensor):
                soup_state[key] = (tensor / n_adapters).to(torch.float16)

        # 4. Sauvegarde atomique
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_file(soup_state, output_path)

        forge_logger.info(f"[SoupManager] Fusion Model Soup réussie : {output_path}")
        return output_path