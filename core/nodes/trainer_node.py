"""
trainer_node.py — Compilateur de Compétences (Skill Compiler).

Transforme un dataset brut (TXT, JSONL) en un module d'expertise portable (.skill).
Ce module sera ensuite injecté via RAG dans le LLMNode.
"""

import os
import time
import json
from typing import Callable, Optional

from core.logger import forge_logger
from core.node import BaseNode
from core.types import DataPacket, DataType


class TrainerNode(BaseNode):
    """
    Usine de spécialisation : dataset brut → module d'expertise (.skill)
    """

    def __init__(self, name: str = "Skill Compiler", output_dir: str = "storage/models"):
        super().__init__(name)
        self.dataset_path: Optional[str] = None
        self.base_model_path: Optional[str] = None # Conservé pour compatibilité IHM
        self.output_dir = output_dir
        self.output_skill_path: Optional[str] = None

        self._progress_callback: Optional[Callable] = None
        self.model_loaded = True # Pas besoin de charger de LLM pour compiler un texte

    # ------------------------------------------------------------------
    # Entrées
    # ------------------------------------------------------------------

    def set_inputs(self, dataset_packet: DataPacket, model_packet: DataPacket) -> None:
        """
        Reçoit le dataset depuis l'interface. 
        (Le model_packet est ignoré dans cette version RAG, mais gardé pour l'IHM).
        """
        self.dataset_path = dataset_packet.content
        self.base_model_path = model_packet.content
        
        forge_logger.log_node_event(self.name, "INPUTS_SET", f"dataset={self.dataset_path}")

    def set_progress_callback(self, callback: Callable) -> None:
        self._progress_callback = callback

    # ------------------------------------------------------------------
    # Traitement
    # ------------------------------------------------------------------

    def process(self) -> DataPacket:
        if not self.dataset_path:
            raise ValueError(f"[{self.name}] Aucun dataset fourni.")

        forge_logger.log_node_event(self.name, "COMPILATION_START")
        result = self._compile_skill()
        forge_logger.log_node_event(self.name, "COMPILATION_END", result.content)
        return result

    # ------------------------------------------------------------------
    # Compilation du Skill (RAG)
    # ------------------------------------------------------------------

    def _compile_skill(self) -> DataPacket:
        forge_logger.info(f"[{self.name}] --- DÉBUT DE LA COMPILATION DU SKILL ---")
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if os.path.isabs(self.output_dir):
            abs_output_dir = self.output_dir
        else:
            abs_output_dir = os.path.join(project_root, self.output_dir.rstrip("/"))

        os.makedirs(abs_output_dir, exist_ok=True)

        # 1. Lecture du dataset
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
        except Exception as e:
            raise RuntimeError(f"Impossible de lire le dataset : {e}")

        # 2. Animation de la barre de progression pour l'IHM (Simulation du formatage)
        total_steps = 4
        steps_info = ["Analyse du texte...", "Nettoyage des données...", "Structuration de la base de connaissances...", "Génération du fichier .skill..."]
        
        for i in range(total_steps):
            time.sleep(0.5) # Simule le temps de traitement
            forge_logger.info(f"[{self.name}] Étape {i + 1}/{total_steps} — {steps_info[i]}")
            if self._progress_callback:
                self._progress_callback(i + 1, total_steps, 0.0)

        # 3. Création du fichier .skill
        dataset_stem = os.path.splitext(os.path.basename(self.dataset_path))[0]
        self.output_skill_path = os.path.join(abs_output_dir, f"{dataset_stem}.skill")

        skill_data = {
            "neural_forge_version": "2.0",
            "type": "rag_skill",
            "name": dataset_stem.replace("_", " ").title(),
            "description": f"Compétence générée automatiquement à partir de {os.path.basename(self.dataset_path)}",
            "system_prompt": (
                f"Tu es désormais un expert absolu sur le sujet : '{dataset_stem}'. "
                "Utilise les connaissances factuelles ci-dessous pour répondre aux questions de l'utilisateur de manière précise. "
                "Si la réponse ne se trouve pas dans ces connaissances, utilise ton intelligence générale mais précise-le."
            ),
            "knowledge_base": raw_content[:15000] # On limite à ~15k caractères par sécurité
        }

        with open(self.output_skill_path, "w", encoding="utf-8") as f:
            json.dump(skill_data, f, indent=2, ensure_ascii=False)

        forge_logger.info(f"[{self.name}] ✅ Compétence forgée avec succès : {self.output_skill_path}")

        return DataPacket(
            DataType.TEXT,
            self.output_skill_path,
            metadata={
                "type": "skill",
                "skill_name": skill_data["name"]
            },
            source=self.name,
        )