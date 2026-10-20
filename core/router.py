"""
router.py — Décisionnaire centralisé de Neural Forge.
Délègue les requêtes textuelles ou visuelles vers les bons pipelines.
"""

import os
from enum import Enum, auto
from typing import Tuple


class ExecutionMode(Enum):
    """Énumération stricte des pipelines d'exécution possibles."""
    TEXT_ONLY = auto()
    MULTIMODAL_DIRECT = auto()
    ORCHESTRATED = auto()
    BLOCKED_ERROR = auto()


class PipelineRouter:
    """Détermine le flux d'exécution selon la configuration matérielle et la requête."""

    @staticmethod
    def determine_pipeline(
        has_image: bool,
        main_is_multimodal: bool,
        vision_expert_loaded: bool,
        force_orchestration: bool,
    ) -> Tuple[ExecutionMode, str]:
        if not has_image:
            return ExecutionMode.TEXT_ONLY, "Mode texte pur."

        if not main_is_multimodal and not vision_expert_loaded:
            return ExecutionMode.BLOCKED_ERROR, (
                "Erreur : Vous tentez de lire une image, mais votre Moteur Principal "
                "est purement textuel et aucun Expert Visuel n'a été chargé."
            )

        if force_orchestration:
            if vision_expert_loaded:
                return ExecutionMode.ORCHESTRATED, "Orchestration forcée activée par l'utilisateur."
            if main_is_multimodal:
                return ExecutionMode.MULTIMODAL_DIRECT, (
                    "Avertissement : L'Orchestration a été ignorée car aucun Expert Visuel n'est chargé. "
                    "Utilisation directe du Moteur Principal Multimodal."
                )
            return ExecutionMode.BLOCKED_ERROR, "Erreur fatale de configuration visuelle."

        if main_is_multimodal:
            return ExecutionMode.MULTIMODAL_DIRECT, "Moteur Principal Multimodal détecté. Traitement direct priorisé."

        return ExecutionMode.ORCHESTRATED, "Moteur Principal textuel détecté. Activation de l'Expert Visuel en renfort."


class DocumentRouter:
    """Aiguilleur rapide (OCR vs VLM) pour l'analyse de documents et d'images."""

    COMPLEX_KEYWORDS = ("mcd", "schéma", "diagramme", "graphe", "architecture", "latex", "code")
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    @classmethod
    def audit(cls, path: str, user_prompt: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        prompt_lower = user_prompt.lower()

        if ext in cls.IMAGE_EXTENSIONS:
            if any(kw in prompt_lower for kw in cls.COMPLEX_KEYWORDS):
                return "HEAVY_TRACK"
            return "FAST_TRACK"

        if ext == ".pdf":
            try:
                import fitz
                with fitz.open(path) as doc:
                    if len(doc) > 0 and len(doc[0].get_text().strip()) < 50:
                        return "HEAVY_TRACK"
            except Exception:
                pass
            return "FAST_TRACK"

        return "FAST_TRACK"