"""
router.py — Décisionnaire centralisé de Neural Forge.
Délègue les requêtes textuelles ou visuelles vers les bons pipelines.
"""

import os
from enum import Enum, auto
from typing import Tuple

class ExecutionMode(Enum):
    TEXT_ONLY = auto()
    MULTIMODAL_DIRECT = auto()
    ORCHESTRATED = auto()
    BLOCKED_ERROR = auto()

class PipelineRouter:
    
    @staticmethod
    def determine_pipeline(
        has_image: bool, 
        main_is_multimodal: bool, 
        vision_expert_loaded: bool, 
        force_orchestration: bool
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
            elif main_is_multimodal:
                return ExecutionMode.MULTIMODAL_DIRECT, (
                    "Avertissement : L'Orchestration a été ignorée car aucun Expert Visuel n'est chargé. "
                    "Utilisation directe du Moteur Principal Multimodal."
                )
            else:
                return ExecutionMode.BLOCKED_ERROR, "Erreur fatale de configuration visuelle."

        if main_is_multimodal:
            return ExecutionMode.MULTIMODAL_DIRECT, "Moteur Principal Multimodal détecté. Traitement direct priorisé."
        else:
            return ExecutionMode.ORCHESTRATED, "Moteur Principal textuel détecté. Activation de l'Expert Visuel en renfort."

class DocumentRouter:
    COMPLEX_KEYWORDS = ['mcd', 'schéma', 'diagramme', 'graphe', 'architecture', 'latex', 'code']

    @classmethod
    def audit(cls, path: str, user_prompt: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        prompt_lower = user_prompt.lower()

        if ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            if any(kw in prompt_lower for kw in cls.COMPLEX_KEYWORDS):
                return "HEAVY_TRACK"
            return "FAST_TRACK"

        if ext == '.pdf':
            try:
                import fitz
                with fitz.open(path) as doc:
                    if len(doc) > 0:
                        text = doc[0].get_text().strip()
                        if len(text) < 50:
                            return "HEAVY_TRACK"
            except Exception:
                pass
            return "FAST_TRACK"

        return "FAST_TRACK"








"""
from enum import Enum, auto
from typing import Tuple

class ExecutionMode(Enum):
    ""Énumération stricte des pipelines d'exécution possibles.""
    TEXT_ONLY = auto()           # Mode classique (Texte -> LLM)
    MULTIMODAL_DIRECT = auto()   # Le Moteur Principal traite l'image tout seul
    ORCHESTRATED = auto()        # Pipeline Composite (OCR + VLM Expert -> LLM)
    BLOCKED_ERROR = auto()       # Requête impossible (ex: Image sans modèle vision)

class PipelineRouter:
    ""
    Classe utilitaire (sans état) chargée de déterminer le flux d'exécution 
    selon la configuration matérielle chargée et la requête utilisateur.
    ""
    
    @staticmethod
    def determine_pipeline(
        has_image: bool, 
        main_is_multimodal: bool, 
        vision_expert_loaded: bool, 
        force_orchestration: bool
    ) -> Tuple[ExecutionMode, str]:
        ""
        Analyse les entrées et retourne le mode d'exécution avec un message de statut explicatif.
        ""
        
        # Règle 1 : Si l'utilisateur n'envoie pas d'image, on fait du texte pur.
        if not has_image:
            return ExecutionMode.TEXT_ONLY, "Mode texte pur."

        # --- À partir d'ici, nous sommes certains qu'une image est jointe ---

        # Règle 2 : Le Mur (Aucune IA ne sait lire les images)
        if not main_is_multimodal and not vision_expert_loaded:
            return ExecutionMode.BLOCKED_ERROR, (
                "Erreur : Vous tentez de lire une image, mais votre Moteur Principal "
                "est purement textuel et aucun Expert Visuel n'a été chargé."
            )

        # Règle 3 : L'utilisateur force le comportement (Priorité absolue)
        if force_orchestration:
            if vision_expert_loaded:
                return ExecutionMode.ORCHESTRATED, "Orchestration forcée activée par l'utilisateur."
            elif main_is_multimodal:
                # Il a coché la case, mais a oublié de charger l'expert visuel. 
                # On le sauve en utilisant son modèle multimodal principal.
                return ExecutionMode.MULTIMODAL_DIRECT, (
                    "Avertissement : L'Orchestration a été ignorée car aucun Expert Visuel n'est chargé. "
                    "Utilisation directe du Moteur Principal Multimodal."
                )
            else:
                # Cas théoriquement impossible couvert par la Règle 2, mais sécurisé ici
                return ExecutionMode.BLOCKED_ERROR, "Erreur fatale de configuration visuelle."

        # Règle 4 : Auto-Détection Intelligente
        if main_is_multimodal:
            return ExecutionMode.MULTIMODAL_DIRECT, "Moteur Principal Multimodal détecté. Traitement direct priorisé."
        else:
            # Le principal est textuel, mais l'expert visuel est là.
            return ExecutionMode.ORCHESTRATED, "Moteur Principal textuel détecté. Activation de l'Expert Visuel en renfort."
"""