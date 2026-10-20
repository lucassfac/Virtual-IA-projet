"""
test_router.py — Tests unitaires pytest pour PipelineRouter et DocumentRouter.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.router import DocumentRouter, ExecutionMode, PipelineRouter


def test_text_only():
    """Si aucune image n'est envoyée, on reste en texte pur."""
    mode, _ = PipelineRouter.determine_pipeline(
        has_image=False,
        main_is_multimodal=True,
        vision_expert_loaded=True,
        force_orchestration=False,
    )
    assert mode == ExecutionMode.TEXT_ONLY


def test_blocked_error_missing_vision():
    """L'envoi d'une image doit être bloqué si aucun modèle ne peut la lire."""
    mode, _ = PipelineRouter.determine_pipeline(
        has_image=True,
        main_is_multimodal=False,
        vision_expert_loaded=False,
        force_orchestration=False,
    )
    assert mode == ExecutionMode.BLOCKED_ERROR


def test_auto_detect_multimodal():
    """Un Moteur Principal multimodal doit prendre en charge l'image par défaut."""
    mode, _ = PipelineRouter.determine_pipeline(
        has_image=True,
        main_is_multimodal=True,
        vision_expert_loaded=False,
        force_orchestration=False,
    )
    assert mode == ExecutionMode.MULTIMODAL_DIRECT


def test_auto_detect_orchestration():
    """Un Moteur Textuel couplé à un Expert Visuel doit déclencher l'orchestration."""
    mode, _ = PipelineRouter.determine_pipeline(
        has_image=True,
        main_is_multimodal=False,
        vision_expert_loaded=True,
        force_orchestration=False,
    )
    assert mode == ExecutionMode.ORCHESTRATED


def test_force_orchestration():
    """La case à cocher doit forcer l'orchestration même si le Moteur Principal est multimodal."""
    mode, _ = PipelineRouter.determine_pipeline(
        has_image=True,
        main_is_multimodal=True,
        vision_expert_loaded=True,
        force_orchestration=True,
    )
    assert mode == ExecutionMode.ORCHESTRATED


def test_document_router_audit_keywords():
    """Une image demandant l'analyse d'un MCD ou schéma doit passer en HEAVY_TRACK."""
    assert DocumentRouter.audit("exo.png", "Analyse ce MCD en SQL") == "HEAVY_TRACK"
    assert DocumentRouter.audit("facture.jpg", "Lis cette facture") == "FAST_TRACK"