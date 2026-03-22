"""
session_manager.py — Persistance des conversations et des paramètres.

Sauvegarde dans .neural_forge_config.json (déjà utilisé pour le token HF) :
  - last_model_path / last_lora_path  → rechargement auto au démarrage
  - conversations[]                   → historique des discussions
"""

import json
import os
import time
from typing import List, Optional

from core.model_manager import _CONFIG_FILE


# ── Paramètres (modèle, lora) ─────────────────────────────────────────

def save_last_model(model_path: str, lora_path: str = "") -> None:
    config = _load()
    config["last_model_path"] = model_path
    config["last_lora_path"]  = lora_path
    _save(config)


def load_last_model() -> tuple[str, str]:
    """Retourne (model_path, lora_path) du dernier modèle utilisé."""
    config = _load()
    return (
        config.get("last_model_path", ""),
        config.get("last_lora_path",  ""),
    )


# ── Conversations ─────────────────────────────────────────────────────

def save_conversation(title: str, messages_html: str) -> str:
    """
    Sauvegarde une conversation.
    Retourne l'id de la conversation.
    """
    config = _load()
    convs  = config.get("conversations", [])

    conv_id = str(int(time.time()))
    convs.insert(0, {
        "id":        conv_id,
        "title":     title[:60],
        "date":      time.strftime("%d/%m/%Y %H:%M"),
        "html":      messages_html,
    })
    # Garder les 50 dernières conversations
    config["conversations"] = convs[:50]
    _save(config)
    return conv_id


def list_conversations() -> List[dict]:
    """Retourne la liste des conversations sauvegardées (plus récentes en premier)."""
    return _load().get("conversations", [])


def get_conversation(conv_id: str) -> Optional[str]:
    """Retourne le HTML d'une conversation par son id."""
    for c in _load().get("conversations", []):
        if c["id"] == conv_id:
            return c["html"]
    return None


def delete_conversation(conv_id: str) -> None:
    config = _load()
    config["conversations"] = [
        c for c in config.get("conversations", [])
        if c["id"] != conv_id
    ]
    _save(config)


def delete_all_conversations() -> None:
    config = _load()
    config["conversations"] = []
    _save(config)


# ── Helpers ───────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(config: dict) -> None:
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
