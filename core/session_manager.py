"""
session_manager.py — Persistance des conversations et des paramètres.
"""

import json
import os
import time
from typing import List, Optional

from core.model_manager import _CONFIG_FILE


# ── Paramètres (modèle, skill) ─────────────────────────────────────────

def save_last_model(model_path: str, skill_path: str = "") -> None:
    config = _load()
    config["last_model_path"] = model_path
    config["last_skill_path"] = skill_path
    _save(config)


def load_last_model() -> tuple[str, str]:
    """Retourne (model_path, skill_path) du dernier modèle utilisé."""
    config = _load()
    return (
        config.get("last_model_path", ""),
        config.get("last_skill_path", ""),
    )

def save_vision_model(llava_path: str, mmproj_path: str) -> None:
    config = _load()
    config["last_llava_path"] = llava_path
    config["last_mmproj_path"] = mmproj_path
    _save(config)

def load_vision_model() -> tuple[str, str]:
    config = _load()
    return (
        config.get("last_llava_path", ""),
        config.get("last_mmproj_path", ""),
    )


# ── Conversations ─────────────────────────────────────────────────────

def save_conversation(title: str, messages_html: str) -> str:
    """Sauvegarde une conversation."""
    config = _load()
    convs  = config.get("conversations", [])

    conv_id = str(int(time.time()))
    convs.insert(0, {
        "id":        conv_id,
        "title":     title[:60],
        "date":      time.strftime("%d/%m/%Y %H:%M"),
        "html":      messages_html,
    })
    config["conversations"] = convs[:50]
    _save(config)
    return conv_id


def list_conversations() -> List[dict]:
    return _load().get("conversations", [])


def get_conversation(conv_id: str) -> Optional[str]:
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