"""
session_manager.py — Gestionnaire de persistance locale (JSON).
Sauvegarde les conversations (avec mise à jour Auto-Save), les derniers modèles et le token HF.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.logger import forge_logger

# Racine absolue du projet (requise par core/model_manager.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SESSION_DIR = os.path.join(_PROJECT_ROOT, "storage", "sessions")
CONV_FILE = os.path.join(SESSION_DIR, "conversations.json")
CONFIG_FILE = os.path.join(SESSION_DIR, "config.json")


def _ensure_dir() -> None:
    os.makedirs(SESSION_DIR, exist_ok=True)


def _read_json(filepath: str, default: Any) -> Any:
    _ensure_dir()
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        forge_logger.error(f"[SessionManager] Erreur lecture {filepath} : {e}")
        return default


def _write_json(filepath: str, data: Any) -> None:
    _ensure_dir()
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        forge_logger.error(f"[SessionManager] Erreur écriture {filepath} : {e}")


# ── GESTION DES CONVERSATIONS (AUTO-SAVE & HISTORIQUE) ────────────────

def save_conversation(
    title: str,
    html_content: str,
    raw_history: Optional[list] = None,
    conv_id: Optional[str] = None,
) -> str:
    """Crée une nouvelle conversation ou met à jour celle existante si conv_id est fourni."""
    convs: List[Dict[str, Any]] = _read_json(CONV_FILE, default=[])
    target_id = conv_id or f"conv_{uuid.uuid4().hex[:8]}"
    now_str = datetime.now().strftime("%d/%m %H:%M")
    clean_title = (title or "Conversation").strip()[:60]

    updated = False
    for c in convs:
        if c.get("id") == target_id:
            c["html"] = html_content
            c["raw_history"] = raw_history or []
            c["date"] = now_str
            if clean_title and clean_title != "Conversation":
                c["title"] = clean_title
            updated = True
            break

    if not updated:
        convs.insert(0, {
            "id": target_id,
            "title": clean_title,
            "date": now_str,
            "html": html_content,
            "raw_history": raw_history or [],
        })

    _write_json(CONV_FILE, convs)
    return target_id


def list_conversations() -> List[Dict[str, str]]:
    """Retourne les métadonnées des conversations sauvegardées."""
    convs = _read_json(CONV_FILE, default=[])
    return [
        {
            "id": c.get("id", ""),
            "title": c.get("title", "Sans titre"),
            "date": c.get("date", ""),
        }
        for c in convs
        if isinstance(c, dict) and "id" in c
    ]


def get_conversation(conv_id: str) -> Optional[Tuple[str, list]]:
    """Récupère le HTML et l'historique brut d'une conversation par son ID."""
    convs = _read_json(CONV_FILE, default=[])
    for c in convs:
        if c.get("id") == conv_id:
            return c.get("html", ""), c.get("raw_history", [])
    return None


def delete_all_conversations() -> None:
    """Purge l'intégralité de l'historique des conversations."""
    _write_json(CONV_FILE, [])
    forge_logger.info("[SessionManager] Historique des conversations purgé.")


# ── GESTION DE LA CONFIGURATION & DES MODÈLES ─────────────────────────

def save_last_model(model_path: str, skill_path: str = "", mmproj_path: str = "") -> None:
    cfg = _read_json(CONFIG_FILE, default={})
    cfg["last_model"] = {
        "model_path": model_path,
        "skill_path": skill_path,
        "mmproj_path": mmproj_path,
    }
    _write_json(CONFIG_FILE, cfg)


def load_last_model(include_mmproj: bool = False):
    cfg = _read_json(CONFIG_FILE, default={})
    last = cfg.get("last_model", {})
    if isinstance(last, dict):
        if include_mmproj:
            return (
                last.get("model_path", ""),
                last.get("skill_path", ""),
                last.get("mmproj_path", ""),
            )
        return last.get("model_path", ""), last.get("skill_path", "")
    return ("", "", "") if include_mmproj else ("", "")


def save_vision_model(model_path: str, mmproj_path: str) -> None:
    cfg = _read_json(CONFIG_FILE, default={})
    cfg["vision_model"] = {
        "model_path": model_path,
        "mmproj_path": mmproj_path,
    }
    _write_json(CONFIG_FILE, cfg)


def load_vision_model() -> Tuple[str, str]:
    cfg = _read_json(CONFIG_FILE, default={})
    vis = cfg.get("vision_model", {})
    if isinstance(vis, dict):
        return vis.get("model_path", ""), vis.get("mmproj_path", "")
    return "", ""


# ── GESTION DU TOKEN HUGGING FACE ─────────────────────────────────────

def save_hf_token(token: str) -> None:
    cfg = _read_json(CONFIG_FILE, default={})
    cfg["hf_token"] = token.strip()
    _write_json(CONFIG_FILE, cfg)


def load_hf_token() -> str:
    cfg = _read_json(CONFIG_FILE, default={})
    return cfg.get("hf_token", "")


def clear_hf_token() -> None:
    cfg = _read_json(CONFIG_FILE, default={})
    cfg.pop("hf_token", None)
    _write_json(CONFIG_FILE, cfg)