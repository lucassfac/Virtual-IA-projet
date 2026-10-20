"""
test_session_manager.py — Tests unitaires de la persistance (Conversations, Auto-Save, Config).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import session_manager


@pytest.fixture(autouse=True)
def isolated_session_files(tmp_path, monkeypatch):
    """Redirige tous les fichiers JSON vers un dossier temporaire isolé pendant les tests."""
    sess_dir = str(tmp_path / "sessions")
    conv_file = os.path.join(sess_dir, "conversations.json")
    cfg_file = os.path.join(sess_dir, "config.json")

    monkeypatch.setattr(session_manager, "SESSION_DIR", sess_dir)
    monkeypatch.setattr(session_manager, "CONV_FILE", conv_file)
    monkeypatch.setattr(session_manager, "CONFIG_FILE", cfg_file)
    yield


def test_save_and_load_hf_token():
    """Vérifie la sauvegarde, la lecture et la suppression du token HuggingFace."""
    assert session_manager.load_hf_token() == ""

    session_manager.save_hf_token("  hf_123abc  ")
    assert session_manager.load_hf_token() == "hf_123abc"

    session_manager.clear_hf_token()
    assert session_manager.load_hf_token() == ""


def test_save_and_get_conversation():
    """Vérifie la création d'une conversation et la récupération de son raw_history."""
    raw_hist = [{"role": "user", "content": "Bonjour"}]
    conv_id = session_manager.save_conversation("Titre Test", "<p>HTML</p>", raw_hist)

    convs = session_manager.list_conversations()
    assert len(convs) == 1
    assert convs[0]["id"] == conv_id
    assert convs[0]["title"] == "Titre Test"

    html, restored_hist = session_manager.get_conversation(conv_id)
    assert html == "<p>HTML</p>"
    assert restored_hist == raw_hist


def test_auto_save_updates_existing_conversation_without_duplicate():
    """Si conv_id est fourni (Auto-Save), la conversation existante doit être mise à jour sans doublon."""
    raw_v1 = [{"role": "user", "content": "Msg 1"}]
    conv_id = session_manager.save_conversation("Ma Session", "<p>V1</p>", raw_v1)

    raw_v2 = raw_v1 + [{"role": "assistant", "content": "Rep 1"}]
    same_id = session_manager.save_conversation("Ma Session", "<p>V2</p>", raw_v2, conv_id=conv_id)

    assert same_id == conv_id
    convs = session_manager.list_conversations()
    assert len(convs) == 1  # Aucun doublon créé

    html, restored_hist = session_manager.get_conversation(conv_id)
    assert html == "<p>V2</p>"
    assert len(restored_hist) == 2


def test_save_and_load_last_model_with_mmproj():
    """Vérifie la persistance du triplet (modèle, skill, mmproj)."""
    session_manager.save_last_model("/m.gguf", "/s.safetensors", "/p.mmproj")

    m, s = session_manager.load_last_model()
    assert (m, s) == ("/m.gguf", "/s.safetensors")

    m3, s3, p3 = session_manager.load_last_model(include_mmproj=True)
    assert (m3, s3, p3) == ("/m.gguf", "/s.safetensors", "/p.mmproj")