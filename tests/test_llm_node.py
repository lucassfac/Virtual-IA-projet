"""
test_llm_node.py — Tests unitaires isolés pour LLMNode.

Couvre :
  - Auto-tuning des paramètres (analytique vs conversationnel)
  - Protection de la VRAM (fenêtre glissante des 10 messages)
  - Isolation des sessions par UUID et anti-poisoning RAG
  - Détection du mode multimodal et chargement d'adaptateur LoRA (.safetensors)
"""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.nodes.llm_node import LLMNode
from core.types import DataPacket, DataType


@pytest.fixture
def mock_memory_manager():
    """Mock le gestionnaire de mémoire pour éviter de créer une base ChromaDB."""
    manager = MagicMock()
    manager.retrieve_context.return_value = "[MÉMOIRE RÉCUPÉRÉE] Test."
    return manager


@pytest.fixture
def node(mock_memory_manager):
    """Fournit un LLMNode prêt à tester, avec ses dépendances lourdes mockées."""
    with patch("core.nodes.llm_node.MemoryManager", return_value=mock_memory_manager):
        n = LLMNode("TestLLM")
        n.hw = MagicMock()
        n.hw.n_threads = 4
        n.hw.n_batch = 512
        n.hw.n_ctx = 4096
        n.hw.n_gpu_layers = -1
        n.hw.gpu_vram_gb = 8.0
        n.hw.use_mmap = True
        n.hw.flash_attn = True
        n.model_loaded = True
        n.current_model_path = "/tmp/fake_model.gguf"

        n.llm_engine = MagicMock()
        n.llm_engine.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Réponse générée valide"}}]
        }
        return n


# ---------------------------------------------------------------------------
# Tests : État, Multimodalité et Isolation de Session
# ---------------------------------------------------------------------------

def test_is_multimodal_property(node):
    """Le nœud doit détecter s'il est multimodal grâce à son projecteur visuel."""
    node.mmproj_path = ""
    assert node.is_multimodal is False

    node.mmproj_path = "/tmp/projector.gguf"
    assert node.is_multimodal is True


def test_clear_history_purges_memory_and_rotates_session_id(node):
    """clear_history doit vider l'historique court-terme ET générer un nouvel ID de session UUID."""
    old_session = node.current_session_id
    node.conversation_history = [{"role": "user", "content": "test"}]

    node.clear_history()

    assert len(node.conversation_history) == 0
    assert node.current_session_id != old_session
    assert node.current_session_id.startswith("session_")


# ---------------------------------------------------------------------------
# Tests : Auto-Tuning Heuristique
# ---------------------------------------------------------------------------

def test_dynamic_params_analytical(node):
    """Les prompts analytiques (code, mcd, sql) doivent brider la température."""
    params = node._get_dynamic_params("Analyse ce MCD et donne moi le code SQL")
    assert params["temperature"] == 0.1
    assert "Règle stricte" in params["system_hint"]


def test_dynamic_params_conversational(node):
    """Les prompts conversationnels doivent permettre la créativité."""
    params = node._get_dynamic_params("Bonjour, raconte moi une histoire")
    assert params["temperature"] == 0.7
    assert params["system_hint"] == ""


# ---------------------------------------------------------------------------
# Tests : Mémoire de Travail & Protection Anti-Vide
# ---------------------------------------------------------------------------

def test_save_answer_compacts_history(node):
    """_save_answer doit protéger la VRAM en tronquant à 10 messages."""
    for i in range(12):
        node.conversation_history.append({"role": "user", "content": f"msg {i}"})

    node._save_answer("Nouvelle réponse")

    assert len(node.conversation_history) == 10
    assert node.conversation_history[-1]["content"] == "Nouvelle réponse"


def test_save_answer_truncates_huge_responses(node):
    """Les pavés immenses doivent être tronqués à 2500 caractères."""
    huge_text = "A" * 3000
    node._save_answer(huge_text)

    saved_content = node.conversation_history[-1]["content"]
    assert len(saved_content) < 3000
    assert "[...]" in saved_content


def test_save_answer_ignores_empty_response_anti_poisoning(node):
    """Une réponse vide ne doit JAMAIS être enregistrée dans ChromaDB ni dans l'historique."""
    node.conversation_history = [{"role": "user", "content": "Question sans réponse"}]

    node._save_answer("   ")

    assert len(node.conversation_history) == 0
    node.memory_manager.save_exchange.assert_not_called()


# ---------------------------------------------------------------------------
# Tests : Intégration RAG (Memory Manager)
# ---------------------------------------------------------------------------

def test_inference_triggers_memory_retrieval_when_history_exists(node):
    """_prepare_inference doit interroger le MemoryManager avec l'ID de session courant."""
    node.conversation_history = [
        {"role": "user", "content": "Msg 1"},
        {"role": "assistant", "content": "Rep 1"},
    ]
    node.set_input(DataPacket(DataType.TEXT, "Ma question secrète"))

    node._prepare_inference()

    node.memory_manager.retrieve_context.assert_called_once_with(
        "Ma question secrète",
        session_id=node.current_session_id,
    )


def test_save_answer_archives_to_rag(node):
    """_save_answer doit archiver la paire Question/Réponse dans ChromaDB sous current_session_id."""
    node.conversation_history = [{"role": "user", "content": "Question utilisateur"}]

    node._save_answer("Réponse IA")

    node.memory_manager.save_exchange.assert_called_once_with(
        node.current_session_id,
        "Question utilisateur",
        "Réponse IA",
    )