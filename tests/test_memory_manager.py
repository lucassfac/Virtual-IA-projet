"""
test_memory_manager.py — Tests unitaires isolés pour le Memory Manager (RAG).
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory_manager import MemoryManager


@patch("core.memory_manager.chromadb")
def test_memory_manager_init_success(mock_chroma):
    """L'initialisation doit créer le client et la collection si ChromaDB est présent."""
    mock_client = MagicMock()
    mock_chroma.PersistentClient.return_value = mock_client
    
    manager = MemoryManager(storage_path="/tmp/fake_mem")
    
    assert manager.is_active is True
    mock_chroma.PersistentClient.assert_called_once()
    mock_client.get_or_create_collection.assert_called_once_with(
        name="neural_forge_conversations",
        embedding_function=manager._embedding_fn,
        metadata={"description": "Mémoire épisodique des conversations de Neural Forge"}
    )


@patch("core.memory_manager.chromadb", None)
def test_memory_manager_disabled_if_no_chromadb():
    """Si ChromaDB n'est pas installé, le manager doit se désactiver gracieusement."""
    manager = MemoryManager()
    assert manager.is_active is False


@patch("core.memory_manager.chromadb")
def test_save_exchange(mock_chroma):
    """Vérifie le formatage et l'insertion d'un échange dans la collection vectorielle."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chroma.PersistentClient.return_value = mock_client

    manager = MemoryManager()
    manager.save_exchange("sess_123", "Bonjour", "Salut !")

    # Vérifie que la méthode add a été appelée avec le bon format
    mock_collection.add.assert_called_once()
    args = mock_collection.add.call_args[1]
    assert "Utilisateur : Bonjour\nAssistant : Salut !" in args["documents"]
    assert args["metadatas"] == [{"session_id": "sess_123"}]
    assert args["ids"] == ["sess_123_1"]


@patch("core.memory_manager.chromadb")
def test_retrieve_context(mock_chroma):
    """Vérifie que la récupération formate correctement le bloc de contexte."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    # On simule un retour de requête ChromaDB
    mock_collection.query.return_value = {
        "documents": [["Utilisateur : Code ?\nAssistant : 4042"]]
    }
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chroma.PersistentClient.return_value = mock_client

    manager = MemoryManager()
    context = manager.retrieve_context("Quel est le code ?", session_id="sess_123")

    assert "[MÉMOIRE ÉPISODIQUE RÉCUPÉRÉE" in context
    assert "Utilisateur : Code ?\nAssistant : 4042" in context
    assert "[FIN DE LA MÉMOIRE RÉCUPÉRÉE]" in context
    
    mock_collection.query.assert_called_once_with(
        query_texts=["Quel est le code ?"],
        n_results=1,
        where={"session_id": "sess_123"}
    )