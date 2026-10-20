"""
test_document_reader.py — Tests de l'extraction et du formatage de documents.
"""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.document_reader import (
    DocumentReadError,
    InjectionStrategy,
    MAX_INJECT_CHARS,
    read_document,
)


@patch("os.path.isfile", return_value=False)
def test_read_missing_document_raises(mock_isfile):
    """Doit lever une erreur si le document n'existe pas."""
    with pytest.raises(DocumentReadError):
        read_document("/path/imaginaire.txt")


@patch("os.path.isfile", return_value=True)
@patch("core.document_reader._extract_text")
def test_short_document_direct_injection(mock_extract, mock_isfile):
    """Un document court doit utiliser la stratégie DIRECT sans altération."""
    mock_extract.return_value = "Ceci est un document court."

    result = read_document("/tmp/doc.txt", direct_threshold=100)

    assert result.strategy == InjectionStrategy.DIRECT
    assert result.content == "Ceci est un document court."
    assert result.original_chars == 27
    assert result.was_summarized is False


@patch("os.path.isfile", return_value=True)
@patch("core.document_reader._extract_text")
def test_long_document_truncated_without_llm(mock_extract, mock_isfile):
    """Un document long sans LLM fourni doit être tronqué avec un avertissement."""
    long_text = "A" * (MAX_INJECT_CHARS + 5000)
    mock_extract.return_value = long_text

    result = read_document("/tmp/doc.txt")

    assert result.strategy == InjectionStrategy.DIRECT
    assert len(result.content) < len(long_text)
    assert "[⚠ Document tronqué" in result.content


@patch("os.path.isfile", return_value=True)
@patch("core.document_reader._extract_text")
def test_long_document_summarized_without_polluting_history(mock_extract, mock_isfile):
    """Le résumé automatique par LLM ne doit pas polluer conversation_history."""
    mock_extract.return_value = " Paragraphe long. " * 1000

    fake_llm = MagicMock()
    fake_llm.model_loaded = True
    fake_llm.conversation_history = [{"role": "user", "content": "Discussion initiale"}]
    fake_llm.llm_engine.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Résumé concis."}}]
    }

    result = read_document("/tmp/long.txt", llm_node=fake_llm, direct_threshold=500)

    assert result.strategy == InjectionStrategy.SUMMARY
    assert result.was_summarized is True
    assert "Résumé concis." in result.content
    assert len(fake_llm.conversation_history) == 1