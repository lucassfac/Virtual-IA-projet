"""
test_file_utils.py — Tests unitaires isolés pour la validation et sécurisation des chemins.
"""

import os
import sys
from unittest.mock import patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.node import ModelLoadError
from core.utils.file_utils import check_safe_path


def test_empty_path_raises_error():
    """Un chemin vide doit lever une erreur immédiate."""
    with pytest.raises(ModelLoadError, match="chemin vide"):
        check_safe_path("", "TestLabel")


@patch("os.path.exists", return_value=False)
def test_missing_file_raises_error(mock_exists):
    """Un fichier inexistant doit lever une erreur."""
    with pytest.raises(ModelLoadError, match="introuvable"):
        check_safe_path("/chemin/imaginaire.txt", "Fichier")


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
@patch("os.access", return_value=False)
def test_unreadable_file_raises_error(mock_access, mock_isfile, mock_exists):
    """Un fichier sans permissions de lecture doit lever une erreur."""
    with pytest.raises(ModelLoadError, match="non lisible"):
        check_safe_path("/root/secret.txt", "Fichier")


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
@patch("os.access", return_value=True)
def test_safe_path_returned_as_is(mock_access, mock_isfile, mock_exists):
    """Un chemin sans espace et valide est retourné tel quel."""
    result = check_safe_path("/opt/models/model.gguf", "Modele")
    assert result == "/opt/models/model.gguf"


@patch("os.path.isfile", return_value=True)
@patch("os.access", return_value=True)
@patch("os.symlink")
@patch("os.path.islink", return_value=False)
def test_path_with_spaces_creates_symlink(mock_islink, mock_symlink, mock_access, mock_isfile):
    """Un chemin avec des espaces doit générer un symlink unique sans espace."""
    original_path = "/home/user/Mon Dossier/model.gguf"

    with patch("os.path.exists", side_effect=lambda p: p == original_path):
        result = check_safe_path(original_path, "Modele Test")

    mock_symlink.assert_called_once()
    assert " " not in os.path.basename(result)
    assert result.endswith(".gguf")
    assert "nf_modele_test" in os.path.basename(result)


@patch("os.path.isfile", return_value=True)
@patch("os.access", return_value=True)
@patch("os.symlink", side_effect=OSError("Symlink bloqué"))
def test_symlink_failure_returns_original_path(mock_symlink, mock_access, mock_isfile):
    """Si la création du symlink échoue (OSError), retourne le chemin d'origine."""
    original_path = "/home/user/Mon Dossier/model.gguf"

    with patch("os.path.exists", side_effect=lambda p: p == original_path):
        result = check_safe_path(original_path, "Modele")

    assert result == original_path