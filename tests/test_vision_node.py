"""
test_vision_node.py — Tests unitaires pour VisionNode (multimodal).

Stratégie : MockVisionNode remplace le moteur LLaVA réel par un MagicMock
pour que les tests s'exécutent sans modèle téléchargé.

Couvre :
  - Rejet des types non-image lors de _run_inference
  - Image manquante (FileNotFoundError)
  - set_prompt()
  - Contenu et métadonnées de la réponse
  - Chargement avec fichier mmproj manquant
  - get_status() après chargement
"""

import os
import sys
import tempfile

import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.node import InvalidDataTypeError, ModelLoadError
from core.nodes.vision_node import VisionNode
from core.types import DataPacket, DataType


# ---------------------------------------------------------------------------
# Mock : VisionNode sans vrai LLaVA
# ---------------------------------------------------------------------------


class MockVisionNode(VisionNode):
    """VisionNode avec moteur LLaVA simulé."""

    def load_model(self, model_path: str, mmproj_path: str) -> None:  # type: ignore
        self.current_model_path = model_path
        self.mmproj_path = mmproj_path
        self.llm_engine = MagicMock()
        self.llm_engine.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Un chat orange allongé sur un canapé en velours."
                    }
                }
            ]
        }
        self.model_loaded = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def node():
    n = MockVisionNode("VisionTest")
    n.load_model("/tmp/llava.gguf", "/tmp/mmproj.gguf")
    return n


@pytest.fixture
def temp_image(tmp_path):
    """Crée un fichier PNG factice pour les tests nécessitant un vrai fichier."""
    img = tmp_path / "test.png"
    # PNG minimal (1×1 pixel blanc)
    img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
        b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return str(img)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vision_inference_rejects_text_packet(node):
    """_run_inference() doit lever InvalidDataTypeError si le paquet est TEXT."""
    node.input_packet = DataPacket(DataType.TEXT, "pas une image")
    with pytest.raises(InvalidDataTypeError):
        node._run_inference()


def test_vision_inference_rejects_missing_image(node):
    """_run_inference() doit lever FileNotFoundError si l'image est absente."""
    node.input_packet = DataPacket(
        DataType.IMAGE_PATH, "/tmp/image_inexistante_12345.jpg"
    )
    with pytest.raises(FileNotFoundError):
        node._run_inference()


def test_vision_inference_returns_text_packet(node, temp_image):
    """Une inférence réussie doit retourner un DataPacket TEXT."""
    node.input_packet = DataPacket(DataType.IMAGE_PATH, temp_image)
    result = node._run_inference()

    assert result.data_type == DataType.TEXT
    assert len(result.content) > 0
    assert result.source == "VisionTest"


def test_vision_inference_metadata(node, temp_image):
    """Le DataPacket de sortie doit contenir image_path et prompt dans ses métadonnées."""
    node.set_prompt("Décris les couleurs dominantes.")
    node.input_packet = DataPacket(DataType.IMAGE_PATH, temp_image)
    result = node._run_inference()

    assert result.metadata["image_path"] == temp_image
    assert result.metadata["prompt"] == "Décris les couleurs dominantes."
    assert result.metadata["model"] == "/tmp/llava.gguf"


def test_set_prompt_updates_question(node):
    """set_prompt() doit mettre à jour text_prompt."""
    node.set_prompt("Quelle est la couleur dominante ?")
    assert node.text_prompt == "Quelle est la couleur dominante ?"


def test_default_prompt_is_set():
    """Le prompt par défaut doit être une description générale."""
    node = MockVisionNode("V")
    node.load_model("/tmp/m.gguf", "/tmp/p.gguf")
    assert "Décris" in node.text_prompt


def test_load_model_missing_mmproj_raises(tmp_path):
    """load_model() sur VisionNode réel doit échouer si mmproj absent.
    On crée un vrai fichier modèle factice pour passer la première vérification
    et atteindre le check du mmproj manquant.
    """
    fake_model = tmp_path / "fake_model.gguf"
    fake_model.write_text("fake")   # Le fichier existe, le mmproj non

    node = VisionNode("VReal")
    with pytest.raises(ModelLoadError, match="mmproj"):
        node.load_model(str(fake_model), "/tmp/missing_mmproj_xyz.gguf")


def test_load_model_missing_model_raises():
    """load_model() sur VisionNode réel doit échouer si modèle absent."""
    node = VisionNode("VReal")
    with pytest.raises(ModelLoadError):
        node.load_model("/tmp/missing_model_xyz.gguf", "/tmp/fake_mmproj.gguf")


def test_get_status_after_load(node):
    """get_status() doit indiquer model_loaded=True après chargement."""
    status = node.get_status()
    assert status["model_loaded"] is True
    assert "image_path" in status["accepted_types"]


def test_accepted_types_only_image():
    """VisionNode ne doit accepter que IMAGE_PATH en entrée."""
    node = MockVisionNode("V")
    node.load_model("/tmp/m.gguf", "/tmp/p.gguf")

    from core.node import InvalidDataTypeError

    with pytest.raises(InvalidDataTypeError):
        node.set_input(DataPacket(DataType.TEXT, "un texte"))

    with pytest.raises(InvalidDataTypeError):
        node.set_input(DataPacket(DataType.AUDIO_PATH, "/audio.wav"))


def test_full_pipeline_via_process(node, temp_image):
    """process() complet via set_input + process() doit fonctionner."""
    packet = DataPacket(DataType.IMAGE_PATH, temp_image)
    node.set_input(packet)
    result = node.process()

    assert result is not None
    assert result.data_type == DataType.TEXT
