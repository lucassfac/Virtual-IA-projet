"""
test_vision_node.py — Tests unitaires pour VisionNode (multimodal).
"""

import os
import sys
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.node import InvalidDataTypeError, ModelLoadError
from core.nodes.vision_node import VisionNode
from core.types import DataPacket, DataType


class MockVisionNode(VisionNode):
    """VisionNode avec moteur VLM simulé pour exécution rapide hors-ligne."""

    def load_model(self, model_path: str, mmproj_path: str) -> None:  # type: ignore
        self.current_model_path = model_path
        self.mmproj_path = mmproj_path
        self.llm_engine = MagicMock()
        self.llm_engine.create_chat_completion.return_value = {
            "choices": [
                {"message": {"content": "Un chat orange allongé sur un canapé en velours."}}
            ]
        }
        self.model_loaded = True


@pytest.fixture
def node():
    n = MockVisionNode("VisionTest")
    n.load_model("/tmp/llava.gguf", "/tmp/mmproj.gguf")
    return n


@pytest.fixture
def temp_image(tmp_path):
    """Crée un fichier PNG valide (1×1 pixel) pour les tests d'encodage Base64."""
    img = tmp_path / "test.png"
    img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
        b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return str(img)


def test_vision_set_input_rejects_non_image(node):
    """set_input() doit lever InvalidDataTypeError pour TEXT et AUDIO_PATH."""
    with pytest.raises(InvalidDataTypeError):
        node.set_input(DataPacket(DataType.TEXT, "pas une image"))
    with pytest.raises(InvalidDataTypeError):
        node.set_input(DataPacket(DataType.AUDIO_PATH, "/audio.wav"))


def test_vision_inference_rejects_missing_image(node):
    """_run_inference() doit lever FileNotFoundError si le fichier image n'existe pas."""
    node.input_packet = DataPacket(DataType.IMAGE_PATH, "/tmp/image_inexistante_12345.jpg")
    with pytest.raises(FileNotFoundError):
        node._run_inference()


def test_vision_inference_metadata_and_analytical_tuning(node, temp_image):
    """Vérifie les métadonnées et l'abaissement de température (0.1) sur un prompt analytique."""
    node.set_prompt("Analyse ce MCD et lis le code SQL.")
    node.set_input(DataPacket(DataType.IMAGE_PATH, temp_image))
    result = node.process()

    assert result.metadata["image_path"] == temp_image
    assert result.metadata["temp_used"] == 0.1
    assert result.metadata["model"] == "/tmp/llava.gguf"


def test_set_prompt_and_default_prompt(node):
    """Vérifie le prompt par défaut et sa modification via set_prompt()."""
    fresh_node = MockVisionNode("V")
    assert "Décris" in fresh_node.text_prompt

    node.set_prompt("Quelle est la couleur dominante ?")
    assert node.text_prompt == "Quelle est la couleur dominante ?"


def test_load_model_missing_files_raise(tmp_path):
    """load_model() sur VisionNode réel doit lever ModelLoadError si modèle ou mmproj est absent."""
    real_node = VisionNode("VReal")
    with pytest.raises(ModelLoadError):
        real_node.load_model("/tmp/missing_model_xyz.gguf", "/tmp/fake_mmproj.gguf")

    fake_model = tmp_path / "fake_model.gguf"
    fake_model.write_text("fake")
    with pytest.raises(ModelLoadError, match="mmproj"):
        real_node.load_model(str(fake_model), "/tmp/missing_mmproj_xyz.gguf")


def test_unload_model_clears_vram_state(node):
    """unload_model() doit fermer le moteur et remettre model_loaded à False."""
    assert node.model_loaded is True
    node.unload_model()
    assert node.model_loaded is False
    assert node.llm_engine is None
    assert node.current_model_path is None