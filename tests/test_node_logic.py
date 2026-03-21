"""
test_node_logic.py — Tests unitaires pour BaseNode et ses règles de sécurité.

Stratégie : MockNode ne charge jamais un vrai modèle → tests rapides et isolés.

Couvre :
  - Règles de sécurité (modèle manquant, input manquant)
  - Validation du type de DataPacket
  - Rejet d'inputs non-DataPacket
  - Chemin nominal complet
  - get_status() et reset()
  - Traçabilité des paquets après process()
  - Comportement de __repr__
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.node import (
    AIModelMissingError,
    BaseNode,
    InputMissingError,
    InvalidDataTypeError,
    ModelLoadError,
)
from core.types import DataPacket, DataType


# ---------------------------------------------------------------------------
# Nœuds de test (Mocks légers)
# ---------------------------------------------------------------------------


class TextOnlyNode(BaseNode):
    """Nœud qui n'accepte que TEXT — simule LLMNode."""

    ACCEPTED_TYPES = (DataType.TEXT,)

    def _run_inference(self) -> DataPacket:
        return DataPacket(DataType.TEXT, "Succès", source=self.name)


class MultiTypeNode(BaseNode):
    """Nœud qui accepte TEXT et IMAGE_PATH — simule VisionNode."""

    ACCEPTED_TYPES = (DataType.TEXT, DataType.IMAGE_PATH)

    def _run_inference(self) -> DataPacket:
        return DataPacket(DataType.TEXT, "ok", source=self.name)


class AnyTypeNode(BaseNode):
    """Nœud joker qui accepte tout — simule un nœud de debug."""

    ACCEPTED_TYPES = (DataType.ANY,)

    def _run_inference(self) -> DataPacket:
        return DataPacket(DataType.ANY, "anything", source=self.name)


# ---------------------------------------------------------------------------
# Règles de sécurité
# ---------------------------------------------------------------------------


def test_run_without_model_raises_ai_model_missing():
    """process() sans modèle chargé doit lever AIModelMissingError."""
    node = TextOnlyNode("TestNode")
    with pytest.raises(AIModelMissingError):
        node.process()


def test_run_without_input_raises_input_missing():
    """process() sans input doit lever InputMissingError."""
    node = TextOnlyNode("TestNode")
    node.load_model("/tmp/fake.gguf")
    with pytest.raises(InputMissingError):
        node.process()


def test_run_nominal():
    """Chemin heureux : modèle + input → résultat correct."""
    node = TextOnlyNode("TestNode")
    node.load_model("/tmp/fake.gguf")
    node.set_input(DataPacket(DataType.TEXT, "Bonjour"))
    result = node.process()

    assert result is not None
    assert result.content == "Succès"
    assert result.data_type == DataType.TEXT
    assert result.source == "TestNode"


# ---------------------------------------------------------------------------
# Validation des types d'entrée
# ---------------------------------------------------------------------------


def test_wrong_data_type_rejected():
    """Un nœud TEXT doit rejeter IMAGE_PATH."""
    node = TextOnlyNode("TestNode")
    with pytest.raises(InvalidDataTypeError):
        node.set_input(DataPacket(DataType.IMAGE_PATH, "/tmp/img.jpg"))


def test_wrong_data_type_error_message():
    """Le message d'erreur doit nommer le type reçu et les types attendus."""
    node = TextOnlyNode("TestNode")
    with pytest.raises(InvalidDataTypeError) as exc_info:
        node.set_input(DataPacket(DataType.IMAGE_PATH, "/img.jpg"))
    assert "image_path" in str(exc_info.value)
    assert "text" in str(exc_info.value)


def test_multitype_node_accepts_text_and_image():
    """Un nœud déclarant deux types doit accepter les deux."""
    node = MultiTypeNode("Multi")
    node.load_model("/tmp/fake.gguf")

    node.set_input(DataPacket(DataType.TEXT, "prompt"))
    r1 = node.process()
    assert r1 is not None

    node.reset()
    node.set_input(DataPacket(DataType.IMAGE_PATH, "/img.jpg"))
    r2 = node.process()
    assert r2 is not None


def test_set_input_rejects_non_packet():
    """set_input() doit lever TypeError pour tout objet non-DataPacket."""
    node = TextOnlyNode("TestNode")
    with pytest.raises(TypeError):
        node.set_input("une chaîne")
    with pytest.raises(TypeError):
        node.set_input(42)
    with pytest.raises(TypeError):
        node.set_input(None)


# ---------------------------------------------------------------------------
# load_model() — chemins invalides
# ---------------------------------------------------------------------------


def test_load_model_empty_path_raises():
    """load_model() doit rejeter une chaîne vide."""
    node = TextOnlyNode("TestNode")
    with pytest.raises(ModelLoadError):
        node.load_model("")


def test_load_model_none_raises():
    """load_model() doit rejeter None."""
    node = TextOnlyNode("TestNode")
    with pytest.raises(ModelLoadError):
        node.load_model(None)


def test_load_model_sets_flag():
    """Après load_model() réussi, model_loaded doit être True."""
    node = TextOnlyNode("TestNode")
    assert node.model_loaded is False
    node.load_model("/tmp/any_path.gguf")
    assert node.model_loaded is True


# ---------------------------------------------------------------------------
# get_status()
# ---------------------------------------------------------------------------


def test_get_status_initial_state():
    node = TextOnlyNode("MyNode")
    status = node.get_status()

    assert status["name"] == "MyNode"
    assert status["class"] == "TextOnlyNode"
    assert status["model_loaded"] is False
    assert status["has_input"] is False
    assert status["has_output"] is False
    assert "text" in status["accepted_types"]


def test_get_status_after_load():
    node = TextOnlyNode("MyNode")
    node.load_model("/tmp/m.gguf")
    assert node.get_status()["model_loaded"] is True


def test_get_status_after_set_input():
    node = TextOnlyNode("MyNode")
    node.load_model("/tmp/m.gguf")
    node.set_input(DataPacket(DataType.TEXT, "hi"))
    assert node.get_status()["has_input"] is True


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_input_and_output():
    """reset() doit vider les buffers mais garder le modèle chargé."""
    node = TextOnlyNode("TestNode")
    node.load_model("/tmp/m.gguf")
    node.set_input(DataPacket(DataType.TEXT, "test"))
    node.process()

    node.reset()

    assert node.input_packet is None
    assert node.output_packet is None
    assert node.model_loaded is True   # Le modèle reste chargé


def test_process_fails_after_reset_without_new_input():
    """Après reset(), process() sans nouvel input doit échouer."""
    node = TextOnlyNode("TestNode")
    node.load_model("/tmp/m.gguf")
    node.set_input(DataPacket(DataType.TEXT, "test"))
    node.process()
    node.reset()

    with pytest.raises(InputMissingError):
        node.process()


# ---------------------------------------------------------------------------
# Traçabilité
# ---------------------------------------------------------------------------


def test_packet_has_trace_after_process():
    """Un DataPacket doit avoir au moins une entrée de trace après process()."""
    node = TextOnlyNode("TestNode")
    node.load_model("/tmp/m.gguf")
    packet = DataPacket(DataType.TEXT, "Hello")
    node.set_input(packet)
    node.process()

    assert len(packet._processing_log) >= 1
    nodes_in_trace = [step["node"] for step in packet._processing_log]
    assert "TestNode" in nodes_in_trace


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_contains_name_and_flag():
    node = TextOnlyNode("MyNode")
    r = repr(node)
    assert "MyNode" in r
    assert "model_loaded" in r
    assert "False" in r

    node.load_model("/tmp/m.gguf")
    assert "True" in repr(node)
