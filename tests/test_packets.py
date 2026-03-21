"""
test_packets.py — Tests unitaires pour DataPacket et DataType.

Couvre :
  - Création pour chaque type
  - Unicité des IDs
  - Ordre des timestamps
  - Métadonnées par défaut
  - Traçabilité (log_step / get_trace)
  - Source field
  - __repr__
"""

import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.types import DataPacket, DataType


# ---------------------------------------------------------------------------
# Création de base
# ---------------------------------------------------------------------------


def test_packet_creation_text():
    """Crée un paquet TEXT et vérifie tous ses champs obligatoires."""
    content = "Test du prompt utilisateur"
    packet = DataPacket(DataType.TEXT, content)

    assert packet.data_type == DataType.TEXT
    assert packet.content == content
    assert packet.id is not None
    assert len(packet.id) == 36          # Format UUID4 : 8-4-4-4-12 + 4 tirets
    assert packet.timestamp > 0


def test_packet_creation_image_with_metadata():
    """Crée un paquet IMAGE_PATH avec métadonnées."""
    path = "/tmp/photo.jpg"
    meta = {"resolution": "1080p", "source": "webcam"}
    packet = DataPacket(DataType.IMAGE_PATH, path, metadata=meta)

    assert packet.data_type == DataType.IMAGE_PATH
    assert packet.content == path
    assert packet.metadata["resolution"] == "1080p"
    assert packet.metadata["source"] == "webcam"


def test_packet_creation_audio():
    """Crée un paquet AUDIO_PATH."""
    packet = DataPacket(DataType.AUDIO_PATH, "/tmp/audio.wav")
    assert packet.data_type == DataType.AUDIO_PATH


def test_all_data_types_usable():
    """Tous les membres de DataType doivent pouvoir créer un DataPacket."""
    for dtype in DataType:
        p = DataPacket(dtype, "content")
        assert p.data_type == dtype


# ---------------------------------------------------------------------------
# Unicité et ordonnancement
# ---------------------------------------------------------------------------


def test_packet_unique_ids():
    """Chaque paquet doit avoir un ID distinct."""
    packets = [DataPacket(DataType.TEXT, "msg") for _ in range(20)]
    ids = {p.id for p in packets}
    assert len(ids) == 20


def test_packet_timestamp_ordering():
    """Un paquet créé plus tard doit avoir un timestamp supérieur."""
    p1 = DataPacket(DataType.TEXT, "premier")
    time.sleep(0.01)
    p2 = DataPacket(DataType.TEXT, "second")
    assert p2.timestamp > p1.timestamp


# ---------------------------------------------------------------------------
# Métadonnées
# ---------------------------------------------------------------------------


def test_packet_default_metadata_is_empty_dict():
    """Sans metadata fournie, le champ doit être un dict vide (pas None)."""
    packet = DataPacket(DataType.TEXT, "test")
    assert isinstance(packet.metadata, dict)
    assert packet.metadata == {}


def test_packet_metadata_isolation():
    """Les métadonnées de deux paquets ne doivent pas se contaminer."""
    p1 = DataPacket(DataType.TEXT, "a", metadata={"k": 1})
    p2 = DataPacket(DataType.TEXT, "b")
    p2.metadata["k"] = 99
    assert p1.metadata["k"] == 1   # p1 non affecté


# ---------------------------------------------------------------------------
# Traçabilité
# ---------------------------------------------------------------------------


def test_packet_log_step_records_entry():
    """log_step() doit enregistrer nœud et action."""
    packet = DataPacket(DataType.TEXT, "data")
    packet.log_step("NodeA", "reçu_en_entrée")
    assert len(packet._processing_log) == 1
    assert packet._processing_log[0]["node"] == "NodeA"
    assert packet._processing_log[0]["action"] == "reçu_en_entrée"


def test_packet_log_multiple_steps():
    """Plusieurs étapes doivent être empilées dans l'ordre."""
    packet = DataPacket(DataType.TEXT, "data")
    packet.log_step("NodeA", "reçu")
    packet.log_step("NodeB", "traité")
    packet.log_step("NodeC", "sorti")
    assert len(packet._processing_log) == 3
    assert packet._processing_log[2]["node"] == "NodeC"


def test_packet_get_trace_contains_steps():
    """get_trace() doit retourner une chaîne avec les nœuds et actions."""
    packet = DataPacket(DataType.TEXT, "data")
    packet.log_step("LLMNode", "inférence")
    trace = packet.get_trace()
    assert "LLMNode" in trace
    assert "inférence" in trace


def test_packet_get_trace_empty():
    """get_trace() sans étape doit retourner un message clair."""
    packet = DataPacket(DataType.TEXT, "data")
    trace = packet.get_trace()
    assert "Aucune" in trace


# ---------------------------------------------------------------------------
# Source et repr
# ---------------------------------------------------------------------------


def test_packet_source_stored():
    packet = DataPacket(DataType.TEXT, "hello", source="LLMNode")
    assert packet.source == "LLMNode"


def test_packet_default_source():
    packet = DataPacket(DataType.TEXT, "hello")
    assert packet.source == "unknown"


def test_packet_repr_contains_type_and_source():
    packet = DataPacket(DataType.TEXT, "Bonjour", source="TestNode")
    r = repr(packet)
    assert "text" in r
    assert "TestNode" in r
