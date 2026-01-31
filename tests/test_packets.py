import pytest
import sys
import os

# --- ASTUCE D'IMPORTATION ---
# On dit à Python : "Regarde dans le dossier parent (..) pour trouver 'core'"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.types import DataPacket, DataType

def test_packet_creation_text():
    """Test 1 : Vérifie qu'on peut créer un paquet de TEXTE."""
    message = "Test du prompt utilisateur"
    packet = DataPacket(DataType.TEXT, message)

    # Vérifications (Assertions)
    assert packet.data_type == DataType.TEXT
    assert packet.content == "Test du prompt utilisateur"
    assert packet.id is not None  # Il doit avoir un ID unique
    assert packet.timestamp > 0   # Le temps doit être enregistré

def test_packet_creation_image():
    """Test 2 : Vérifie qu'on peut créer un paquet IMAGE avec métadonnées."""
    fake_path = "/tmp/photo_chat.jpg"
    meta = {"resolution": "1080p", "source": "webcam"}
    
    packet = DataPacket(DataType.IMAGE_PATH, fake_path, metadata=meta)

    assert packet.data_type == DataType.IMAGE_PATH
    assert packet.content == fake_path
    assert packet.metadata["resolution"] == "1080p"

def test_packet_repr():
    """Test 3 : Vérifie que l'affichage console est joli."""
    packet = DataPacket(DataType.TEXT, "Bonjour")
    # On vérifie que la conversion en string contient bien le type
    assert "text" in str(packet)