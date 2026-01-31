from enum import Enum
import uuid
import time
from typing import Any, Dict, Optional

# 1. Liste des types de données acceptés par ton moteur
class DataType(Enum):
    TEXT = "text"              # Pour le Chatbot
    IMAGE_PATH = "image_path"  # Pour la Webcam / LLaVA (On passe le chemin du fichier)
    AUDIO_PATH = "audio_path"  # Pour le Micro / Whisper
    any = "any"                # Joker (pour les nœuds de debug)

# 2. L'enveloppe qui va voyager de nœud en nœud
class DataPacket:
    def __init__(self, data_type: DataType, content: Any, metadata: Optional[Dict] = None):
        self.id = str(uuid.uuid4())      # ID unique pour tracer le paquet (debug)
        self.timestamp = time.time()     # Pour mesurer la latence
        self.data_type = data_type       # Le type (ex: DataType.TEXT)
        self.content = content           # La donnée réelle (ex: "Bonjour" ou "/tmp/img.jpg")
        self.metadata = metadata or {}   # Infos bonus (ex: {"confidence": 0.98})

    def __repr__(self):
        # Pour un affichage propre dans la console quand tu feras print(packet)
        return f"<DataPacket type={self.data_type.value} content={str(self.content)[:20]}...>"