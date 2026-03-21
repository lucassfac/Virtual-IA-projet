from enum import Enum
import uuid
import time
from typing import Any, Dict, List, Optional


class DataType(Enum):
    TEXT = "text"
    IMAGE_PATH = "image_path"
    AUDIO_PATH = "audio_path"
    ANY = "any"


class DataPacket:
    """
    Enveloppe universelle qui circule entre tous les nœuds du pipeline.
    Embarque traçabilité, horodatage et métadonnées.
    """

    def __init__(
        self,
        data_type: DataType,
        content: Any,
        metadata: Optional[Dict] = None,
        source: str = "unknown",
    ):
        self.id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.data_type = data_type
        self.content = content
        self.metadata: Dict = metadata or {}
        self.source = source
        self._processing_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Traceability
    # ------------------------------------------------------------------

    def log_step(self, node_name: str, action: str) -> None:
        """Enregistre une étape de traitement pour la traçabilité complète."""
        self._processing_log.append(
            {
                "timestamp": time.time(),
                "node": node_name,
                "action": action,
            }
        )

    def get_trace(self) -> str:
        """Retourne un journal lisible du parcours du paquet."""
        if not self._processing_log:
            return f"[{self.id[:8]}] Aucune étape enregistrée."
        lines = [f"[{self.id[:8]}] Trace du paquet :"]
        for step in self._processing_log:
            t = time.strftime("%H:%M:%S", time.localtime(step["timestamp"]))
            lines.append(f"  {t} | {step['node']} → {step['action']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<DataPacket id={self.id[:8]} "
            f"type={self.data_type.value} "
            f"src={self.source} "
            f"content={str(self.content)[:30]}...>"
        )
