"""
vision_node.py — Nœud d'analyse d'image via LLaVA (multimodal).

Modèles compatibles : LLaVA-1.5, BakLLaVA, tout modèle LLaVA au format GGUF.
Requiert deux fichiers :
  - model_path  : le modèle de langage  (ex: llava-1.5-7b-q4.gguf)
  - mmproj_path : le projecteur visuel  (ex: llava-1.5-7b-mmproj-f16.gguf)
"""

import base64
import os
from typing import Optional

from core.logger import forge_logger
from core.node import BaseNode, InvalidDataTypeError, ModelLoadError
from core.types import DataPacket, DataType

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    Llama = None
    Llava15ChatHandler = None


class VisionNode(BaseNode):
    """
    Nœud multimodal : analyse une image et répond à une question textuelle.
    L'image est fournie via un DataPacket de type IMAGE_PATH.
    La question est définie via set_prompt() (défaut : description générale).
    """

    ACCEPTED_TYPES = (DataType.IMAGE_PATH,)

    # Extensions supportées → type MIME
    _MIME_MAP = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    def __init__(self, name: str = "Vision Analyzer"):
        super().__init__(name)
        self.llm_engine: Optional[object] = None
        self.current_model_path: Optional[str] = None
        self.mmproj_path: Optional[str] = None
        self.text_prompt: str = "Décris cette image en détail."

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------

    def load_model(self, model_path: str, mmproj_path: str) -> None:
        """
        Charge le modèle LLaVA + son projecteur visuel.

        :param model_path:  Chemin vers le .gguf du modèle LLaVA
        :param mmproj_path: Chemin vers le .gguf du projecteur multimodal
        """
        if Llama is None:
            raise ModelLoadError(
                "llama-cpp-python n'est pas installé. "
                "Lancez : pip install llama-cpp-python"
            )

        if not os.path.exists(model_path):
            raise ModelLoadError(
                f"[{self.name}] Modèle introuvable : '{model_path}'"
            )
        if not os.path.exists(mmproj_path):
            raise ModelLoadError(
                f"[{self.name}] Projecteur multimodal introuvable : '{mmproj_path}'"
            )

        forge_logger.log_node_event(
            self.name, "LOAD_VISION_MODEL",
            f"model={model_path}, mmproj={mmproj_path}"
        )

        try:
            chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
            self.llm_engine = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=4096,
                verbose=False,
            )
            self.current_model_path = model_path
            self.mmproj_path = mmproj_path
            super().load_model(model_path)

        except Exception as e:
            raise ModelLoadError(
                f"[{self.name}] Échec du chargement vision : {e}"
            )

    # ------------------------------------------------------------------
    # Configuration du prompt textuel
    # ------------------------------------------------------------------

    def set_prompt(self, prompt: str) -> None:
        """
        Définit la question à poser sur l'image.
        Exemple : "Quels objets dangereux vois-tu ?"
        """
        self.text_prompt = prompt
        forge_logger.log_node_event(
            self.name, "PROMPT_SET", prompt[:80]
        )

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------

    def _run_inference(self) -> DataPacket:
        if self.input_packet.data_type != DataType.IMAGE_PATH:
            raise InvalidDataTypeError(
                f"[{self.name}] VisionNode requiert IMAGE_PATH, "
                f"reçu : {self.input_packet.data_type.value}"
            )

        image_path: str = self.input_packet.content
        forge_logger.log_node_event(
            self.name, "VISION_INFERENCE_START", image_path
        )

        image_data_url = self._image_to_data_url(image_path)

        response = self.llm_engine.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                        {"type": "text", "text": self.text_prompt},
                    ],
                }
            ],
            max_tokens=400,
        )

        result_text: str = (
            response["choices"][0]["message"]["content"].strip()
        )

        if not result_text:
            result_text = "(Analyse vide — vérifiez le modèle LLaVA et le mmproj)"
            forge_logger.warning(f"[{self.name}] Réponse vision vide")

        forge_logger.log_node_event(
            self.name, "VISION_INFERENCE_END",
            f"response_len={len(result_text)}"
        )

        return DataPacket(
            DataType.TEXT,
            result_text,
            metadata={
                "image_path": image_path,
                "prompt": self.text_prompt,
                "model": self.current_model_path,
                "mmproj": self.mmproj_path,
            },
            source=self.name,
        )

    # ------------------------------------------------------------------
    # Utilitaire privé
    # ------------------------------------------------------------------

    def _image_to_data_url(self, image_path: str) -> str:
        """
        Convertit un fichier image en data URL base64.
        Format : data:<mime>;base64,<données>
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"[{self.name}] Image introuvable : '{image_path}'"
            )

        ext = os.path.splitext(image_path)[1].lower()
        mime_type = self._MIME_MAP.get(ext, "image/jpeg")

        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{b64_data}"
