"""
llm_node.py — Nœud d'inférence texte via llama-cpp-python.

Nouveautés vs v1 :
  - Streaming token par token (stream_inference)
  - ModelLoadError propre (FileNotFoundError capturé)
  - Métadonnées riches dans le DataPacket de sortie
  - ACCEPTED_TYPES restreint à TEXT
"""

from typing import Generator, Optional

from core.logger import forge_logger
from core.node import AIModelMissingError, BaseNode, ModelLoadError
from core.types import DataPacket, DataType

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # Permet d'importer le module même sans llama-cpp installé


class LLMNode(BaseNode):
    """
    Nœud d'inférence textuelle.
    Supporte un adaptateur LoRA optionnel pour la spécialisation métier.
    """

    ACCEPTED_TYPES = (DataType.TEXT,)

    def __init__(self, name: str = "LLM Generator"):
        super().__init__(name)
        self.llm_engine: Optional[object] = None
        self.current_model_path: Optional[str] = None
        self.current_lora_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------

    def load_model(self, model_path: str, lora_path: Optional[str] = None) -> None:
        """
        Charge le modèle GGUF, avec LoRA optionnel.

        :param model_path: Chemin vers le fichier .gguf
        :param lora_path:  Chemin vers l'adaptateur .lora (spécialisation)
        """
        if Llama is None:
            raise ModelLoadError(
                "llama-cpp-python n'est pas installé. "
                "Lancez : pip install llama-cpp-python"
            )

        forge_logger.log_node_event(
            self.name, "LOAD_MODEL",
            f"model={model_path}, lora={lora_path}"
        )

        try:
            self.llm_engine = Llama(
                model_path=model_path,
                lora_path=lora_path,
                n_ctx=2048,
                verbose=False,
            )
            self.current_model_path = model_path
            self.current_lora_path = lora_path

            if lora_path:
                forge_logger.log_node_event(
                    self.name, "LORA_APPLIED", lora_path
                )

            super().load_model(model_path)

        except FileNotFoundError:
            raise ModelLoadError(
                f"[{self.name}] Fichier modèle introuvable : '{model_path}'"
            )
        except Exception as e:
            raise ModelLoadError(
                f"[{self.name}] Échec du chargement : {e}"
            )

    # ------------------------------------------------------------------
    # Inférence bloquante
    # ------------------------------------------------------------------

    def _run_inference(self) -> DataPacket:
        prompt = self.input_packet.content
        full_prompt = f"Question: {prompt}\nRéponse:"

        forge_logger.log_node_event(
            self.name, "INFERENCE_START",
            f"prompt_len={len(prompt)}"
        )

        output = self.llm_engine(
            full_prompt,
            max_tokens=200,
            stop=["Question:"],
            echo=False,
        )

        response_text: str = output["choices"][0]["text"].strip()
        finish_reason: str = output["choices"][0].get("finish_reason", "unknown")

        if not response_text:
            response_text = "(Réponse vide — essayez de reformuler le prompt)"
            forge_logger.warning(f"[{self.name}] Réponse vide retournée")

        forge_logger.log_node_event(
            self.name, "INFERENCE_END",
            f"response_len={len(response_text)}, finish={finish_reason}"
        )

        return DataPacket(
            DataType.TEXT,
            response_text,
            metadata={
                "model": self.current_model_path,
                "lora": self.current_lora_path,
                "prompt_tokens": len(full_prompt.split()),
                "finish_reason": finish_reason,
            },
            source=self.name,
        )

    # ------------------------------------------------------------------
    # Streaming token par token (pour l'IHM PyQt6)
    # ------------------------------------------------------------------

    def stream_inference(self, packet: DataPacket) -> Generator[str, None, None]:
        """
        Générateur yielding un token à la fois.
        Conçu pour alimenter un QTextEdit en temps réel via un QThread.

        Usage :
            for token in node.stream_inference(packet):
                text_edit.insertPlainText(token)
        """
        self.set_input(packet)

        if not self.model_loaded:
            raise AIModelMissingError(
                f"[{self.name}] Aucun modèle chargé pour le streaming."
            )

        prompt = self.input_packet.content
        full_prompt = f"Question: {prompt}\nRéponse:"

        forge_logger.log_node_event(self.name, "STREAM_START")

        stream = self.llm_engine(
            full_prompt,
            max_tokens=200,
            stop=["Question:"],
            echo=False,
            stream=True,
        )

        for chunk in stream:
            token: str = chunk["choices"][0]["text"]
            if token:
                yield token

        forge_logger.log_node_event(self.name, "STREAM_END")
