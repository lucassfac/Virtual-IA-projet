"""
llm_node.py — Nœud d'inférence texte via llama-cpp-python.
"""

import json
import os
from typing import Generator, Optional

from core.logger import forge_logger
from core.node import AIModelMissingError, BaseNode, ModelLoadError
from core.types import DataPacket, DataType

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class LLMNode(BaseNode):
    ACCEPTED_TYPES = (DataType.TEXT,)

    def __init__(self, name: str = "LLM Generator"):
        super().__init__(name)
        self.llm_engine: Optional[object] = None
        self.current_model_path: Optional[str] = None
        self.current_lora_path: Optional[str] = None
        self._lora_is_simulated: bool = False

    @staticmethod
    def _is_real_lora(path: str) -> bool:
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
            if magic == b"GGUF":
                return True
            if magic[:1] == b"{":
                return False
            return True
        except OSError:
            return False

    def load_model(self, model_path: str, lora_path: Optional[str] = None) -> None:
        if Llama is None:
            raise ModelLoadError("llama-cpp-python n'est pas installé.")

        effective_lora = None
        self._lora_is_simulated = False

        if lora_path:
            if self._is_real_lora(lora_path):
                effective_lora = lora_path
                forge_logger.log_node_event(self.name, "LORA_REAL", lora_path)
            else:
                # Adaptateur simulé → on charge sans LoRA, on logue discrètement
                self._lora_is_simulated = True
                forge_logger.warning(
                    f"[{self.name}] LoRA simulé ignoré : '{os.path.basename(lora_path)}'"
                )

        forge_logger.log_node_event(
            self.name, "LOAD_MODEL",
            f"model={model_path}, lora={effective_lora}"
        )

        try:
            import os as _os
            # Max 4 threads : au-delà le gain est marginal
            # et le CPU sature à 99% avec des modèles 4B+
            n_cpu = _os.cpu_count() or 4
            n_threads = min(n_cpu, 4)
            self.llm_engine = Llama(
                model_path=model_path,
                lora_path=effective_lora,
                n_ctx=2048,
                n_threads=n_threads,
                n_threads_batch=n_threads,
                n_batch=256,
                n_gpu_layers=-1,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
            )
            self.current_model_path = model_path
            self.current_lora_path = lora_path

            if effective_lora:
                forge_logger.log_node_event(self.name, "LORA_APPLIED", effective_lora)

            super().load_model(model_path)

        except FileNotFoundError:
            raise ModelLoadError(f"[{self.name}] Modèle introuvable : '{model_path}'")
        except Exception as e:
            raise ModelLoadError(f"[{self.name}] Échec du chargement : {e}")

    def _run_inference(self) -> DataPacket:
        prompt = self.input_packet.content
        full_prompt = f"Question: {prompt}\nRéponse:"
        forge_logger.log_node_event(self.name, "INFERENCE_START", f"len={len(prompt)}")

        output = self.llm_engine(
            full_prompt,
            max_tokens=2048,
            stop=["Question:"],
            echo=False,
            temperature=0.7,
            repeat_penalty=1.1,
        )
        response_text: str = output["choices"][0]["text"].strip()
        finish_reason: str = output["choices"][0].get("finish_reason", "unknown")

        if not response_text:
            response_text = "(Réponse vide — essayez de reformuler le prompt)"

        forge_logger.log_node_event(self.name, "INFERENCE_END",
                                    f"len={len(response_text)}, finish={finish_reason}")

        return DataPacket(
            DataType.TEXT, response_text,
            metadata={
                "model": self.current_model_path,
                "lora": self.current_lora_path,
                "lora_applied": not self._lora_is_simulated,
                "finish_reason": finish_reason,
            },
            source=self.name,
        )

    def stream_inference(self, packet: DataPacket) -> Generator[str, None, None]:
        self.set_input(packet)
        if not self.model_loaded:
            raise AIModelMissingError(f"[{self.name}] Aucun modèle chargé.")

        prompt = self.input_packet.content
        full_prompt = f"Question: {prompt}\nRéponse:"
        forge_logger.log_node_event(self.name, "STREAM_START")

        stream = self.llm_engine(
            full_prompt,
            max_tokens=2048,
            stop=["Question:"],
            echo=False,
            stream=True,
            temperature=0.7,
            repeat_penalty=1.1,
        )
        for chunk in stream:
            token: str = chunk["choices"][0]["text"]
            if token:
                yield token

        forge_logger.log_node_event(self.name, "STREAM_END")
