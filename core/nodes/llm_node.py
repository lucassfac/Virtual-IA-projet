"""
llm_node.py — Moteur d'inférence haute performance Neural Forge.
Version 4.3 : Clean Code & Centralisation des utilitaires.
"""

import gc
import json
import base64
import mimetypes
import os
from typing import Generator, Optional, List, Dict, Union

from core.logger import forge_logger
from core.node import AIModelMissingError, BaseNode, ModelLoadError
from core.types import DataPacket, DataType
from core.utils.hardware import get_profile, HardwareProfile
from core.utils.file_utils import check_safe_path

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    Llama = None
    Llava15ChatHandler = None

class LLMNode(BaseNode):
    ACCEPTED_TYPES = (DataType.TEXT, DataType.IMAGE_PATH)

    def __init__(self, name: str = "LLM Generator"):
        super().__init__(name)
        self.llm_engine: Optional[object] = None
        self.draft_engine: Optional[object] = None
        self.current_model_path: Optional[str] = None
        self.current_skill_path: Optional[str] = None
        self.skill_system_prompt: str = ""
        self.skill_knowledge_base: str = ""
        self._turbo_active: bool = False
        self.hw: Optional[HardwareProfile] = None
        self.conversation_history: List[Dict[str, Union[str, list]]] = []

    @property
    def is_multimodal(self) -> bool:
        if not self.model_loaded: return False
        proj = getattr(self, "mmproj_path", "")
        return bool(proj and isinstance(proj, str) and proj.strip() != "")

    def unload_model(self) -> None:
        if self.llm_engine:
            try: self.llm_engine.close()
            except Exception: pass
            self.llm_engine = None
            
        if self.draft_engine:
            try: self.draft_engine.close()
            except Exception: pass
            self.draft_engine = None
            
        self.current_model_path = None
        self.current_skill_path = None
        self.skill_system_prompt = ""
        self.skill_knowledge_base = ""
        self.model_loaded = False
        gc.collect() 
        forge_logger.info(f"[{self.name}] Moteur déchargé, VRAM libérée.")

    def clear_history(self) -> None:
        self.conversation_history = []
        forge_logger.info(f"[{self.name}] Historique effacé.")

    def _get_skill_context(self) -> list:
        if self.skill_system_prompt and self.skill_knowledge_base:
            return [{"role": "system", "content": f"{self.skill_system_prompt}\n\nBASE DE CONNAISSANCES:\n{self.skill_knowledge_base}"}]
        return []

    def load_model(self, model_path: str, skill_path: Optional[str] = None, draft_model_path: Optional[str] = None, turbo: bool = False) -> None:
        if Llama is None: raise ModelLoadError("llama-cpp-python n'est pas installé.")

        self.unload_model()
        self.hw = get_profile()

        safe_model_path = check_safe_path(model_path, "Modèle Principal")
        safe_draft_path = check_safe_path(draft_model_path, "Modèle Draft") if draft_model_path else None
        
        if skill_path and os.path.exists(skill_path):
            try:
                with open(skill_path, "r", encoding="utf-8") as f:
                    skill_data = json.load(f)
                    if skill_data.get("type") == "rag_skill":
                        self.current_skill_path = skill_path
                        self.skill_system_prompt = skill_data.get("system_prompt", "")
                        self.skill_knowledge_base = skill_data.get("knowledge_base", "")
            except Exception as e:
                forge_logger.error(f"[{self.name}] Erreur de lecture du skill : {e}")

        self._turbo_active = bool(turbo and safe_draft_path and self.hw.turbo_eligible)

        kwargs = {
            "model_path": safe_model_path, "n_ctx": 4096, 
            "n_threads": self.hw.n_threads, "n_threads_batch": self.hw.n_threads,
            "n_batch": self.hw.n_batch, "n_gpu_layers": 15,
            "use_mmap": self.hw.use_mmap, "logits_all": False, "verbose": False,
        }

        if getattr(self.hw, 'flash_attn', False):
            kwargs["flash_attn"] = True

        proj = getattr(self, "mmproj_path", "")
        if proj and os.path.exists(proj) and Llava15ChatHandler is not None:
            try:
                kwargs["chat_handler"] = Llava15ChatHandler(clip_model_path=proj)
                forge_logger.info(f"[{self.name}] Projecteur visuel mmproj activé.")
            except Exception as e:
                forge_logger.error(f"[{self.name}] Échec du chargement mmproj : {e}")

        try:
            self.llm_engine = Llama(**kwargs)
        except Exception as e:
            forge_logger.warning(f"[{self.name}] Échec GPU ({e}). Fallback CPU...")
            kwargs["n_gpu_layers"] = 0
            kwargs["n_ctx"] = 2048 
            try:
                self.llm_engine = Llama(**kwargs)
            except Exception as fallback_err:
                raise ModelLoadError(f"Échec total : {fallback_err}")

        self.current_model_path = model_path
        self.model_loaded = True

    def _prepare_message_content(self) -> Union[str, list]:
        if self.input_packet.data_type == DataType.IMAGE_PATH:
            user_text = self.input_packet.metadata.get("prompt", "Décris cette image.")
            if not self.is_multimodal:
                return f"[Instruction système : Vous êtes un modèle textuel pur. Informez l'utilisateur que vous ne pouvez pas voir son image.]\nRequête : {user_text}"

            img_path = self.input_packet.content
            try:
                with open(img_path, "rb") as img_file:
                    base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                mime_type = mimetypes.guess_type(img_path)[0] or "image/jpeg"
                return [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}},
                    {"type": "text", "text": user_text}
                ]
            except Exception as e:
                forge_logger.error(f"[{self.name}] Échec encodage Base64 : {e}")
                return user_text
        return self.input_packet.content

    def _run_inference(self) -> DataPacket:
        if self.conversation_history and self.conversation_history[-1]["role"] == "user":
            self.conversation_history.pop()

        self.conversation_history.append({"role": "user", "content": self._prepare_message_content()})
        self.conversation_history = self.conversation_history[-10:]

        forge_logger.log_node_event(self.name, "INFERENCE_START", "Génération en cours...")
        
        output = self.llm_engine.create_chat_completion(
            messages=self._get_skill_context() + self.conversation_history,
            max_tokens=1024,
            temperature=0.3 if self.current_skill_path else 0.7,
            stream=False
        )
        
        answer: str = output["choices"][0]["message"]["content"].strip()
        self.conversation_history.append({"role": "assistant", "content": answer})

        return DataPacket(DataType.TEXT, answer, metadata={"model": self.current_model_path}, source=self.name)

    def stream_inference(self, packet: DataPacket) -> Generator[str, None, None]:
        self.set_input(packet)
        if not self.model_loaded: raise AIModelMissingError(f"[{self.name}] Aucun modèle chargé.")

        if self.conversation_history and self.conversation_history[-1]["role"] == "user":
            self.conversation_history.pop()

        self.conversation_history.append({"role": "user", "content": self._prepare_message_content()})
        self.conversation_history = self.conversation_history[-10:]

        stream = self.llm_engine.create_chat_completion(
            messages=self._get_skill_context() + self.conversation_history,
            max_tokens=1024, stream=True,
            temperature=0.3 if self.current_skill_path else 0.7,
        )
        
        full_answer = ""
        for chunk in stream:
            tok = chunk["choices"][0].get("delta", {}).get("content", "")
            if tok:
                full_answer += tok
                yield tok

        self.conversation_history.append({"role": "assistant", "content": full_answer})

    def get_status(self) -> dict:
        base = super().get_status()
        if self.hw: base.update({"cuda_ok": self.hw.cuda_ok, "n_ctx": self.hw.n_ctx})
        return base
