"""
llm_node.py — Moteur d'inférence haute performance Neural Forge.
Version 4.7 : Isolation de session UUID, Protection RAG Anti-Vide et Support Qwen2.5-VL.
"""

import base64
import gc
import json
import mimetypes
import os
import uuid
from typing import Dict, Generator, List, Optional, Union

from core.logger import forge_logger
from core.memory_manager import MemoryManager
from core.node import AIModelMissingError, BaseNode, ModelLoadError
from core.types import DataPacket, DataType
from core.utils.file_utils import check_safe_path
from core.utils.hardware import HardwareProfile, get_profile

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

        self.max_tokens_limit: int = 1024
        self.conversation_history: List[Dict[str, Union[str, list]]] = []
        self.memory_manager = MemoryManager()
        self.current_session_id: str = f"session_{uuid.uuid4().hex[:8]}"

    @property
    def is_multimodal(self) -> bool:
        if not self.model_loaded:
            return False
        proj = getattr(self, "mmproj_path", "")
        return bool(proj and isinstance(proj, str) and proj.strip() != "")

    def unload_model(self) -> None:
        """Libère la VRAM et réinitialise les états."""
        for engine in [self.llm_engine, self.draft_engine]:
            if engine:
                try:
                    engine.close()
                except Exception:
                    pass

        self.llm_engine = self.draft_engine = None
        self.current_model_path = self.current_skill_path = None
        self.skill_system_prompt = self.skill_knowledge_base = ""
        self.model_loaded = False
        gc.collect()
        forge_logger.info(f"[{self.name}] Moteur déchargé, VRAM libérée.")

    def clear_history(self) -> None:
        """Vide la mémoire court-terme et génère un nouvel ID de session isolé."""
        self.conversation_history.clear()
        self.current_session_id = f"session_{uuid.uuid4().hex[:8]}"
        forge_logger.info(
            f"[{self.name}] Historique effacé. Nouvelle session : {self.current_session_id}"
        )

    def load_model(
        self,
        model_path: str,
        skill_path: Optional[str] = None,
        draft_model_path: Optional[str] = None,
        turbo: bool = False,
    ) -> None:
        if Llama is None:
            raise ModelLoadError("llama-cpp-python n'est pas installé.")

        self.unload_model()
        self.hw = get_profile()

        safe_model_path = check_safe_path(model_path, "Modèle Principal")
        self._load_skill(skill_path)
        self._turbo_active = bool(turbo and draft_model_path and self.hw.turbo_eligible)

        kwargs = self._build_engine_kwargs(safe_model_path)
        self._setup_vision_projector(kwargs, safe_model_path)
        self._initialize_engine(kwargs)

        self.current_model_path = model_path
        self.model_loaded = True

    # ── SOUS-MÉTHODES D'INITIALISATION (SRP) ──

    DEFAULT_SYSTEM_PROMPT = (
        "Tu es un assistant IA fiable, précis et rigoureux exécuté localement dans Neural Forge. "
        "Réponds toujours dans la langue de l'utilisateur. "
        "RÈGLE D'OR : Sois factuel. Si tu ne connais pas une information, une œuvre ou un détail précis, "
        "dis-le honnêtement sans jamais inventer de faits, de noms ou de citations."
    )

    def _load_skill(self, skill_path: Optional[str]) -> None:
        self.current_lora_path = None
        if not skill_path or not os.path.exists(skill_path):
            return

        # 1. Cas d'un Adaptateur LoRA ou Model Soup (.safetensors / .bin)
        if skill_path.lower().endswith((".safetensors", ".bin")):
            self.current_skill_path = skill_path
            self.current_lora_path = check_safe_path(skill_path, "Adaptateur LoRA")
            forge_logger.info(f"[{self.name}] Adaptateur LoRA/Soup prêt : {os.path.basename(skill_path)}")
            return

        # 2. Cas d'une Compétence RAG (.skill JSON)
        try:
            with open(skill_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("type") == "rag_skill":
                    self.current_skill_path = skill_path
                    self.skill_system_prompt = data.get("system_prompt", "")
                    self.skill_knowledge_base = data.get("knowledge_base", "")
        except Exception as e:
            forge_logger.error(f"[{self.name}] Erreur lecture skill : {e}")

    def _build_engine_kwargs(self, model_path: str) -> dict:
        # Calcul adaptatif des couches GPU selon la VRAM réelle de l'utilisateur
        gpu_layers = self.hw.n_gpu_layers
        if self.hw.gpu_vram_gb > 0 and os.path.exists(model_path):
            model_gb = os.path.getsize(model_path) / (1024**3)
            proj = getattr(self, "mmproj_path", "")
            proj_gb = os.path.getsize(proj) / (1024**3) if proj and os.path.exists(proj) else 0.0
            total_needed = model_gb + proj_gb + 1.0
            if total_needed > self.hw.gpu_vram_gb:
                # Décharge partielle intelligente sur GPU + RAM CPU pour éviter tout Out-Of-Memory
                ratio = max(0.2, (self.hw.gpu_vram_gb - proj_gb - 0.8) / max(0.5, model_gb))
                gpu_layers = max(8, int(32 * min(1.0, ratio)))

        kwargs = {
            "model_path": model_path,
            "n_ctx": getattr(self, "context_window", self.hw.n_ctx),
            "n_threads": self.hw.n_threads,
            "n_threads_batch": self.hw.n_threads,
            "n_batch": self.hw.n_batch,
            "n_gpu_layers": gpu_layers,
            "use_mmap": self.hw.use_mmap,
            "logits_all": False,
            "verbose": False,
        }
        if getattr(self.hw, "flash_attn", False):
            kwargs["flash_attn"] = True
        if getattr(self, "current_lora_path", None):
            kwargs["lora_path"] = self.current_lora_path
        return kwargs

    def _setup_vision_projector(self, kwargs: dict, model_path: str) -> None:
        proj = getattr(self, "mmproj_path", "")
        if not (proj and os.path.exists(proj)):
            return

        nom_fichier = os.path.basename(model_path).lower()
        try:
            if "moondream" in nom_fichier:
                from llama_cpp.llama_chat_format import MoondreamChatHandler
                kwargs["chat_handler"] = MoondreamChatHandler(clip_model_path=proj)
            elif "qwen" in nom_fichier:
                # Évite que Llava15ChatHandler n'écrase le template ChatML avec du Vicuna (USER/ASSISTANT)
                try:
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                    kwargs["chat_handler"] = Qwen25VLChatHandler(clip_model_path=proj)
                except ImportError:
                    from llama_cpp.llama_chat_format import Llava16ChatHandler
                    kwargs["chat_handler"] = Llava16ChatHandler(clip_model_path=proj)
                kwargs["chat_format"] = "chatml"
            elif Llava15ChatHandler is not None:
                kwargs["chat_handler"] = Llava15ChatHandler(clip_model_path=proj)
                if "mistral" in nom_fichier or "mixtral" in nom_fichier:
                    kwargs["chat_format"] = "mistral"
                else:
                    kwargs["chat_format"] = "vicuna"

            forge_logger.info(
                f"[{self.name}] Projecteur visuel activé (Format: {kwargs.get('chat_format', 'natif')})"
            )
        except Exception as e:
            forge_logger.error(f"[{self.name}] Échec mmproj : {e}")

    def _initialize_engine(self, kwargs: dict) -> None:
        try:
            self.llm_engine = Llama(**kwargs)
        except Exception as e:
            forge_logger.warning(f"[{self.name}] Échec GPU ({e}). Fallback CPU...")
            kwargs["n_gpu_layers"] = 0
            kwargs["n_ctx"] = 2048
            try:
                self.llm_engine = Llama(**kwargs)
            except Exception as fallback_err:
                raise ModelLoadError(f"Échec total de l'inférence : {fallback_err}")

    # ── HEURISTIQUE ET MÉMOIRE JIT ──

    def _get_dynamic_params(self, text: str) -> dict:
        """Ajuste les paramètres d'inférence selon l'intention utilisateur."""
        mots_stricts = [
            "lis", "texte", "exo", "exercice", "mcd", "sql",
            "code", "chiffre", "traduis", "analyse", "précis",
        ]
        est_analytique = any(kw in text.lower() for kw in mots_stricts)

        if est_analytique:
            return {
                "temperature": 0.1,
                "repeat_penalty": 1.05,
                "frequency_penalty": 0.0,
                "system_hint": (
                    "\n\n[Règle stricte : Sois direct. Ne rédige AUCUNE conclusion générale, "
                    "aucun résumé, et arrête-toi immédiatement après ta réponse.]"
                ),
            }
        return {
            "temperature": 0.3 if self.current_skill_path else 0.7,
            "repeat_penalty": 1.15,
            "frequency_penalty": 0.2,
            "system_hint": "",
        }

    def _get_skill_context(self) -> list:
            if self.skill_system_prompt and self.skill_knowledge_base:
                return [
                    {
                        "role": "system",
                        "content": (
                            f"{self.skill_system_prompt}\n\n"
                            f"BASE DE CONNAISSANCES:\n{self.skill_knowledge_base}"
                        ),
                    }
                ]
            return [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]

    def _build_current_message(self, user_text: str, system_hint: str) -> dict:
        """Construit le message en utilisant un pointeur FILE_REF pour les images."""
        prompt_final = user_text + system_hint

        if self.input_packet.data_type == DataType.IMAGE_PATH:
            if not self.is_multimodal:
                return {
                    "role": "user",
                    "content": f"[Alerte: Modèle purement textuel. Image ignorée.]\nRequête : {prompt_final}",
                }

            img_path = self.input_packet.content
            return {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"FILE_REF:{img_path}"}},
                    {"type": "text", "text": prompt_final},
                ],
            }
        return {"role": "user", "content": prompt_final}

    # ── HYDRATATION JUST-IN-TIME (JIT) ──

    def _hydrate_messages(self, messages: list) -> list:
        """Remplace les pointeurs FILE_REF par la véritable image Base64 uniquement pour la plus récente."""
        last_img_idx = self._find_last_image_index(messages)
        return [self._hydrate_single(msg, i == last_img_idx) for i, msg in enumerate(messages)]

    def _find_last_image_index(self, messages: list) -> int:
        for i in reversed(range(len(messages))):
            if isinstance(messages[i].get("content"), list):
                return i
        return -1

    def _hydrate_single(self, msg: dict, is_last_image: bool) -> dict:
        if not isinstance(msg.get("content"), list):
            return msg

        new_content = []
        for item in msg["content"]:
            if (
                item.get("type") == "image_url"
                and str(item.get("image_url", {}).get("url", "")).startswith("FILE_REF:")
            ):
                img_path = item["image_url"]["url"].split("FILE_REF:")[1]
                if is_last_image:
                    new_content.append(self._load_image_to_base64(img_path))
                else:
                    new_content.append(
                        {
                            "type": "text",
                            "text": f"[Image précédente mémorisée: {os.path.basename(img_path)}]",
                        }
                    )
            else:
                new_content.append(item)

        return {"role": msg["role"], "content": new_content}

    def _load_image_to_base64(self, img_path: str) -> dict:
        try:
            with open(img_path, "rb") as img_file:
                b64 = base64.b64encode(img_file.read()).decode("utf-8")
            mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        except Exception as e:
            forge_logger.error(f"[LLMNode] Erreur d'accès à l'image {img_path}: {e}")
            return {"type": "text", "text": "[Image introuvable sur le disque]"}

    # ── INFÉRENCE (DRY) ──

    def _prepare_inference(self) -> tuple[list, dict]:
        if not self.model_loaded:
            raise AIModelMissingError(f"[{self.name}] Aucun modèle chargé.")

        user_text = (
            self.input_packet.content
            if self.input_packet.data_type == DataType.TEXT
            else self.input_packet.metadata.get("prompt", "Décris cette image.")
        )
        params = self._get_dynamic_params(user_text)

        # Interrogation du RAG uniquement si la session a déjà de l'historique ou a été chargée
        retrieved_memory = ""
        if len(self.conversation_history) >= 2:
            retrieved_memory = self.memory_manager.retrieve_context(
                user_text, session_id=self.current_session_id
            )

        combined_system_hint = retrieved_memory + params["system_hint"]

        if self.conversation_history and self.conversation_history[-1]["role"] == "user":
            self.conversation_history.pop()

        self.conversation_history.append(self._build_current_message(user_text, ""))

        msgs_to_send = self._get_skill_context() + self.conversation_history[:-1]
        msgs_to_send.append(self._build_current_message(user_text, combined_system_hint))

        return self._hydrate_messages(msgs_to_send), params

    def _save_answer(self, answer: str) -> None:
        """Sauvegarde la réponse uniquement si elle est valide (évite de corrompre ChromaDB)."""
        clean_ans = answer.strip()

        if not clean_ans or len(clean_ans) < 2:
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            forge_logger.warning(f"[{self.name}] Réponse vide ignorée.")
            return

        if self.conversation_history and self.conversation_history[-1]["role"] == "user":
            last_user_query = self.conversation_history[-1]["content"]
            if isinstance(last_user_query, list):
                last_user_query = " ".join(
                    [i.get("text", "") for i in last_user_query if i.get("type") == "text"]
                )

            self.memory_manager.save_exchange(
                self.current_session_id, last_user_query, clean_ans
            )

        safe_answer = clean_ans if len(clean_ans) < 2500 else clean_ans[:2500] + "\n\n[...]"
        self.conversation_history.append({"role": "assistant", "content": safe_answer})
        self.conversation_history = self.conversation_history[-10:]

    def _run_inference(self) -> DataPacket:
        hydrated_msgs, params = self._prepare_inference()
        forge_logger.log_node_event(
            self.name, "INFERENCE_START", f"max_tokens={self.max_tokens_limit}"
        )

        stop_words = [
            "<|im_end|>", "<|endoftext|>", "USER:",
            "User:", "Observation:",
        ]

        output = self.llm_engine.create_chat_completion(
            messages=hydrated_msgs,
            max_tokens=self.max_tokens_limit,
            temperature=params["temperature"],
            repeat_penalty=params["repeat_penalty"],
            frequency_penalty=params["frequency_penalty"],
            stop=stop_words,
            stream=False,
        )

        answer = output["choices"][0]["message"]["content"].strip()
        self._save_answer(answer)

        return DataPacket(
            DataType.TEXT,
            answer,
            metadata={"model": self.current_model_path},
            source=self.name,
        )

    def stream_inference(self, packet: DataPacket) -> Generator[str, None, None]:
        self.set_input(packet)
        hydrated_msgs, params = self._prepare_inference()

        stop_words = [
            "<|im_end|>", "<|endoftext|>", "USER:",
            "User:", "Observation:",
        ]

        stream = self.llm_engine.create_chat_completion(
            messages=hydrated_msgs,
            max_tokens=self.max_tokens_limit,
            temperature=params["temperature"],
            repeat_penalty=params["repeat_penalty"],
            frequency_penalty=params["frequency_penalty"],
            stop=stop_words,
            stream=True,
        )

        full_answer = ""
        for chunk in stream:
            tok = chunk["choices"][0].get("delta", {}).get("content", "")
            if tok:
                full_answer += tok
                yield tok

        self._save_answer(full_answer)

    def get_status(self) -> dict:
        base = super().get_status()
        if self.hw:
            base.update({"cuda_ok": self.hw.cuda_ok, "n_ctx": self.hw.n_ctx})
        return base