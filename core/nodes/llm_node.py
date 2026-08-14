"""
llm_node.py — Moteur d'inférence haute performance Neural Forge.
Version 4.0 : Système de "Skills" RAG dynamique (Fichiers .skill) et Fallback GPU.
"""

import os
import gc
import json
from typing import Generator, Optional, List, Dict

from core.logger import forge_logger
from core.node import AIModelMissingError, BaseNode, ModelLoadError
from core.types import DataPacket, DataType
from core.utils.hardware import get_profile, HardwareProfile

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# ── Helpers chemins ───────────────────────────────────────────────────

def _check_path(path: str, label: str) -> str:
    if not path:
        raise ModelLoadError(f"{label} : chemin vide.")
    if not os.path.exists(path):
        raise ModelLoadError(f"{label} introuvable : '{path}'")
    if not os.access(path, os.R_OK):
        raise ModelLoadError(f"{label} non lisible (permissions).")

    if " " in path:
        forge_logger.warning(f"[{label}] Espaces détectés. Création d'un symlink...")
        return _make_space_free_path(path, label)
    return path

def _make_space_free_path(original: str, label: str) -> str:
    try:
        ext = os.path.splitext(original)[1]
        safe_name = f"nf_{label.lower().replace(' ', '_')}{ext}"
        safe_path = os.path.join("/tmp", safe_name)
        if os.path.islink(safe_path) and os.readlink(safe_path) != original:
            os.remove(safe_path)
        if not os.path.exists(safe_path):
            os.symlink(original, safe_path)
        return safe_path
    except Exception:
        return original


# ── LLMNode ───────────────────────────────────────────────────────────

class LLMNode(BaseNode):

    ACCEPTED_TYPES = (DataType.TEXT,)

    def __init__(self, name: str = "LLM Generator"):
        super().__init__(name)
        self.llm_engine:         Optional[object]          = None
        self.draft_engine:       Optional[object]          = None
        self.current_model_path: Optional[str]             = None
        
        # NOUVEAU : Variables pour les compétences (Skills)
        self.current_skill_path: Optional[str]             = None
        self.skill_system_prompt: str                      = ""
        self.skill_knowledge_base: str                     = ""
        
        self._turbo_active:      bool                      = False
        self.hw:                 Optional[HardwareProfile] = None

        self.conversation_history: List[Dict[str, str]]    = []

    def unload_model(self) -> None:
        """Décharge le modèle et FORCE la destruction du contexte C++."""
        if self.llm_engine is not None:
            try:
                self.llm_engine.close()
            except Exception:
                pass
            del self.llm_engine
            self.llm_engine = None
            
        if self.draft_engine is not None:
            try:
                self.draft_engine.close()
            except Exception:
                pass
            del self.draft_engine
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

    # ── Moteur Skill (RAG) ────────────────────────────────────────────

    def _get_skill_context(self) -> list:
        """Prépare le prompt système si une compétence est équipée."""
        if self.skill_system_prompt and self.skill_knowledge_base:
            forge_logger.info(f"[{self.name}] Skill actif injecté dans le contexte.")
            return [{
                "role": "system",
                "content": f"{self.skill_system_prompt}\n\nBASE DE CONNAISSANCES:\n{self.skill_knowledge_base}"
            }]
        return []

    # ── Chargement ────────────────────────────────────────────────────

    def load_model(self, model_path: str, skill_path: Optional[str] = None, draft_model_path: Optional[str] = None, turbo: bool = False) -> None:
        if Llama is None:
            raise ModelLoadError("llama-cpp-python n'est pas installé.")

        self.unload_model()
        self.hw = get_profile()

        safe_model_path = _check_path(model_path, "Modèle Principal")
        safe_draft_path = _check_path(draft_model_path, "Modèle Draft") if draft_model_path else None
        
        # ── Lecture du fichier .skill ──
        if skill_path and os.path.exists(skill_path):
            try:
                with open(skill_path, "r", encoding="utf-8") as f:
                    skill_data = json.load(f)
                    if skill_data.get("type") == "rag_skill":
                        self.current_skill_path = skill_path
                        self.skill_system_prompt = skill_data.get("system_prompt", "")
                        self.skill_knowledge_base = skill_data.get("knowledge_base", "")
                        forge_logger.info(f"[{self.name}] Compétence '{skill_data.get('name')}' chargée avec succès.")
                    else:
                        forge_logger.warning(f"[{self.name}] Fichier skill invalide ignoré.")
            except Exception as e:
                forge_logger.error(f"[{self.name}] Erreur de lecture du skill : {e}")

        self._turbo_active = False
        if turbo and safe_draft_path and self.hw.turbo_eligible:
            self._turbo_active = True

        # Context Window à 4096 pour supporter la base de connaissances du Skill
        kwargs = dict(
            model_path      = safe_model_path,
            n_ctx           = 4096, 
            n_threads       = self.hw.n_threads,
            n_threads_batch = self.hw.n_threads,
            n_batch         = self.hw.n_batch,
            n_gpu_layers    = self.hw.n_gpu_layers,
            use_mmap        = self.hw.use_mmap,
            logits_all      = False,
            verbose         = False,
        )
        if self.hw.flash_attn:
            kwargs["flash_attn"] = True

        try:
            self.llm_engine = Llama(**kwargs)
        except Exception as e:
            forge_logger.warning(f"[{self.name}] Échec GPU. Fallback CPU...")
            kwargs["n_gpu_layers"] = 0
            kwargs["n_ctx"] = 2048 
            try:
                self.llm_engine = Llama(**kwargs)
            except Exception as fallback_err:
                raise ModelLoadError(f"Échec total : {fallback_err}")

        self.current_model_path = model_path
        self.model_loaded = True

    # ── Inférence ─────────────────────────────────────────────────────

    def _run_inference(self) -> DataPacket:
        prompt = self.input_packet.content
        self.conversation_history.append({"role": "user", "content": prompt})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        forge_logger.log_node_event(self.name, "INFERENCE_START", "Génération en cours...")

        # INJECTION DU SKILL
        messages_to_send = self._get_skill_context() + self.conversation_history

        output = self.llm_engine.create_chat_completion(
            messages=messages_to_send,
            max_tokens=1024,
            temperature=0.3 if self.current_skill_path else 0.7, # Température basse si un skill est actif
            stream=False
        )
        
        answer: str = output["choices"][0]["message"]["content"].strip()
        self.conversation_history.append({"role": "assistant", "content": answer})

        return DataPacket(DataType.TEXT, answer, metadata={"model": self.current_model_path}, source=self.name)

    def stream_inference(self, packet: DataPacket) -> Generator[str, None, None]:
        self.set_input(packet)
        if not self.model_loaded:
            raise AIModelMissingError(f"[{self.name}] Aucun modèle chargé.")

        self.conversation_history.append({"role": "user", "content": self.input_packet.content})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        # INJECTION DU SKILL
        messages_to_send = self._get_skill_context() + self.conversation_history

        stream = self.llm_engine.create_chat_completion(
            messages=messages_to_send,
            max_tokens=1024,
            stream=True,
            temperature=0.3 if self.current_skill_path else 0.7,
        )
        
        full_answer = ""
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            tok = delta.get("content", "")
            if tok:
                full_answer += tok
                yield tok

        self.conversation_history.append({"role": "assistant", "content": full_answer})

    def get_status(self) -> dict:
        base = super().get_status()
        if self.hw:
            base.update({"cuda_ok": self.hw.gpu.cuda_ok, "n_ctx": self.hw.n_ctx})
        return base