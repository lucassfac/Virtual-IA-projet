"""
llm_node.py — Moteur d'inférence haute performance Neural Forge.
Paramètres auto-détectés depuis hardware.py à chaque load_model().
"""

import os
from typing import Generator, Optional

from core.logger import forge_logger
from core.node import AIModelMissingError, BaseNode, ModelLoadError
from core.types import DataPacket, DataType
from core.utils.hardware import get_profile, HardwareProfile

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class LLMNode(BaseNode):

    ACCEPTED_TYPES = (DataType.TEXT,)

    def __init__(self, name: str = "LLM Generator"):
        super().__init__(name)
        self.llm_engine:         Optional[object]          = None
        self.draft_engine:       Optional[object]          = None
        self.current_model_path: Optional[str]             = None
        self.current_lora_path:  Optional[str]             = None
        self._lora_is_simulated: bool                      = False
        self._turbo_active:      bool                      = False
        self.hw:                 Optional[HardwareProfile] = None

    # ── LoRA validation ───────────────────────────────────────────────

    @staticmethod
    def _is_real_lora(path: str) -> bool:
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
            return magic == b"GGUF" or magic[:3] == b"ggm"
        except OSError:
            return False

    # ── Chargement ────────────────────────────────────────────────────

    def load_model(
        self,
        model_path:       str,
        lora_path:        Optional[str] = None,
        draft_model_path: Optional[str] = None,
        turbo:            bool          = False,
    ) -> None:
        if Llama is None:
            raise ModelLoadError("llama-cpp-python n'est pas installé.")

        self.hw = get_profile()

        # ── LoRA ──
        effective_lora = None
        self._lora_is_simulated = False
        if lora_path:
            if self._is_real_lora(lora_path):
                effective_lora = lora_path
            else:
                self._lora_is_simulated = True
                forge_logger.warning(
                    f"[{self.name}] LoRA simulé ignoré : "
                    f"'{os.path.basename(lora_path)}'"
                )

        # ── Turbo gate ──
        self._turbo_active = False
        if turbo:
            if not self.hw.turbo_eligible:
                forge_logger.warning(
                    f"[{self.name}] Turbo refusé — RAM {self.hw.ram_total_gb} Go < 12 Go."
                )
            elif not draft_model_path or not os.path.exists(draft_model_path):
                forge_logger.warning(
                    f"[{self.name}] Turbo refusé — pas de modèle draft."
                )
            else:
                self._turbo_active = True

        mode = "TURBO" if self._turbo_active else "STANDARD"
        forge_logger.log_node_event(
            self.name, "LOAD_MODEL",
            f"[{mode}] {os.path.basename(model_path)}  "
            f"threads={self.hw.n_threads}  ctx={self.hw.n_ctx}  "
            f"gpu={self.hw.n_gpu_layers}  batch={self.hw.n_batch}  "
            f"flash={self.hw.flash_attn}"
        )

        # ── Moteur principal ──
        try:
            kwargs = dict(
                model_path    = model_path,
                lora_path     = effective_lora,
                n_ctx         = self.hw.n_ctx,
                n_threads     = self.hw.n_threads,
                n_threads_batch = self.hw.n_threads,
                n_batch       = self.hw.n_batch,
                n_gpu_layers  = self.hw.n_gpu_layers,
                use_mmap      = self.hw.use_mmap,
                use_mlock     = False,
                verbose       = False,
            )
            # Paramètres optionnels selon version llama-cpp
            if self.hw.flash_attn:
                kwargs["flash_attn"] = True
            try:
                kwargs["n_ubatch"] = self.hw.n_ubatch
            except Exception:
                pass

            self.llm_engine = Llama(**kwargs)

        except TypeError as e:
            # Vieille API — retire les kwargs inconnus
            for k in ("flash_attn", "n_ubatch"):
                kwargs.pop(k, None)
            self.llm_engine = Llama(**kwargs)
        except FileNotFoundError:
            raise ModelLoadError(
                f"[{self.name}] Introuvable : '{model_path}'"
            )
        except Exception as e:
            raise ModelLoadError(f"[{self.name}] Chargement échoué : {e}")

        # ── Modèle draft (Turbo) ──
        self.draft_engine = None
        if self._turbo_active:
            try:
                self.draft_engine = Llama(
                    model_path   = draft_model_path,
                    n_ctx        = self.hw.n_ctx,
                    n_threads    = self.hw.n_threads,
                    n_gpu_layers = self.hw.n_gpu_layers,
                    use_mmap     = True,
                    verbose      = False,
                )
                forge_logger.log_node_event(
                    self.name, "TURBO_DRAFT_OK",
                    os.path.basename(draft_model_path)
                )
            except Exception as e:
                forge_logger.warning(
                    f"[{self.name}] Draft échoué ({e}) — retour Standard."
                )
                self._turbo_active = False
                self.draft_engine  = None

        self.current_model_path = model_path
        self.current_lora_path  = lora_path
        super().load_model(model_path)
        forge_logger.log_node_event(
            self.name, "READY",
            f"[{'TURBO' if self._turbo_active else 'STANDARD'}] "
            f"{os.path.basename(model_path)}"
        )

    # ── Inférence bloquante ───────────────────────────────────────────

    def _run_inference(self) -> DataPacket:
        prompt = self.input_packet.content
        output = self.llm_engine(
            f"Question: {prompt}\nRéponse:",
            max_tokens=2048,
            stop=["Question:"],
            echo=False,
            temperature=0.7,
            repeat_penalty=1.1,
        )
        text: str   = output["choices"][0]["text"].strip() or "(Réponse vide)"
        finish: str = output["choices"][0].get("finish_reason", "unknown")
        return DataPacket(
            DataType.TEXT, text,
            metadata={
                "model":         self.current_model_path,
                "lora":          self.current_lora_path,
                "lora_applied":  not self._lora_is_simulated,
                "turbo":         self._turbo_active,
                "finish_reason": finish,
            },
            source=self.name,
        )

    # ── Streaming ─────────────────────────────────────────────────────

    def stream_inference(self, packet: DataPacket) -> Generator[str, None, None]:
        self.set_input(packet)
        if not self.model_loaded:
            raise AIModelMissingError(f"[{self.name}] Aucun modèle chargé.")

        forge_logger.log_node_event(
            self.name, "STREAM_START",
            f"{'TURBO' if self._turbo_active else 'STANDARD'}"
        )

        stream = self.llm_engine(
            f"Question: {self.input_packet.content}\nRéponse:",
            max_tokens=2048,
            stop=["Question:"],
            echo=False,
            stream=True,
            temperature=0.7,
            repeat_penalty=1.1,
        )
        for chunk in stream:
            tok: str = chunk["choices"][0]["text"]
            if tok:
                yield tok

        forge_logger.log_node_event(self.name, "STREAM_END")

    # ── Statut ────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        base = super().get_status()
        if self.hw:
            base.update({
                "turbo_active":   self._turbo_active,
                "turbo_eligible": self.hw.turbo_eligible,
                "ram_total_gb":   self.hw.ram_total_gb,
                "gpu":            self.hw.gpu.name,
                "gpu_vram_gb":    self.hw.gpu.vram_gb,
                "cuda_ok":        self.hw.gpu.cuda_ok,
                "n_threads":      self.hw.n_threads,
                "n_ctx":          self.hw.n_ctx,
                "flash_attn":     self.hw.flash_attn,
                "draft_loaded":   self.draft_engine is not None,
            })
        return base
