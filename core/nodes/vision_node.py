"""
vision_node.py — Nœud d'analyse d'image (LLaVA multimodal).
"""

import base64
import os
import io
from typing import Optional

from core.logger import forge_logger
from core.node import BaseNode, ModelLoadError
from core.types import DataPacket, DataType
from core.utils.hardware import get_profile
from core.utils.file_utils import check_safe_path

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    Llama = None
    Llava15ChatHandler = None


class VisionNode(BaseNode):
    ACCEPTED_TYPES = (DataType.IMAGE_PATH,)
    _MIME_MAP = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    def __init__(self, name: str = "Vision Analyzer"):
        super().__init__(name)
        self.llm_engine: Optional[object] = None
        self.chat_handler: Optional[object] = None
        self.current_model_path: Optional[str] = None
        self.mmproj_path: Optional[str] = None
        self.text_prompt: str = "Décris cette image avec le plus de détails possible, en lisant tout le texte visible."
        self.high_res_mode: bool = True 

    def set_high_res_mode(self, enabled: bool):
        self.high_res_mode = enabled

    def unload_model(self) -> None:
        if self.llm_engine:
            try: self.llm_engine.close()
            except Exception: pass
            self.llm_engine = None
            
        if getattr(self, 'chat_handler', None):
            self.chat_handler = None
            
        self.current_model_path = None
        self.mmproj_path = None
        self.model_loaded = False
        import gc
        gc.collect()
        forge_logger.info(f"[{self.name}] Moteur Vision déchargé.")

    def load_model(self, model_path: str, mmproj_path: str) -> None: 
        if Llama is None: raise ModelLoadError("llama-cpp-python sans support Vision.")

        safe_model  = check_safe_path(model_path,  "Modèle Vision")
        safe_mmproj = check_safe_path(mmproj_path, "Projecteur mmproj")
        hw = get_profile()
        nom_fichier = os.path.basename(safe_model).lower()
        
        try:
            if "moondream" in nom_fichier:
                from llama_cpp.llama_chat_format import MoondreamChatHandler
                self.chat_handler = MoondreamChatHandler(clip_model_path=safe_mmproj)
                format_detecte = None
            else:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                self.chat_handler = Llava15ChatHandler(clip_model_path=safe_mmproj)
                format_detecte = "mistral" if ("mistral" in nom_fichier or "mixtral" in nom_fichier) else ("chatml" if any(k in nom_fichier for k in ["chatml", "qwen", "hermes"]) else "vicuna")
        except ImportError:
            forge_logger.error("Handler spécifique introuvable. Mettez à jour llama-cpp-python.")
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            self.chat_handler = Llava15ChatHandler(clip_model_path=safe_mmproj)
            format_detecte = "vicuna"
            
        forge_logger.info(f"[{self.name}] Moteur Vision configuré pour : {nom_fichier}")

        try:
            self.llm_engine = Llama(
                model_path=safe_model, chat_handler=self.chat_handler,
                chat_format=format_detecte, n_ctx=4096,
                n_gpu_layers=hw.n_gpu_layers, verbose=False,
            )
        except Exception as e:
            raise ModelLoadError(f"Échec du chargement Vision :\n  {e}")

        self.current_model_path = model_path
        self.mmproj_path = mmproj_path
        super().load_model(model_path)

    def set_prompt(self, prompt: str) -> None:
        self.text_prompt = prompt

    def _run_inference(self) -> DataPacket:
        image_path: str = self.input_packet.content
        data_url = self._image_to_data_url(image_path)

        prompt_bas = self.text_prompt.lower()
        temp = 0.1 if any(kw in prompt_bas for kw in ["lis", "texte", "exo", "exercice", "mcd", "sql", "code", "chiffre"]) else 0.4

        try:
            response = self.llm_engine.create_chat_completion(
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": self.text_prompt}]}],
                max_tokens=1024, temperature=temp, repeat_penalty=1.2, frequency_penalty=0.5,
            )
            result_text: str = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Erreur pendant l'analyse LLaVA : {e}")

        if not result_text or "MSGMSG" in result_text:
            result_text = "(Le module visuel a rencontré une difficulté technique sur cette zone de l'image.)"

        return DataPacket(DataType.TEXT, result_text, metadata={"image_path": image_path, "model": self.current_model_path, "temp_used": temp}, source=self.name)

    def _image_to_data_url(self, image_path: str) -> str:
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = self._MIME_MAP.get(ext, "image/jpeg")
        from PIL import Image

        dim = (672, 672) if self.high_res_mode else (336, 336)
        qual = 95 if self.high_res_mode else 85
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB': img = img.convert('RGB')
                img.thumbnail(dim)
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=qual)
                b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            forge_logger.error(f"[VisionNode] Erreur downsampling : {e}")
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{b64}"