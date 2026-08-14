"""
vision_node.py — Nœud d'analyse d'image (LLaVA multimodal).
"""

import base64
import os
import shutil
from typing import Optional

from core.logger import forge_logger
from core.node import BaseNode, InvalidDataTypeError, ModelLoadError
from core.types import DataPacket, DataType
from core.utils.hardware import get_profile

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    Llama = None
    Llava15ChatHandler = None

def _check_path(path: str, label: str) -> str:
    if not path:
        raise ModelLoadError(f"{label} : chemin vide.")
    if not os.path.exists(path):
        raise ModelLoadError(f"{label} introuvable :\n  {path}")
    if not os.access(path, os.R_OK):
        raise ModelLoadError(f"{label} non lisible (permissions).")

    if " " in path:
        forge_logger.warning(f"[VisionNode] Espaces détectés : {path}")
        safe_path = _make_space_free_path(path, label)
        return safe_path
    return path

def _make_space_free_path(original: str, label: str) -> str:
    try:
        ext       = os.path.splitext(original)[1]
        safe_name = f"nf_{label.lower().replace(' ', '_')}{ext}"
        safe_path = os.path.join("/tmp", safe_name)
        if os.path.islink(safe_path) and os.readlink(safe_path) != original:
            os.remove(safe_path)
        if not os.path.exists(safe_path):
            os.symlink(original, safe_path)
        return safe_path
    except Exception:
        return original

class VisionNode(BaseNode):
    ACCEPTED_TYPES = (DataType.IMAGE_PATH,)
    _MIME_MAP = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    def __init__(self, name: str = "Vision Analyzer"):
        super().__init__(name)
        self.llm_engine:          Optional[object] = None
        self.chat_handler:        Optional[object] = None
        self.current_model_path:  Optional[str]    = None
        self.mmproj_path:         Optional[str]    = None
        self.text_prompt:         str              = "Décris cette image avec le plus de détails possible, en lisant tout le texte visible."

        self.high_res_mode:       bool             = True 

    def set_high_res_mode(self, enabled: bool):
        self.high_res_mode = enabled

    def unload_model(self) -> None:
        if self.llm_engine is not None:
            try:
                self.llm_engine.close()
            except Exception:
                pass
            del self.llm_engine
            self.llm_engine = None
            
        if getattr(self, 'chat_handler', None) is not None:
            del self.chat_handler
            self.chat_handler = None
            
        self.current_model_path = None
        self.mmproj_path = None
        self.model_loaded = False
        import gc
        gc.collect()
        forge_logger.info(f"[{self.name}] Moteur Vision déchargé, VRAM libérée.")

    def load_model(self, model_path: str, mmproj_path: str) -> None: 
        if Llama is None or Llava15ChatHandler is None:
            raise ModelLoadError("llama-cpp-python sans support LLaVA.")

        safe_model  = _check_path(model_path,  "Modèle LLaVA")
        safe_mmproj = _check_path(mmproj_path, "Projecteur mmproj")
        hw = get_profile()

        try:
            self.chat_handler = Llava15ChatHandler(clip_model_path=safe_mmproj)
        except Exception as e:
            raise ModelLoadError(f"Erreur projecteur CLIP : {e}")

        nom_fichier = os.path.basename(safe_model).lower()
        format_detecte = "vicuna"
        if "mistral" in nom_fichier or "mixtral" in nom_fichier:
            format_detecte = "mistral"
        elif "chatml" in nom_fichier or "qwen" in nom_fichier or "hermes" in nom_fichier:
            format_detecte = "chatml"
            
        forge_logger.info(f"[{self.name}] Auto-détection format vision : {format_detecte}")

        try:
            self.llm_engine = Llama(
                model_path=safe_model,
                chat_handler=self.chat_handler,
                chat_format=format_detecte,
                n_ctx=4096,
                n_gpu_layers=hw.n_gpu_layers,
                verbose=False,
            )
        except Exception as e:
            raise ModelLoadError(f"Échec du chargement LLaVA :\n  {e}")

        self.current_model_path = model_path
        self.mmproj_path        = mmproj_path
        super().load_model(model_path)

    def set_prompt(self, prompt: str) -> None:
        self.text_prompt = prompt

    def _run_inference(self) -> DataPacket:
        image_path: str = self.input_packet.content
        data_url = self._image_to_data_url(image_path)

        prompt_bas = self.text_prompt.lower()
        mots_precision = ["lis", "texte", "exo", "exercice", "mcd", "sql", "code", "chiffre"]
        temp = 0.1 if any(kw in prompt_bas for kw in mots_precision) else 0.4

        try:
            response = self.llm_engine.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}}, # ✅ IMAGE EN PREMIER (Indispensable)
                        {"type": "text", "text": self.text_prompt},           # ✅ TEXTE EN DEUXIÈME
                    ],
                }],
                max_tokens=1024,
                temperature=temp,
                repeat_penalty=1.2,
                frequency_penalty=0.5,
            )
            result_text: str = response["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            raise RuntimeError(f"Erreur pendant l'analyse LLaVA : {e}")

        if not result_text or "MSGMSG" in result_text:
            result_text = "(Le module visuel a rencontré une difficulté technique sur cette zone de l'image.)"

        return DataPacket(
            DataType.TEXT, result_text,
            metadata={"image_path": image_path, "model": self.current_model_path, "temp_used": temp},
            source=self.name,
        )

    def _image_to_data_url(self, image_path: str) -> str:
        """Choisit dynamiquement la résolution de l'image (Vitesse vs Précision)."""
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = self._MIME_MAP.get(ext, "image/jpeg")

        if self.high_res_mode:
            # MODE PRÉCISION (MCD, Textes) -> Garde l'image intacte pour le Slicing
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{b64}"
            
        else:
            # MODE RAPIDE (Paysages, Objets) -> Downsampling à 336x336
            from PIL import Image
            import io
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.thumbnail((336, 336))
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{b64}"
            except Exception as e:
                forge_logger.error(f"[VisionNode] Erreur downsampling : {e}")
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                return f"data:{mime_type};base64,{b64}"