"""
vision_node.py — Nœud d'analyse d'image (LLaVA multimodal).

Corrections v2 :
  - Vérification os.path.exists() avant tout chargement
  - Détection et gestion des espaces dans les chemins (WSL critique)
  - try/except granulaires par étape (clip, llava, inférence)
  - Messages d'erreur actionnables dans l'IHM
"""

import base64
import os
import shutil
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


# ── Helpers chemins ───────────────────────────────────────────────────

def _check_path(path: str, label: str) -> str:
    """
    Vérifie qu'un chemin existe et est lisible.
    Si le chemin contient des espaces sous WSL, tente de créer
    un lien symbolique dans /tmp (sans espaces) pour llama-cpp.

    Retourne le chemin sûr à utiliser.
    Lève ModelLoadError avec un message clair si le fichier est absent.
    """
    if not path:
        raise ModelLoadError(f"{label} : chemin vide.")

    if not os.path.exists(path):
        raise ModelLoadError(
            f"{label} introuvable :\n"
            f"  {path}\n\n"
            f"Vérifiez que le fichier existe dans le dossier models/."
        )

    if not os.access(path, os.R_OK):
        raise ModelLoadError(
            f"{label} non lisible (permissions insuffisantes) :\n  {path}"
        )

    # ── Gestion des espaces dans le chemin (problème fréquent sous WSL) ──
    if " " in path:
        forge_logger.warning(
            f"[VisionNode] Le chemin contient des espaces, "
            f"ce qui peut faire échouer llama-cpp : {path}"
        )
        safe_path = _make_space_free_path(path, label)
        if safe_path != path:
            forge_logger.info(
                f"[VisionNode] Lien symbolique créé : {safe_path}"
            )
            return safe_path

    return path


def _make_space_free_path(original: str, label: str) -> str:
    """
    Crée un lien symbolique dans /tmp sans espaces.
    Ex: '/mnt/c/Mon Dossier/model.gguf' → '/tmp/nf_model.gguf'
    """
    try:
        ext      = os.path.splitext(original)[1]
        safe_name = f"nf_{label.lower().replace(' ', '_')}{ext}"
        safe_path = os.path.join("/tmp", safe_name)

        # Supprimer l'ancien lien si obsolète
        if os.path.islink(safe_path) and os.readlink(safe_path) != original:
            os.remove(safe_path)

        if not os.path.exists(safe_path):
            os.symlink(original, safe_path)

        return safe_path
    except Exception as e:
        forge_logger.warning(
            f"[VisionNode] Impossible de créer le lien symlink ({e}). "
            "Tentative avec le chemin original."
        )
        return original


# ── VisionNode ────────────────────────────────────────────────────────

class VisionNode(BaseNode):
    """
    Nœud multimodal LLaVA : image → texte.
    Compatible LLaVA 1.5 et 1.6 (Mistral, Vicuna).
    """

    ACCEPTED_TYPES = (DataType.IMAGE_PATH,)

    _MIME_MAP = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    def __init__(self, name: str = "Vision Analyzer"):
        super().__init__(name)
        self.llm_engine:          Optional[object] = None
        self.current_model_path:  Optional[str]    = None
        self.mmproj_path:         Optional[str]    = None
        self.text_prompt:         str              = "Décris cette image en détail."

    # ── Chargement ────────────────────────────────────────────────────

    def load_model(self, model_path: str, mmproj_path: str) -> None:  # type: ignore
        """
        Charge le modèle LLaVA + son projecteur multimodal.

        Erreurs possibles et messages correspondants :
          - Fichier absent        → chemin affiché + conseil
          - Espaces dans le chemin → symlink automatique dans /tmp
          - Mismatch mmproj       → conseil de télécharger le bon mmproj
          - llama-cpp sans LLaVA  → commande de recompilation affichée
        """
        if Llama is None or Llava15ChatHandler is None:
            raise ModelLoadError(
                "llama-cpp-python n'est pas installé ou ne supporte pas LLaVA.\n\n"
                "Réinstallez avec :\n"
                "  pip uninstall llama-cpp-python -y\n"
                '  CMAKE_ARGS="-DLLAVA_BUILD=ON" pip install llama-cpp-python --no-cache-dir'
            )

        # ── Étape 1 : Vérification et nettoyage des chemins ──
        try:
            safe_model  = _check_path(model_path,  "Modèle LLaVA")
            safe_mmproj = _check_path(mmproj_path, "Projecteur mmproj")
        except ModelLoadError:
            raise

        forge_logger.log_node_event(
            self.name, "LOAD_VISION_MODEL",
            f"model={os.path.basename(safe_model)}  "
            f"mmproj={os.path.basename(safe_mmproj)}"
        )

        # ── Étape 2 : Chargement du projecteur CLIP ──
        try:
            chat_handler = Llava15ChatHandler(clip_model_path=safe_mmproj)
        except Exception as e:
            err_str = str(e).lower()
            if "mismatch" in err_str or "n_embd" in err_str:
                raise ModelLoadError(
                    "Incompatibilité modèle / mmproj.\n\n"
                    "Le fichier mmproj ne correspond pas au modèle LLaVA.\n"
                    "Chaque modèle LLaVA a son propre projecteur :\n\n"
                    "  llava-v1.6-mistral-7b  →  cjpais/llava-1.6-mistral-7b-gguf\n"
                    "                              fichier : mmproj-model-f16.gguf\n\n"
                    "  llava-v1.5-7b          →  mys/gguf-llava-v1.5-7b\n"
                    "                              fichier : mmproj-model-f16.gguf\n\n"
                    f"Erreur brute : {e}"
                )
            elif "failed to load" in err_str or "mtmd" in err_str:
                raise ModelLoadError(
                    "Échec du chargement du projecteur multimodal (mmproj).\n\n"
                    "Causes possibles :\n"
                    "1. Fichier mmproj corrompu ou incomplet.\n"
                    "2. llama-cpp-python compilé sans support LLaVA.\n\n"
                    "Solution :\n"
                    "  pip uninstall llama-cpp-python -y\n"
                    '  CMAKE_ARGS="-DLLAVA_BUILD=ON" '
                    "pip install llama-cpp-python --no-cache-dir\n\n"
                    f"Erreur brute : {e}"
                )
            else:
                raise ModelLoadError(
                    f"Erreur projecteur CLIP : {e}\n\n"
                    f"Fichier mmproj : {safe_mmproj}"
                )

        # ── Étape 3 : Chargement du modèle LLaVA ──
        try:
            self.llm_engine = Llama(
                model_path=safe_model,
                chat_handler=chat_handler,
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False,
            )
        except FileNotFoundError:
            raise ModelLoadError(
                f"Modèle LLaVA introuvable après symlink :\n  {safe_model}\n"
                "Vérifiez que le fichier est accessible."
            )
        except Exception as e:
            raise ModelLoadError(
                f"Échec du chargement LLaVA :\n  {e}\n\n"
                f"Modèle  : {safe_model}\n"
                f"mmproj  : {safe_mmproj}"
            )

        self.current_model_path = model_path   # conserver le chemin original
        self.mmproj_path        = mmproj_path
        super().load_model(model_path)
        forge_logger.log_node_event(self.name, "VISION_READY",
                                    os.path.basename(model_path))

    # ── Prompt ────────────────────────────────────────────────────────

    def set_prompt(self, prompt: str) -> None:
        self.text_prompt = prompt
        forge_logger.log_node_event(self.name, "PROMPT_SET", prompt[:80])

    # ── Inférence ─────────────────────────────────────────────────────

    def _run_inference(self) -> DataPacket:
        if self.input_packet.data_type != DataType.IMAGE_PATH:
            raise InvalidDataTypeError(
                f"VisionNode requiert IMAGE_PATH, "
                f"reçu : {self.input_packet.data_type.value}"
            )

        image_path: str = self.input_packet.content

        # Vérification de l'image avant envoi
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image introuvable : '{image_path}'"
            )

        forge_logger.log_node_event(self.name, "VISION_INFERENCE_START",
                                    os.path.basename(image_path))

        # Conversion base64
        try:
            data_url = self._image_to_data_url(image_path)
        except Exception as e:
            raise RuntimeError(f"Impossible de lire l'image : {e}")

        # Appel LLaVA
        try:
            response = self.llm_engine.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": data_url}},
                        {"type": "text", "text": self.text_prompt},
                    ],
                }],
                max_tokens=800,
            )
            result_text: str = (
                response["choices"][0]["message"]["content"].strip()
            )
        except Exception as e:
            raise RuntimeError(
                f"Erreur pendant l'analyse LLaVA : {e}\n"
                "Le modèle ou le mmproj pourrait être incompatible."
            )

        if not result_text:
            result_text = "(Analyse vide — vérifiez le modèle LLaVA et le mmproj)"

        forge_logger.log_node_event(self.name, "VISION_INFERENCE_END",
                                    f"len={len(result_text)}")

        return DataPacket(
            DataType.TEXT, result_text,
            metadata={
                "image_path": image_path,
                "prompt":     self.text_prompt,
                "model":      self.current_model_path,
                "mmproj":     self.mmproj_path,
            },
            source=self.name,
        )

    # ── Utilitaire ────────────────────────────────────────────────────

    def _image_to_data_url(self, image_path: str) -> str:
        ext       = os.path.splitext(image_path)[1].lower()
        mime_type = self._MIME_MAP.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"
