"""
eel_main.py — Point d'entrée Neural Forge avec interface Eel (HTML/JS).

Installe Eel si nécessaire :
    pip install eel

Lancement :
    python eel_main.py
"""

import os
import sys
import threading
import tkinter.filedialog as fd
import tkinter
from typing import Optional

import eel

# ── Paths ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.nodes.llm_node   import LLMNode
from core.nodes.vision_node import VisionNode
from core.types             import DataPacket, DataType
from core.session_manager   import (
    save_last_model, load_last_model,
    save_conversation, list_conversations,
    get_conversation,
)
from core.model_manager     import save_hf_token
from core.utils.hardware    import get_profile

# ── Init Eel ───────────────────────────────────────────────────────────
eel.init('web')

# ── Nœuds partagés ────────────────────────────────────────────────────
llm_node    = LLMNode(name="LLM-Principal")
vision_node = VisionNode(name="Vision-Principal")

# Worker stream en cours
_stream_thread:  Optional[threading.Thread] = None
_stop_requested: bool                        = False


# ── Matériel ───────────────────────────────────────────────────────────

@eel.expose
def get_hw_info():
    hw = get_profile()
    return {
        "badge":          hw.badge_text,
        "color":          hw.badge_color,
        "turbo_eligible": hw.turbo_eligible,
        "turbo_reason":   hw.turbo_reason,
        "ram_gb":         hw.ram_total_gb,
        "gpu":            hw.gpu.name,
    }


# ── Chargement modèle ─────────────────────────────────────────────────

@eel.expose
def load_model(model_path: str, lora_path: str = "",
               draft_path: str = "", turbo: bool = False):
    """Charge le LLMNode depuis l'interface."""
    try:
        llm_node.load_model(
            model_path=model_path,
            lora_path=lora_path or None,
            draft_model_path=draft_path or None,
            turbo=turbo,
        )
        name = os.path.basename(model_path)
        save_last_model(model_path, lora_path)
        eel.set_model_status('llm', True, name)
        return {"ok": True, "name": name, "message": f"Prêt : {name}"}
    except Exception as e:
        return {"ok": False, "name": "", "message": str(e)}


@eel.expose
def get_last_model():
    model_path, lora_path = load_last_model()
    return {"model_path": model_path, "lora_path": lora_path}


# ── Envoi message & streaming ─────────────────────────────────────────

@eel.expose
def send_message(text: str, turbo: bool = False,
                 file_path: str = None, file_type: str = None):
    """
    Reçoit le message de l'UI.
    Lance le streaming dans un thread séparé pour ne pas bloquer Eel.
    """
    global _stream_thread, _stop_requested
    _stop_requested = False

    def _run():
        global _stop_requested
        try:
            if file_type == 'image' and file_path and vision_node.model_loaded:
                # Vision — pas de streaming, réponse complète
                vision_node.set_prompt(text or "Décris cette image en détail.")
                packet = DataPacket(DataType.IMAGE_PATH, file_path, source="user")
                vision_node.set_input(packet)
                result = vision_node.process()
                eel.display_token(result.content)
                eel.stream_done(None)
                return

            if file_type == 'doc' and file_path:
                from core.document_reader import read_document, DocumentReadError
                try:
                    doc = read_document(file_path, llm_node=llm_node)
                    prompt = doc.build_prompt(text)
                except DocumentReadError as e:
                    eel.stream_done(str(e))
                    return
            else:
                prompt = text or "Bonjour !"

            if not llm_node.model_loaded:
                eel.stream_done("Aucun modèle chargé — ouvrez Paramètres.")
                return

            packet = DataPacket(DataType.TEXT, prompt, source="user")
            for token in llm_node.stream_inference(packet):
                if _stop_requested:
                    break
                eel.display_token(token)

            eel.stream_done(None)

        except Exception as e:
            eel.stream_done(str(e))

    _stream_thread = threading.Thread(target=_run, daemon=True)
    _stream_thread.start()


@eel.expose
def stop_generation():
    global _stop_requested
    _stop_requested = True


# ── Fichiers / parcourir ──────────────────────────────────────────────

@eel.expose
def browse_file(file_type: str) -> str:
    """
    Ouvre un dialog de sélection de fichier natif.
    Retourne le chemin sélectionné ou '' si annulé.
    """
    root = tkinter.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    filters = {
        "model":    [("GGUF models", "*.gguf"), ("All files", "*.*")],
        "lora":     [("LoRA adapters", "*.lora *.bin"), ("All files", "*.*")],
        "draft":    [("GGUF models", "*.gguf"), ("All files", "*.*")],
        "image":    [("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        "document": [("Documents", "*.txt *.md *.pdf *.docx *.json *.jsonl"), ("All files", "*.*")],
    }

    path = fd.askopenfilename(
        parent=root,
        title=f"Sélectionner un fichier",
        filetypes=filters.get(file_type, [("All files", "*.*")]),
        initialdir=os.path.join(os.path.dirname(__file__), "models")
        if file_type in ("model", "lora", "draft") else os.path.expanduser("~"),
    )
    root.destroy()
    return path or ""


# ── Token HF ─────────────────────────────────────────────────────────

@eel.expose
def save_hf_token_js(token: str):
    save_hf_token(token)
    return True


# ── Conversations ─────────────────────────────────────────────────────

@eel.expose
def save_conversation_js(title: str, html: str):
    return save_conversation(title, html)

@eel.expose
def list_conversations_js():
    return list_conversations()

@eel.expose
def get_conversation_js(conv_id: str):
    return get_conversation(conv_id)


# ── Lancement ─────────────────────────────────────────────────────────

def start():
    hw = get_profile()
    print(f"[Neural Forge] Profil matériel : {hw.profile.value.upper()}")
    print(f"[Neural Forge] RAM={hw.ram_total_gb}Go  GPU={hw.gpu.name}  CUDA={hw.gpu.cuda_ok}")
    print(f"[Neural Forge] Turbo={'Disponible' if hw.turbo_eligible else 'Verrouillé'}")
    
    print("\n" + "="*50)
    print("🚀 INTERFACE PRÊTE !")
    print("👉 Copie ce lien dans Chrome (sur Windows) : http://localhost:8765/index.html")
    print("="*50 + "\n")

    # mode=None empêche Eel de chercher Chrome/Firefox
    eel.start(
        'index.html',
        mode=None, # <--- C'EST LA CLÉ
        port=8765,
        host='localhost',
        block=True
    )

if __name__ == '__main__':
    try:
        start()
    except Exception as e:
        print(f"Erreur au lancement : {e}")
