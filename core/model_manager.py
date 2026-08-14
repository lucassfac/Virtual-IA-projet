"""
model_manager.py — Gestionnaire de modèles locaux pour Neural Forge.

Fonctions :
  - Recherche de modèles GGUF sur HuggingFace Hub (API publique, sans token)
  - Téléchargement avec progression (callback)
  - Inventaire des modèles installés localement
  - Suppression de modèles
"""

import json
import os
import urllib.request
import urllib.parse
from typing import Callable, List, Optional

# Fichier de config local (non versionné)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_FILE  = os.path.join(_PROJECT_ROOT, ".neural_forge_config.json")

# Modification vers le nouveau dossier de stockage
DEFAULT_MODELS_DIR = os.path.join(_PROJECT_ROOT, "storage", "models")


def get_project_root() -> str:
    """Retourne le chemin absolu de la racine du projet (où se trouve app.py)."""
    return _PROJECT_ROOT


def get_models_dir() -> str:
    """Retourne le chemin absolu du dossier models/."""
    path = os.path.join(_PROJECT_ROOT, "models")
    os.makedirs(path, exist_ok=True)
    return path


# ── Constantes ────────────────────────────────────────────────────────
HF_API_MODELS = "https://huggingface.co/api/models"
HF_API_FILES  = "https://huggingface.co/api/models/{repo_id}"
HF_CDN        = "https://huggingface.co/{repo_id}/resolve/main/{filename}"


# Modèles mis en avant (suggestions rapides)
FEATURED_MODELS = [
    {
        "name": "TinyLlama 1.1B Chat",
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb": 0.67,
        "description": "Ultra-léger, idéal pour les tests — 100% libre",
        "tags": ["chat", "tiny"],
    },
    {
        "name": "Mistral 7B Instruct v0.2",
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_gb": 4.37,
        "description": "Mistral AI — excellent rapport qualité/taille",
        "tags": ["chat", "instruct", "mistral"],
    },
    {
        "name": "Phi-3 Mini 3.8B",
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "size_gb": 2.18,
        "description": "Microsoft Phi-3 — très performant pour sa taille",
        "tags": ["chat", "microsoft", "phi"],
    },
    {
        "name": "LLaMA 3.2 3B Instruct",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 1.91,
        "description": "Meta LLaMA 3.2 — 3B instruct, licence permissive",
        "tags": ["chat", "meta", "llama"],
    },
    {
        "name": "Qwen 2.5 3B Instruct",
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_gb": 1.93,
        "description": "Alibaba Qwen 2.5 — multilingue (FR, EN, ZH…)",
        "tags": ["chat", "multilingual", "qwen"],
    },
    {
        "name": "DeepSeek R1 1.5B",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "size_gb": 0.99,
        "description": "DeepSeek R1 distillé — raisonnement pas-à-pas",
        "tags": ["reasoning", "deepseek"],
    },
    {
        "name": "OpenHermes 2.5 Mistral 7B",
        "repo_id": "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
        "filename": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        "size_gb": 4.37,
        "description": "Fine-tune Mistral optimisé pour les instructions",
        "tags": ["chat", "instruct", "mistral"],
    },
    {
        "name": "Neural Chat 7B",
        "repo_id": "TheBloke/neural-chat-7B-v3-1-GGUF",
        "filename": "neural-chat-7b-v3-1.Q4_K_M.gguf",
        "size_gb": 4.11,
        "description": "Intel Neural Chat — optimisé CPU",
        "tags": ["chat", "intel"],
    },
    {
        "name": "Gemma 3 1B  🔒",
        "repo_id": "bartowski/gemma-3-1b-it-GGUF",
        "filename": "gemma-3-1b-it-Q4_K_M.gguf",
        "size_gb": 0.81,
        "description": "Google Gemma 3 1B — token HuggingFace requis",
        "tags": ["chat", "google", "gemma"],
    },
    {
        "name": "Gemma 3 4B  🔒",
        "repo_id": "bartowski/gemma-3-4b-it-GGUF",
        "filename": "gemma-3-4b-it-Q4_K_M.gguf",
        "size_gb": 2.53,
        "description": "Google Gemma 3 4B — token HuggingFace requis",
        "tags": ["chat", "google", "gemma"],
    },
]


# ── Exceptions ────────────────────────────────────────────────────────

class DownloadError(Exception):
    pass

class SearchError(Exception):
    pass



# ── Persistance du token HuggingFace ─────────────────────────────────

def save_hf_token(token: str) -> None:
    """Sauvegarde le token HuggingFace dans le fichier de config local."""
    config = _load_config()
    config["hf_token"] = token.strip()
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_hf_token() -> str:
    """Retourne le token HuggingFace sauvegardé, ou chaîne vide."""
    return _load_config().get("hf_token", "")


def clear_hf_token() -> None:
    """Supprime le token sauvegardé."""
    config = _load_config()
    config.pop("hf_token", None)
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _load_config() -> dict:
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _auth_headers(token: str = "") -> dict:
    """Headers HTTP avec Authorization si token fourni."""
    headers = {"User-Agent": "NeuralForge/0.1"}
    t = token or load_hf_token()
    if t:
        headers["Authorization"] = f"Bearer {t}"
    return headers


# ── Recherche HuggingFace ─────────────────────────────────────────────

def search_models(query: str, limit: int = 20) -> List[dict]:
    """
    Recherche des modèles GGUF sur HuggingFace.
    Retourne une liste de dicts avec : name, repo_id, description, downloads, tags.
    """
    params = urllib.parse.urlencode({
        "search": query,
        "filter": "gguf",
        "limit": limit,
        "sort": "downloads",
        "direction": -1,
    })
    url = f"{HF_API_MODELS}?{params}"

    try:
        req = urllib.request.Request(url, headers=_auth_headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise SearchError(f"Erreur réseau : {e}")

    results = []
    for item in data:
        repo_id = item.get("id", "")
        tags = item.get("tags", [])
        results.append({
            "name": repo_id.split("/")[-1],
            "repo_id": repo_id,
            "description": item.get("description") or "Pas de description",
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "tags": tags,
            "private": item.get("private", False),
        })
    return results


def get_repo_gguf_files(repo_id: str) -> List[dict]:
    """
    Retourne la liste des fichiers .gguf disponibles dans un repo HuggingFace.
    """
    url = HF_API_FILES.format(repo_id=repo_id)
    try:
        req = urllib.request.Request(url, headers=_auth_headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise SearchError(f"Impossible de récupérer les fichiers de {repo_id} : {e}")

    siblings = data.get("siblings", [])
    files = []
    for s in siblings:
        fname = s.get("rfilename", "")
        if fname.lower().endswith(".gguf"):
            files.append({
                "filename": fname,
                "size": s.get("size", 0),
                "size_gb": round(s.get("size", 0) / 1024**3, 2),
            })
    return sorted(files, key=lambda x: x["filename"])


# ── Téléchargement ────────────────────────────────────────────────────

def download_model(
    repo_id: str,
    filename: str,
    dest_dir: str = DEFAULT_MODELS_DIR,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    cancel_flag: Optional[list] = None,
    token: str = "",
) -> str:
    """
    Télécharge un fichier GGUF depuis HuggingFace.

    :param repo_id:     ex. "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    :param filename:    ex. "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    :param dest_dir:    Dossier de destination
    :param progress_cb: callback(bytes_downloaded, total_bytes)
    :param cancel_flag: liste d'un booléen [False] — mettre à [True] pour annuler
    :returns:           Chemin absolu du fichier téléchargé
    :raises DownloadError: en cas d'échec
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    local_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    url = HF_CDN.format(repo_id=repo_id, filename=filename)

    try:
        # ── Étape 1 : HEAD pour connaître la taille exacte ──
        head_req = urllib.request.Request(
            url, method="HEAD", headers=_auth_headers(token)
        )
        try:
            with urllib.request.urlopen(head_req, timeout=10) as hr:
                remote_size = int(hr.headers.get("Content-Length", 0))
        except Exception:
            remote_size = 0

        # ── Étape 2 : Fichier déjà complet ? ──
        if remote_size > 0 and local_size >= remote_size:
            if progress_cb:
                progress_cb(remote_size, remote_size)
            return os.path.abspath(dest_path)

        # ── Étape 3 : Téléchargement (ou reprise) ──
        req = urllib.request.Request(url, headers=_auth_headers(token))
        # Reprise seulement si on connaît la taille totale
        if local_size > 0 and remote_size > 0:
            req.add_header("Range", f"bytes={local_size}-")

        bytes_done = local_size   # point de départ pour la progression

        with urllib.request.urlopen(req, timeout=60) as resp:
            status = getattr(resp, "status", 200)

            # 416 = déjà complet selon le serveur
            if status == 416:
                if progress_cb and remote_size > 0:
                    progress_cb(remote_size, remote_size)
                return os.path.abspath(dest_path)

            content_length = int(resp.headers.get("Content-Length", 0) or 0)

            # Taille totale = octets déjà présents + octets restants
            if remote_size > 0:
                total = remote_size
            elif content_length > 0:
                total = local_size + content_length
            else:
                total = 0

            chunk = 65536
            mode = "ab" if local_size > 0 else "wb"

            with open(dest_path, mode) as f:
                while True:
                    if cancel_flag and cancel_flag[0]:
                        raise DownloadError("Téléchargement annulé.")
                    data = resp.read(chunk)
                    if not data:
                        break
                    f.write(data)
                    bytes_done += len(data)
                    if progress_cb and total > 0:
                        # Toujours entre 0 et total
                        progress_cb(min(bytes_done, total), total)

    except DownloadError:
        raise
    except Exception as e:
        raise DownloadError(f"Erreur téléchargement : {e}")

    return os.path.abspath(dest_path)


# ── Inventaire local ──────────────────────────────────────────────────

def list_local_models(models_dir: str = DEFAULT_MODELS_DIR) -> List[dict]:
    """
    Retourne la liste des modèles installés dans models_dir.
    """
    if not os.path.exists(models_dir):
        return []

    models = []
    for fname in sorted(os.listdir(models_dir)):
        if not fname.lower().endswith(".gguf"):
            continue
        path = os.path.join(models_dir, fname)
        stat = os.stat(path)
        models.append({
            "filename": fname,
            "path": os.path.abspath(path),
            "size_gb": round(stat.st_size / 1024**3, 2),
            "size_mb": round(stat.st_size / 1024**2, 0),
        })
    return models


def delete_model(path: str) -> None:
    """Supprime un fichier modèle."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    os.remove(path)


def get_models_dir_size(models_dir: str = DEFAULT_MODELS_DIR) -> float:
    """Retourne la taille totale du dossier models en Go."""
    if not os.path.exists(models_dir):
        return 0.0
    total = sum(
        os.path.getsize(os.path.join(models_dir, f))
        for f in os.listdir(models_dir)
        if os.path.isfile(os.path.join(models_dir, f))
    )
    return round(total / 1024**3, 2)
