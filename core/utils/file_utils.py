"""
file_utils.py — Utilitaires de gestion des fichiers pour Neural Forge.
"""
import os
from core.logger import forge_logger
from core.node import ModelLoadError

def check_safe_path(path: str, label: str) -> str:
    """Vérifie la validité d'un chemin et crée un lien symbolique si des espaces sont présents."""
    if not path:
        raise ModelLoadError(f"{label} : chemin vide.")
    if not os.path.exists(path):
        raise ModelLoadError(f"{label} introuvable : '{path}'")
    if not os.access(path, os.R_OK):
        raise ModelLoadError(f"{label} non lisible (permissions).")

    if " " in path:
        forge_logger.warning(f"[{label}] Espaces détectés. Création d'un symlink sécurisé...")
        try:
            ext = os.path.splitext(path)[1]
            safe_name = f"nf_{label.lower().replace(' ', '_')}{ext}"
            safe_path = os.path.join("/tmp", safe_name)
            if os.path.islink(safe_path) and os.readlink(safe_path) != path:
                os.remove(safe_path)
            if not os.path.exists(safe_path):
                os.symlink(path, safe_path)
            return safe_path
        except Exception:
            return path
    return path