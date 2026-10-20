"""
file_utils.py — Utilitaires de validation et sécurisation des chemins de fichiers.
"""

import hashlib
import os
import tempfile

from core.logger import forge_logger
from core.node import ModelLoadError


def check_safe_path(path: str, label: str = "Fichier") -> str:
    """
    Vérifie l'existence et les droits de lecture d'un fichier.
    Crée un lien symbolique sans espace dans le dossier temporaire OS si nécessaire.
    """
    if not path or not isinstance(path, str) or not path.strip():
        raise ModelLoadError(f"{label} : chemin vide.")

    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise ModelLoadError(f"{label} introuvable : '{path}'")
    if not os.path.isfile(abs_path):
        raise ModelLoadError(f"{label} n'est pas un fichier valide : '{path}'")
    if not os.access(abs_path, os.R_OK):
        raise ModelLoadError(f"{label} non lisible (permissions insuffisantes).")

    if " " in abs_path:
        forge_logger.warning(f"[{label}] Espaces détectés dans le chemin. Création d'un symlink sécurisé...")
        try:
            ext = os.path.splitext(abs_path)[1]
            path_hash = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:8]
            safe_label = "".join(c if c.isalnum() else "_" for c in label.lower())
            safe_name = f"nf_{safe_label}_{path_hash}{ext}"
            safe_path = os.path.join(tempfile.gettempdir(), safe_name)

            if os.path.islink(safe_path) and os.readlink(safe_path) != abs_path:
                os.remove(safe_path)
            if not os.path.exists(safe_path):
                os.symlink(abs_path, safe_path)
            return safe_path
        except OSError as exc:
            forge_logger.warning(f"[{label}] Impossible de créer le symlink ({exc}), utilisation du chemin direct.")
            return abs_path

    return abs_path