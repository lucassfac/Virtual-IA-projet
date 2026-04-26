"""
hardware_check.py — Détection matérielle pour Neural Forge.
Utilisé pour activer/bloquer le mode Turbo (Speculative Decoding).
"""

import os
from dataclasses import dataclass

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

TURBO_RAM_THRESHOLD_GB = 12.0   # RAM minimale pour le mode Turbo
TURBO_VRAM_THRESHOLD_GB = 6.0   # VRAM recommandée (non bloquante)


@dataclass
class HardwareSummary:
    ram_total_gb:   float
    ram_used_gb:    float
    cpu_cores:      int
    cpu_cores_phys: int
    gpu_vram_gb:    float
    gpu_name:       str
    turbo_eligible: bool      # RAM >= 12 Go
    turbo_reason:   str       # Message affiché dans le tooltip


def get_hardware_summary() -> HardwareSummary:
    """
    Lit les specs de la machine et détermine l'éligibilité Turbo.
    Ne plante jamais — retourne des valeurs par défaut si psutil absent.
    """
    # ── RAM ──
    if _HAS_PSUTIL:
        mem = psutil.virtual_memory()
        ram_total_gb = mem.total / 1024**3
        ram_used_gb  = mem.used  / 1024**3
        cpu_cores      = psutil.cpu_count(logical=True)  or 1
        cpu_cores_phys = psutil.cpu_count(logical=False) or 1
    else:
        ram_total_gb   = 8.0
        ram_used_gb    = 4.0
        cpu_cores      = os.cpu_count() or 4
        cpu_cores_phys = max(1, cpu_cores // 2)

    # ── GPU / VRAM ──
    gpu_vram_gb = 0.0
    gpu_name    = "Aucun GPU détecté"
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(", ")
            gpu_name    = parts[0].strip()
            gpu_vram_gb = int(parts[1].strip()) / 1024
    except Exception:
        pass

    # ── Éligibilité Turbo ──
    turbo_eligible = ram_total_gb >= TURBO_RAM_THRESHOLD_GB

    if turbo_eligible:
        turbo_reason = (
            f"Turbo disponible — {ram_total_gb:.1f} Go RAM détectés.\n"
            f"Speculative Decoding actif avec un modèle draft."
        )
    else:
        needed = TURBO_RAM_THRESHOLD_GB - ram_total_gb
        turbo_reason = (
            f"Mode Turbo verrouillé.\n"
            f"RAM détectée : {ram_total_gb:.1f} Go\n"
            f"RAM requise  : {TURBO_RAM_THRESHOLD_GB:.0f} Go minimum\n"
            f"Il vous manque {needed:.1f} Go pour activer le Speculative Decoding."
        )

    return HardwareSummary(
        ram_total_gb=round(ram_total_gb, 1),
        ram_used_gb=round(ram_used_gb, 1),
        cpu_cores=cpu_cores,
        cpu_cores_phys=cpu_cores_phys,
        gpu_vram_gb=round(gpu_vram_gb, 1),
        gpu_name=gpu_name,
        turbo_eligible=turbo_eligible,
        turbo_reason=turbo_reason,
    )


# Instance unique — chargée une seule fois au démarrage
_cached: HardwareSummary | None = None


def get_hw() -> HardwareSummary:
    """Retourne le résumé matériel mis en cache (évite les appels répétés)."""
    global _cached
    if _cached is None:
        _cached = get_hardware_summary()
    return _cached
