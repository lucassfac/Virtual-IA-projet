"""
hardware.py — Découverte matérielle complète pour Neural Forge.
Détecte RAM, CPU, GPU (NVIDIA/AMD/Intel Arc) et construit le profil optimal.
"""

import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class PerfProfile(Enum):
    ECO         = "eco"         # < 12 Go RAM  — CPU only, ctx réduit
    TURBO_READY = "turbo_ready" # ≥ 12 Go RAM  — Speculative Decoding possible


@dataclass
class GPUInfo:
    name:       str
    vram_gb:    float
    vendor:     str    # "nvidia" | "amd" | "intel" | "none"
    cuda_ok:    bool   # llama-cpp compilé avec CUDA ?
    vulkan_ok:  bool


@dataclass
class HardwareProfile:
    # Identité
    profile:        PerfProfile
    # RAM
    ram_total_gb:   float
    ram_available_gb: float
    # CPU
    cpu_cores_phys: int
    cpu_cores_log:  int
    cpu_freq_mhz:   float
    # GPU
    gpu:            GPUInfo
    # Paramètres llama-cpp optimaux calculés
    n_threads:      int
    n_batch:        int
    n_ubatch:       int
    n_ctx:          int
    n_gpu_layers:   int
    flash_attn:     bool
    use_mmap:       bool
    # Turbo
    turbo_eligible: bool
    turbo_reason:   str
    # Badge UI
    badge_text:     str
    badge_color:    str   # hex


# ── Détection GPU ─────────────────────────────────────────────────────

def _detect_gpu() -> GPUInfo:
    """Détecte NVIDIA, AMD (ROCm) ou Intel Arc via subprocess."""

    # ── NVIDIA ──
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(", ")
            name     = parts[0].strip()
            vram_mb  = int(parts[1].strip())
            vram_gb  = round(vram_mb / 1024, 1)

            # Vérifie si llama-cpp a été compilé avec CUDA
            cuda_ok = _check_llama_backend("cuda")

            return GPUInfo(
                name=name, vram_gb=vram_gb,
                vendor="nvidia", cuda_ok=cuda_ok, vulkan_ok=False
            )
    except Exception:
        pass

    # ── AMD ROCm ──
    try:
        r = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=4
        )
        if r.returncode == 0:
            import json
            data = json.loads(r.stdout)
            vram_mb = int(list(data.values())[0].get("VRAM Total Memory (B)", 0)) // 1024**2
            return GPUInfo(
                name="AMD GPU (ROCm)", vram_gb=round(vram_mb/1024, 1),
                vendor="amd", cuda_ok=False, vulkan_ok=True
            )
    except Exception:
        pass

    # ── Intel Arc / Vulkan ──
    try:
        r = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=4
        )
        if r.returncode == 0 and "Intel" in r.stdout:
            return GPUInfo(
                name="Intel Arc (Vulkan)", vram_gb=0.0,
                vendor="intel", cuda_ok=False, vulkan_ok=True
            )
    except Exception:
        pass

    return GPUInfo(
        name="Aucun GPU détecté", vram_gb=0.0,
        vendor="none", cuda_ok=False, vulkan_ok=False
    )


def _check_llama_backend(backend: str) -> bool:
    """
    Vérifie si llama-cpp-python a été compilé avec un backend donné.
    Rapide : on inspecte juste les symbols du .so sans charger de modèle.
    """
    try:
        import llama_cpp
        lib_path = os.path.dirname(llama_cpp.__file__)
        for fname in os.listdir(lib_path):
            if fname.endswith(".so") or fname.endswith(".pyd"):
                full = os.path.join(lib_path, fname)
                r = subprocess.run(
                    ["nm", "-D", full],
                    capture_output=True, text=True, timeout=3
                )
                if backend == "cuda" and "cuda" in r.stdout.lower():
                    return True
                if backend == "vulkan" and "vulkan" in r.stdout.lower():
                    return True
    except Exception:
        pass
    return False


# ── Profil complet ────────────────────────────────────────────────────

def build_hardware_profile() -> HardwareProfile:
    """
    Construit le profil matériel complet et calcule les paramètres optimaux.
    """
    # ── RAM ──
    if _HAS_PSUTIL:
        mem              = psutil.virtual_memory()
        ram_total_gb     = mem.total     / 1024**3
        ram_available_gb = mem.available / 1024**3
        cpu_phys         = psutil.cpu_count(logical=False) or 2
        cpu_log          = psutil.cpu_count(logical=True)  or 4
        try:
            freq = psutil.cpu_freq()
            cpu_freq_mhz = freq.current if freq else 0.0
        except Exception:
            cpu_freq_mhz = 0.0
    else:
        ram_total_gb     = 8.0
        ram_available_gb = 4.0
        cpu_phys         = max(1, (os.cpu_count() or 4) // 2)
        cpu_log          = os.cpu_count() or 4
        cpu_freq_mhz     = 0.0

    gpu = _detect_gpu()

    # ── Profil ──
    turbo_eligible = ram_total_gb >= 12.0
    profile        = PerfProfile.TURBO_READY if turbo_eligible else PerfProfile.ECO

    # ── Paramètres llama-cpp optimaux ──
    #
    # n_threads : cœurs physiques // 2
    #   Au-delà, la contention mémoire annule le gain (mesuré sur DDR5)
    n_threads = max(2, cpu_phys // 2)

    # n_ctx : 2048 — bon équilibre qualité/vitesse pour tous les usages courants
    n_ctx = 2048

    # n_batch / n_ubatch : optimisés pour DDR5 (bande passante ~80 Go/s)
    n_batch  = 512
    n_ubatch = 256

    # GPU layers
    if gpu.vendor == "none":
        n_gpu_layers = 0
    else:
        n_gpu_layers = -1   # tout sur GPU

    # Flash Attention : gain ~30% sur les longues séquences
    flash_attn = (gpu.vram_gb >= 4 or gpu.vendor in ("amd", "intel"))

    # use_mmap : chargement instantané depuis le disque
    use_mmap = True

    # ── Turbo tooltip ──
    if turbo_eligible:
        turbo_reason = (
            f"Turbo disponible — {ram_total_gb:.1f} Go RAM.\n"
            f"Ajoutez un modèle draft (ex: TinyLlama 1.1B)\n"
            f"pour activer le Speculative Decoding."
        )
    else:
        needed = 12.0 - ram_total_gb
        turbo_reason = (
            f"Turbo verrouillé.\n"
            f"RAM détectée : {ram_total_gb:.1f} Go\n"
            f"Requis       : 12 Go minimum\n"
            f"Manque       : {needed:.1f} Go"
        )

    # ── Badge UI ──
    if gpu.cuda_ok:
        badge_text  = f"CUDA  {gpu.vram_gb}Go VRAM"
        badge_color = "#30D158"
    elif gpu.vulkan_ok:
        badge_text  = f"Vulkan  {gpu.vendor.upper()}"
        badge_color = "#FF9F0A"
    elif gpu.vendor != "none":
        badge_text  = f"GPU  {gpu.name[:20]}"
        badge_color = "#FF9F0A"
    else:
        badge_text  = f"CPU only  {ram_total_gb:.0f}Go RAM"
        badge_color = "#636366"

    return HardwareProfile(
        profile=profile,
        ram_total_gb=round(ram_total_gb, 1),
        ram_available_gb=round(ram_available_gb, 1),
        cpu_cores_phys=cpu_phys,
        cpu_cores_log=cpu_log,
        cpu_freq_mhz=round(cpu_freq_mhz, 0),
        gpu=gpu,
        n_threads=n_threads,
        n_batch=n_batch,
        n_ubatch=n_ubatch,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        flash_attn=flash_attn,
        use_mmap=use_mmap,
        turbo_eligible=turbo_eligible,
        turbo_reason=turbo_reason,
        badge_text=badge_text,
        badge_color=badge_color,
    )


# ── Singleton ─────────────────────────────────────────────────────────
_profile: Optional[HardwareProfile] = None


def get_profile() -> HardwareProfile:
    """Retourne le profil mis en cache (calculé une seule fois au démarrage)."""
    global _profile
    if _profile is None:
        _profile = build_hardware_profile()
    return _profile
