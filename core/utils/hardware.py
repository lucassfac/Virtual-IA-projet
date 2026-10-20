"""
hardware.py — Découverte matérielle centralisée pour Neural Forge.
Détecte RAM, CPU, GPU et construit le profil optimal d'exécution.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

TURBO_RAM_THRESHOLD_GB = 12.0


@dataclass
class HardwareProfile:
    # RAM & CPU
    ram_total_gb: float
    ram_available_gb: float
    cpu_cores_phys: int
    cpu_cores_log: int
    cpu_freq_mhz: float
    # GPU
    gpu_name: str
    gpu_vram_gb: float
    gpu_vendor: str
    cuda_ok: bool
    vulkan_ok: bool
    # Paramètres llama-cpp optimaux calculés
    n_threads: int
    n_batch: int
    n_ubatch: int
    n_ctx: int
    n_gpu_layers: int
    flash_attn: bool
    use_mmap: bool
    # Turbo (Speculative Decoding)
    turbo_eligible: bool
    turbo_reason: str


def _check_llama_backend(backend: str) -> bool:
    """Vérifie si llama-cpp-python a été compilé avec l'accélération GPU."""
    try:
        import llama_cpp
        if hasattr(llama_cpp, "llama_supports_gpu_offload"):
            if llama_cpp.llama_supports_gpu_offload():
                return True
        lib_path = os.path.dirname(llama_cpp.__file__)
        for fname in os.listdir(lib_path):
            if fname.endswith((".so", ".pyd", ".dll", ".dylib")):
                if backend in fname.lower():
                    return True
                r = subprocess.run(
                    ["nm", "-D", os.path.join(lib_path, fname)],
                    capture_output=True, text=True, timeout=2
                )
                if r.returncode == 0 and backend in r.stdout.lower():
                    return True
    except Exception:
        pass
    return False


def _detect_gpu() -> dict:
    """Détecte le GPU (NVIDIA CUDA ou AMD ROCm) et retourne ses caractéristiques."""
    # ── NVIDIA ──
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if r.returncode == 0 and r.stdout.strip():
            first_line = r.stdout.strip().splitlines()[0]
            parts = first_line.split(", ")
            if len(parts) >= 2:
                return {
                    "name": parts[0].strip(),
                    "vram_gb": round(float(parts[1].strip()) / 1024, 1),
                    "vendor": "nvidia",
                    "cuda_ok": _check_llama_backend("cuda"),
                    "vulkan_ok": False,
                }
    except Exception:
        pass

    # ── AMD ROCm ──
    try:
        r = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=3
        )
        if r.returncode == 0 and r.stdout.strip():
            data = json.loads(r.stdout)
            vram_bytes = int(list(data.values())[0].get("VRAM Total Memory (B)", 0))
            return {
                "name": "AMD GPU (ROCm)",
                "vram_gb": round(vram_bytes / 1024**3, 1),
                "vendor": "amd",
                "cuda_ok": False,
                "vulkan_ok": True,
            }
    except Exception:
        pass

    return {
        "name": "Aucun GPU détecté",
        "vram_gb": 0.0,
        "vendor": "none",
        "cuda_ok": False,
        "vulkan_ok": False,
    }


def build_hardware_profile() -> HardwareProfile:
    if _HAS_PSUTIL:
        mem = psutil.virtual_memory()
        ram_total = mem.total / 1024**3
        ram_avail = mem.available / 1024**3
        cpu_phys = psutil.cpu_count(logical=False) or 2
        cpu_log = psutil.cpu_count(logical=True) or 4
        try:
            freq = psutil.cpu_freq()
            cpu_freq = freq.current if freq else 0.0
        except Exception:
            cpu_freq = 0.0
    else:
        ram_total, ram_avail = 8.0, 4.0
        cpu_log = os.cpu_count() or 4
        cpu_phys = max(1, cpu_log // 2)
        cpu_freq = 0.0

    gpu = _detect_gpu()
    turbo_eligible = ram_total >= TURBO_RAM_THRESHOLD_GB
    turbo_reason = (
        "Turbo disponible."
        if turbo_eligible
        else f"Turbo verrouillé (Requiert {TURBO_RAM_THRESHOLD_GB} Go RAM, {ram_total:.1f} Go détectés)."
    )

    return HardwareProfile(
        ram_total_gb=round(ram_total, 1),
        ram_available_gb=round(ram_avail, 1),
        cpu_cores_phys=cpu_phys,
        cpu_cores_log=cpu_log,
        cpu_freq_mhz=round(cpu_freq, 0),
        gpu_name=gpu["name"],
        gpu_vram_gb=gpu["vram_gb"],
        gpu_vendor=gpu["vendor"],
        cuda_ok=gpu["cuda_ok"],
        vulkan_ok=gpu["vulkan_ok"],
        n_threads=max(2, cpu_phys // 2),
        n_batch=512,
        n_ubatch=256,
        n_ctx=4096,
        n_gpu_layers=-1 if gpu["vendor"] != "none" else 0,
        flash_attn=(gpu["vram_gb"] >= 4.0 or gpu["vendor"] in ("amd", "intel")),
        use_mmap=True,
        turbo_eligible=turbo_eligible,
        turbo_reason=turbo_reason,
    )


_profile: Optional[HardwareProfile] = None


def get_profile() -> HardwareProfile:
    global _profile
    if _profile is None:
        _profile = build_hardware_profile()
    return _profile


get_hw = get_profile