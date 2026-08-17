#!/usr/bin/env python3
"""
install.py — Installateur intelligent de Neural Forge.
Analyse le matériel (RAM, GPU NVIDIA/ROCm) et compile llama-cpp-python en conséquence.
"""

import os
import sys
import subprocess
import platform
import shutil
import argparse
import time
from pathlib import Path

LLAMA_VERSION = "0.3.18"

# ── Gestion des couleurs ANSI ────────────────────────────────────────
_NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")

def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"

def bold(t):    return _c("1",      t)
def dim(t):     return _c("2",      t)
def green(t):   return _c("1;32",   t)
def yellow(t):  return _c("1;33",   t)
def red(t):     return _c("1;31",   t)
def blue(t):    return _c("1;34",   t)
def cyan(t):    return _c("1;36",   t)
def magenta(t): return _c("1;35",   t)

# ── Helpers de journalisation console ────────────────────────────────
def banner():
    art = r"""
  _   _                      _   _____
 | \ | | ___ _   _ _ __ __ _| | |  ___|__  _ __ __ _  ___
 |  \| |/ _ \ | | | '__/ _` | | | |_ / _ \| '__/ _` |/ _ \
 | |\  |  __/ |_| | | | (_| | | |  _| (_) | | | (_| |  __/
 |_| \_|\___|\__,_|_|  \__,_|_| |_|  \___/|_|  \__, |\___|
                                                 |___/
"""
    print(magenta(art))
    print(bold("  Neural Forge — Installateur intelligent"))
    print(dim("  Edge AI Studio · Installation adaptative au matériel\n"))
    print("  " + "─" * 54 + "\n")

def section(title: str):
    print(f"\n  {cyan('▸')} {bold(title)}")
    print(f"  {'─' * 52}")

def ok(msg: str):   print(f"  {green('✓')}  {msg}")
def warn(msg: str): print(f"  {yellow('⚠')}  {msg}")
def err(msg: str):  print(f"  {red('✗')}  {msg}")
def info(msg: str): print(f"  {dim('·')}  {msg}")
def step(msg: str): print(f"\n  {blue('→')}  {bold(msg)}")


# ── Détection du matériel ────────────────────────────────────────────

class HardwareInfo:
    def __init__(self):
        self.cuda_available = False
        self.nvcc_path = ""
        self.cuda_version = ""
        self.gpu_name = ""
        self.gpu_vram_gb = 0.0
        self.cuda_arch = ""
        self.rocm_available = False
        self.python_version = platform.python_version()
        self.os_name = platform.system()
        self.ram_gb = 0.0
        self._detect()

    def _detect(self):
        self._detect_ram()
        self._detect_nvidia()
        self._detect_rocm()

    def _detect_ram(self):
        try:
            import psutil
            self.ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
        except ImportError:
            try:
                if self.os_name == "Linux":
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if "MemTotal" in line:
                                self.ram_gb = round(int(line.split()[1]) / 1024**2, 1)
                                break
            except Exception:
                self.ram_gb = 0.0

    def _detect_nvidia(self):
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if os.path.exists(nvcc):
            self.nvcc_path = nvcc
            try:
                r = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=5)
                for line in r.stdout.splitlines():
                    if "release" in line.lower():
                        self.cuda_version = line.split("release")[-1].strip().split(",")[0].strip().split()[0]
            except Exception:
                pass

        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                parts = [p.strip() for p in r.stdout.strip().split(",")]
                self.gpu_name = parts[0] if len(parts) > 0 else ""
                self.gpu_vram_gb = round(int(parts[1]) / 1024, 1) if len(parts) > 1 else 0
                compute = parts[2] if len(parts) > 2 else ""
                self.cuda_arch = compute.replace(".", "")
                self.cuda_available = True
        except Exception:
            pass

        if self.cuda_available and not self.nvcc_path:
            warn("GPU NVIDIA détecté mais nvcc absent — installation CPU recommandée.")

    def _detect_rocm(self):
        try:
            r = subprocess.run(["rocm-smi", "--showid"], capture_output=True, text=True, timeout=4)
            self.rocm_available = r.returncode == 0
        except Exception:
            pass

    def cuda_arch_flag(self) -> tuple[str, str]:
        arch_map = {
            "61": "GTX 10xx (Pascal)", "70": "V100/Titan (Volta)",
            "75": "RTX 20xx (Turing)", "80": "A100 (Ampere)",
            "86": "RTX 30xx (Ampere)", "87": "Jetson Orin",
            "89": "RTX 40xx (Ada)",    "90": "H100 (Hopper)",
        }
        arch = self.cuda_arch or "86"
        return arch, arch_map.get(arch, arch)


def print_hw_report(hw: HardwareInfo):
    section("Détection du matériel")
    info(f"Système  : {hw.os_name}  ·  Python {hw.python_version}")
    info(f"RAM      : {hw.ram_gb} Go" if hw.ram_gb else "RAM      : inconnue")

    if hw.cuda_available:
        ok(f"GPU NVIDIA  : {hw.gpu_name}  ({hw.gpu_vram_gb} Go VRAM)")
        ok(f"CUDA        : v{hw.cuda_version}")
        arch, label = hw.cuda_arch_flag()
        ok(f"Architecture: sm_{arch}  ({label})")
    else:
        info("Aucun GPU NVIDIA compatible détecté — mode CPU par défaut.")


def choose_install_mode(hw: HardwareInfo, args) -> str:
    if args.cpu:
        step("Mode forcé : CPU (--cpu)")
        return "cpu"
    if args.cuda:
        if not hw.cuda_available:
            warn("--cuda demandé mais aucun GPU NVIDIA détecté — fallback CPU.")
            return "cpu"
        step("Mode forcé : CUDA (--cuda)")
        return "cuda"
    if not hw.cuda_available:
        return "cpu"

    section("Choix du mode d'installation")
    print(f"\n  {yellow('!')}  GPU NVIDIA détecté : {bold(hw.gpu_name)}")
    print(f"      Accélération CUDA disponible (v{hw.cuda_version})\n")
    
    if args.quiet:
        return "cuda"

    try:
        choice = input(f"  Installer avec l'accélération GPU (CUDA) ? {bold('[Y/n]')}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "cuda"

    return "cpu" if choice in ("n", "no", "non") else "cuda"


# ── Configuration et Exécution Pip ────────────────────────────────────

def build_llama_cmd(mode: str, hw: HardwareInfo) -> dict:
    pkg = f"llama-cpp-python=={LLAMA_VERSION}"
    if mode == "cuda":
        arch, _ = hw.cuda_arch_flag()
        nvcc = hw.nvcc_path or "/usr/local/cuda/bin/nvcc"
        return {
            "env_vars": {
                "CUDACXX": nvcc,
                "CMAKE_ARGS": f"-DGGML_CUDA=ON -DGGML_LLAVA=ON -DCMAKE_CUDA_ARCHITECTURES={arch}",
                "FORCE_CMAKE": "1",
            },
            "pip_args": [pkg, "--no-cache-dir"],
            "display": f"CUDA (sm_{arch}) + LLaVA",
        }
    else:
        return {
            "env_vars": {
                "CMAKE_ARGS": "-DGGML_LLAVA=ON",
                "FORCE_CMAKE": "1",
            },
            "pip_args": [pkg, "--no-cache-dir"],
            "display": "CPU + LLaVA",
        }


def run_pip(args: list, env_extra: dict = None, label: str = "") -> bool:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    cmd = [sys.executable, "-m", "pip", "install"] + args
    step(f"Installation : {label or ' '.join(args[:2])}")
    
    start = time.time()
    try:
        proc = subprocess.run(cmd, env=env)
        elapsed = time.time() - start
        if proc.returncode == 0:
            ok(f"Installé en {elapsed:.0f}s")
            return True
        err(f"Échec (code {proc.returncode})")
        return False
    except Exception as e:
        err(f"Erreur : {e}")
        return False


def install_requirements() -> bool:
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        return True

    section("Dépendances (requirements.txt)")
    lines = req_file.read_text().splitlines()
    filtered = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#") and "llama-cpp-python" not in l.lower()]

    for pkg in filtered:
        run_pip([pkg], label=pkg)
    return True


def verify_install(mode: str) -> bool:
    section("Vérification")
    step("Test d'import llama_cpp…")
    try:
        result = subprocess.run([sys.executable, "-c", "import llama_cpp; print('OK', llama_cpp.__version__)"], capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            ok("llama_cpp importé avec succès")
            return True
        err("Import échoué")
        return False
    except Exception as e:
        err(f"Erreur de vérification : {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Installateur Neural Forge")
    parser.add_argument("--cpu", action="store_true", help="Forcer le mode CPU")
    parser.add_argument("--cuda", action="store_true", help="Forcer le mode CUDA")
    parser.add_argument("--quiet", action="store_true", help="Pas de prompt interactif")
    args = parser.parse_args()

    banner()
    hw = HardwareInfo()
    print_hw_report(hw)

    mode = choose_install_mode(hw, args)
    llama_cfg = build_llama_cmd(mode, hw)
    
    llama_ok = run_pip(llama_cfg["pip_args"], env_extra=llama_cfg["env_vars"], label=f"llama-cpp-python [{llama_cfg['display']}]")

    if not llama_ok and mode == "cuda":
        warn("Tentative de repli sur le mode CPU…")
        fallback_cfg = build_llama_cmd("cpu", hw)
        llama_ok = run_pip(fallback_cfg["pip_args"], env_extra=fallback_cfg["env_vars"], label="llama-cpp-python [CPU fallback]")
        mode = "cpu"

    install_requirements()
    success = verify_install(mode) if llama_ok else False

    section("Résumé")
    if success:
        ok(f"Neural Forge installé avec succès en mode {mode.upper()}")
        print(f"\n  {bold('Démarrage :')} {cyan('python app.py')}\n")
    else:
        err("Installation incomplète.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()