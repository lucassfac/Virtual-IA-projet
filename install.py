#!/usr/bin/env python3
"""
install.py — Installateur intelligent de Neural Forge.

Usage :
    python install.py          # Interactif (recommandé)
    python install.py --cpu    # Force le mode CPU
    python install.py --cuda   # Force le mode CUDA (sans prompt)
    python install.py --quiet  # Pas de confirmation
"""

import os
import sys
import subprocess
import platform
import shutil
import argparse
import time
from pathlib import Path

# ── Version cible llama-cpp-python ────────────────────────────────────
LLAMA_VERSION = "0.3.18"

# ── Couleurs ANSI ─────────────────────────────────────────────────────
# Désactivées automatiquement si le terminal ne les supporte pas
_NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")

def _c(code: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def bold(t):    return _c("1",      t)
def dim(t):     return _c("2",      t)
def green(t):   return _c("1;32",   t)
def yellow(t):  return _c("1;33",   t)
def red(t):     return _c("1;31",   t)
def blue(t):    return _c("1;34",   t)
def cyan(t):    return _c("1;36",   t)
def magenta(t): return _c("1;35",   t)

# ── Helpers d'affichage ───────────────────────────────────────────────

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
    print("  " + "─" * 54)
    print()

def section(title: str):
    print(f"\n  {cyan('▸')} {bold(title)}")
    print(f"  {'─' * 52}")

def ok(msg: str):
    print(f"  {green('✓')}  {msg}")

def warn(msg: str):
    print(f"  {yellow('⚠')}  {msg}")

def err(msg: str):
    print(f"  {red('✗')}  {msg}")

def info(msg: str):
    print(f"  {dim('·')}  {msg}")

def step(msg: str):
    print(f"\n  {blue('→')}  {bold(msg)}")

def progress(msg: str):
    print(f"  {dim('...')} {msg}", end="", flush=True)

def done():
    print(f"  {green('done')}")


# ── Détection matérielle ──────────────────────────────────────────────

class HardwareInfo:
    def __init__(self):
        self.cuda_available   = False
        self.nvcc_path        = ""
        self.cuda_version     = ""
        self.gpu_name         = ""
        self.gpu_vram_gb      = 0.0
        self.cuda_arch        = ""
        self.rocm_available   = False
        self.python_version   = platform.python_version()
        self.os_name          = platform.system()
        self.ram_gb           = 0.0
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
            # Fallback sans psutil
            try:
                if self.os_name == "Linux":
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if "MemTotal" in line:
                                kb = int(line.split()[1])
                                self.ram_gb = round(kb / 1024**2, 1)
                                break
            except Exception:
                self.ram_gb = 0.0

    def _detect_nvidia(self):
        # ── nvcc ──
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if os.path.exists(nvcc):
            self.nvcc_path = nvcc
            try:
                r = subprocess.run(
                    [nvcc, "--version"],
                    capture_output=True, text=True, timeout=5
                )
                for line in r.stdout.splitlines():
                    if "release" in line.lower():
                        # "Cuda compilation tools, release 12.1"
                        parts = line.split("release")[-1].strip().split(",")[0].strip()
                        self.cuda_version = parts.split()[0]
            except Exception:
                pass

        # ── nvidia-smi ──
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                parts = [p.strip() for p in r.stdout.strip().split(",")]
                self.gpu_name    = parts[0] if len(parts) > 0 else ""
                self.gpu_vram_gb = round(int(parts[1]) / 1024, 1) if len(parts) > 1 else 0
                compute          = parts[2] if len(parts) > 2 else ""
                # compute_cap "8.6" → arch "86"
                self.cuda_arch   = compute.replace(".", "")
                self.cuda_available = True
        except Exception:
            pass

        # Si nvidia-smi OK mais nvcc absent → CUDA partielle
        if self.cuda_available and not self.nvcc_path:
            warn("GPU NVIDIA détecté mais nvcc absent — installation CPU recommandée.")
            warn("Installez le CUDA Toolkit : https://developer.nvidia.com/cuda-downloads")

    def _detect_rocm(self):
        try:
            r = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True, text=True, timeout=4
            )
            self.rocm_available = r.returncode == 0
        except Exception:
            pass

    def cuda_arch_flag(self) -> str:
        """Retourne le flag CMAKE_CUDA_ARCHITECTURES selon la carte."""
        arch_map = {
            "61": "GTX 10xx (Pascal)",  "70": "V100/Titan (Volta)",
            "75": "RTX 20xx (Turing)",  "80": "A100 (Ampere)",
            "86": "RTX 30xx (Ampere)",  "87": "Jetson Orin",
            "89": "RTX 40xx (Ada)",     "90": "H100 (Hopper)",
        }
        arch = self.cuda_arch or "86"  # default Ampere si inconnu
        label = arch_map.get(arch, arch)
        return arch, label


# ── Affichage du rapport matériel ─────────────────────────────────────

def print_hw_report(hw: HardwareInfo):
    section("Détection du matériel")

    info(f"Système  : {hw.os_name}  ·  Python {hw.python_version}")
    info(f"RAM      : {hw.ram_gb} Go" if hw.ram_gb else "RAM      : inconnue (psutil absent)")

    if hw.cuda_available:
        ok(f"GPU NVIDIA  : {hw.gpu_name}  ({hw.gpu_vram_gb} Go VRAM)")
        ok(f"nvcc        : {hw.nvcc_path}")
        ok(f"CUDA        : v{hw.cuda_version}")
        arch, label = hw.cuda_arch_flag()
        ok(f"Architecture: sm_{arch}  ({label})")
    elif hw.rocm_available:
        ok("GPU AMD (ROCm) détecté")
        warn("Support ROCm expérimental — installation CPU recommandée.")
    else:
        info("Aucun GPU NVIDIA/ROCm détecté — mode CPU.")


# ── Choix du mode d'installation ─────────────────────────────────────

def choose_install_mode(hw: HardwareInfo, args) -> str:
    """Retourne 'cuda' | 'cpu'."""

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

    # ── Prompt interactif ──
    section("Choix du mode d'installation")
    print(f"\n  {yellow('!')}  GPU NVIDIA détecté : {bold(hw.gpu_name)}")
    print(f"      Accélération CUDA disponible (v{hw.cuda_version})\n")
    print("  Modes disponibles :")
    print(f"    {green('[C]')} CUDA  — {bold('recommandé')} · génération 5-20× plus rapide")
    print(f"    {dim('[P]')} CPU   — compatible partout · plus lent\n")

    if args.quiet:
        print(f"  {dim('(--quiet : CUDA sélectionné automatiquement)')}")
        return "cuda"

    try:
        choice = input(
            f"  Installer avec l'accélération GPU (CUDA) ? "
            f"{bold('[Y/n]')}: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "cuda"

    if choice in ("", "y", "yes", "o", "oui"):
        return "cuda"
    elif choice in ("n", "no", "non"):
        return "cpu"
    else:
        warn(f"Réponse non reconnue ('{choice}') — CUDA sélectionné par défaut.")
        return "cuda"


# ── Construction des commandes pip ────────────────────────────────────

def build_llama_cmd(mode: str, hw: HardwareInfo) -> list:
    pkg = f"llama-cpp-python=={LLAMA_VERSION}"

    if mode == "cuda":
        arch, _ = hw.cuda_arch_flag()
        nvcc    = hw.nvcc_path or "/usr/local/cuda/bin/nvcc"
        env_prefix = (
            f"CUDACXX={nvcc} "
            f'CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_LLAVA=ON '
            f'-DCMAKE_CUDA_ARCHITECTURES={arch}"'
        )
        return {
            "env_vars": {
                "CUDACXX": nvcc,
                "CMAKE_ARGS": (
                    f"-DGGML_CUDA=ON -DGGML_LLAVA=ON "
                    f"-DCMAKE_CUDA_ARCHITECTURES={arch}"
                ),
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


# ── Exécution pip ─────────────────────────────────────────────────────

def run_pip(args: list, env_extra: dict = None, label: str = "") -> bool:
    """Lance pip avec les arguments donnés. Retourne True si succès."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    cmd = [sys.executable, "-m", "pip", "install"] + args

    step(f"Installation : {label or ' '.join(args[:2])}")
    info(f"Commande : {' '.join(cmd)}")
    print()

    start = time.time()
    try:
        proc = subprocess.run(cmd, env=env)
        elapsed = time.time() - start

        if proc.returncode == 0:
            ok(f"Installé en {elapsed:.0f}s")
            return True
        else:
            err(f"Échec (code {proc.returncode}) après {elapsed:.0f}s")
            return False

    except FileNotFoundError:
        err("pip introuvable — vérifiez votre environnement Python.")
        return False
    except KeyboardInterrupt:
        print()
        warn("Installation interrompue par l'utilisateur.")
        return False
    except Exception as e:
        err(f"Erreur inattendue : {e}")
        return False


def install_requirements() -> bool:
    """Installe le reste des paquets depuis requirements.txt."""
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        warn("requirements.txt introuvable — passage à l'étape suivante.")
        return True

    section("Dépendances (requirements.txt)")

    # Lire et filtrer llama-cpp-python (déjà installé)
    lines = req_file.read_text().splitlines()
    filtered = [
        l for l in lines
        if l.strip()
        and not l.strip().startswith("#")
        and "llama-cpp-python" not in l.lower()
    ]

    if not filtered:
        info("Aucune dépendance supplémentaire.")
        return True

    for pkg_line in filtered:
        pkg = pkg_line.split("#")[0].strip()
        if not pkg:
            continue
        ok_flag = run_pip([pkg], label=pkg)
        if not ok_flag:
            warn(f"Échec de {pkg} — continuez manuellement avec : pip install {pkg}")

    return True


# ── Vérification post-installation ───────────────────────────────────

def verify_install(mode: str) -> bool:
    section("Vérification")

    step("Test d'import llama_cpp…")
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import llama_cpp; print('OK', llama_cpp.__version__)"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            ok(f"llama_cpp {result.stdout.strip().split()[-1]} importé")
        else:
            err(f"Import échoué : {result.stderr.strip()[:200]}")
            return False
    except Exception as e:
        err(f"Impossible de vérifier : {e}")
        return False

    if mode == "cuda":
        step("Vérification support CUDA dans llama_cpp…")
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 "from llama_cpp import Llama; "
                 "import llama_cpp._libs; print('llama_cpp ok')"],
                capture_output=True, text=True, timeout=15
            )
            # Vérifier les symboles CUDA dans le .so
            check = subprocess.run(
                [sys.executable, "-c",
                 "import llama_cpp, os, subprocess\n"
                 "lib = os.path.dirname(llama_cpp.__file__)\n"
                 "import glob\n"
                 "sos = glob.glob(lib + '/**/*.so*', recursive=True) + glob.glob(lib + '/**/*.pyd', recursive=True)\n"
                 "found = False\n"
                 "for so in sos:\n"
                 "    r = subprocess.run(['nm','-D',so],capture_output=True,text=True)\n"
                 "    if 'cuda' in r.stdout.lower(): found=True; break\n"
                 "print('CUDA_IN_BINARY' if found else 'CPU_ONLY')"
                ],
                capture_output=True, text=True, timeout=10
            )
            if "CUDA_IN_BINARY" in check.stdout:
                ok("Symboles CUDA détectés dans llama_cpp — GPU actif ✓")
            else:
                warn("Symboles CUDA absents — la bibliothèque a été compilée en mode CPU.")
                warn("Relancez : python install.py --cuda")
        except Exception:
            warn("Impossible de vérifier les symboles CUDA.")

    step("Test LLaVA (llama_chat_format)…")
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "from llama_cpp.llama_chat_format import Llava15ChatHandler; print('LLaVA OK')"],
            capture_output=True, text=True, timeout=10
        )
        if "LLaVA OK" in result.stdout:
            ok("Support LLaVA disponible")
        else:
            warn("Support LLaVA absent — réinstallez avec GGML_LLAVA=ON")
    except Exception:
        warn("Impossible de vérifier LLaVA.")

    return True


# ── Résumé final ──────────────────────────────────────────────────────

def print_summary(mode: str, hw: HardwareInfo, success: bool):
    section("Résumé")
    print()

    if success:
        ok(f"Neural Forge installé en mode {bold(mode.upper())}")
        print()
        if mode == "cuda":
            ok(f"GPU : {hw.gpu_name}  ({hw.gpu_vram_gb} Go VRAM)")
            ok(f"Accélération CUDA active — génération rapide ⚡")
        else:
            ok("Mode CPU — compatible sur toutes les machines")
            if hw.cuda_available:
                info("Pour activer CUDA ultérieurement : python install.py --cuda")
        print()
        print(f"  {bold('Démarrage :')}  {cyan('python app.py')}")
        print(f"  {bold('Interface web :')} {cyan('python eel_main.py')}")
    else:
        err("Installation incomplète — consultez les messages ci-dessus.")
        print()
        info("En cas de problème avec llama-cpp-python :")
        info("  pip install llama-cpp-python --extra-index-url")
        info("  https://abetlen.github.io/llama-cpp-python/whl/cpu")

    print()
    print("  " + "─" * 54)
    print()


# ── Point d'entrée ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Installateur Neural Forge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cpu",   action="store_true", help="Forcer le mode CPU")
    parser.add_argument("--cuda",  action="store_true", help="Forcer le mode CUDA")
    parser.add_argument("--quiet", action="store_true", help="Pas de prompt interactif")
    args = parser.parse_args()

    banner()

    # ── Détection matérielle ──
    progress("Analyse du matériel…")
    hw = HardwareInfo()
    print()

    print_hw_report(hw)

    # ── Choix du mode ──
    mode = choose_install_mode(hw, args)

    section(f"Installation — mode {mode.upper()}")
    info(f"llama-cpp-python v{LLAMA_VERSION}")

    # ── llama-cpp-python ──
    llama_cfg = build_llama_cmd(mode, hw)
    llama_ok  = run_pip(
        llama_cfg["pip_args"],
        env_extra=llama_cfg["env_vars"],
        label=f"llama-cpp-python v{LLAMA_VERSION}  [{llama_cfg['display']}]",
    )

    if not llama_ok:
        warn("Échec avec les paramètres optimaux — tentative avec le fallback CPU…")
        fallback_cfg = build_llama_cmd("cpu", hw)
        llama_ok = run_pip(
            fallback_cfg["pip_args"],
            env_extra=fallback_cfg["env_vars"],
            label=f"llama-cpp-python v{LLAMA_VERSION}  [CPU fallback]",
        )
        if llama_ok:
            mode = "cpu"
            warn("Installé en mode CPU (le mode CUDA a échoué).")

    # ── Autres dépendances ──
    install_requirements()

    # ── Vérification ──
    success = verify_install(mode) if llama_ok else False

    # ── Résumé ──
    print_summary(mode, hw, success)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
