"""
system_monitor.py — Monitoring CPU / RAM / GPU en temps réel.
Met à jour la status bar toutes les 2 secondes via QTimer.
Dépendance optionnelle : psutil (pip install psutil)
"""

import os
from PyQt6.QtCore import QTimer, QObject, pyqtSignal


class SystemMonitor(QObject):
    """
    Émet stats_updated(cpu_pct, ram_pct, ram_used_gb, ram_total_gb, gpu_info)
    toutes les 2 secondes.
    """

    stats_updated = pyqtSignal(float, float, float, float, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._poll)
        self._has_psutil = self._check_psutil()
        self._has_gpu    = self._check_gpu()

    def start(self):
        self._timer.start()
        self._poll()  # Première mesure immédiate

    def stop(self):
        self._timer.stop()

    def _check_psutil(self) -> bool:
        try:
            import psutil
            return True
        except ImportError:
            return False

    def _check_gpu(self) -> bool:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False

    def _poll(self):
        cpu_pct = 0.0
        ram_pct = 0.0
        ram_used_gb = 0.0
        ram_total_gb = 0.0
        gpu_info = ""

        if self._has_psutil:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            ram_pct = mem.percent
            ram_used_gb  = mem.used  / 1024**3
            ram_total_gb = mem.total / 1024**3

        if self._has_gpu:
            try:
                import subprocess
                r = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                if r.returncode == 0:
                    parts = r.stdout.strip().split(", ")
                    if len(parts) >= 3:
                        gpu_use  = int(parts[0])
                        vram_used  = int(parts[1])
                        vram_total = int(parts[2])
                        gpu_info = f"GPU {gpu_use}%  VRAM {vram_used}/{vram_total} Mo"
            except Exception:
                pass

        self.stats_updated.emit(cpu_pct, ram_pct, ram_used_gb, ram_total_gb, gpu_info)
