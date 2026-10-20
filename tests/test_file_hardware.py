"""
test_hardware.py — Tests unitaires pour la découverte matérielle.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.utils import hardware
from core.utils.hardware import build_hardware_profile, get_profile, TURBO_RAM_THRESHOLD_GB


@pytest.fixture(autouse=True)
def reset_singleton():
    """Réinitialise le Singleton _profile avant chaque test pour éviter la contamination."""
    hardware._profile = None
    yield
    hardware._profile = None


@patch("core.utils.hardware._detect_gpu")
@patch("psutil.virtual_memory")
@patch("psutil.cpu_count")
def test_build_hardware_profile_turbo_eligible(mock_cpu, mock_vm, mock_gpu):
    """Si la RAM totale est >= 12 Go, le mode Turbo doit être éligible."""
    # Simulation de 16 Go de RAM
    mock_vm.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3)
    mock_cpu.return_value = 8
    mock_gpu.return_value = {"name": "Test GPU", "vram_gb": 8.0, "vendor": "nvidia", "cuda_ok": True, "vulkan_ok": False}
    
    profile = build_hardware_profile()
    
    assert profile.ram_total_gb == 16.0
    assert profile.turbo_eligible is True
    assert "Turbo disponible" in profile.turbo_reason


@patch("core.utils.hardware._detect_gpu")
@patch("psutil.virtual_memory")
@patch("psutil.cpu_count")
def test_build_hardware_profile_turbo_locked(mock_cpu, mock_vm, mock_gpu):
    """Si la RAM totale est < 12 Go, le mode Turbo doit être verrouillé."""
    # Simulation de 8 Go de RAM
    mock_vm.return_value = MagicMock(total=8 * 1024**3, available=4 * 1024**3)
    mock_cpu.return_value = 4
    mock_gpu.return_value = {"name": "Test GPU", "vram_gb": 2.0, "vendor": "amd", "cuda_ok": False, "vulkan_ok": True}
    
    profile = build_hardware_profile()
    
    assert profile.ram_total_gb == 8.0
    assert profile.turbo_eligible is False
    assert "Turbo verrouillé" in profile.turbo_reason


@patch("core.utils.hardware.build_hardware_profile")
def test_get_profile_is_singleton(mock_build):
    """get_profile() ne doit construire le profil qu'une seule fois (Singleton)."""
    mock_build.return_value = "FakeProfile"
    
    p1 = get_profile()
    p2 = get_profile()
    
    assert p1 == "FakeProfile"
    assert p1 is p2
    mock_build.assert_called_once()