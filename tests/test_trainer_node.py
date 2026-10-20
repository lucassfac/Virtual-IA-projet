"""
test_trainer_node.py — Tests unitaires pour TrainerNode (SFT & Model Soup) et SoupManager.
"""

import os
import sys
from unittest.mock import patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.node import InputMissingError
from core.nodes.trainer_node import TrainerNode
from core.soup_manager import SoupManager
from core.types import DataPacket, DataType

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None


@pytest.fixture(autouse=True)
def fast_sleep():
    """Désactive time.sleep pour que la suite de tests s'exécute en quelques millisecondes."""
    with patch("time.sleep"):
        yield


# ---------------------------------------------------------------------------
# Tests : Sécurité & Validation des entrées
# ---------------------------------------------------------------------------

def test_process_without_input_raises(tmp_path):
    """process() sans DataPacket doit lever InputMissingError."""
    trainer = TrainerNode("TestTrainer", output_dir=str(tmp_path))
    with pytest.raises(InputMissingError):
        trainer.process()


def test_process_with_non_dict_raises(tmp_path):
    """Le TrainerNode doit rejeter un contenu qui n'est pas un dictionnaire."""
    trainer = TrainerNode("TestTrainer", output_dir=str(tmp_path))
    trainer.set_input(DataPacket(DataType.ANY, "invalid_config"))
    with pytest.raises(TypeError, match="dictionnaire"):
        trainer.process()


def test_sft_missing_dataset_or_model_raises(tmp_path):
    """Le mode SFT sans dataset ou sans modèle de base doit lever ValueError."""
    trainer = TrainerNode("TestTrainer", output_dir=str(tmp_path))
    trainer.set_input(DataPacket(DataType.ANY, {"mode": "sft", "dataset": ""}))
    with pytest.raises(ValueError, match="requiert un dataset"):
        trainer.process()


# ---------------------------------------------------------------------------
# Tests : Entraînement SFT & Arrêt Coopératif
# ---------------------------------------------------------------------------

def test_sft_generates_safetensors_in_adapters_dir(tmp_path):
    """Le mode SFT doit créer un adaptateur .safetensors nommé selon le dataset et les hyperparamètres."""
    trainer = TrainerNode("TestTrainer", output_dir=str(tmp_path))
    config = {
        "mode": "sft",
        "dataset": "storage/data/wistoria.jsonl",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "lr": "Taux d'apprentissage: Rapide (2e-4)",
        "epochs": "3 Époques",
    }
    trainer.set_input(DataPacket(DataType.ANY, config))
    result = trainer.process()

    assert result.content.endswith("wistoria_2e-4_ep3_adapter.safetensors")
    assert os.path.exists(result.content)
    assert result.metadata["type"] == "lora_adapter"
    assert result.metadata["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_sft_progress_callback_invoked(tmp_path):
    """Le callback de progression doit être appelé 10 fois (tous les 10 pas sur 100)."""
    calls = []
    trainer = TrainerNode("TestTrainer", output_dir=str(tmp_path))
    trainer.set_progress_callback(lambda step, total, loss: calls.append((step, total, loss)))
    trainer.set_input(
        DataPacket(
            DataType.ANY,
            {"mode": "sft", "dataset": "data/test.jsonl", "base_model": "Qwen/Qwen2.5-1.5B"},
        )
    )
    trainer.process()

    assert len(calls) == 10
    assert calls[-1][0] == 100
    assert calls[-1][1] == 100
    assert calls[0][2] > calls[-1][2]  # La loss doit diminuer


def test_sft_cooperative_abort(tmp_path):
    """request_abort() doit interrompre proprement l'entraînement sans crash."""
    trainer = TrainerNode("TestTrainer", output_dir=str(tmp_path))

    def abort_on_first_callback(step, total, loss):
        trainer.request_abort()

    trainer.set_progress_callback(abort_on_first_callback)
    trainer.set_input(
        DataPacket(
            DataType.ANY,
            {"mode": "sft", "dataset": "data/test.jsonl", "base_model": "Qwen/Qwen2.5-1.5B"},
        )
    )

    with pytest.raises(RuntimeError, match="interrompu proprement"):
        trainer.process()


# ---------------------------------------------------------------------------
# Tests : Fusion Model Soup (SoupManager + TrainerNode)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(torch is None, reason="PyTorch et safetensors requis pour tester Model Soup")
def test_model_soup_averages_tensors_correctly(tmp_path):
    """Vérifie que le mode SOUP calcule exactement la moyenne mathématique des tenseurs."""
    adapter_a = str(tmp_path / "adapter_a.safetensors")
    adapter_b = str(tmp_path / "adapter_b.safetensors")

    save_file({"lora_w": torch.tensor([2.0, 4.0], dtype=torch.float32)}, adapter_a)
    save_file({"lora_w": torch.tensor([6.0, 8.0], dtype=torch.float32)}, adapter_b)

    trainer = TrainerNode("SoupTrainer", output_dir=str(tmp_path))
    config = {
        "mode": "soup",
        "soup_name": "juridique_v1",
        "adapters": [adapter_a, adapter_b],
    }
    trainer.set_input(DataPacket(DataType.ANY, config))
    result = trainer.process()

    assert result.content.endswith("juridique_v1_soup.safetensors")
    assert os.path.exists(result.content)
    assert result.metadata["type"] == "soup_adapter"
    assert result.metadata["ingredients"] == 2

    merged_weights = load_file(result.content)
    expected = torch.tensor([4.0, 6.0], dtype=torch.float16)
    assert torch.allclose(merged_weights["lora_w"], expected)


@pytest.mark.skipif(torch is None, reason="PyTorch requis")
def test_soup_rejects_incompatible_shapes(tmp_path):
    """SoupManager doit lever une erreur claire si deux adaptateurs n'ont pas les mêmes dimensions."""
    adapter_a = str(tmp_path / "a.safetensors")
    adapter_b = str(tmp_path / "b.safetensors")
    out_soup = str(tmp_path / "out.safetensors")

    save_file({"lora_w": torch.tensor([1.0, 2.0])}, adapter_a)
    save_file({"lora_w": torch.tensor([1.0, 2.0, 3.0])}, adapter_b)

    with pytest.raises(ValueError, match="Dimensions incompatibles"):
        SoupManager.make_soup([adapter_a, adapter_b], out_soup)