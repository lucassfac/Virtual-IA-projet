"""
test_trainer_node.py — Tests unitaires pour TrainerNode.

Couvre :
  - Sécurité (dataset manquant, modèle manquant)
  - set_inputs() : rejet des mauvais types
  - Chemin nominal : résultat, extension, métadonnées
  - Callback de progression : appelé à chaque étape, loss décroissante
  - Nom de l'adaptateur dérivé du dataset
  - reset() après entraînement
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.nodes.trainer_node import TrainerNode
from core.types import DataPacket, DataType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_trainer(output_dir="/tmp") -> TrainerNode:
    return TrainerNode("TestTrainer", output_dir=output_dir)


def make_inputs():
    return (
        DataPacket(DataType.TEXT, "data/juridique.jsonl"),
        DataPacket(DataType.TEXT, "models/tinyllama.gguf"),
    )


# ---------------------------------------------------------------------------
# Règles de sécurité
# ---------------------------------------------------------------------------


def test_process_without_dataset_raises():
    """process() sans dataset doit lever ValueError mentionnant 'dataset'."""
    trainer = make_trainer()
    trainer.base_model_path = "models/base.gguf"   # Modèle fourni
    with pytest.raises(ValueError, match="dataset"):
        trainer.process()


def test_process_without_model_raises():
    """process() sans modèle de base doit lever ValueError mentionnant 'modèle'."""
    trainer = make_trainer()
    trainer.dataset_path = "data/train.jsonl"   # Dataset fourni
    with pytest.raises(ValueError, match="modèle"):
        trainer.process()


def test_process_without_any_input_raises():
    """process() sans rien du tout doit lever ValueError."""
    trainer = make_trainer()
    with pytest.raises(ValueError):
        trainer.process()


# ---------------------------------------------------------------------------
# Validation des types dans set_inputs()
# ---------------------------------------------------------------------------


def test_set_inputs_rejects_image_dataset():
    """set_inputs() doit rejeter un dataset de type IMAGE_PATH."""
    trainer = make_trainer()
    with pytest.raises(TypeError):
        trainer.set_inputs(
            DataPacket(DataType.IMAGE_PATH, "/img.jpg"),
            DataPacket(DataType.TEXT, "model.gguf"),
        )


def test_set_inputs_rejects_image_model():
    """set_inputs() doit rejeter un modèle de type IMAGE_PATH."""
    trainer = make_trainer()
    with pytest.raises(TypeError):
        trainer.set_inputs(
            DataPacket(DataType.TEXT, "data.jsonl"),
            DataPacket(DataType.IMAGE_PATH, "/model.jpg"),
        )


# ---------------------------------------------------------------------------
# Chemin nominal
# ---------------------------------------------------------------------------


def test_nominal_returns_datapacket():
    """L'entraînement doit retourner un DataPacket TEXT."""
    trainer = make_trainer()
    trainer.set_inputs(*make_inputs())
    result = trainer.process()

    assert result is not None
    assert result.data_type == DataType.TEXT


def test_nominal_output_is_lora_file():
    """Le chemin retourné doit se terminer par .lora."""
    trainer = make_trainer()
    trainer.set_inputs(*make_inputs())
    result = trainer.process()
    assert result.content.endswith(".lora")


def test_adapter_name_derived_from_dataset():
    """Le nom de l'adaptateur doit contenir le nom du dataset (sans extension)."""
    trainer = make_trainer()
    trainer.set_inputs(
        DataPacket(DataType.TEXT, "data/contrats_clients.jsonl"),
        DataPacket(DataType.TEXT, "models/base.gguf"),
    )
    result = trainer.process()
    assert "contrats_clients" in result.content


def test_output_metadata_complete():
    """Les métadonnées doivent contenir base_model, dataset, steps, final_loss."""
    trainer = make_trainer()
    trainer.set_inputs(*make_inputs())
    result = trainer.process()

    meta = result.metadata
    assert "base_model" in meta
    assert "dataset" in meta
    assert "steps" in meta
    assert "final_loss" in meta


def test_output_metadata_values():
    """Les valeurs des métadonnées doivent correspondre aux inputs."""
    dataset_pkt = DataPacket(DataType.TEXT, "data/juridique.jsonl")
    model_pkt = DataPacket(DataType.TEXT, "models/mistral.gguf")

    trainer = make_trainer()
    trainer.set_inputs(dataset_pkt, model_pkt)
    result = trainer.process()

    assert result.metadata["base_model"] == "models/mistral.gguf"
    assert result.metadata["dataset"] == "data/juridique.jsonl"
    assert result.metadata["final_loss"] < 1.0


# ---------------------------------------------------------------------------
# Callback de progression
# ---------------------------------------------------------------------------


def test_progress_callback_called_each_step():
    """Le callback doit être appelé autant de fois qu'il y a d'étapes."""
    steps = []

    def on_progress(step, total, loss):
        steps.append((step, total, loss))

    trainer = make_trainer()
    trainer.set_inputs(*make_inputs())
    trainer.set_progress_callback(on_progress)
    trainer.process()

    assert len(steps) == 5   # 5 étapes simulées


def test_progress_callback_last_step():
    """Le dernier appel du callback doit correspondre à l'étape finale."""
    calls = []

    trainer = make_trainer()
    trainer.set_inputs(*make_inputs())
    trainer.set_progress_callback(lambda s, t, l: calls.append((s, t, l)))
    trainer.process()

    last = calls[-1]
    assert last[0] == last[1]   # step == total


def test_progress_loss_decreasing():
    """La loss doit décroître d'étape en étape (courbe normale)."""
    losses = []

    trainer = make_trainer()
    trainer.set_inputs(*make_inputs())
    trainer.set_progress_callback(lambda s, t, l: losses.append(l))
    trainer.process()

    for i in range(1, len(losses)):
        assert losses[i] < losses[i - 1], (
            f"La loss devrait décroître : étape {i} = {losses[i]} "
            f">= étape {i-1} = {losses[i-1]}"
        )


def test_no_callback_does_not_crash():
    """Sans callback enregistré, process() doit fonctionner normalement."""
    trainer = make_trainer()
    trainer.set_inputs(*make_inputs())
    result = trainer.process()   # Ne doit pas lever d'exception
    assert result is not None


# ---------------------------------------------------------------------------
# Traçabilité des paquets
# ---------------------------------------------------------------------------


def test_input_packets_have_trace():
    """Après set_inputs(), les paquets doivent avoir une entrée de trace."""
    dataset_pkt, model_pkt = make_inputs()
    trainer = make_trainer()
    trainer.set_inputs(dataset_pkt, model_pkt)

    assert len(dataset_pkt._processing_log) >= 1
    assert len(model_pkt._processing_log) >= 1


def test_output_source_is_trainer_name():
    """Le DataPacket de sortie doit avoir le nom du trainer comme source."""
    trainer = make_trainer()
    trainer.set_inputs(*make_inputs())
    result = trainer.process()
    assert result.source == "TestTrainer"
