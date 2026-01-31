import pytest
import sys
import os

# Importation du dossier core
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.node import BaseNode, AIModelMissingError, InputMissingError
from core.types import DataPacket, DataType

# --- MOCKING ---
# On crée un faux nœud simple juste pour tester la logique de base
# sans avoir besoin de charger un vrai Llama (ce qui serait lent)
class MockNode(BaseNode):
    def _run_inference(self):
        return "Succès"

# --- LES TESTS ---

def test_run_without_model_should_fail():
    """Test : L'application ne peut pas se lancer sans modèle préinstallé."""
    node = MockNode("TestNode")
    
    # On essaie de lancer process() SANS avoir fait load_model()
    # On s'attend à ce que ça lève une erreur AIModelMissingError
    with pytest.raises(AIModelMissingError):
        node.process()

def test_run_without_input_should_fail():
    """Test : L'application ne doit pas lancer le modèle sans entrée préalable."""
    node = MockNode("TestNode")
    
    # 1. On charge le modèle (donc la règle 1 est OK)
    node.load_model("/tmp/fake_model.gguf")
    
    # 2. MAIS on ne lui donne pas d'input (Règle 2 KO)
    with pytest.raises(InputMissingError):
        node.process()

def test_nominal_case():
    """Test : Tout va bien (Modèle OK + Input OK)."""
    node = MockNode("TestNode")
    
    # 1. Setup complet
    node.load_model("/tmp/fake_model.gguf")
    packet = DataPacket(DataType.TEXT, "Bonjour")
    node.set_input(packet)
    
    # 2. Exécution
    result = node.process()
    assert result == "Succès"