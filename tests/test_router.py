import unittest
from core.router import PipelineRouter, ExecutionMode

class TestPipelineRouter(unittest.TestCase):
    
    def test_text_only(self):
        """Si aucune image n'est envoyée, on reste en texte pur."""
        mode, _ = PipelineRouter.determine_pipeline(
            has_image=False, main_is_multimodal=True, 
            vision_expert_loaded=True, force_orchestration=False
        )
        self.assertEqual(mode, ExecutionMode.TEXT_ONLY)

    def test_blocked_error_missing_vision(self):
        """L'envoi d'une image doit être bloqué si aucun modèle ne peut la lire."""
        mode, _ = PipelineRouter.determine_pipeline(
            has_image=True, main_is_multimodal=False, 
            vision_expert_loaded=False, force_orchestration=False
        )
        self.assertEqual(mode, ExecutionMode.BLOCKED_ERROR)

    def test_auto_detect_multimodal(self):
        """Un Moteur Principal multimodal doit prendre en charge l'image par défaut."""
        mode, _ = PipelineRouter.determine_pipeline(
            has_image=True, main_is_multimodal=True, 
            vision_expert_loaded=False, force_orchestration=False
        )
        self.assertEqual(mode, ExecutionMode.MULTIMODAL_DIRECT)

    def test_auto_detect_orchestration(self):
        """Un Moteur Textuel couplé à un Expert Visuel doit déclencher l'orchestration."""
        mode, _ = PipelineRouter.determine_pipeline(
            has_image=True, main_is_multimodal=False, 
            vision_expert_loaded=True, force_orchestration=False
        )
        self.assertEqual(mode, ExecutionMode.ORCHESTRATED)

    def test_force_orchestration(self):
        """La case à cocher doit forcer l'orchestration même si le Moteur Principal est multimodal."""
        mode, _ = PipelineRouter.determine_pipeline(
            has_image=True, main_is_multimodal=True, 
            vision_expert_loaded=True, force_orchestration=True
        )
        self.assertEqual(mode, ExecutionMode.ORCHESTRATED)

if __name__ == '__main__':
    unittest.main(verbosity=2)