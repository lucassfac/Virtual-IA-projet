import time
from core.node import BaseNode
from core.types import DataPacket, DataType

class TrainerNode(BaseNode):
    def __init__(self, name: str = "LoRA Trainer"):
        super().__init__(name)
        # On initialise les variables à None
        self.dataset_path = None
        self.base_model_path = None
        self.output_adapter_path = "models/mon_expert.lora"

        # ASTUCE : On "trompe" le système parent.
        # Comme ce nœud est une usine et non un cerveau, on dit qu'il est "toujours prêt".
        self.model_loaded = True 

    def set_inputs(self, dataset_packet: DataPacket, model_packet: DataPacket):
        """Récupère les ingrédients."""
        if dataset_packet.data_type == DataType.TEXT:
            self.dataset_path = dataset_packet.content
        
        if model_packet.data_type == DataType.TEXT:
            self.base_model_path = model_packet.content

    def process(self):
        """
        SURCHARGE : On remplace la méthode process() du parent.
        Le Trainer a ses propres règles de sécurité.
        """
        # 1. Vérification spécifique au Training
        if not self.dataset_path:
            raise ValueError(f"[{self.name}] Erreur : Pas de dataset fourni !")
        
        if not self.base_model_path:
            raise ValueError(f"[{self.name}] Erreur : Pas de modèle de base fourni !")

        # 2. On lance directement l'inférence (l'entraînement)
        return self._run_inference()

    def _run_inference(self):
        """L'entraînement simulé."""
        print(f"[{self.name}] --- DÉBUT DE L'ENTRAÎNEMENT ---")
        print(f"[{self.name}] Base : {self.base_model_path}")
        print(f"[{self.name}] Données : {self.dataset_path}")
        
        steps = 3
        for i in range(steps):
            time.sleep(0.5) # Simulation du calcul GPU
            print(f"[{self.name}] Étape {i+1}/{steps} : Optimisation des poids (LoRA)...")

        print(f"[{self.name}] ✅ Sauvegarde de l'adaptateur vers {self.output_adapter_path}")
        
        # On renvoie le chemin du fichier créé
        return DataPacket(DataType.TEXT, self.output_adapter_path)