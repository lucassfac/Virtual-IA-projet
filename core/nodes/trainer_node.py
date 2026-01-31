import time
from core.node import BaseNode
from core.types import DataPacket, DataType

class TrainerNode(BaseNode):
    def __init__(self, name: str = "LoRA Trainer"):
        super().__init__(name)
        self.dataset_path = None
        self.base_model_path = None
        self.output_adapter_path = "models/mon_expert.lora" # Où on sauvegarde le résultat

    def set_inputs(self, dataset_packet: DataPacket, model_packet: DataPacket):
        """Ce nœud a besoin de DEUX ingrédients : Données + Cerveau de base"""
        if dataset_packet.data_type == DataType.TEXT: # Idéalement un chemin vers un JSONL
            self.dataset_path = dataset_packet.content
        
        if model_packet.data_type == DataType.TEXT: # Le chemin du modèle de base
            self.base_model_path = model_packet.content

    def _run_inference(self):
        """
        Lancer l'entraînement (Fine-Tuning).
        """
        print(f"[{self.name}] --- DÉBUT DE L'ENTRAÎNEMENT ---")
        print(f"[{self.name}] Base : {self.base_model_path}")
        print(f"[{self.name}] Données : {self.dataset_path}")
        
        # ICI : C'est là qu'on mettrait le code lourd avec la librairie 'PEFT' ou 'Unsloth'.
        # Pour le test, on simule le travail.
        
        steps = 5
        for i in range(steps):
            time.sleep(0.5) # On simule du calcul GPU
            print(f"[{self.name}] Étape {i+1}/{steps} : Optimisation des poids...")

        print(f"[{self.name}] Sauvegarde de l'adaptateur LoRA vers {self.output_adapter_path}")
        
        # On renvoie le chemin du fichier créé pour que le LLM puisse l'utiliser
        return DataPacket(DataType.TEXT, self.output_adapter_path)