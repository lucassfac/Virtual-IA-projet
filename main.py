from core.nodes.llm_node import LLMNode
from core.nodes.trainer_node import TrainerNode
from core.types import DataPacket, DataType
import os

def main():
    print("=== TEST DU PIPELINE LoRA (Spécialisation) ===\n")

    # --- PHASE 1 : CRÉATION DE L'EXPERT (Training) ---
    trainer = TrainerNode(name="Trainer-Juridique")
    
    # Inputs fictifs pour le test
    dataset = DataPacket(DataType.TEXT, "documents_juridiques.jsonl")
    base_model = DataPacket(DataType.TEXT, "models/tinyllama.gguf")
    
    trainer.set_inputs(dataset, base_model)
    
    # Lancement de l'entraînement simulé
    packet_resultat_training = trainer.process()
    path_lora_cree = packet_resultat_training.content
    
    print(f"\n✅ SUCCESS: Fichier LoRA créé à : {path_lora_cree}\n")

    # --- PHASE 2 : UTILISATION DE L'EXPERT (Inférence) ---
    llm_node = LLMNode(name="Mistral-Avocat")
    
    # On charge le modèle AVEC le chemin du LoRA qu'on vient de créer
    # (Note: Dans ce test simulé, le fichier .lora n'existe pas vraiment physiquement
    # donc on va mettre lora_path=None juste pour éviter que llama.cpp ne plante,
    # mais la logique du code est prête pour quand tu auras un vrai fichier .lora)
    
    # Pour le test réel sans planter : on ne met pas le lora_path car le fichier est fictif
    # Mais dans la vraie vie : lora_path=path_lora_cree
    llm_node.load_model("models/tinyllama.gguf", lora_path=None) 

    # Test de question
    user_input = DataPacket(DataType.TEXT, "Que dit l'article 12 du code civil ?")
    llm_node.set_input(user_input)
    
    reponse = llm_node.process()
    
    print(f"--- Réponse de l'IA Spécialisée ---")
    print(reponse.content)

if __name__ == "__main__":
    main()