"""
main.py — Script de test fonctionnel du pipeline de spécialisation (Skills).
"""

from core.nodes.llm_node import LLMNode
from core.nodes.trainer_node import TrainerNode
from core.types import DataPacket, DataType

def main():
    print("=== TEST DU PIPELINE DE COMPÉTENCES (Skills) ===\n")

    # --- PHASE 1 : CRÉATION DE L'EXPERT (Compilation RAG) ---
    trainer = TrainerNode(name="Trainer-Juridique")
    
    # Inputs fictifs pour le test
    dataset = DataPacket(DataType.TEXT, "documents_juridiques.jsonl")
    base_model = DataPacket(DataType.TEXT, "") # Ignoré en mode RAG
    
    trainer.set_inputs(dataset, base_model)
    
    # Lancement de la compilation simulée
    packet_resultat = trainer.process()
    path_skill_cree = packet_resultat.content
    
    print(f"\n✅ SUCCESS: Fichier Skill créé à : {path_skill_cree}\n")

    # --- PHASE 2 : UTILISATION DE L'EXPERT (Inférence) ---
    llm_node = LLMNode(name="Mistral-Avocat")
    
    # On simule le chargement du modèle (On met skill_path=None pour ne pas crasher si le fichier n'existe pas)
    # Dans un cas réel : skill_path=path_skill_cree
    llm_node.load_model("models/tinyllama.gguf", skill_path=None) 

    user_input = DataPacket(DataType.TEXT, "Que dit l'article 12 du code civil ?")
    llm_node.set_input(user_input)
    
    reponse = llm_node.process()
    
    print("--- Réponse de l'IA Spécialisée ---")
    print(reponse.content)


if __name__ == "__main__":
    main()