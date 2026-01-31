from core.nodes.llm_node import LLMNode
from core.types import DataPacket, DataType
import os

def main():
    print("=== DÉMARRAGE DE NEURAL FORGE (CLI MODE) ===")

    # 1. Création du Nœud
    my_node = LLMNode(name="Mistral-Local")

    # 2. Chargement du Modèle (Celui que tu as téléchargé tout à l'heure)
    # Assure-toi que le fichier est bien là !
    model_path = "models/tinyllama.gguf" 
    
    if not os.path.exists(model_path):
        print(f"Erreur : Le modèle {model_path} n'existe pas.")
        return

    my_node.load_model(model_path)

    # 3. Simulation de l'entrée utilisateur (Comme si ça venait de l'interface)
    user_input = DataPacket(DataType.TEXT, "Quelle est la capitale de la France ?")
    my_node.set_input(user_input)

    # 4. Exécution du moteur
    result_packet = my_node.process()

    # 5. Affichage du résultat
    print("\n--- RÉSULTAT FINAL ---")
    print(f"IA a répondu : {result_packet.content}")
    print(f"Type de donnée : {result_packet.data_type}")

if __name__ == "__main__":
    main()