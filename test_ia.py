from llama_cpp import Llama
import os

# 1. Définir le chemin du modèle
model_path = "models/tinyllama.gguf"

# Vérification de sécurité
if not os.path.exists(model_path):
    print(f"ERREUR: Le fichier {model_path} est introuvable !")
    exit()

print("--- Chargement du modèle en cours (ça peut prendre quelques secondes)... ---")

# 2. Initialiser le cerveau (Inférence)
# n_ctx=2048 : C'est la mémoire à court terme (combien de mots il retient)
# verbose=False : Pour éviter qu'il inonde ton terminal de logs techniques
llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

print("--- Modèle chargé ! Génération de la réponse... ---")

# 3. Poser une question (Le Prompt)
question = "Q: Explique moi en une phrase ce qu'est une Intelligence Artificielle. A: "

# 4. Lancer la génération
output = llm(
    question, 
    max_tokens=100,  # Limite la longueur de la réponse
    stop=["Q:", "\n"], # Lui dire de s'arrêter s'il essaie de se parler à lui-même
    echo=True        # Afficher la question + la réponse
)

# 5. Afficher le résultat propre
print("\n=== RÉSULTAT ===")
print(output['choices'][0]['text'])