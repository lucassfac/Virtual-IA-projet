from llama_cpp import Llama
from core.node import BaseNode
from core.types import DataPacket, DataType

class LLMNode(BaseNode):
    def __init__(self, name: str = "LLM Generator"):
        super().__init__(name)
        self.llm_engine = None  # C'est ici qu'on stockera l'objet Llama

    def load_model(self, model_path: str):
        """
        Surcharge de la méthode parent.
        Ici, on charge VRAIMENT le modèle en RAM avec llama-cpp.
        """
        print(f"[{self.name}] Chargement du modèle {model_path}...")
        try:
            # n_ctx=2048 : La mémoire de conversation
            # n_threads=4 : Utilise 4 cœurs du CPU pour aller vite
            self.llm_engine = Llama(model_path=model_path, n_ctx=2048, verbose=False)
            
            # Important : On dit au parent (BaseNode) que c'est bon
            super().load_model(model_path)
            
        except Exception as e:
            print(f"ERREUR FATALE: Impossible de charger le modèle. {e}")
            raise e

    def _run_inference(self):
        prompt_utilisateur = self.input_packet.content
        
        # MODIFICATION 1 : On enlève le "A:" pour laisser l'IA compléter naturellement
        # TinyLlama préfère souvent compléter une phrase.
        full_prompt = f"Question: {prompt_utilisateur}\nRéponse:"

        print(f"[{self.name}] Génération en cours...")
        
        output = self.llm_engine(
            full_prompt, 
            max_tokens=100,
            # MODIFICATION 2 : On retire "\n" de la liste stop !
            # On lui permet de faire des paragraphes.
            stop=["Question:"], 
            echo=False
        )
        
        # On nettoie le texte (strip enlève les espaces inutiles au début/fin)
        reponse_texte = output['choices'][0]['text'].strip()
        
        # DEBUG : Si c'est vide, on veut le savoir
        if not reponse_texte:
            reponse_texte = "(L'IA a renvoyé une réponse vide, essaie de changer le prompt)"

        return DataPacket(DataType.TEXT, reponse_texte)