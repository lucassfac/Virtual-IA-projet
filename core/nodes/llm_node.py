from llama_cpp import Llama
from core.node import BaseNode
from core.types import DataPacket, DataType

class LLMNode(BaseNode):
    def __init__(self, name: str = "LLM Generator"):
        super().__init__(name)
        self.llm_engine = None

    # MODIFICATION : On ajoute un argument optionnel lora_path
    def load_model(self, model_path: str, lora_path: str = None):
        """
        Charge le modèle, et optionnellement un adaptateur LoRA (Spécialisation).
        """
        print(f"[{self.name}] Chargement du modèle {model_path}...")
        
        if lora_path:
            print(f"[{self.name}] 🧬 APPLICATION DE LA SPÉCIALISATION (LoRA) : {lora_path}")
        
        try:
            # MODIFICATION ICI : On passe lora_path au moteur
            self.llm_engine = Llama(
                model_path=model_path,
                lora_path=lora_path,  # C'est ici que la magie opère
                n_ctx=2048, 
                verbose=False
            )
            super().load_model(model_path)
            
        except Exception as e:
            print(f"ERREUR FATALE: {e}")
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