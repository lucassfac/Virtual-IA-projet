"""
memory_manager.py — Gestionnaire de Mémoire à Long Terme (RAG Épisodique).
Indexation vectorielle et récupération sémantique (VectorStoreRetrieverMemory).
"""

import os
from typing import List, Dict, Optional
from core.logger import forge_logger

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None


class MemoryManager:
    """
    Gère la persistance et la récupération sémantique des historiques de conversation.
    Fonctionne comme une base de données vectorielle locale.
    """

    def __init__(self, storage_path: str = "storage/memory", collection_name: str = "neural_forge_conversations"):
        if chromadb is None:
            forge_logger.error("[MemoryManager] ChromaDB n'est pas installé. La mémoire à long terme est désactivée.")
            self._client = None
            return

        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        try:
            # Initialisation du client local ChromaDB
            self._client = chromadb.PersistentClient(
                path=self.storage_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Utilisation du modèle d'embedding multilingue léger par défaut (parfait pour le français)
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_fn,
                metadata={"description": "Mémoire épisodique des conversations de Neural Forge"}
            )
            forge_logger.info(f"[MemoryManager] Collection '{collection_name}' initialisée avec succès.")
            
        except Exception as e:
            forge_logger.error(f"[MemoryManager] Échec de l'initialisation de ChromaDB : {e}")
            self._client = None

    @property
    def is_active(self) -> bool:
        return self._client is not None

    def save_exchange(self, session_id: str, user_query: str, ai_response: str, metadata: Optional[Dict] = None) -> None:
        """
        Sauvegarde un échange (Question/Réponse) sous forme de vecteur mathématique.
        """
        if not self.is_active:
            return

        try:
            # Création du "Chunk" (morceau de contexte)
            document = f"Utilisateur : {user_query}\nAssistant : {ai_response}"
            
            # Identifiant unique basé sur la session et le nombre d'éléments existants
            doc_id = f"{session_id}_{self._collection.count() + 1}"
            
            meta = {"session_id": session_id}
            if metadata:
                meta.update(metadata)

            self._collection.add(
                documents=[document],
                metadatas=[meta],
                ids=[doc_id]
            )
        except Exception as e:
            forge_logger.error(f"[MemoryManager] Erreur lors de la sauvegarde de l'échange : {e}")

    def retrieve_context(self, current_query: str, session_id: Optional[str] = None, n_results: int = 2) -> str:
        """
        Recherche dans la base vectorielle les échanges passés sémantiquement proches de la question actuelle.
        Retourne un bloc de contexte formaté prêt à être injecté dans le prompt système.
        """
        if not self.is_active or self._collection.count() == 0:
            return ""

        try:
            # Filtre optionnel pour ne chercher que dans la session en cours ou restaurée
            where_clause = {"session_id": session_id} if session_id else None

            # Recherche des vecteurs les plus proches de la question
            results = self._collection.query(
                query_texts=[current_query],
                n_results=min(n_results, self._collection.count()),
                where=where_clause
            )

            documents = results.get("documents", [[]])[0]
            
            if not documents:
                return ""

            # Formatage sémantique strict pour éviter la confusion du LLM
            context_block = "\n\n[MÉMOIRE ÉPISODIQUE RÉCUPÉRÉE DES DISCUSSIONS PRÉCÉDENTES]\n"
            for doc in documents:
                context_block += f"---\n{doc}\n"
            context_block += "---\n[FIN DE LA MÉMOIRE RÉCUPÉRÉE]\n"

            return context_block

        except Exception as e:
            forge_logger.error(f"[MemoryManager] Erreur lors de la récupération du contexte : {e}")
            return ""