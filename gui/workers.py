"""
workers.py — QThread workers pour découpler l'IA de l'interface graphique.

Chaque worker :
  - Hérite de QThread
  - Émet des signaux PyQt6 (résultat, erreur, progression)
  - Ne touche JAMAIS directement aux widgets (thread-safe)

Signaux disponibles :
  InferenceWorker  → result(str), error(str), finished()
  StreamWorker     → token(str), error(str), finished()
  VisionWorker     → result(str), error(str), finished()
  TrainingWorker   → progress(int, int, float), finished(str), error(str)
"""

from PyQt6.QtCore import QThread, pyqtSignal


class InferenceWorker(QThread):
    """
    Lance une inférence LLM bloquante dans un thread séparé.
    Utilisé quand le streaming n'est pas nécessaire.
    """

    result = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, llm_node, packet):
        super().__init__()
        self.llm_node = llm_node
        self.packet = packet

    def run(self):
        try:
            self.llm_node.set_input(self.packet)
            result_packet = self.llm_node.process()
            self.result.emit(result_packet.content)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class StreamWorker(QThread):
    """
    Lance une inférence LLM en streaming (token par token).
    Supporte l'annulation via cancel().
    """

    token    = pyqtSignal(str)
    error    = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, llm_node, packet):
        super().__init__()
        self.llm_node  = llm_node
        self.packet    = packet
        self._cancelled = False

    def cancel(self):
        """Demande l'arrêt propre de la génération."""
        self._cancelled = True

    def run(self):
        try:
            for tok in self.llm_node.stream_inference(self.packet):
                if self._cancelled:
                    break
                self.token.emit(tok)
        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))
        finally:
            self.finished.emit()


class VisionWorker(QThread):
    """
    Lance l'analyse d'image (LLaVA) dans un thread séparé.
    """

    result = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, vision_node, packet):
        super().__init__()
        self.vision_node = vision_node
        self.packet = packet

    def run(self):
        try:
            self.vision_node.set_input(self.packet)
            result_packet = self.vision_node.process()
            self.result.emit(result_packet.content)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class TrainingWorker(QThread):
    """
    Lance l'entraînement LoRA dans un thread séparé.

    Signaux :
      progress(step, total, loss)  — mis à jour à chaque étape
      finished(lora_path)          — chemin du fichier .lora produit
      error(message)               — en cas d'échec
    """

    progress = pyqtSignal(int, int, float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, trainer_node):
        super().__init__()
        self.trainer_node = trainer_node

    def run(self):
        try:
            self.trainer_node.set_progress_callback(
                lambda step, total, loss: self.progress.emit(step, total, loss)
            )
            result_packet = self.trainer_node.process()
            self.finished.emit(result_packet.content)
        except Exception as e:
            self.error.emit(str(e))


class ModelLoadWorker(QThread):
    """
    Charge un modèle LLM dans un thread séparé (peut prendre plusieurs secondes).
    """

    success = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, llm_node, model_path, lora_path=None):
        super().__init__()
        self.llm_node = llm_node
        self.model_path = model_path
        self.lora_path = lora_path

    def run(self):
        try:
            self.llm_node.load_model(self.model_path, self.lora_path)
            self.success.emit(self.model_path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class SearchWorker(QThread):
    """Recherche de modèles HuggingFace dans un thread."""
    results = pyqtSignal(list)
    error   = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self):
        try:
            from core.model_manager import search_models
            data = search_models(self.query)
            self.results.emit(data)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class DownloadWorker(QThread):
    """Téléchargement d'un modèle GGUF depuis HuggingFace."""
    progress  = pyqtSignal(int, int)   # (bytes_done, bytes_total)
    success   = pyqtSignal(str)        # chemin absolu
    error     = pyqtSignal(str)
    finished  = pyqtSignal()

    def __init__(self, repo_id: str, filename: str, dest_dir: str, token: str = ""):
        super().__init__()
        self.repo_id  = repo_id
        self.filename = filename
        self.dest_dir = dest_dir
        self.token    = token
        self._cancel  = [False]

    def cancel(self):
        self._cancel[0] = True

    def run(self):
        try:
            from core.model_manager import download_model
            path = download_model(
                self.repo_id, self.filename, self.dest_dir,
                progress_cb=lambda d, t: self.progress.emit(d, t),
                cancel_flag=self._cancel,
                token=self.token,
            )
            self.success.emit(path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
