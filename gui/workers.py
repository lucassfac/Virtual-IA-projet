"""
workers.py — QThread workers pour découpler l'IA de l'interface graphique.
"""

from PyQt6.QtCore import QThread, pyqtSignal

class InferenceWorker(QThread):
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
    token    = pyqtSignal(str)
    error    = pyqtSignal(str)
    finished = pyqtSignal()
    def __init__(self, llm_node, packet):
        super().__init__()
        self.llm_node  = llm_node
        self.packet    = packet
        self._cancelled = False
    def cancel(self):
        self._cancelled = True
    def run(self):
        try:
            for tok in self.llm_node.stream_inference(self.packet):
                if self._cancelled: break
                self.token.emit(tok)
        except Exception as e:
            if not self._cancelled: self.error.emit(str(e))
        finally:
            self.finished.emit()

class VisionWorker(QThread):
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
    progress = pyqtSignal(int, int, float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    def __init__(self, trainer_node):
        super().__init__()
        self.trainer_node = trainer_node
    def run(self):
        try:
            self.trainer_node.set_progress_callback(lambda step, total, loss: self.progress.emit(step, total, loss))
            result_packet = self.trainer_node.process()
            self.finished.emit(result_packet.content)
        except Exception as e:
            self.error.emit(str(e))

class ModelLoadWorker(QThread):
    success = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    def __init__(self, llm_node, model_path, skill_path=None):
        super().__init__()
        self.llm_node = llm_node
        self.model_path = model_path
        self.skill_path = skill_path
    def run(self):
        try:
            self.llm_node.load_model(self.model_path, self.skill_path)
            self.success.emit(self.model_path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

class SearchWorker(QThread):
    results = pyqtSignal(list)
    error   = pyqtSignal(str)
    finished = pyqtSignal()
    def __init__(self, query: str):
        super().__init__()
        self.query = query
    def run(self):
        try:
            from core.model_manager import search_models
            self.results.emit(search_models(self.query))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

class DownloadWorker(QThread):
    progress  = pyqtSignal(int, int)
    success   = pyqtSignal(str)
    error     = pyqtSignal(str)
    finished  = pyqtSignal()
    def __init__(self, repo_id: str, filename: str, dest_dir: str, token: str = ""):
        super().__init__()
        self.repo_id, self.filename, self.dest_dir, self.token = repo_id, filename, dest_dir, token
        self._cancel  = [False]
    def cancel(self): self._cancel[0] = True
    def run(self):
        try:
            from core.model_manager import download_model
            path = download_model(self.repo_id, self.filename, self.dest_dir, progress_cb=lambda d, t: self.progress.emit(d, t), cancel_flag=self._cancel, token=self.token)
            self.success.emit(path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

# ── LE CHEF D'ORCHESTRE EST ICI ──
class SwapOrchestratorWorker(QThread):
    status   = pyqtSignal(str)
    token    = pyqtSignal(str)
    error    = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, llm_node, vision_node, image_path, text_prompt, llm_paths, vis_paths):
        super().__init__()
        self.llm_node = llm_node
        self.vision_node = vision_node
        self.image_path = image_path
        self.text_prompt = text_prompt
        self.llm_model_path, self.llm_skill_path = llm_paths
        self.vis_model_path, self.vis_mmproj_path = vis_paths
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from core.types import DataPacket, DataType
        try:
            self.status.emit("🔄 [1/5] Déchargement de Gemma (Libération VRAM)...")
            self.llm_node.unload_model()

            self.status.emit("👁️ [2/5] Chargement de LLaVA...")
            self.vision_node.load_model(self.vis_model_path, self.vis_mmproj_path)

            prompt_lower = self.text_prompt.lower()
            mots_haute_precision = ["texte", "exo", "exercice", "mcd", "sql", "schéma", "diagramme", "code", "lis"]
            needs_high_res = any(mot in prompt_lower for mot in mots_haute_precision)
            self.vision_node.set_high_res_mode(needs_high_res)

            if needs_high_res:
                self.status.emit("🔍 [3/5] Extraction HD (Lecture précise, ~30s)...")
            else:
                self.status.emit("⚡ [3/5] Extraction Rapide (Vue globale, ~3s)...")

            vision_instruction = (
                "Analyse cette image d'exercice de base de données.\n"
                "PARTIE 1 : Transcris mot pour mot tout le texte de l'exercice (les questions a, b, c, etc.) en bas de l'image.\n"
                "PARTIE 2 : Décris le schéma (entités, relations, et surtout les chiffres des cardinalités comme 1,1 ou 0,N)."
            )
            self.vision_node.set_prompt(vision_instruction)
            
            img_packet = DataPacket(DataType.IMAGE_PATH, self.image_path, source="orchestrator")
            self.vision_node.set_input(img_packet)
            vision_result = self.vision_node.process()
            img_desc = vision_result.content

            self.status.emit("🔄 [4/5] Déchargement de LLaVA (Libération VRAM)...")
            self.vision_node.unload_model()

            self.status.emit("🧠 [5/5] Rechargement de Gemma...")
            self.llm_node.load_model(self.llm_model_path, skill_path=self.llm_skill_path)

            self.status.emit("✨ Résolution finale par le Cerveau...")
            
            # ✅ PROMPT DE RELAIS "AUTORITAIRE" (Empêche Gemma de dire qu'elle ne voit pas l'image)
            final_prompt = (
                "Voici les données extraites d'une image d'exercice MCD :\n\n"
                f"### TEXTE ET STRUCTURE LUS PAR LE MODULE VISION ###\n{img_desc}\n"
                "###################################################\n\n"
                "CONSIGNE : Réponds point par point aux questions (a, b, c, d, e, f) présentes dans le texte ci-dessus. "
                "Utilise la structure du MCD décrite pour justifier chaque réponse. "
                f"Demande de l'utilisateur : {self.text_prompt}"
            )
            text_packet = DataPacket(DataType.TEXT, final_prompt, source="orchestrator")

            for tok in self.llm_node.stream_inference(text_packet):
                if self._cancelled: break
                self.token.emit(tok)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()