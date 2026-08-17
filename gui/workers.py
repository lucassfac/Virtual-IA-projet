"""
workers.py — QThread workers pour découpler l'IA de l'interface graphique.
Centralise toutes les tâches asynchrones (Inférence, Vision, Réseau).
"""

from PyQt6.QtCore import QThread, pyqtSignal

# ── INFERENCE ─────────────────────────────────────────────────────────

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

# ── RÉSEAU ET FICHIERS (HuggingFace) ──────────────────────────────────

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
        self.repo_id = repo_id
        self.filename = filename
        self.dest_dir = dest_dir
        self.token = token
        self._cancel  = [False]
        
    def cancel(self): 
        self._cancel[0] = True
        
    def run(self):
        try:
            from core.model_manager import download_model
            path = download_model(
                self.repo_id, 
                self.filename, 
                self.dest_dir, 
                progress_cb=lambda d, t: self.progress.emit(d, t), 
                cancel_flag=self._cancel, 
                token=self.token
            )
            self.success.emit(path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

# ── AUTRES TÂCHES ─────────────────────────────────────────────────────

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
    
    def __init__(self, llm_node, model_path, mmproj_path=None, skill_path=None, draft_path=None, turbo=False):
        super().__init__()
        self.llm_node = llm_node
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.skill_path = skill_path
        self.draft_path = draft_path
        self.turbo = turbo
        
    def run(self):
        try:
            # On injecte le projecteur visuel dans le nœud s'il existe
            self.llm_node.mmproj_path = self.mmproj_path 
            self.llm_node.load_model(
                self.model_path, 
                skill_path=self.skill_path,
                draft_model_path=self.draft_path, 
                turbo=self.turbo
            )
            self.success.emit(self.model_path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

# ── LE CHEF D'ORCHESTRE ───────────────────────────────────────────────
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
            # 1. MODE LLaVA SOLO (Force Brute HD)
            if not self.llm_node.model_loaded:
                self.status.emit("👁️ [1/3] Chargement de LLaVA en VRAM...")
                self.vision_node.load_model(self.vis_model_path, self.vis_mmproj_path)
                self.vision_node.set_high_res_mode(True) 
                self.status.emit("⏳ [2/3] Analyse HD en cours (1 à 3 minutes)...")
                self.vision_node.set_prompt(self.text_prompt)
                
                img_packet = DataPacket(DataType.IMAGE_PATH, self.image_path, source="orchestrator")
                self.vision_node.set_input(img_packet)
                vision_result = self.vision_node.process()
                
                self.status.emit("🔄 [3/3] Libération de la VRAM...")
                self.vision_node.unload_model()
                self.token.emit(vision_result.content)
                return

            # 2. MODE ORCHESTRATION (Gemma + LLaVA SD + OCR)
            self.status.emit("🔍 [1/6] Radar textuel (OCR Rapide)...")
            ocr_text, ocr_failed = "", False
            try:
                from core.document_reader import _read_image_fast
                res = _read_image_fast(self.image_path).strip()
                if "échoué" in res.lower() or "erreur" in res.lower():
                    ocr_failed = True
                else:
                    ocr_text = res
            except Exception:
                ocr_failed = True

            has_text = len(ocr_text) > 15

            if has_text:
                self.status.emit("📄 Texte extrait (OCR). Analyse visuelle des éléments graphiques...")
                vision_instruction = (
                    "Le texte de cette image a déjà été lu par un autre outil. Ton rôle est de le compléter. "
                    "1. Décris la mise en page générale.\n"
                    "2. S'il y a des schémas, graphiques ou diagrammes : traduis leur structure formelle en texte clair.\n"
                    "3. S'il y a des photographies : décris leur contenu visuel en détail.\n"
                    "Ne perds pas de temps à retranscrire le texte."
                )
                final_template = (
                    "Voici les données extraites de l'image :\n\n"
                    f"### TEXTE EXACT (OCR) ###\n{ocr_text}\n\n"
                    f"### STRUCTURE ET VISUEL (VLM) ###\n{{img_desc}}\n\n"
                    f"CONSIGNE : Réponds à la question : {self.text_prompt}"
                )
            elif ocr_failed:
                self.status.emit("⚠️ OCR inactif. Le modèle Vision prend le relais complet...")
                vision_instruction = "Transcris le texte visible, puis décris l'image en détail."
                final_template = f"Voici l'analyse de l'image :\n\n### ANALYSE ###\n{{img_desc}}\n\nCONSIGNE : Réponds à : {self.text_prompt}"
            else:
                self.status.emit("🖼️ Aucun texte détecté. Bascule en analyse visuelle pure...")
                vision_instruction = "Il n'y a pas de texte à lire sur cette image. Décris visuellement, fidèlement et en détail le sujet principal."
                final_template = f"Voici la description détaillée d'une photographie :\n\n### IMAGE ###\n{{img_desc}}\n\nCONSIGNE : Réponds à : {self.text_prompt}"

            self.status.emit("🔄 [2/6] Déchargement du moteur principal...")
            self.llm_node.unload_model()

            self.status.emit("👁️ [3/6] Analyse structurelle (Vision)...")
            self.vision_node.load_model(self.vis_model_path, self.vis_mmproj_path)
            self.vision_node.set_high_res_mode(False) 
            
            self.vision_node.set_prompt(vision_instruction)
            img_packet = DataPacket(DataType.IMAGE_PATH, self.image_path, source="orchestrator")
            self.vision_node.set_input(img_packet)
            vision_result = self.vision_node.process()
            
            self.status.emit("🔄 [4/6] Déchargement du modèle Vision...")
            self.vision_node.unload_model()

            self.status.emit("🧠 [5/6] Rechargement du moteur principal...")
            self.llm_node.load_model(self.llm_model_path, skill_path=self.llm_skill_path)

            self.status.emit("✨ [6/6] Synthèse IA...")
            final_prompt = final_template.format(img_desc=vision_result.content)
            text_packet = DataPacket(DataType.TEXT, final_prompt, source="orchestrator")
            
            for tok in self.llm_node.stream_inference(text_packet):
                if self._cancelled: break
                self.token.emit(tok)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()