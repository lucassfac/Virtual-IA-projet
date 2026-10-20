# 🧠 Neural Forge — Local Composite AI & Model Soup Studio

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6%20(Sonoma%20Dark)-41CD52?logo=qt&logoColor=white)
![Inference](https://img.shields.io/badge/Engine-llama.cpp%20%7C%20GGUF-FF6F00)
![Training](https://img.shields.io/badge/Fine--Tuning-PyTorch%20%7C%20PEFT%20%7C%20Model%20Soup-EE4C2C?logo=pytorch&logoColor=white)
![RAG](https://img.shields.io/badge/Memory-ChromaDB%20Vector%20Store-00A67E)
![Tests](https://img.shields.io/badge/Tests-Pytest%20Isolated-0A84FF?logo=pytest&logoColor=white)

**Neural Forge** est une station de travail d'Intelligence Artificielle **100 % locale, modulaire et respectueuse de la confidentialité**, conçue pour exécuter, orchestrer, spécialiser (SFT LoRA) et fusionner (*Model Soup*) des modèles de langage (LLM) et de vision (VLM) sur du matériel grand public.

---

## ✨ Fonctionnalités Clés

1. **Architecture Composite & Routage Intelligent (`PipelineRouter`)**
   * Support natif des modèles textuels et multimodaux (**Qwen2.5-VL, LLaVA, Moondream2, Gemma 3, Mistral, DeepSeek R1, Llama 3.2**).
   * **Swap VRAM Dynamique (`SwapOrchestratorWorker`)** : Permet aux cartes graphiques modestes (4 à 8 Go de VRAM) de combiner OCR rapide (`Tesseract`), analyse structurelle par VLM dédié et synthèse par LLM principal en déchargeant/rechargeant dynamiquement les modèles en mémoire vidéo.
2. **Mémoire Épisodique à Double Étage (Short-Term & RAG Long-Term)**
   * **Court-terme** : Fenêtre glissante stricte (10 messages) avec hydratation *Just-In-Time* (JIT) des images en Base64 pour éviter toute saturation du contexte (`n_ctx`).
   * **Long-terme (`MemoryManager`)** : Indexation vectorielle locale sous **ChromaDB** (`paraphrase-multilingual-MiniLM-L12-v2`) isolée par identifiant de session unique (`UUID4`) avec protection anti-empoisonnement.
3. **Forge Neurale : SFT LoRA & Fusion Mathématique (*Model Soup*)**
   * **Étape 1 (SFT)** : Entraînement supervisé d'adaptateurs légers (`.safetensors`) à partir de datasets `.jsonl` via Hugging Face `PEFT` / `TRL`.
   * **Étape 2 (Model Soup)** : Interpolation arithmétique en précision `float32` $\rightarrow$ `float16` de plusieurs adaptateurs LoRA (`SoupManager`) permettant de combiner différentes compétences sans coût supplémentaire en VRAM lors de l'inférence.
4. **Analyse Documentaire Hybride (`document_reader`)**
   * Extraction native multi-format (`.pdf`, `.docx`, `.txt`, `.md`, `.json`, `.jsonl`, `.csv`).
   * Bascule automatique entre injection directe et **résumé Map-Reduce par chunks** hors-historique pour les documents volumineux.

---

## 📂 Architecture du Dépôt

```text
Virtual-IA-projet/
├── app.py                        # Point d'entrée + Hook global anti-crash (excepthook)
├── requirements.txt              # Dépendances logicielles verrouillées par domaine
├── core/                         # Backend pur (Zéro dépendance PyQt — Prêt pour API/WebUI)
│   ├── node.py                   # Classe abstraite BaseNode (Contrats & Gardes de sécurité)
│   ├── types.py                  # Enveloppe universelle DataPacket (Traçabilité UUID)
│   ├── router.py                 # Aiguilleurs PipelineRouter & DocumentRouter (Fast/Heavy Track)
│   ├── soup_manager.py           # Moteur mathématique d'interpolation de tenseurs (.safetensors)
│   ├── memory_manager.py         # RAG épisodique vectoriel sous ChromaDB
│   ├── document_reader.py        # Extracteur PDF/DOCX/OCR & Résumeur Map-Reduce sans état
│   ├── model_manager.py          # Client API Hugging Face & Téléchargeur HTTP avec reprise (Range)
│   ├── session_manager.py        # Persistance JSON (Auto-Save de sessions, Config, Token HF)
│   ├── logger.py                 # Logger Singleton dual (Console INFO + Fichier DEBUG)
│   ├── nodes/
│   │   ├── llm_node.py           # Moteur d'inférence GGUF, LoRA, Auto-Tuning & Hydratation JIT
│   │   ├── vision_node.py        # Nœud VLM dédié (LLaVA / Qwen-VL / Moondream)
│   │   └── trainer_node.py       # Orchestrateur d'entraînement SFT et de fusion Model Soup
│   └── utils/
│       ├── hardware.py           # Découverte matérielle (CPU, RAM, GPU CUDA/ROCm) & Profilage
│       └── file_utils.py         # Validation d'accès et création de symlinks sécurisés
├── gui/                          # Interface Graphique Desktop (PyQt6 — Thème Apple Sonoma)
│   ├── main_window.py            # Fenêtre principale, Sidebar animée & Monitoring temps réel
│   ├── workers.py                # QThreads asynchrones (Stream, Vision, Training, SwapOrchestrator)
│   ├── download_dialog.py        # Sélecteur de quantification GGUF & Téléchargeur style Steam
│   ├── widgets.py                # Composants réutilisables (FilePickerRow)
│   ├── system_monitor.py         # Sonde périodique CPU / RAM / GPU (nvidia-smi)
│   ├── style.py                  # Feuille de style globale QSS
│   └── tabs/
│       ├── chat_tab.py           # Interface Chat multimodale avec Auto-Save de session
│       ├── models_tab.py         # Boutique Hugging Face avec autocomplétion
│       ├── training_tab.py       # Studio d'entraînement SFT & Model Soup
│       ├── settings_tab.py       # Configuration matérielle, sécurité VRAM & chargement
│       └── vision_tab.py         # Laboratoire de test visuel isolé
├── storage/                      # Données locales (Ignorées par Git)
│   ├── data/                     # Datasets d'entraînement (.jsonl)
│   ├── logs/                     # Journaux d'exécution horodatés (.log)
│   ├── memory/                   # Base vectorielle persistante ChromaDB
│   ├── sessions/                 # Historique des conversations & configuration JSON
│   └── models/
│       ├── gguf/                 # Modèles quantifiés pour l'inférence (.gguf & .mmproj)
│       ├── base_hf/              # Modèles complets non-compressés pour le SFT
│       ├── adapters/             # Adaptateurs LoRA unitaires (.safetensors)
│       └── soups/                # Adaptateurs fusionnés par Model Soup (.safetensors)
└── tests/                        # Suite de tests unitaires isolés (Pytest)
```

---

## 🚀 Installation & Configuration

### 1. Prérequis Système
* **OS** : Linux (Ubuntu/Debian recommandé), macOS ou Windows.
* **Python** : `3.10` à `3.12`.
* **Outils système (Optionnel pour l'OCR rapide)** :
  ```bash
  sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng
  ```

### 2. Environnement Virtuel & Dépendances
```bash
git clone <url-du-depot>
cd Virtual-IA-projet
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Activation de l'Accélération GPU (NVIDIA CUDA)
Pour compiler `llama-cpp-python` avec le support matériel CUDA :
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 4. Lancement de l'Application & des Tests
```bash
# Lancer l'interface graphique Neural Forge
python app.py

# Exécuter la suite de tests unitaires avec Pytest
pytest -v
```

---

## 📐 Documentation d'Ingénierie & Gestion de Projet

Bien que développé en autonomie complète (Lead Developer unique), **Neural Forge** est structuré selon les standards d'ingénierie logicielle industrielle (**PBS, WBS, OBS**) afin de garantir sa maintenabilité, sa testabilité et sa future migration vers une interface Web moderne.

### 1. PBS — Product Breakdown Structure (Découpage Produit)

Le système est décomposé en **5 sous-produits indépendants** :

```text
[P0] Station IA Locale Neural Forge
 ├── [P1] Noyau d'Exécution & Transport (Core Pipeline)
 │    ├── P1.1 : Protocole DataPacket (UUID, Timestamp, Trace log)
 │    ├── P1.2 : Contrat abstrait BaseNode (Validation de types, Gardes d'état)
 │    └── P1.3 : Sondes matérielles (HardwareProfile) & Sécurisation OS (file_utils)
 ├── [P2] Moteurs d'Inférence & Orchestration Multimodale
 │    ├── P2.1 : LLMNode (GGUF, Auto-tuning analytique, Hydratation image JIT)
 │    ├── P2.2 : VisionNode (Handlers Qwen2.5-VL, LLaVA, Moondream + Downsampling PIL)
 │    └── P2.3 : PipelineRouter & SwapOrchestrator (Swap VRAM dynamique OCR + VLM + LLM)
 ├── [P3] Sous-Système Cognitif (Mémoire & Documents)
 │    ├── P3.1 : MemoryManager (ChromaDB, Embeddings multilingues, Isolation par session)
 │    ├── P3.2 : DocumentReader (Extracteurs PDF/DOCX/JSONL + Résumé Map-Reduce sans état)
 │    └── P3.3 : SessionManager (Persistance JSON & Auto-Save incrémental)
 ├── [P4] Forge d'Apprentissage (Training & Model Soup)
 │    ├── P4.1 : TrainerNode SFT (Fine-tuning LoRA via PyTorch/PEFT/TRL)
 │    └── P4.2 : SoupManager (Interpolation mathématique de tenseurs .safetensors)
 └── [P5] Interface Utilisateur & Asynchronisme (GUI Layer)
      ├── P5.1 : Pool de QThreads (StreamWorker, TrainingWorker, DownloadWorker)
      └── P5.2 : Vues modulaires (Chat, Bibliothèque HF, Entraînement, Paramètres)
```

---

### 2. OBS — Organization Breakdown Structure (Rôles & Responsabilités)

Dans le cadre d'un développement par un ingénieur unique, l'architecture sépare strictement les **4 casquettes métiers (Rôles)** pour éviter tout couplage fort dans le code :

| Code Rôle | Casquette Métier | Périmètre de responsabilité dans le dépôt |
| :--- | :--- | :--- |
| **ARCH-SYS** | **Architecte Système & MLOps** | Conception de `BaseNode`, `DataPacket`, profilage matériel (`hardware.py`), gestion de la VRAM et sécurisation des threads (`workers.py`). |
| **ENG-AI** | **Ingénieur IA (NLP / Vision / SFT)** | Intégration `llama-cpp-python`, prompt engineering anti-hallucination, pipeline SFT PyTorch/PEFT (`trainer_node.py`) et algorithme `SoupManager`. |
| **ENG-DATA** | **Ingénieur Données & RAG** | Gestion de la base vectorielle ChromaDB (`memory_manager.py`), pipeline d'extraction documentaire (`document_reader.py`) et persistance (`session_manager.py`). |
| **ENG-UI / QA** | **Ingénieur Frontend & Qualité (QA)** | Conception des onglets PyQt6 (`gui/`), ergonomie UX, prévention des freezes d'interface et maintien de la suite de tests `pytest` (`tests/`). |

---

### 3. WBS — Work Breakdown Structure (Lotissement des Travaux)

| Lot (WP) | Tâches Techniques Réalisées | Rôle (OBS) | Statut |
| :--- | :--- | :--- | :---: |
| **WP1 : Socle & Architecture** | Création du pattern `BaseNode` / `DataPacket`, logger singleton dual, détection automatique CPU/RAM/VRAM et calcul adaptatif des couches GPU (`n_gpu_layers`). | `ARCH-SYS` | ✅ Terminé |
| **WP2 : Inférence & Multimodalité** | Intégration GGUF en streaming, support natif Qwen2.5-VL / LLaVA / Moondream, création du `PipelineRouter` et du `SwapOrchestratorWorker` (OCR + VLM + LLM). | `ENG-AI` | ✅ Terminé |
| **WP3 : RAG & Ingestion Documentaire** | Intégration ChromaDB avec isolation par `UUID` de session, protection contre les réponses vides, lecture PDF/DOCX et résumé automatique Map-Reduce hors-historique. | `ENG-DATA` | ✅ Terminé |
| **WP4 : Studio SFT & Model Soup** | Arborescence dédiée (`base_hf`, `adapters`, `soups`), catalogue de modèles SFT, arrêt coopératif et fusion mathématique `SoupManager` en `float32`/`float16`. | `ENG-AI` | ✅ Terminé |
| **WP5 : Stabilisation UI & QA** | Auto-sauvegarde transparente des conversations, prévention des débordements 32-bits (> 2 Go) au téléchargement, hook global anti-crash et suite de 10 fichiers `pytest`. | `ENG-UI / QA` | ✅ Terminé |
| **WP6 : Migration UI Moderne (Roadmap)** | Découplage complet de `core/` permettant le branchement d'une interface Web moderne (FastAPI / React ou PyWebView) en remplacement de la couche PyQt6. | `ENG-UI / QA` | 🔜 Planifié |

---

### 4. Matrice des Exigences & Solutions Techniques Adoptées

| ID | Exigence / Problématique Rencontrée | Solution d'Ingénierie Implémentée | Fichier(s) Cible(s) |
| :--- | :--- | :--- | :--- |
| **REQ-01** | **Exécution sur matériel limité (4–8 Go VRAM)** sans crash *Out-Of-Memory* lors de l'analyse d'images. | Calcul dynamique du ratio de couches GPU (`_build_engine_kwargs`) + **Swap VRAM séquentiel** : déchargement du LLM $\rightarrow$ chargement VLM $\rightarrow$ analyse $\rightarrow$ déchargement VLM $\rightarrow$ rechargement LLM. | `llm_node.py`, `workers.py`, `settings_tab.py` |
| **REQ-02** | **Incompatibilité du format `.gguf` avec l'entraînement PyTorch** tout en gardant une UX intuitive. | Séparation claire : catalogue de poids complets Hugging Face (`base_hf/`) pour l'entraînement SFT $\rightarrow$ génération d'adaptateurs `.safetensors` (`adapters/` & `soups/`) greffés à la volée sur le `.gguf` via `lora_path`. | `training_tab.py`, `trainer_node.py`, `llm_node.py` |
| **REQ-03** | **Pollution du contexte et de ChromaDB** entre deux discussions différentes ou lors d'un échec de génération. | Génération d'un `session_<uuid>` unique à chaque nouvelle discussion, filtrage strict `where={"session_id": ...}` et rejet automatique des réponses vides (`len < 2`). | `llm_node.py`, `memory_manager.py`, `chat_tab.py` |
| **REQ-04** | **Conflit de templates multimodaux** (ex: Qwen2.5-VL muet à cause du prompt Vicuna forcé par `Llava15ChatHandler`). | Détection dynamique de l'architecture du modèle chargé et aiguillage vers `Qwen25VLChatHandler` (`chatml`), `MoondreamChatHandler` ou `Llava15ChatHandler` + retrait de `"Assistant:"` des `stop_words`. | `llm_node.py`, `vision_node.py` |
| **REQ-05** | **Réduction des hallucinations** sur les modèles locaux de petite taille (1.5B à 4B paramètres). | Injection d'un `DEFAULT_SYSTEM_PROMPT` d'ancrage factuel strict + abaissement automatique de la température à `0.1` dès qu'un mot-clé analytique (`code`, `sql`, `mcd`, `exercice`) est détecté. | `llm_node.py`, `vision_node.py` |
| **REQ-06** | **Stabilité C++/Qt sous Linux** (prévention des crashs `Aborted (core dumped)` sur les threads et gros téléchargements). | Remplacement de `QThread.terminate()` par un drapeau coopératif `request_abort()`, signaux `pyqtSignal(object, object)` 64-bits pour les fichiers $> 2\text{ Go}$, et `sys.excepthook` global. | `app.py`, `download_dialog.py`, `training_tab.py` |
