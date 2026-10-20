# ============================================================================
# Makefile — Automatisation de l'environnement Neural Forge
# ============================================================================

PYTHON ?= python3
VENV_DIR := venv
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
VENV_PYTEST := $(VENV_DIR)/bin/pytest

.PHONY: help venv install install-cuda install-cpu run test clean purge-cache

help:
	@echo "=================================================================="
	@echo " 🧠 Neural Forge — Commandes disponibles :"
	@echo "=================================================================="
	@echo "  make install       : Crée le venv et lance l'installateur intelligent (Auto GPU/CPU)"
	@echo "  make install-cuda  : Force l'installation avec compilation GPU NVIDIA (CUDA)"
	@echo "  make install-cpu   : Force l'installation en mode CPU uniquement"
	@echo "  make run           : Lance l'application graphique Neural Forge (app.py)"
	@echo "  make test          : Exécute la suite complète de tests unitaires (pytest)"
	@echo "  make clean         : Nettoie les fichiers temporaires et caches (__pycache__)"
	@echo "  make purge-cache   : Supprime les modèles d'entraînement lourds (base_hf/)"
	@echo "=================================================================="

# Crée l'environnement virtuel s'il n'existe pas encore
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "→ Création de l'environnement virtuel ($(VENV_DIR))..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
		$(VENV_PIP) install --upgrade pip; \
	fi

install: venv
	@$(VENV_PY) install.py

install-cuda: venv
	@$(VENV_PY) install.py --cuda

install-cpu: venv
	@$(VENV_PY) install.py --cpu

run: venv
	@$(VENV_PY) app.py

test: venv
	@$(VENV_PYTEST) -v

clean:
	@echo "→ Nettoyage des caches Python et Pytest..."
	@find . -maxdepth 3 -name "__pycache__" -exec rm -rf {} +
	@rm -rf .pytest_cache tests/.pytest_cache .coverage htmlcov
	@echo "✓ Dépôt nettoyé."

purge-cache:
	@echo "→ Purge du cache des modèles d'entraînement Hugging Face (base_hf)..."
	@rm -rf storage/models/base_hf/*
	@echo "✓ Espace disque libéré."