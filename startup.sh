#!/bin/bash
echo "🚀 Démarrage de Turf Analyzer..."
echo "Version Python: $(python --version)"
echo "Installation des dépendances..."
pip install -r requirements.txt --quiet
echo "Lancement du serveur..."
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --log-level debug
