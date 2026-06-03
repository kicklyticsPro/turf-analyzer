"""
scripts/daily_update.py — Mise à jour automatique quotidienne (Stats + Modèle).
Utilisation : python scripts/daily_update.py
"""
import os
import sys
from datetime import datetime, timedelta

# Ajouter le dossier racine au path pour importer app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app

def daily_update():
    print(f"--- MISE À JOUR QUOTIDIENNE : {datetime.now().strftime('%Y-%m-%d %H:%M')} ---")
    
    # 1. Calcul des statistiques sur les 180 derniers jours (incluant hier)
    print("Étape 1 : Recalcul des statistiques (Elo, Forme, Pedigree)...")
    try:
        # compute_all_stats sauvegarde automatiquement les fichiers .pkl dans le cache
        app.compute_all_stats(max_days=180, use_cache=False)
        print("✅ Statistiques mises à jour avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du calcul des stats : {e}")
        return

    # 2. Ré-entraînement du modèle IA (XGBoost par défaut)
    print("\nÉtape 2 : Ré-entraînement du modèle Machine Learning (30 derniers jours)...")
    try:
        # On entraîne sur les 30 derniers jours pour capturer les tendances récentes
        info = app.train_ml_model(days_back=30, model_type="xgb", xgb_n_trees=100)
        if info:
            print(f"✅ Modèle ré-entraîné : LogLoss={info['log_loss']:.4f}, AUC={info['auc']:.4f}")
        else:
            print("⚠️ L'entraînement a échoué (pas assez de données ?).")
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement du modèle : {e}")
        return

    print("\n--- MISE À JOUR TERMINÉE AVEC SUCCÈS ---")

if __name__ == "__main__":
    daily_update()
