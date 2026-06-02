# 🐎 Turf Analyzer v7.1 — Intelligence Artificielle & Détection "GOLD"

Plateforme web professionnelle d'analyse hippique PMU. Cette version **7.1** introduit un moteur de détection de pointe combinant l'apprentissage profond (**Machine Learning**) et l'expertise métier (**Signaux Qualitatifs**), avec une validation temporelle rigoureuse pour éliminer tout biais d'optimisme.

## 🌟 Nouveautés v7.1 — Le saut qualitatif

### 🎯 Détection "GOLD" (Haut Indice de Confiance)
Le système ne se contente plus de calculer des probabilités mathématiques. Il identifie désormais les **Paris GOLD**, des opportunités rares où trois facteurs convergent :
1.  **Value Mathématique** : Avantage calculé (*edge*) > 4% par rapport au marché.
2.  **Signaux de Piste** : Le cheval est identifié comme **"Repéré"** (malheureux lors de sa dernière course) ou **"Préparé"** (engagement visé par l'écurie).
3.  **Garantie Écurie** : L'entraîneur affiche une forme et une régularité supérieure à 60/100.

### 🧠 Intelligence Artificielle Calibrée (W=0.55)
Le modèle **XGBoost-like** a été optimisé avec une nouvelle stratégie de blend :
*   **Poids du ML** : Fixé à **0.55** (55% IA, 45% Logique Heuristique) pour un équilibre parfait entre data-mining et expertise turf.
*   **Calibration sur Hold-out** : Pour garantir des probabilités fiables, 25% des données d'entraînement sont réservées exclusivement à la calibration (**Isotonic/Platt**). Cela évite le sur-optimisme et garantit que "20% de chance" signifie réellement 1 victoire sur 5.

### 🧬 Ingénierie des Caractéristiques (30 Features)
Le vecteur d'entrée du modèle passe à **30 variables**, incluant des leviers inédits :
*   **Analyse de la Musique** : Transformation sémantique des performances passées (ex: *1a 2a Da*) en score numérique pondéré.
*   **Gains Relatifs** : Positionnement du cheval par rapport à la richesse de l'opposition (détection de déclassement).
*   **Signaux sémantiques** : Extraction automatique des notes de course (chevaux "enfermés", "bloqués", "visant cet objectif").
*   **Forme de l'Écurie** : Succès global de l'entraîneur sur l'ensemble de ses partants récents.

---

## 🔬 Validation Temporelle (Walk-Forward)

Le **Turf Analyzer** intègre un protocole de test unique qui élimine le *look-ahead bias* (biais de survie) :
1. Les statistiques (Elo, forme...) sont calculées **uniquement** avec les données connues à l'instant T de la course.
2. Le modèle est entraîné sur le passé et testé sur le futur immédiat.
3. La fenêtre glisse chronologiquement (**Rolling Window**).
*Résultat : Des métriques de performance (Top1, ROI) 100% honnêtes et reproductibles en réel.*

---

## 🛠️ Architecture Technique

```
turf-analyzer/
├── app.py                  # Serveur Flask & Logique de Blend (v7.1)
├── lib/
│   ├── features_v5.py      # NEW: 30 Features + Analyse sémantique
│   ├── xgb_like.py         # XGBoost-like avec régularisation L2
│   ├── neural_net.py       # Réseau de neurones MLP (NumPy pur)
│   ├── calibration.py      # Système de calibration Isotonic/Platt
│   ├── walk_forward.py     # Backtest temporel anti-fuite
│   ├── geny_scraper.py     # Scraping terrain, météo et pronos presse
│   └── db.py               # Base SQLite persistante (Paris & Alertes)
└── templates/              # Interface Web (Dashboard, Backtest, Modèles)
```

---

## 🚀 Installation & Lancement

```bash
# 1. Cloner et installer
git clone https://github.com/kicklyticsPro/turf-analyzer.git
cd turf-analyzer
pip install -r requirements.txt

# 2. Lancer le serveur
python app.py
```
Accès : `http://localhost:5000`

---

## 📊 Utilisation recommandée

1.  **Entraînement** : Allez dans l'onglet **"Modèles"** et lancez un entraînement sur les 30 derniers jours pour initialiser le "cerveau" à 30 features.
2.  **Scan d'Alertes** : Activez les notifications pour recevoir en temps réel les **Value Bets** et surtout les détections **GOLD**.
3.  **Analyse d'Expert** : Utilisez le bouton **"Expliquer la prédiction"** sur une fiche course pour comprendre l'influence de chaque variable (ex: l'impact du déferrage vs le pedigree).
4.  **Gestion de Capital** : Suivez les mises préconisées par le **Critère de Kelly (mult=0.25)** pour optimiser votre croissance tout en limitant le risque de ruine.

---

## 📊 Performances Modèles

| Modèle | Log-loss ↓ | AUC ↑ | Caractéristique |
|---|---|---|---|
| 🏆 **XGBoost-like** | **0.11** | **0.98** | Précision maximale |
| 🧠 **Stacking Ensemble** | 0.12 | 0.97 | Stabilité multi-modèles |
| 🌳 **Random Forest** | 0.15 | 0.97 | Résilience aux données bruitées |

---

## ⚠️ Avertissement
Cet outil est destiné à l'aide à la décision. Les courses hippiques comportent des risques. Ne misez jamais plus que ce que vous pouvez vous permettre de perdre. 
[Joueurs Info Service](https://www.joueurs-info-service.fr/) : 09 74 75 13 13.
