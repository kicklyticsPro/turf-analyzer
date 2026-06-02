"""
v6 - AutoML : recherche d'hyperparamètres + cross-validation + métriques avancées.
"""
import math
import random
from itertools import product


# ============================================================
#  Métriques
# ============================================================
def log_loss(y_true, y_pred, eps=1e-15):
    """Log-loss binaire. Plus c'est bas, mieux c'est."""
    import numpy as np
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()


def brier_score(y_true, y_pred):
    """Brier score = mean((y_true - y_pred)^2). Calibration globale."""
    import numpy as np
    return ((np.asarray(y_true) - np.asarray(y_pred)) ** 2).mean()


def roc_auc(y_true, y_pred):
    """AUC approximé par tri (Mann-Whitney U statistic)."""
    pairs = sorted(zip(y_pred, y_true))
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Somme des rangs des positifs
    sum_ranks_pos = 0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            sum_ranks_pos += rank
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def calibration_curve(y_true, y_pred, n_bins=10):
    """Retourne (proba_predite_moy, taux_observe) par bin. Pour graph de calibration."""
    import numpy as np
    pred = np.asarray(y_pred)
    true = np.asarray(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        mask = (pred >= bins[i]) & (pred < bins[i+1])
        if mask.sum() > 0:
            result.append({
                "bin_min": float(bins[i]),
                "bin_max": float(bins[i+1]),
                "n": int(mask.sum()),
                "avg_pred": float(pred[mask].mean()),
                "actual_rate": float(true[mask].mean()),
            })
    return result


def evaluate_model(model, X, y):
    """Évalue un modèle sur (X, y). Retourne un dict de métriques."""
    import numpy as np
    if hasattr(model, "predict_proba_batch"):
        preds = model.predict_proba_batch(X)
    else:
        preds = np.array([model.predict_one(x) for x in X])
    return {
        "log_loss": float(log_loss(y, preds)),
        "auc": float(roc_auc(y, preds)),
        "brier": float(brier_score(y, preds)),
        "calibration": calibration_curve(y, preds),
    }


# ============================================================
#  Cross-Validation temporelle
# ============================================================
def temporal_kfold(X, y, dates, n_folds=5):
    """
    K-fold temporel : on coupe l'historique en N parties chronologiques.
    fold k : train sur tout ce qui précède, test sur le fold k.
    Évite la fuite future→passé qu'un K-fold random aurait.
    """
    import numpy as np
    n = len(y)
    # Trie par date
    order = sorted(range(n), key=lambda i: dates[i])
    fold_size = n // (n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_end = k * fold_size
        test_start = train_end
        test_end = min((k + 1) * fold_size, n)
        train_idx = [order[i] for i in range(train_end)]
        test_idx = [order[i] for i in range(test_start, test_end)]
        if len(train_idx) >= 100 and len(test_idx) >= 30:
            folds.append((train_idx, test_idx))
    return folds


def cross_validate(model_factory, X, y, dates=None, n_folds=3):
    """
    Évalue un modèle par CV. model_factory() doit retourner une nouvelle instance.
    Retourne moyenne et std des métriques.
    """
    import numpy as np
    if dates:
        folds = temporal_kfold(X, y, dates, n_folds=n_folds)
    else:
        # Random folds
        n = len(y)
        idx = list(range(n))
        random.shuffle(idx)
        fold_size = n // n_folds
        folds = []
        for k in range(n_folds):
            test_idx = idx[k * fold_size:(k + 1) * fold_size]
            train_idx = [i for i in idx if i not in set(test_idx)]
            folds.append((train_idx, test_idx))

    metrics_list = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_tr = [X[i] for i in train_idx]
        y_tr = [y[i] for i in train_idx]
        X_te = [X[i] for i in test_idx]
        y_te = [y[i] for i in test_idx]
        model = model_factory()
        model.fit(X_tr, y_tr)
        m = evaluate_model(model, X_te, y_te)
        metrics_list.append(m)

    # Moyennes
    return {
        "n_folds": len(metrics_list),
        "log_loss_mean": np.mean([m["log_loss"] for m in metrics_list]),
        "log_loss_std": np.std([m["log_loss"] for m in metrics_list]),
        "auc_mean": np.mean([m["auc"] for m in metrics_list]),
        "auc_std": np.std([m["auc"] for m in metrics_list]),
        "brier_mean": np.mean([m["brier"] for m in metrics_list]),
        "folds": metrics_list,
    }


# ============================================================
#  Random search d'hyperparamètres
# ============================================================
def random_search(model_class, param_grid, X, y, n_iter=10, n_folds=3, seed=42):
    """
    Cherche les meilleurs hyperparamètres par random search + CV.
    param_grid : {param_name: [possible_values]}
    Retourne {best_params, best_score, all_results}
    """
    random.seed(seed)
    keys = list(param_grid.keys())
    all_combos = list(product(*[param_grid[k] for k in keys]))
    if n_iter >= len(all_combos):
        sampled = all_combos  # grid search
    else:
        sampled = random.sample(all_combos, n_iter)

    results = []
    best_score = float('inf')
    best_params = None

    for combo in sampled:
        params = dict(zip(keys, combo))
        try:
            def factory():
                return model_class(**params)
            cv = cross_validate(factory, X, y, n_folds=n_folds)
            score = cv["log_loss_mean"]
            results.append({"params": params, "score": score,
                          "auc": cv["auc_mean"]})
            if score < best_score:
                best_score = score
                best_params = params
        except Exception as e:
            results.append({"params": params, "error": str(e)})

    results.sort(key=lambda r: r.get("score", float('inf')))
    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": results[:10],  # top 10
    }


# ============================================================
#  Stacking ensemble (méta-modèle)
# ============================================================
class StackingEnsemble:
    """
    Combine plusieurs modèles via un méta-modèle (logistic).
    Entrée du méta : prédictions des modèles de base.
    """
    def __init__(self, base_models, meta_lr=0.05, meta_epochs=300):
        self.base_models = base_models  # liste de modèles déjà entraînés
        self.meta_weights = None
        self.meta_bias = 0
        self.meta_lr = meta_lr
        self.meta_epochs = meta_epochs

    def fit_meta(self, X, y):
        """
        Entraîne le méta-modèle sur les prédictions des bases.
        IMPORTANT : utiliser des données NON vues pendant l'entraînement des bases (out-of-fold)
        sinon le méta sur-apprend.
        """
        import numpy as np
        # Récupère les prédictions de chaque base
        preds = []
        for m in self.base_models:
            if hasattr(m, "predict_proba_batch"):
                preds.append(m.predict_proba_batch(X))
            else:
                preds.append([m.predict_one(x) for x in X])
        M = np.array(preds).T  # shape (n_samples, n_models)
        y = np.asarray(y, dtype=float)
        n, d = M.shape

        # Logistic regression simple par gradient descent
        self.meta_weights = np.ones(d) / d  # init équiprobable
        self.meta_bias = 0.0
        for _ in range(self.meta_epochs):
            z = M @ self.meta_weights + self.meta_bias
            p = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            err = p - y
            grad_w = M.T @ err / n
            grad_b = err.mean()
            self.meta_weights -= self.meta_lr * grad_w
            self.meta_bias -= self.meta_lr * grad_b
        return self

    def predict_one(self, x):
        if self.meta_weights is None:
            # Pas entraîné : moyenne simple
            preds = [m.predict_one(x) for m in self.base_models]
            return sum(preds) / len(preds)
        preds = [m.predict_one(x) for m in self.base_models]
        z = sum(p * w for p, w in zip(preds, self.meta_weights)) + self.meta_bias
        return 1 / (1 + math.exp(-max(-500, min(500, z))))

    def get_model_weights(self):
        """Retourne le poids relatif de chaque modèle base après normalisation."""
        if self.meta_weights is None:
            return None
        import numpy as np
        # Softmax des poids pour lisibilité
        w = np.abs(self.meta_weights)
        total = w.sum() or 1
        return (w / total).tolist()


# ============================================================
#  Explainability (SHAP-like simple)
# ============================================================
def feature_importance_perturbation(model, x, feature_names, n_perturb=10):
    """
    Calcule l'importance de chaque feature pour CE cheval par perturbation.
    Pour chaque feature, on la remplace par sa valeur moyenne et on regarde
    l'impact sur la prédiction.
    """
    import numpy as np
    base_pred = model.predict_one(x)
    importances = []
    for i, fname in enumerate(feature_names):
        # Crée x' avec feature i réduit à 50 (neutre)
        x_perturbed = list(x)
        x_perturbed[i] = 50  # valeur neutre
        new_pred = model.predict_one(x_perturbed)
        delta = base_pred - new_pred
        importances.append({
            "feature": fname,
            "value": round(x[i], 1),
            "impact": round(delta * 100, 2),  # en points de %
        })
    # Trie par impact absolu
    importances.sort(key=lambda d: -abs(d["impact"]))
    return importances[:10]  # top 10
