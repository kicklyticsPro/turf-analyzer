"""
v5 - XGBoost-like : GBM avec régularisation L2, subsampling, early stopping.
Implémentation NumPy pur, optimisée vectoriellement.
"""
import math
import random


class RegularizedTree:
    """
    Arbre régressif avec régularisation L2 (lambda) sur les feuilles.
    Score d'un split : gain = (G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)) / 2 - γ
    où G = somme des gradients, H = somme des hessiens.
    """
    def __init__(self, max_depth=4, min_samples=20, lambda_reg=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.tree = None

    def fit(self, X, gradients, hessians):
        import numpy as np
        self.tree = self._build(
            np.asarray(X, dtype=float),
            np.asarray(gradients, dtype=float),
            np.asarray(hessians, dtype=float),
            depth=0
        )

    def _leaf_value(self, g, h):
        """Valeur optimale d'une feuille : -G / (H + λ)"""
        G = g.sum()
        H = h.sum()
        return -G / (H + self.lambda_reg)

    def _split_gain(self, g_l, h_l, g_r, h_r):
        """Gain d'un split."""
        G_L, H_L = g_l.sum(), h_l.sum()
        G_R, H_R = g_r.sum(), h_r.sum()
        gain = 0.5 * (
            G_L**2 / (H_L + self.lambda_reg) +
            G_R**2 / (H_R + self.lambda_reg) -
            (G_L + G_R)**2 / (H_L + H_R + self.lambda_reg)
        ) - self.gamma
        return gain

    def _build(self, X, g, h, depth):
        import numpy as np
        n = len(g)
        if n < self.min_samples or depth >= self.max_depth:
            return {"leaf": True, "value": float(self._leaf_value(g, h))}

        best_split = None
        best_gain = 0
        n_features = X.shape[1]

        for f in range(n_features):
            vals = X[:, f]
            quantiles = np.percentile(vals, [10, 25, 40, 55, 70, 85])
            for q in np.unique(quantiles):
                lm = vals <= q
                rm = ~lm
                if lm.sum() < self.min_samples or rm.sum() < self.min_samples:
                    continue
                gain = self._split_gain(g[lm], h[lm], g[rm], h[rm])
                if gain > best_gain:
                    best_gain = gain
                    best_split = (f, float(q), lm, rm)

        if best_split is None:
            return {"leaf": True, "value": float(self._leaf_value(g, h))}

        f, q, lm, rm = best_split
        return {"leaf": False, "feature": f, "threshold": q,
                "left": self._build(X[lm], g[lm], h[lm], depth + 1),
                "right": self._build(X[rm], g[rm], h[rm], depth + 1)}

    def predict_one(self, x):
        node = self.tree
        while not node["leaf"]:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]

    def predict(self, X):
        import numpy as np
        return np.array([self.predict_one(x) for x in X])


class XGBoostLike:
    """
    GBM régularisé style XGBoost :
      - L2 régularisation sur poids des feuilles
      - γ (gamma) : pénalité par feuille (prune les splits faibles)
      - Subsampling : 50% des échantillons par arbre
      - Early stopping si validation loss n'améliore plus
    """
    def __init__(self, n_trees=100, max_depth=4, learning_rate=0.1,
                 lambda_reg=1.0, gamma=0.1, subsample=0.5,
                 early_stopping=10, val_split=0.15):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = learning_rate
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.subsample = subsample
        self.early_stopping = early_stopping
        self.val_split = val_split
        self.trees = []
        self.init_pred = 0.0
        self.best_n_trees = 0

    def fit(self, X, y, seed=42):
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        # Train/val split
        idx = np.arange(n)
        np.random.shuffle(idx)
        n_val = int(n * self.val_split)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Init : log-odds de la moyenne
        mean_y = max(min(y_train.mean(), 0.99), 0.01)
        self.init_pred = math.log(mean_y / (1 - mean_y))

        F_train = np.full(len(y_train), self.init_pred)
        F_val = np.full(len(y_val), self.init_pred)

        best_val_loss = float('inf')
        no_improve = 0
        best_idx = 0

        for i in range(self.n_trees):
            # Subsampling
            n_sub = max(1, int(len(y_train) * self.subsample))
            sub_idx = np.random.choice(len(y_train), n_sub, replace=False)
            X_sub = X_train[sub_idx]
            y_sub = y_train[sub_idx]
            F_sub = F_train[sub_idx]

            # Gradients & Hessians (log-loss)
            p_sub = 1 / (1 + np.exp(-F_sub))
            g = p_sub - y_sub          # gradient
            h = p_sub * (1 - p_sub)     # hessian

            tree = RegularizedTree(max_depth=self.max_depth,
                                    min_samples=20,
                                    lambda_reg=self.lambda_reg,
                                    gamma=self.gamma)
            tree.fit(X_sub, g, h)
            self.trees.append(tree)

            # Update F
            F_train = F_train + self.lr * tree.predict(X_train)
            F_val = F_val + self.lr * tree.predict(X_val)

            # Validation loss (log-loss)
            p_val = 1 / (1 + np.exp(-F_val))
            p_val = np.clip(p_val, 1e-7, 1 - 1e-7)
            val_loss = -(y_val * np.log(p_val) + (1 - y_val) * np.log(1 - p_val)).mean()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_idx = i
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.early_stopping:
                # Tronquer aux meilleurs arbres
                self.trees = self.trees[:best_idx + 1]
                break

        self.best_n_trees = len(self.trees)
        return self

    def predict_one(self, x):
        F = self.init_pred
        for tree in self.trees:
            F += self.lr * tree.predict_one(x)
        return 1 / (1 + math.exp(-F))

    def predict_proba(self, X):
        import numpy as np
        return np.array([self.predict_one(x) for x in X])

    def to_dict(self):
        return {"type": "xgb", "n_trees": self.n_trees, "max_depth": self.max_depth,
                "lr": self.lr, "lambda_reg": self.lambda_reg, "gamma": self.gamma,
                "subsample": self.subsample, "init_pred": self.init_pred,
                "best_n_trees": self.best_n_trees,
                "trees": [t.tree for t in self.trees]}

    @classmethod
    def from_dict(cls, d):
        m = cls(n_trees=d["n_trees"], max_depth=d["max_depth"],
                learning_rate=d["lr"], lambda_reg=d["lambda_reg"],
                gamma=d["gamma"], subsample=d["subsample"])
        m.init_pred = d["init_pred"]
        m.best_n_trees = d.get("best_n_trees", len(d["trees"]))
        for tree_data in d["trees"]:
            t = RegularizedTree(max_depth=d["max_depth"],
                                 lambda_reg=d["lambda_reg"],
                                 gamma=d["gamma"])
            t.tree = tree_data
            m.trees.append(t)
        return m
