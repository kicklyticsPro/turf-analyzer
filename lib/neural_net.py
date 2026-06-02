"""
v6 - Multi-Layer Perceptron en NumPy pur.
Architecture : Input -> Hidden1 (ReLU) -> Hidden2 (ReLU) -> Output (Sigmoid)
Optimizer : Adam, avec dropout pour régulariser.
"""
import math
import random


class MLPClassifier:
    def __init__(self, hidden_sizes=(32, 16), learning_rate=0.001,
                 epochs=200, batch_size=64, dropout=0.2,
                 lambda_reg=0.0001, early_stopping=15, val_split=0.15):
        self.hidden_sizes = hidden_sizes
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.early_stopping = early_stopping
        self.val_split = val_split

        # Paramètres appris
        self.weights = []  # liste de matrices
        self.biases = []   # liste de vecteurs
        self.mu = None     # normalisation
        self.sigma = None
        self.best_epoch = 0
        self.best_val_loss = float('inf')

    def _init_weights(self, n_features):
        """He initialization pour ReLU."""
        import numpy as np
        sizes = [n_features] + list(self.hidden_sizes) + [1]
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            # He init : sqrt(2/n_in)
            std = math.sqrt(2.0 / sizes[i])
            w = np.random.randn(sizes[i], sizes[i+1]) * std
            b = np.zeros(sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def _relu(x):
        import numpy as np
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x):
        import numpy as np
        return (x > 0).astype(float)

    @staticmethod
    def _sigmoid(x):
        import numpy as np
        # Clip pour éviter overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _forward(self, X, training=False):
        """Forward pass. Retourne (activations, pré-activations, masks dropout)."""
        import numpy as np
        activations = [X]
        pre_acts = []
        dropout_masks = []
        a = X
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            pre_acts.append(z)
            if i < len(self.weights) - 1:
                # Couche cachée : ReLU + dropout
                a = self._relu(z)
                if training and self.dropout > 0:
                    mask = (np.random.rand(*a.shape) > self.dropout).astype(float)
                    mask /= (1 - self.dropout)  # inverted dropout
                    a = a * mask
                    dropout_masks.append(mask)
                else:
                    dropout_masks.append(None)
            else:
                # Couche sortie : sigmoid
                a = self._sigmoid(z)
                dropout_masks.append(None)
            activations.append(a)
        return activations, pre_acts, dropout_masks

    def _backward(self, activations, pre_acts, dropout_masks, y_true):
        """Backpropagation. Retourne (grad_weights, grad_biases)."""
        import numpy as np
        n = len(y_true)
        L = len(self.weights)

        grad_w = [None] * L
        grad_b = [None] * L

        # Erreur sortie (log-loss + sigmoid simplifié) : delta = (p - y)
        delta = activations[-1].flatten() - y_true
        delta = delta.reshape(-1, 1)

        for l in range(L - 1, -1, -1):
            a_prev = activations[l]
            grad_w[l] = a_prev.T @ delta / n + self.lambda_reg * self.weights[l]
            grad_b[l] = delta.mean(axis=0)
            if l > 0:
                # Propager le gradient
                delta = delta @ self.weights[l].T
                if dropout_masks[l-1] is not None:
                    delta = delta * dropout_masks[l-1]
                delta = delta * self._relu_grad(pre_acts[l-1])

        return grad_w, grad_b

    def fit(self, X, y, seed=42, verbose=False):
        import numpy as np
        np.random.seed(seed)
        random.seed(seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Normalisation
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xn = (X - self.mu) / self.sigma

        # Train/val split
        n = len(y)
        idx = np.arange(n)
        np.random.shuffle(idx)
        n_val = int(n * self.val_split)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_train, y_train = Xn[train_idx], y[train_idx]
        X_val, y_val = Xn[val_idx], y[val_idx]

        self._init_weights(Xn.shape[1])

        # Adam state
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        best_weights = None
        best_biases = None
        no_improve = 0
        self.best_val_loss = float('inf')

        for epoch in range(self.epochs):
            # Shuffle batch
            perm = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), self.batch_size):
                batch_idx = perm[i:i + self.batch_size]
                X_b, y_b = X_train[batch_idx], y_train[batch_idx]
                acts, pres, masks = self._forward(X_b, training=True)
                gw, gb = self._backward(acts, pres, masks, y_b)
                t += 1
                # Adam update
                for l in range(len(self.weights)):
                    m_w[l] = beta1 * m_w[l] + (1 - beta1) * gw[l]
                    v_w[l] = beta2 * v_w[l] + (1 - beta2) * (gw[l] ** 2)
                    m_b[l] = beta1 * m_b[l] + (1 - beta1) * gb[l]
                    v_b[l] = beta2 * v_b[l] + (1 - beta2) * (gb[l] ** 2)
                    m_w_hat = m_w[l] / (1 - beta1 ** t)
                    v_w_hat = v_w[l] / (1 - beta2 ** t)
                    m_b_hat = m_b[l] / (1 - beta1 ** t)
                    v_b_hat = v_b[l] / (1 - beta2 ** t)
                    self.weights[l] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
                    self.biases[l] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

            # Validation
            acts_val, _, _ = self._forward(X_val, training=False)
            p_val = acts_val[-1].flatten()
            p_val = np.clip(p_val, 1e-7, 1 - 1e-7)
            val_loss = -(y_val * np.log(p_val) + (1 - y_val) * np.log(1 - p_val)).mean()

            if val_loss < self.best_val_loss - 1e-5:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                no_improve = 0
            else:
                no_improve += 1

            if verbose and epoch % 20 == 0:
                print(f"  Epoch {epoch}: val_loss={val_loss:.4f}")

            if no_improve >= self.early_stopping:
                break

        # Restore best weights
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        return self

    def predict_proba_batch(self, X):
        import numpy as np
        X = np.asarray(X, dtype=float)
        Xn = (X - self.mu) / self.sigma
        acts, _, _ = self._forward(Xn, training=False)
        return acts[-1].flatten()

    def predict_one(self, x):
        return float(self.predict_proba_batch([x])[0])

    def to_dict(self):
        return {
            "type": "mlp",
            "hidden_sizes": list(self.hidden_sizes),
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "mu": self.mu.tolist() if self.mu is not None else None,
            "sigma": self.sigma.tolist() if self.sigma is not None else None,
            "best_epoch": self.best_epoch,
            "best_val_loss": float(self.best_val_loss),
        }

    @classmethod
    def from_dict(cls, d):
        import numpy as np
        m = cls(hidden_sizes=tuple(d["hidden_sizes"]))
        m.weights = [np.array(w) for w in d["weights"]]
        m.biases = [np.array(b) for b in d["biases"]]
        m.mu = np.array(d["mu"]) if d.get("mu") else None
        m.sigma = np.array(d["sigma"]) if d.get("sigma") else None
        m.best_epoch = d.get("best_epoch", 0)
        m.best_val_loss = d.get("best_val_loss", 0)
        return m
