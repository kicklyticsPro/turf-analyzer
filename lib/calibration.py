"""
lib/calibration.py — Méthodes de calibration de probabilités (v7.1).

La calibration corrige les probabilités brutes d'un modèle pour qu'elles
correspondent aux fréquences réellement observées (ex: parmi les chevaux
prédits à 30%, ~30% gagnent effectivement).

Deux méthodes :
  - Isotonic (PAV exact) : non-paramétrique, flexible, robuste si assez de données.
  - Platt scaling (sigmoïde logistique) : paramétrique, lisse, robuste sur peu
    de données.

⚠️ Bonne pratique : TOUJOURS ajuster la calibration sur un *hold-out* distinct
du train du modèle, sinon on sous-estime l'erreur (optimisme).
"""

import math


# ------------------------------------------------------------------
#  Isotonic regression — Pool Adjacent Violators (PAV) exact
# ------------------------------------------------------------------
def fit_isotonic_pav(predictions, actuals):
    """
    Régression isotone exacte (PAV) sur des paires (prediction, label 0/1).
    Retourne une table [(x_seuil, y_calibré), ...] croissante, utilisable par
    apply_isotonic().
    """
    pairs = sorted(zip(predictions, actuals), key=lambda t: t[0])
    if not pairs:
        return [(0.0, 0.0), (1.0, 1.0)]

    # Blocs PAV : chaque bloc = [somme_y, poids, x_moyen]
    xs = [p for p, _ in pairs]
    ys = [float(a) for _, a in pairs]

    # valeurs (moyennes) et poids par point
    values = ys[:]
    weights = [1.0] * len(ys)
    block_x = xs[:]  # x représentatif (moyenne) de chaque bloc

    i = 0
    # Algorithme PAV : fusionne les blocs adjacents qui violent la monotonie
    blocks_val = []
    blocks_w = []
    blocks_x = []
    for v, w, x in zip(values, weights, block_x):
        blocks_val.append(v)
        blocks_w.append(w)
        blocks_x.append([x * w, w])  # somme pondérée de x, poids
        # fusion tant que violation
        while len(blocks_val) >= 2 and blocks_val[-2] > blocks_val[-1]:
            v2 = blocks_val.pop(); w2 = blocks_w.pop(); sx2 = blocks_x.pop()
            v1 = blocks_val.pop(); w1 = blocks_w.pop(); sx1 = blocks_x.pop()
            nw = w1 + w2
            nv = (v1 * w1 + v2 * w2) / nw
            nsx = [sx1[0] + sx2[0], sx1[1] + sx2[1]]
            blocks_val.append(nv); blocks_w.append(nw); blocks_x.append(nsx)

    table = []
    for v, sx in zip(blocks_val, blocks_x):
        x_mean = sx[0] / sx[1] if sx[1] else 0.0
        table.append((x_mean, max(0.0, min(1.0, v))))
    # garantit bornes
    if table[0][0] > 0:
        table.insert(0, (0.0, table[0][1]))
    if table[-1][0] < 1:
        table.append((1.0, table[-1][1]))
    return table


def apply_isotonic(p, table):
    """Interpolation linéaire dans la table isotone."""
    if not table:
        return p
    if p <= table[0][0]:
        return table[0][1]
    if p >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        x0, y0 = table[i]
        x1, y1 = table[i + 1]
        if x0 <= p <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (p - x0) / (x1 - x0)
    return p


# ------------------------------------------------------------------
#  Platt scaling — sigmoïde logistique a*logit(p)+b ajustée par descente
# ------------------------------------------------------------------
def _logit(p, eps=1e-6):
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_platt(predictions, actuals, lr=0.3, epochs=400):
    """
    Platt scaling : ajuste p_cal = sigmoid(a * logit(p) + b) par descente de
    gradient sur la log-loss. Retourne (a, b).
    """
    if not predictions:
        return (1.0, 0.0)
    z = [_logit(p) for p in predictions]
    y = [float(a) for a in actuals]
    n = len(z)
    a, b = 1.0, 0.0
    for _ in range(epochs):
        ga = gb = 0.0
        for zi, yi in zip(z, y):
            pi = _sigmoid(a * zi + b)
            err = pi - yi
            ga += err * zi
            gb += err
        a -= lr * ga / n
        b -= lr * gb / n
    return (a, b)


def apply_platt(p, params):
    a, b = params
    return _sigmoid(a * _logit(p) + b)


# ------------------------------------------------------------------
#  Wrapper unifié
# ------------------------------------------------------------------
class Calibrator:
    """Calibrateur sérialisable supportant isotonic ou platt."""

    def __init__(self, method="isotonic", model=None):
        self.method = method
        self.model = model  # table (isotonic) ou (a,b) (platt)

    @classmethod
    def fit(cls, predictions, actuals, method="isotonic"):
        if method == "platt":
            return cls("platt", fit_platt(predictions, actuals))
        return cls("isotonic", fit_isotonic_pav(predictions, actuals))

    def apply(self, p):
        if self.model is None:
            return p
        if self.method == "platt":
            return apply_platt(p, self.model)
        return apply_isotonic(p, self.model)

    def to_dict(self):
        return {"method": self.method, "model": self.model}

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls("isotonic", None)
        return cls(d.get("method", "isotonic"), d.get("model"))
