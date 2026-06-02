"""
experiments/exp_calibration.py — Optimisation calibration + blend (v7.1).

Utilise les jours mis en cache par harness.py. Walk-forward simple :
on entraîne sur les N premiers jours, on teste sur le reste, en faisant
glisser. Tout en mémoire => instantané.

Compare :
  - calibration: none / isotonic(train) / isotonic(holdout) / platt(holdout)
  - blend heuristique/ML : poids w_ml de 0.0 à 1.0
Métriques: log-loss, Brier, AUC (priorité calibration) + Top1 + ROI value-bet.
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.harness as h
from lib.xgb_like import XGBoostLike
from lib.ml_models import RandomForest, GradientBoosting
from lib.calibration import Calibrator
from lib.automl import log_loss, roc_auc, brier_score
from lib.kelly import kelly_amount


def make_model(kind):
    if kind == "xgb":
        return XGBoostLike(n_trees=80, max_depth=4, learning_rate=0.1,
                           lambda_reg=1.0, gamma=0.1, subsample=0.5,
                           early_stopping=10)
    if kind == "rf":
        return RandomForest(n_trees=40, max_depth=8, min_samples=15)
    return GradientBoosting(n_trees=50, max_depth=3, learning_rate=0.1)


def flatten(days):
    """Concatène toutes les courses de plusieurs jours en liste de courses."""
    races = []
    for _, data in days:
        races.extend(data["races"])
    return races


def races_to_xy(races):
    X, y = [], []
    for r in races:
        for row in r:
            X.append(row["feat"])
            y.append(row["label"])
    return X, y


def normalize_in_race(values):
    s = sum(values) or 1.0
    return [v / s * 100 for v in values]


def eval_config(train_races, test_races, kind="xgb", calib_method="holdout_iso",
                w_ml=0.5, holdout_frac=0.25):
    """Entraîne sur train_races, évalue sur test_races avec une config donnée.
    Retourne un dict de métriques au niveau cheval + course.
    """
    # --- train du modèle (+ éventuel hold-out pour calibration) ---
    X_tr, y_tr = races_to_xy(train_races)
    if sum(y_tr) < 10:
        return None

    if calib_method.startswith("holdout"):
        k = int(len(X_tr) * (1 - holdout_frac))
        X_fit, y_fit = X_tr[:k], y_tr[:k]
        X_cal, y_cal = X_tr[k:], y_tr[k:]
    else:
        X_fit, y_fit = X_tr, y_tr
        X_cal, y_cal = X_tr, y_tr

    model = make_model(kind)
    model.fit(X_fit, y_fit)

    # --- calibration ---
    calib = None
    if calib_method != "none":
        cal_preds = [model.predict_one(x) for x in X_cal]
        meth = "platt" if "platt" in calib_method else "isotonic"
        calib = Calibrator.fit(cal_preds, y_cal, method=meth)

    # --- évaluation course par course ---
    y_true, y_pred = [], []
    top1_ok = top3_ok = ncourses = 0
    vb_n = vb_won = 0
    kelly_mise = kelly_gain = 0.0

    for race in test_races:
        feats = [row["feat"] for row in race]
        raw = [model.predict_one(f) for f in feats]
        if calib:
            raw = [calib.apply(p) for p in raw]
        ml_pct = normalize_in_race(raw)
        heur_pct = [row["chance_heur"] for row in race]

        chance = [w_ml * ml_pct[i] + (1 - w_ml) * heur_pct[i]
                  for i in range(len(race))]
        chance = normalize_in_race(chance)

        # métriques de calibration au niveau cheval
        for i, row in enumerate(race):
            y_true.append(row["label"])
            y_pred.append(chance[i] / 100.0)

        order = sorted(range(len(race)), key=lambda i: -chance[i])
        ncourses += 1
        if race[order[0]]["ordre"] == 1:
            top1_ok += 1
        if any(race[i]["ordre"] == 1 for i in order[:3]):
            top3_ok += 1

        # value bets (mêmes règles que prod: edge>4 & cote>=4, Kelly 0.25)
        for i, row in enumerate(race):
            cote = row["cote"]
            if not cote:
                continue
            edge = chance[i] - row["proba_marche"]
            if edge > 4 and cote >= 4:
                p = chance[i] / 100.0
                km = kelly_amount(p, cote, 100, kelly_mult=0.25)
                if km > 0:
                    vb_n += 1
                    kelly_mise += km
                    if row["ordre"] == 1:
                        vb_won += 1
                        kelly_gain += km * cote

    if not y_true or not any(y_true):
        return None
    return {
        "log_loss": round(log_loss(y_true, y_pred), 4),
        "brier": round(brier_score(y_true, y_pred), 4),
        "auc": round(roc_auc(y_true, y_pred), 4),
        "top1": round(top1_ok / ncourses * 100, 2),
        "top3": round(top3_ok / ncourses * 100, 2),
        "ncourses": ncourses,
        "vb_n": vb_n,
        "vb_winrate": round(vb_won / vb_n * 100, 2) if vb_n else 0.0,
        "kelly_roi": round((kelly_gain - kelly_mise) / kelly_mise * 100, 2) if kelly_mise else 0.0,
    }


def walk(days, train_days, kind, calib_method, w_ml):
    """Walk-forward : fenêtre glissante de train_days jours, test sur le jour suivant.
    Agrège les métriques (pondérées par nb de courses)."""
    nc_tot = 0
    ll_w = br_w = auc_w = 0.0
    top1c = top3c = 0.0
    vb_n = vb_won = 0
    folds = 0
    for i in range(train_days, len(days)):
        train = flatten(days[i - train_days:i])
        res = eval_config(train, days[i][1]["races"],
                          kind=kind, calib_method=calib_method, w_ml=w_ml)
        if not res:
            continue
        folds += 1
        nc = res["ncourses"]
        nc_tot += nc
        top1c += res["top1"] * nc / 100
        top3c += res["top3"] * nc / 100
        ll_w += res["log_loss"] * nc
        br_w += res["brier"] * nc
        auc_w += res["auc"] * nc
        vb_n += res["vb_n"]
        vb_won += res["vb_n"] * res["vb_winrate"] / 100
    nc = nc_tot or 1
    return {
        "folds": folds,
        "ncourses": nc_tot,
        "log_loss": round(ll_w / nc, 4),
        "brier": round(br_w / nc, 4),
        "auc": round(auc_w / nc, 4),
        "top1": round(top1c / nc * 100, 2),
        "top3": round(top3c / nc * 100, 2),
        "vb_n": vb_n,
        "vb_winrate": round(vb_won / vb_n * 100, 2) if vb_n else 0,
    }


if __name__ == "__main__":
    days = h.load_days(datetime(2026, 5, 2), datetime(2026, 5, 31))
    print(f"Jours chargés: {len(days)}  ({sum(len(d['races']) for _,d in days)} courses)\n")

    TRAIN_DAYS = 14  # fenêtre d'entraînement glissante

    print("=" * 78)
    print("A) IMPACT DE LA CALIBRATION (modèle xgb, blend 50/50)")
    print("=" * 78)
    print(f"{'calibration':<20}{'logloss':>9}{'brier':>8}{'auc':>7}{'top1':>7}{'top3':>7}{'vb_n':>6}{'vb_wr':>7}")
    for cm in ["none", "train_iso", "holdout_iso", "holdout_platt"]:
        r = walk(days, TRAIN_DAYS, "xgb", cm, 0.5)
        print(f"{cm:<20}{r['log_loss']:>9}{r['brier']:>8}{r['auc']:>7}{r['top1']:>7}{r['top3']:>7}{r['vb_n']:>6}{r['vb_winrate']:>7}")

    print("\n" + "=" * 78)
    print("B) POIDS DU BLEND heuristique/ML (xgb, holdout_iso)")
    print("=" * 78)
    print(f"{'w_ml':<20}{'logloss':>9}{'brier':>8}{'auc':>7}{'top1':>7}{'top3':>7}{'vb_n':>6}{'vb_wr':>7}")
    for w in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]:
        r = walk(days, TRAIN_DAYS, "xgb", "holdout_iso", w)
        tag = f"{w:.2f}" + (" (heur seule)" if w == 0 else " (ML seul)" if w == 1 else "")
        print(f"{tag:<20}{r['log_loss']:>9}{r['brier']:>8}{r['auc']:>7}{r['top1']:>7}{r['top3']:>7}{r['vb_n']:>6}{r['vb_winrate']:>7}")
