"""
lib/walk_forward.py — Backtest temporel rigoureux (walk-forward analysis).

OBJECTIF
========
Le backtest "naïf" actuel souffre d'un *look-ahead bias* : les statistiques
(Elo, forme, stats driver/cheval...) sont calculées sur les 180 derniers jours
À PARTIR D'AUJOURD'HUI, puis on évalue des courses passées. Le modèle "voit"
donc des informations postérieures à la course qu'il prédit => métriques gonflées.

Le walk-forward corrige cela : pour évaluer une course à la date T, on
n'utilise QUE les données strictement antérieures à T :
    - stats calculées sur [T - stats_window, T - 1]
    - modèle ML entraîné sur des courses de [T - train_window, T - 1]
    - test sur les courses de la fenêtre [T, T + test_window - 1]
puis on fait glisser la fenêtre vers le futur (rolling / expanding).

Ce module ne contient QUE de la logique pure (pas de Flask, pas de réseau)
pour être testable unitairement. La récupération des données et l'entraînement
sont injectés via des callbacks par app.py.
"""

from datetime import datetime, timedelta


# ------------------------------------------------------------------
#  Génération des fenêtres temporelles
# ------------------------------------------------------------------
def generate_windows(ref_date, n_folds, train_window, test_window,
                     gap=0, mode="rolling"):
    """
    Génère la liste des fenêtres (fold) du plus ancien au plus récent.

    Chaque fenêtre couvre une période de TEST située dans le passé. La période
    d'ENTRAÎNEMENT précède strictement la période de test (avec un éventuel
    `gap` de séparation pour éviter toute contamination de bordure).

    Args:
        ref_date    : date de référence (datetime), typiquement "aujourd'hui".
                      Le fold le plus récent teste les jours juste avant ref_date.
        n_folds     : nombre de plis (folds).
        train_window: nb de jours d'entraînement par fold.
        test_window : nb de jours de test par fold.
        gap         : nb de jours laissés entre la fin du train et le début du
                      test (purge anti-fuite). Défaut 0.
        mode        : "rolling" (fenêtre glissante de taille fixe) ou
                      "expanding" (le train commence toujours au point le plus
                      ancien et grandit à chaque fold).

    Returns:
        list[dict] avec, pour chaque fold (ordre chronologique croissant) :
            fold        : index 1..n_folds
            train_start : datetime (inclus)
            train_end   : datetime (inclus)
            test_start  : datetime (inclus)
            test_end    : datetime (inclus)
        Les dates de test sont toujours < ref_date.
    """
    if n_folds < 1:
        raise ValueError("n_folds doit être >= 1")
    if train_window < 1 or test_window < 1:
        raise ValueError("train_window et test_window doivent être >= 1")
    if mode not in ("rolling", "expanding"):
        raise ValueError("mode doit être 'rolling' ou 'expanding'")

    folds = []
    # Le fold le PLUS RÉCENT teste [ref_date - test_window, ref_date - 1].
    # Les folds plus anciens reculent de test_window à chaque cran.
    # On les construit du plus récent au plus ancien, puis on inverse.
    oldest_train_start = None
    for k in range(n_folds):
        test_end = ref_date - timedelta(days=1 + k * test_window)
        test_start = test_end - timedelta(days=test_window - 1)
        train_end = test_start - timedelta(days=1 + gap)
        train_start = train_end - timedelta(days=train_window - 1)
        oldest_train_start = train_start  # se met à jour jusqu'au plus ancien
        folds.append({
            "test_end": test_end,
            "test_start": test_start,
            "train_end": train_end,
            "train_start": train_start,
        })

    folds.reverse()  # ordre chronologique croissant

    # Mode expanding : tous les folds partent du point d'entraînement le + ancien
    if mode == "expanding":
        for f in folds:
            f["train_start"] = oldest_train_start

    for i, f in enumerate(folds, start=1):
        f["fold"] = i
    return folds


def daterange(start, end):
    """Itère sur chaque jour de start à end (inclus)."""
    d = start
    while d <= end:
        yield d
        d = d + timedelta(days=1)


# ------------------------------------------------------------------
#  Agrégation des métriques
# ------------------------------------------------------------------
def _safe_div(a, b):
    return a / b if b else 0.0


def aggregate_fold_metrics(fold_results):
    """
    Agrège les métriques de tous les folds en un résumé global.

    Args:
        fold_results : list[dict], chaque dict produit par un fold avec au moins :
            fold, n_test_courses, n_test_samples,
            top1_correct, top3_correct,            (compteurs entiers)
            log_loss, auc, brier,                  (métriques sur le test du fold)
            kelly_mise, kelly_gain,                (montants Kelly cumulés)
            value_bets, value_bets_won             (compteurs value bets)

    Returns:
        dict avec :
            folds       : la liste passée (pour affichage par fold)
            global      : métriques agrégées sur l'ensemble des folds
    """
    n_courses = sum(f.get("n_test_courses", 0) for f in fold_results)
    n_samples = sum(f.get("n_test_samples", 0) for f in fold_results)
    top1 = sum(f.get("top1_correct", 0) for f in fold_results)
    top3 = sum(f.get("top3_correct", 0) for f in fold_results)

    km = sum(f.get("kelly_mise", 0.0) for f in fold_results)
    kg = sum(f.get("kelly_gain", 0.0) for f in fold_results)
    vb = sum(f.get("value_bets", 0) for f in fold_results)
    vb_won = sum(f.get("value_bets_won", 0) for f in fold_results)

    # Moyenne des métriques de classification pondérée par le nb d'échantillons
    def _weighted(metric):
        num = sum(f.get(metric, 0.0) * f.get("n_test_samples", 0)
                  for f in fold_results)
        return _safe_div(num, n_samples)

    glob = {
        "n_folds": len(fold_results),
        "n_test_courses": n_courses,
        "n_test_samples": n_samples,
        "top1_rate": round(_safe_div(top1, n_courses) * 100, 2),
        "top3_rate": round(_safe_div(top3, n_courses) * 100, 2),
        "log_loss": round(_weighted("log_loss"), 4),
        "auc": round(_weighted("auc"), 4),
        "brier": round(_weighted("brier"), 4),
        "kelly_mise": round(km, 2),
        "kelly_gain": round(kg, 2),
        "kelly_profit": round(kg - km, 2),
        "kelly_roi": round(_safe_div(kg - km, km) * 100, 2) if km else 0.0,
        "value_bets": vb,
        "value_bets_won": vb_won,
        "vb_winrate": round(_safe_div(vb_won, vb) * 100, 2) if vb else 0.0,
    }

    # Stabilité : écart-type des Top1 par fold (un modèle robuste est régulier)
    top1_rates = [f.get("top1_rate") for f in fold_results
                  if f.get("top1_rate") is not None]
    if len(top1_rates) >= 2:
        mean = sum(top1_rates) / len(top1_rates)
        var = sum((r - mean) ** 2 for r in top1_rates) / len(top1_rates)
        glob["top1_std"] = round(var ** 0.5, 2)
    else:
        glob["top1_std"] = 0.0

    return {"folds": fold_results, "global": glob}


def fmt_window(f, date_fmt="%d/%m"):
    """Représentation lisible d'une fenêtre de fold."""
    return {
        "fold": f.get("fold"),
        "train": f"{f['train_start'].strftime(date_fmt)}→{f['train_end'].strftime(date_fmt)}",
        "test": f"{f['test_start'].strftime(date_fmt)}→{f['test_end'].strftime(date_fmt)}",
    }
