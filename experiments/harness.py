"""
experiments/harness.py — Harnais de mesure rapide pour l'optimisation v7.1.

Objectif : itérer sur modèles / calibration / blend SANS re-télécharger ni
re-calculer les stats à chaque essai.

Stratégie :
  1. collect_day(date) télécharge + featurise toutes les courses d'un jour,
     en utilisant des stats calculées à ref_date = lendemain (anti look-ahead),
     et met le résultat en cache disque (experiments/cache/*.pkl).
  2. Les expériences chargent ces caches et testent des combinaisons en mémoire.

Lancer :  python -m experiments.harness collect 2026-04-01 2026-06-01
          python -m experiments.harness exp
"""
import os
import sys
import pickle
from datetime import datetime, timedelta

# Permet "python -m experiments.harness" depuis la racine du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ------------------------------------------------------------------
#  Collecte + cache d'un jour
# ------------------------------------------------------------------
def _day_cache_path(date_str):
    return os.path.join(CACHE_DIR, f"day_{date_str}.pkl")


def collect_day(d, stats_window=120, force=False):
    """Collecte une journée : pour chaque course terminée, on stocke
    [(features, label, cote, ordreArrivee, proba_marche, chance_heur), ...].
    stats calculées à ref_date = d+1 (donc strictement < d+1, i.e. <= d).
    NB: pour être 100% sans fuite intra-jour, on calcule les stats à ref_date=d
    (n'inclut que <= d-1). On teste donc les courses du jour d avec stats <= d-1.
    """
    date_str = app.fmt_date(d)
    path = _day_cache_path(date_str)
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            return pickle.load(f)

    # stats connues la VEILLE de d (aucune course de d incluse)
    bundle = app.compute_all_stats(max_days=stats_window, ref_date=d)
    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = bundle

    try:
        prog = app.get_programme(date_str)
    except Exception:
        prog = None

    races = []
    if prog:
        tasks = []
        for r in prog["programme"]["reunions"]:
            hippo = r["hippodrome"]["libelleCourt"]
            for c in r["courses"]:
                if c.get("arriveeDefinitive"):
                    tasks.append((date_str, r["numOfficiel"], c["numOrdre"],
                                  c.get("distance"), c.get("discipline"),
                                  hippo, c.get("corde", "")))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=20) as ex:
            fetched = list(ex.map(app._fetch_full, tasks))

        for result in fetched:
            if not result:
                continue
            parts, perfs, distance, discipline, hippodrome, type_corde = result
            analyses = app.analyser_course(
                parts, perfs, distance, discipline, hippodrome, type_corde,
                team_stats, horse_stats, elo, elo_hist, horse_races, pedigree,
                use_ml=False, capital=100)  # heuristique seule pour récupérer chance_heur
            if not analyses:
                continue
            if not any(a["ordreArrivee"] == 1 for a in analyses):
                continue
            nb = len(analyses)
            rows = []
            for a in analyses:
                rows.append({
                    "feat": app.featurize(a, nb),
                    "label": 1 if a["ordreArrivee"] == 1 else 0,
                    "cote": a["cote"],
                    "ordre": a["ordreArrivee"],
                    "proba_marche": a["probaMarche"],
                    "chance_heur": a.get("chanceHeur", a["chance"]),
                    "nb": nb,
                })
            races.append(rows)

    out = {"date": date_str, "races": races}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"[collect] {date_str}: {len(races)} courses, "
          f"{sum(len(r) for r in races)} chevaux")
    return out


def collect_range(start, end, stats_window=120, force=False):
    d = start
    total = 0
    while d <= end:
        data = collect_day(d, stats_window=stats_window, force=force)
        total += len(data["races"])
        d += timedelta(days=1)
    print(f"[collect] terminé: {total} courses sur la période.")


def load_days(start, end):
    """Charge les jours déjà mis en cache (ignore les manquants)."""
    days = []
    d = start
    while d <= end:
        p = _day_cache_path(app.fmt_date(d))
        if os.path.exists(p):
            with open(p, "rb") as f:
                days.append((d, pickle.load(f)))
        d += timedelta(days=1)
    return days


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "collect":
        s = datetime.strptime(sys.argv[2], "%Y-%m-%d")
        e = datetime.strptime(sys.argv[3], "%Y-%m-%d")
        collect_range(s, e)
    else:
        print(__doc__)
