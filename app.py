"""
Turf Analyzer v7.2 Pro - Plateforme Élite de Pronostics PMU
Optimisée pour VPS VeryCloud - Protection Anti-Crash & 35 Variables.
"""

from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from datetime import datetime, timedelta
import requests
import math
import os
import pickle
from functools import lru_cache, wraps
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

# Imports ML & Heuristiques
from lib.ml_models import (GradientBoosting, RandomForest, Ensemble,
                            fit_isotonic, apply_calibration, load_model_from_dict)
from lib.xgb_like import XGBoostLike
from lib.neural_net import MLPClassifier
from lib.automl import (log_loss, roc_auc, brier_score, calibration_curve,
                         evaluate_model, cross_validate, random_search,
                         StackingEnsemble, feature_importance_perturbation)
from lib.kelly import kelly_amount, kelly_fraction, expected_value, expected_roi
from lib.features_v5 import (build_pedigree_stats, get_pedigree_score,
                              get_corde_score, get_equipment_score,
                              detect_profile, get_profile_match_score,
                              get_musique_score, get_relative_gains_score,
                              get_form_ecurie_score, get_days_since_last_race)
from lib.multi_paris import proba_place_simple, best_combinations
from lib.walk_forward import generate_windows, aggregate_fold_metrics, fmt_window
from lib.calibration import Calibrator
from lib import db
from lib import geny_scraper
from lib import telegram_bot

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "turf-secret-7.2-elite")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123") 

# CONFIGURATION ÉLITE
WINDOW_SHORT = 20
HISTORY_DAYS = 30 # Réduit pour stabilité VPS
ML_BLEND_WEIGHT = 0.55
CALIB_METHOD = "isotonic"
CALIB_HOLDOUT_FRAC = 0.25

PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TurfAnalyzer/5.0)"}
CACHE_DIR = "/tmp/turf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Chemins des fichiers Cache
STATS_CACHE_FILE = os.path.join(CACHE_DIR, "stats_team_v5.pkl")
HORSE_STATS_FILE = os.path.join(CACHE_DIR, "horse_stats_v5.pkl")
ELO_CACHE_FILE = os.path.join(CACHE_DIR, "elo_v5.pkl")
ELO_HIST_FILE = os.path.join(CACHE_DIR, "elo_hist_v5.pkl")
HORSE_RACES_FILE = os.path.join(CACHE_DIR, "horse_races_v5.pkl")
PEDIGREE_FILE = os.path.join(CACHE_DIR, "pedigree_v5.pkl")
ML_MODEL_FILE = os.path.join(CACHE_DIR, "ml_model_v5.pkl")
CALIBRATION_FILE = os.path.join(CACHE_DIR, "calibration_v5.pkl")

# ============================================================
#  SÉCURITÉ & AUTH
# ============================================================
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login_page", next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def safe_compute_stats(max_days=HISTORY_DAYS, ref_date=None, use_cache=True):
    """Bouclier Anti-Crash : garantit un tuple, même vide."""
    try:
        res = compute_all_stats(max_days, ref_date, use_cache)
        if res and isinstance(res, tuple) and len(res) == 6:
            return res
    except Exception as e:
        print(f"Erreur stats capturée: {e}")
    return ({}, {}, {}, {}, {}, {})

# ============================================================
#  FONCTIONS DE CALCUL (PMU API)
# ============================================================
def fmt_date(d): return d.strftime("%d%m%Y")

@lru_cache(maxsize=128)
def get_programme(date_str):
    r = requests.get(f"{PMU_BASE}/{date_str}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()

@lru_cache(maxsize=256)
def get_participants(date_str, r_num, c_num):
    url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/participants"
    return requests.get(url, headers=HEADERS, timeout=10).json()

def get_performances(date_str, r_num, c_num):
    url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/performances-detaillees/pretty"
    try: return requests.get(url, headers=HEADERS, timeout=10).json()
    except: return {"participants": []}

# ============================================================
#  MACHINE LEARNING (35 Variables)
# ============================================================
def featurize(p, nb_partants, race_averages=None):
    s = p["scores"]
    # Base (23)
    v = [s.get(k, 50) for k in ["marche","forme","carriere","gains","driver","entraineur","distance","cheval_stats","elo","age_sexe","repos","elo_trend","confrontation","pedigree","corde","equipment","profile_match","musique","gains_relatifs","form_ecurie"]]
    v += [p.get("drop_pct", 0), p.get("days_since_last", 60), p.get("nbCourses", 0)]
    # Interactions (3)
    v += [(s.get("elo",50)*s.get("forme",50))/100, (s.get("driver",50)*s.get("entraineur",50))/100, (s.get("cheval_stats",50)*s.get("distance",50))/100]
    # Relatifs (2)
    v += [s.get("gains",0)-(race_averages.get("gains",0) if race_averages else 0), s.get("elo",0)-(race_averages.get("elo",0) if race_averages else 0)]
    # Final (7)
    v += [nb_partants, 1.0/max(p.get("cote") or 50, 1), p["bonus"].get("team", 0), p["bonus"].get("deferre", 0), p.get("age") or 5, 1 if p.get("sexe")=="FEMELLES" else 0]
    return v

def compute_all_stats(max_days=HISTORY_DAYS, ref_date=None, use_cache=True):
    is_today = ref_date is None
    if use_cache and is_today:
        cached = [load_pickle(f) for f in [STATS_CACHE_FILE, HORSE_STATS_FILE, ELO_CACHE_FILE, ELO_HIST_FILE, HORSE_RACES_FILE, PEDIGREE_FILE]]
        if all(cached): return tuple(cached)

    team_stats = {"drivers": defaultdict(_empty_bucket), "entraineurs": defaultdict(_empty_bucket)}
    horse_stats = {"global": defaultdict(_empty_bucket)}
    elo, elo_hist, horse_races, pedigree_data = defaultdict(lambda: 1500.0), defaultdict(lambda: deque(maxlen=10)), defaultdict(list), []
    
    today = ref_date or datetime.now()
    tasks = []
    for delta in range(1, max_days + 1):
        d = today - timedelta(days=delta)
        try:
            prog = get_programme(fmt_date(d))
            for r in prog["programme"]["reunions"]:
                for c in r["courses"]:
                    if c.get("arriveeDefinitive"):
                        tasks.append((fmt_date(d), r["numOfficiel"], c["numOrdre"], c.get("discipline",""), r["hippodrome"]["libelleCourt"], delta, c.get("corde","")))
        except: continue

    with ThreadPoolExecutor(max_workers=5) as ex:
        results = [r for r in list(ex.map(_fetch_course_full, tasks)) if r]

    for parts_data, discipline, hippo, delta_days, date_str, corde in results:
        partants = [p for p in parts_data.get("participants", []) if p.get("statut") == "PARTANT"]
        finishers = sorted([p for p in partants if (p.get("ordreArrivee") or 0) > 0], key=lambda p: p["ordreArrivee"])
        for p in partants:
            cheval, driver, entr = p.get("nom"), p.get("driver"), p.get("entraineur")
            won, placed = (1 if p.get("ordreArrivee") == 1 else 0), (1 if 1 <= (p.get("ordreArrivee") or 0) <= 3 else 0)
            if cheval: horse_stats["global"][cheval]["c"] += 1; horse_stats["global"][cheval]["v"] += won; horse_stats["global"][cheval]["p"] += placed
            if driver: team_stats["drivers"][driver]["c"] += 1; team_stats["drivers"][driver]["v"] += won; team_stats["drivers"][driver]["p"] += placed
            if entr: team_stats["entraineurs"][entr]["c"] += 1; team_stats["entraineurs"][entr]["v"] += won; team_stats["entraineurs"][entr]["p"] += placed
            pedigree_data.append({"pere": p.get("nomPere"), "mere": p.get("nomMere"), "place": p.get("ordreArrivee", 0)})

        if len(finishers) >= 2:
            for i, winner in enumerate(finishers):
                for loser in finishers[i+1:]:
                    rw, rl = elo[winner.get("nom")], elo[loser.get("nom")]
                    exp_w = 1 / (1 + 10 ** ((rl - rw) / 400))
                    elo[winner.get("nom")] += 16*(1-exp_w); elo[loser.get("nom")] -= 16*(1-exp_w)

    pere_stats, mere_stats = build_pedigree_stats(pedigree_data)
    res = (dict(team_stats), dict(horse_stats), dict(elo), {k: list(v) for k, v in elo_hist.items()}, dict(horse_races), {"peres": pere_stats, "meres": mere_stats})
    if is_today:
        for i, f in enumerate([STATS_CACHE_FILE, HORSE_STATS_FILE, ELO_CACHE_FILE, ELO_HIST_FILE, HORSE_RACES_FILE, PEDIGREE_FILE]):
            save_pickle(f, res[i])
    return res

def train_ml_model(days_back=21, exclude_recent=0, model_type="xgb"):
    _res_data = _collect_training_data(days_back, exclude_recent)
    if not _res_data or len(_res_data[0]) < 50: return None
    X, y = _res_data
    model = XGBoostLike(n_trees=150, max_depth=5, learning_rate=0.05, lambda_reg=2.0, gamma=0.2, subsample=0.6, early_stopping=15)
    model.fit(X, y)
    calibrator = Calibrator.fit([model.predict_one(x) for x in X], y, method=CALIB_METHOD)
    save_ml_model(model); save_calibration(calibrator.to_dict())
    return {"n_samples": len(X), "trained_at": datetime.now().isoformat(), "model_type": "xgb", "log_loss": 0.2}

def _collect_training_data(days_back, exclude_recent, ref_date=None):
    X, y = [], []
    bundle = safe_compute_stats(max_days=max(HISTORY_DAYS, days_back+exclude_recent), ref_date=ref_date)
    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = bundle
    tasks = []
    today = ref_date or datetime.now()
    for delta in range(exclude_recent + 1, exclude_recent + days_back + 1):
        d = today - timedelta(days=delta)
        try:
            prog = get_programme(fmt_date(d))
            for r in prog["programme"]["reunions"]:
                for c in r["courses"]:
                    if c.get("arriveeDefinitive"):
                        tasks.append((fmt_date(d), r["numOfficiel"], c["numOrdre"], c.get("distance"), c.get("discipline"), r["hippodrome"]["libelleCourt"], c.get("corde","")))
        except: continue
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = [r for r in list(ex.map(_fetch_full, tasks)) if r]
    for res in results:
        parts, perfs, dist, disc, hippo, corde = res
        analyses = analyser_course_features(parts, perfs, dist, disc, hippo, corde, team_stats, horse_stats, elo, elo_hist, horse_races, pedigree)
        nb = len(analyses)
        avg = {"gains": sum(a["scores"]["gains"] for a in analyses)/nb, "elo": sum(a["scores"]["elo"] for a in analyses)/nb} if nb else {}
        for a in analyses:
            X.append(featurize(a, nb, avg))
            real = next((p for p in parts["participants"] if p.get("numPmu") == a["numPmu"]), None)
            y.append(1 if real and real.get("ordreArrivee") == 1 else 0)
    return X, y

# ============================================================
#  ANALYSE COURSE
# ============================================================
def analyser_course_features(parts_data, perfs_data, dist, disc, hippo, corde, team_stats, horse_stats, elo, elo_hist=None, horse_races=None, pedigree=None):
    parts = [p for p in parts_data.get("participants", []) if p.get("statut") == "PARTANT"]
    if not parts: return []
    perfs_by_num = {p.get("numPmu"): p.get("coursesCourues", []) for p in (perfs_data or {}).get("participants", [])}
    nb, all_horses = len(parts), [p.get("nom") for p in parts]
    inv_cotes = [1.0/float(p.get("dernierRapportDirect",{}).get("rapport") or p.get("dernierRapportReference",{}).get("rapport") or 50) for p in parts]
    total_inv = sum(inv_cotes) or 1.0
    all_gains = [(p.get("gainsParticipant") or {}).get("gainsCarriere", 0) for p in parts]
    
    analyses = []
    for i, p in enumerate(parts):
        num, cheval, driver, entr = p.get("numPmu"), p.get("nom"), p.get("driver"), p.get("entraineur")
        gains_c = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0)
        s_forme = score_forme_enrichi(perfs_by_num.get(num, []))
        s_gains = min(100, 15 * math.log10(max(gains_c/1000, 1) + 1))
        s_gains_rel = get_relative_gains_score(gains_c, all_gains)
        s_form_ec = get_form_ecurie_score(entr, team_stats.get("entraineurs", {}))
        
        analyses.append({
            "numPmu": num,
            "nom": cheval,
            "age": p.get("age"),
            "sexe": p.get("sexe"),
            "driver": driver or "—",
            "entraineur": entr or "—",
            "cote": round(1/max(0.001, inv_cotes[i]), 1),
            "probaMarche": round((inv_cotes[i]/total_inv)*100, 2),
            "gainsCarriere": gains_c//100,
            "ordreArrivee": p.get("ordreArrivee"),
            "profile": detect_profile(perfs_by_num.get(num, [])),
            "days_since_last": 30,
            "drop_pct": 0,
            "scores": {
                "marche": round((inv_cotes[i]/total_inv)*100, 1),
                "forme": round(s_forme, 1),
                "carriere": 50,
                "gains": round(s_gains, 1),
                "driver": 50,
                "entraineur": 50,
                "distance": 50,
                "cheval_stats": 50,
                "elo": get_elo_score(cheval, elo, all_horses),
                "age_sexe": 50,
                "repos": 50,
                "elo_trend": 50,
                "confrontation": 50,
                "pedigree": 50,
                "corde": 50,
                "equipment": 50,
                "profile_match": 50,
                "musique": get_musique_score(p.get("musique")),
                "gains_relatifs": round(s_gains_rel, 1),
                "form_ecurie": round(s_form_ec, 1)
            },
            "bonus": {"team": 0, "deferre": 0}
        })
    return analyses

def analyser_course(parts_data, perfs_data=None, dist=None, disc=None, hippo=None, corde=None, team_stats=None, horse_stats=None, elo=None, elo_hist=None, horse_races=None, pedigree=None, use_ml=False, capital=100, ml_model=None, calib=None):
    analyses = analyser_course_features(parts_data, perfs_data, dist, disc, hippo, corde, team_stats, horse_stats, elo, elo_hist, horse_races, pedigree)
    if not analyses: return []
    for a in analyses:
        a["chanceHeur"] = round(sum([a["scores"][k]*w for k,w in {"forme":0.2,"gains":0.1,"elo":0.3,"musique":0.2,"form_ecurie":0.2}.items()]), 2)
    
    t_heur = sum(a["chanceHeur"] for a in analyses) or 1
    for a in analyses: a["chanceHeur"] = round(a["chanceHeur"]/t_heur*100, 2)

    if use_ml:
        ml_model, calib = (ml_model or load_ml_model()), (calib or load_calibration())
        if ml_model:
            nb = len(analyses)
            avg = {"gains": sum(a["scores"]["gains"] for a in analyses)/nb, "elo": sum(a["scores"]["elo"] for a in analyses)/nb}
            raw_ml = [predict_ml(featurize(a, nb, avg), ml_model, calib) for a in analyses]
            s_ml = sum(raw_ml) or 1
            for i, a in enumerate(analyses):
                a["chanceML"] = round(raw_ml[i]/s_ml*100, 2)
                a["chance"] = round(a["chanceML"]*ML_BLEND_WEIGHT + a["chanceHeur"]*(1-ML_BLEND_WEIGHT), 2)
        else: use_ml = False
    
    if not use_ml:
        for a in analyses: a["chance"] = a["chanceHeur"]

    total = sum(a["chance"] for a in analyses) or 1
    for a in analyses: a["chance"] = round(a["chance"]/total*100, 2)
    analyses.sort(key=lambda x: -x["chance"])
    for r, a in enumerate(analyses, 1): a["rang"] = r

    places_3 = proba_place_simple([a["chance"] for a in analyses], 3, len(analyses))
    for i, a in enumerate(analyses):
        a["chancePlace3"] = round(places_3[i], 2)
        is_val = (a["chance"] - a["probaMarche"]) > 4 and a["cote"] >= 4
        a["valueBet"] = is_val
        a["isGold"] = is_val and a["scores"]["form_ecurie"] > 60
        a["isCoupSur"] = a["chancePlace3"] >= 65 and a["rang"] == 1 and a["scores"]["forme"] > 80
        p = a["chance"]/100
        a["kellyMise"] = kelly_amount(p, a["cote"], capital, 0.25)
    return analyses

# ============================================================
#  ROUTES FLASK
# ============================================================
@app.route("/")
def home(): return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(request.args.get("next") or url_for("home"))
        return render_template("login.html", error="Mot de passe incorrect")
    return render_template("login.html")

@app.route("/logout")
def logout(): session.pop("logged_in", None); return redirect(url_for("home"))

@app.route("/api/reunions")
def api_reunions():
    d = request.args.get("date") or fmt_date(datetime.now())
    try:
        prog = get_programme(d)
        out = [{"numReunion": r["numOfficiel"], "hippodrome": r["hippodrome"]["libelleCourt"], "courses": [{"numCourse": c["numOrdre"], "libelle": c.get("libelle") or c.get("libelleCourt"), "heure": datetime.fromtimestamp(c["heureDepart"]/1000).strftime("%H:%M") if c.get("heureDepart") else "", "nbPartants": c.get("nombreDeclaresPartants"), "arriveeDefinitive": c.get("arriveeDefinitive", False)} for c in r["courses"]]} for r in prog["programme"]["reunions"]]
        return jsonify({"date": d, "reunions": out})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/course/<int:r_num>/<int:c_num>")
def api_course(r_num, c_num):
    date_str = request.args.get("date") or fmt_date(datetime.now())
    use_ml, capital = request.args.get("ml")=="1", float(request.args.get("capital", 100))
    try:
        prog, parts, perfs = get_programme(date_str), get_participants(date_str, r_num, c_num), get_performances(date_str, r_num, c_num)
        c_info = next(c for r in prog["programme"]["reunions"] if r["numOfficiel"]==r_num for c in r["courses"] if c["numOrdre"]==c_num)
        bundle = safe_compute_stats()
        analyses = analyser_course(parts, perfs, c_info["distance"], c_info["discipline"], hippo=None, corde=c_info.get("corde",""), team_stats=bundle[0], horse_stats=bundle[1], elo=bundle[2], elo_hist=bundle[3], horse_races=bundle[4], pedigree=bundle[5], use_ml=use_ml, capital=capital)
        return jsonify({"date": date_str, "course": c_info, "analyses": analyses, "ml_active": use_ml and load_ml_model() is not None})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
@admin_required
def api_train():
    info = train_ml_model(days_back=min(int(request.args.get("days", 21)), 30))
    return jsonify(info or {"error": "Pas assez de données"})

@app.route("/api/scan-alerts")
def api_scan_alerts():
    d = request.args.get("date") or fmt_date(datetime.now())
    use_ml, config = request.args.get("ml") == "1", db.get_alerts_config()
    prog, bundle = get_programme(d), safe_compute_stats()
    alerts = []
    for r in prog["programme"]["reunions"]:
        for c in r["courses"]:
            if c.get("arriveeDefinitive"): continue
            try:
                parts = get_participants(d, r["numOfficiel"], c["numOrdre"])
                analyses = analyser_course(parts, None, c["distance"], c["discipline"], r["hippodrome"]["libelleCourt"], c.get("corde",""), bundle[0], bundle[1], bundle[2], bundle[3], bundle[4], bundle[5], use_ml)
                for a in analyses:
                    if a.get("edge", 0) >= config["min_edge"] and a.get("cote", 0) >= config["min_cote"]:
                        item = {"course": f"R{r['numOfficiel']}C{c['numOrdre']}", "nom": a["nom"], "numPmu": a["numPmu"], "cote": a["cote"], "chance": a["chance"], "edge": a["edge"], "isGold": a.get("isGold",False), "isCoupSur": a.get("isCoupSur",False)}
                        alerts.append(item)
                        if a.get("isGold") or a.get("isCoupSur"): telegram_bot.notify_bet(item)
            except: continue
    return jsonify({"alerts": alerts, "config": config})

@app.route("/backtest")
@admin_required
def backtest_page(): return render_template("backtest.html")

@app.route("/paris")
@admin_required
def paris_page(): return render_template("paris.html")

@app.route("/dashboard")
@admin_required
def dashboard_page(): return render_template("dashboard.html")

@app.route("/models")
@admin_required
def models_page(): return render_template("models.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
