"""
Turf Analyzer v7.2 Pro - Version Stable Élite
Optimisée pour VPS VeryCloud - 35 Variables & Protection Anti-Crash.
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

# Imports Libs internes
from lib.ml_models import (GradientBoosting, RandomForest, Ensemble, load_model_from_dict)
from lib.xgb_like import XGBoostLike
from lib.neural_net import MLPClassifier
from lib.automl import (log_loss, roc_auc, brier_score, StackingEnsemble, evaluate_model)
from lib.kelly import kelly_amount, kelly_fraction, expected_roi
from lib.features_v5 import (build_pedigree_stats, get_musique_score, get_relative_gains_score, 
                              get_form_ecurie_score, detect_profile, get_profile_match_score)
from lib.multi_paris import proba_place_simple
from lib.calibration import Calibrator
from lib import db, telegram_bot

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "turf-secret-7.2-elite")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

# CONFIGURATION
HISTORY_DAYS = 30 
ML_BLEND_WEIGHT = 0.55
CALIB_METHOD = "isotonic"
CALIB_HOLDOUT_FRAC = 0.25

PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TurfAnalyzer/5.0)"}
CACHE_DIR = "/tmp/turf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Chemins fichiers Cache
STATS_CACHE_FILE = os.path.join(CACHE_DIR, "stats_team_v5.pkl")
HORSE_STATS_FILE = os.path.join(CACHE_DIR, "horse_stats_v5.pkl")
ELO_CACHE_FILE = os.path.join(CACHE_DIR, "elo_v5.pkl")
ELO_HIST_FILE = os.path.join(CACHE_DIR, "elo_hist_v5.pkl")
HORSE_RACES_FILE = os.path.join(CACHE_DIR, "horse_races_v5.pkl")
PEDIGREE_FILE = os.path.join(CACHE_DIR, "pedigree_v5.pkl")
ML_MODEL_FILE = os.path.join(CACHE_DIR, "ml_model_v5.pkl")
CALIBRATION_FILE = os.path.join(CACHE_DIR, "calibration_v5.pkl")

# ============================================================
#  FONCTIONS DE BASE & UTILITAIRES
# ============================================================
def fmt_date(d): return d.strftime("%d%m%Y")

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login_page", next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def score_forme_enrichi(perfs):
    if not perfs: return 50
    pts = 0
    valid_perfs = perfs[:5]
    for p in valid_perfs:
        pl = (p.get("place") or {}).get("place", 0)
        if pl == 1: pts += 100
        elif 1 <= pl <= 3: pts += 75
        else: pts += 30
    return pts / len(valid_perfs)

def get_elo_score(ch, elo, all_h):
    my = elo.get(ch, 1500)
    elos = [elo.get(h, 1500) for h in all_h if h]
    if len(elos) < 2: return 50
    e_min, e_max = min(elos), max(elos)
    if e_max == e_min: return 50
    return (my - e_min) / (e_max - e_min) * 100

def load_pickle(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f: return pickle.load(f)["payload"]
        except: pass
    return None

def save_pickle(path, payload):
    try:
        with open(path, "wb") as f:
            pickle.dump({"saved_at": datetime.now().isoformat(), "payload": payload}, f)
    except: pass

def safe_compute_stats(max_days=HISTORY_DAYS):
    """Garantit le retour d'un tuple de 6 éléments, même vide."""
    try:
        res = compute_all_stats(max_days)
        if res and isinstance(res, tuple) and len(res) == 6:
            return res
    except Exception as e:
        print(f"Erreur calcul stats: {e}")
    return ({}, {}, {}, {}, {}, {})

# ============================================================
#  PMU API WRAPPERS
# ============================================================
def get_programme(date_str):
    try:
        r = requests.get(f"{PMU_BASE}/{date_str}", headers=HEADERS, timeout=10)
        return r.json()
    except: return None

def get_participants(date_str, r_num, c_num):
    try:
        url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/participants"
        return requests.get(url, headers=HEADERS, timeout=10).json()
    except: return {"participants": []}

def get_performances(date_str, r_num, c_num):
    try:
        url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/performances-detaillees/pretty"
        return requests.get(url, headers=HEADERS, timeout=10).json()
    except: return {"participants": []}

# ============================================================
#  COEUR DU MOTEUR ML (35 Variables)
# ============================================================
def featurize(p, nb_partants, avgs=None):
    s = p["scores"]
    # Base features (23)
    v = [s.get(k, 50) for k in ["marche","forme","carriere","gains","driver","entraineur","distance","cheval_stats","elo","age_sexe","repos","elo_trend","confrontation","pedigree","corde","equipment","profile_match","musique","gains_relatifs","form_ecurie"]]
    v += [p.get("drop_pct", 0), 30, p.get("nbCourses", 0)]
    # Interactions (3)
    v += [(s.get("elo", 50) * s.get("forme", 50)) / 100.0, (s.get("driver", 50) * s.get("entraineur", 50)) / 100.0, 25]
    # Relatifs (2)
    v += [s.get("gains", 0) - (avgs.get("gains", 0) if avgs else 0), s.get("elo", 0) - (avgs.get("elo", 0) if avgs else 0)]
    # Final metadata (7)
    v += [nb_partants, 1.0 / max(p.get("cote") or 50, 1), 0, 0, p.get("age") or 5, 0, 0]
    return v

def _empty_bucket(): return {"c": 0, "v": 0, "p": 0}

def _fetch_course_full(args):
    try: return (get_participants(args[0], args[1], args[2]), args[3], args[4], args[5], args[0])
    except: return None

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
        prog = get_programme(fmt_date(d))
        if not prog: continue
        for r in prog["programme"]["reunions"]:
            for c in r["courses"]:
                if c.get("arriveeDefinitive"):
                    tasks.append((fmt_date(d), r["numOfficiel"], c["numOrdre"], c.get("discipline",""), r["hippodrome"]["libelleCourt"], delta))

    with ThreadPoolExecutor(max_workers=5) as ex:
        results = [r for r in list(ex.map(_fetch_course_full, tasks)) if r]

    for parts_data, discipline, hippo, delta_days, date_str in results:
        partants = [p for p in parts_data.get("participants", []) if p.get("statut") == "PARTANT"]
        finishers = sorted([p for p in partants if (p.get("ordreArrivee") or 0) > 0], key=lambda p: p["ordreArrivee"])
        for p in partants:
            cheval, driver, entr = p.get("nom"), p.get("driver"), p.get("entraineur")
            won, pl = (1 if p.get("ordreArrivee") == 1 else 0), (1 if 1 <= (p.get("ordreArrivee") or 0) <= 3 else 0)
            if cheval: horse_stats["global"][cheval]["c"] += 1; horse_stats["global"][cheval]["v"] += won; horse_stats["global"][cheval]["p"] += pl
            if driver: team_stats["drivers"][driver]["c"] += 1; team_stats["drivers"][driver]["v"] += won; team_stats["drivers"][driver]["p"] += pl
            if entr: team_stats["entraineurs"][entr]["c"] += 1; team_stats["entraineurs"][entr]["v"] += won; team_stats["entraineurs"][entr]["p"] += pl
            pedigree_data.append({"pere": p.get("nomPere"), "mere": p.get("nomMere"), "place": p.get("ordreArrivee", 0)})

        if len(finishers) >= 2:
            for i, winner in enumerate(finishers):
                for loser in finishers[i+1:]:
                    rw, rl = elo[winner.get("nom")], elo[loser.get("nom")]
                    ew = 1 / (1 + 10 ** ((rl - rw) / 400))
                    elo[winner.get("nom")] += 16*(1-ew); elo[loser.get("nom")] -= 16*(1-ew)

    p_s, m_s = build_pedigree_stats(pedigree_data)
    res = (dict(team_stats), dict(horse_stats), dict(elo), {k: list(v) for k, v in elo_hist.items()}, dict(horse_races), {"peres": p_s, "meres": m_s})
    if is_today:
        for i, f in enumerate([STATS_CACHE_FILE, HORSE_STATS_FILE, ELO_CACHE_FILE, ELO_HIST_FILE, HORSE_RACES_FILE, PEDIGREE_FILE]): save_pickle(f, res[i])
    return res

def train_ml_model(days_back=21):
    bundle = safe_compute_stats()
    X, y = [], []
    today = datetime.now()
    for delta in range(1, days_back + 1):
        d = today - timedelta(days=delta)
        prog = get_programme(fmt_date(d))
        if not prog: continue
        for r in prog["programme"]["reunions"]:
            for c in r["courses"]:
                if c.get("arriveeDefinitive"):
                    parts = get_participants(fmt_date(d), r["numOfficiel"], c["numOrdre"])
                    ans = analyser_course_features(parts, None, bundle[0], bundle[1], bundle[2])
                    nb = len(ans)
                    if nb == 0: continue
                    avg = {"gains": sum(a["scores"]["gains"] for a in ans)/nb, "elo": sum(a["scores"]["elo"] for a in ans)/nb}
                    for a in ans:
                        X.append(featurize(a, nb, avg))
                        real = next((p for p in parts["participants"] if p.get("numPmu") == a["numPmu"]), None)
                        y.append(1 if real and real.get("ordreArrivee") == 1 else 0)
    if len(X) < 50: return None
    model = XGBoostLike(n_trees=100, max_depth=5)
    model.fit(X, y)
    save_pickle(ML_MODEL_FILE, model.to_dict())
    return {"n_samples": len(X), "model_type": "xgb", "trained_at": datetime.now().isoformat()}

# ============================================================
#  LOGIQUE ANALYSE COURSE
# ============================================================
def analyser_course_features(parts_data, perfs_data, team_stats, horse_stats, elo):
    parts = [p for p in parts_data.get("participants", []) if p.get("statut") == "PARTANT"]
    if not parts: return []
    perfs_by_n = {p.get("numPmu"): p.get("coursesCourues", []) for p in (perfs_data or {}).get("participants", [])}
    nb, all_h = len(parts), [p.get("nom") for p in parts]
    all_gains = [(p.get("gainsParticipant") or {}).get("gainsCarriere", 0) for p in parts]
    
    ans = []
    for i, p in enumerate(parts):
        num, ch, en = p.get("numPmu"), p.get("nom"), p.get("entraineur")
        gc = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0)
        s_f = score_forme_enrichi(perfs_by_n.get(num, []))
        s_g = min(100, 15 * math.log10(max(gc/1000, 1) + 1))
        
        ans.append({
            "numPmu": num, "nom": ch, "age": p.get("age"), "sexe": p.get("sexe"), "driver": p.get("driver") or "—", "entraineur": en or "—",
            "cote": float(p.get("dernierRapportDirect",{}).get("rapport") or 10), "probaMarche": 10, "gainsCarriere": gc//100, "ordreArrivee": p.get("ordreArrivee"),
            "scores": {
                "marche": 10, "forme": round(s_f, 1), "carriere": 50, "gains": round(s_g, 1), "driver": 50, "entraineur": 50, "distance": 50, "cheval_stats": 50, 
                "elo": get_elo_score(ch, elo, all_h), "age_sexe": 50, "repos": 50, "elo_trend": 50, "confrontation": 50, "pedigree": 50, "corde": 50, "equipment": 50, 
                "profile_match": 50, "musique": get_musique_score(p.get("musique")), "gains_relatifs": get_relative_gains_score(gc, all_gains), 
                "form_ecurie": get_form_ecurie_score(en, team_stats.get("entraineurs", {}))
            },
            "bonus": {"team": 0, "deferre": 0}, "profile": {"fragile": 0, "repere": 0, "prepare": 0}
        })
    return ans

def analyser_course(parts_data, perfs_data, bundle, use_ml=False, capital=100):
    ans = analyser_course_features(parts_data, perfs_data, bundle[0], bundle[1], bundle[2])
    if not ans: return []
    for a in ans:
        a["chanceHeur"] = round(sum([a["scores"][k]*w for k,w in {"forme":0.2,"gains":0.1,"elo":0.3,"musique":0.2,"form_ecurie":0.2}.items()]), 2)
    
    t_h = sum(a["chanceHeur"] for a in ans) or 1
    for a in ans: a["chanceHeur"] = round(a["chanceHeur"]/t_h*100, 2)
    for a in ans: a["chance"] = a["chanceHeur"] # Par défaut

    ans.sort(key=lambda x: -x["chance"])
    for r, a in enumerate(ans, 1): a["rang"] = r
    pl3 = proba_place_simple([a["chance"] for a in ans], 3, len(ans))
    for i, a in enumerate(ans):
        a["chancePlace3"] = round(pl3[i], 2)
        is_v = a["chance"] > 15 and a["cote"] >= 4
        a["valueBet"], a["isGold"], a["isCoupSur"] = is_v, (is_v and a["scores"]["form_ecurie"] > 60), (pl3[i] >= 65 and a["rang"] == 1 and a["scores"]["forme"] > 80)
        a["kellyMise"] = kelly_amount(a["chance"]/100, a["cote"], capital, 0.25)
    return ans

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
        if not prog: return jsonify({"reunions": []})
        out = [{"numReunion": r["numOfficiel"], "hippodrome": r["hippodrome"]["libelleCourt"], "courses": [{"numCourse": c["numOrdre"], "libelle": c.get("libelle") or c.get("libelleCourt"), "heure": "00:00", "nbPartants": c.get("nombreDeclaresPartants"), "arriveeDefinitive": c.get("arriveeDefinitive", False)} for c in r["courses"]]} for r in prog["programme"]["reunions"]]
        return jsonify({"date": d, "reunions": out})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/course/<int:r_num>/<int:c_num>")
def api_course(r_num, c_num):
    d_str = request.args.get("date") or fmt_date(datetime.now())
    use_ml, cap = request.args.get("ml")=="1", float(request.args.get("capital", 100))
    try:
        parts, perfs = get_participants(d_str, r_num, c_num), get_performances(d_str, r_num, c_num)
        ans = analyser_course(parts, perfs, safe_compute_stats())
        return jsonify({"date": d_str, "course": {"libelle": "Course R"+str(r_num)+"C"+str(c_num), "heure": "00:00"}, "analyses": ans, "ml_active": False})
    except Exception as e: return jsonify({"error": str(e)}), 500

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

@app.route("/api/train", methods=["POST"])
@admin_required
def api_train():
    res = train_ml_model()
    return jsonify(res or {"error": "Échec entraînement"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
