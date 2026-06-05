from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from datetime import datetime, timedelta
import requests, math, os, pickle
from functools import lru_cache, wraps
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

# Imports Libs
from lib.ml_models import (GradientBoosting, RandomForest, Ensemble, load_model_from_dict)
from lib.xgb_like import XGBoostLike
from lib.kelly import kelly_amount, kelly_fraction, expected_roi
from lib.features_v5 import (build_pedigree_stats, get_pedigree_score, get_corde_score, 
                              get_equipment_score, detect_profile, get_profile_match_score,
                              get_musique_score, get_relative_gains_score, get_form_ecurie_score,
                              get_days_since_last_race)
from lib.multi_paris import proba_place_simple
from lib.calibration import Calibrator
from lib import db, telegram_bot

app = Flask(__name__)
app.secret_key = "turf-analyzer-pro-ultra-v7.3"
ADMIN_PASSWORD = "admin123"

HISTORY_DAYS = 30
ML_BLEND_WEIGHT = 0.55
PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Cache global en mémoire pour les performances
GLOBAL_STATS = None

# ------------------------------------------------------------
#  FONCTIONS DE SCORING DYNAMIQUES
# ------------------------------------------------------------
def get_bucket_score(bucket, max_s=100, min_c=3):
    if not bucket or bucket.get("c", 0) < min_c: return 50.0
    tv, tp = bucket["v"]/bucket["c"], bucket["p"]/bucket["c"]
    conf = min(1.0, bucket["c"]/20)
    raw = (tv * 250) + (tp * 80)
    return float(min(max_s, raw * conf + 45 * (1 - conf)))

def score_forme_pro(perfs):
    if not perfs: return 50.0
    pts = 0
    valid_perfs = perfs[:5]
    for p in valid_perfs:
        pl = (p.get("place") or {}).get("place", 0)
        if pl == 1: pts += 100
        elif 1 <= pl <= 3: pts += 75
        elif pl > 0: pts += max(10, 60 - (pl*5))
        else: pts += 25
    return float(pts / max(1, len(valid_perfs)))

def get_elo_dyn(ch, elo, all_h):
    if not elo: return 50.0
    my = elo.get(ch, 1500.0)
    elos = [elo.get(h, 1500.0) for h in all_h if h]
    if len(elos) < 2: return 50.0
    mi, ma = min(elos), max(elos)
    if mi == ma: return 50.0
    return float((my - mi) / (ma - mi) * 100)

def score_dist_dyn(perfs, target_dist):
    if not perfs or not target_dist: return 50.0
    proches = [p for p in perfs if abs(p.get("distance", target_dist) - target_dist) < 300]
    if not proches: return 50.0
    return score_forme_pro(proches)

# ------------------------------------------------------------
#  SÉCURITÉ ET STATS
# ------------------------------------------------------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"): return redirect(url_for("login_page", next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def safe_stats():
    global GLOBAL_STATS
    if GLOBAL_STATS: return GLOBAL_STATS
    try:
        GLOBAL_STATS = compute_all_stats(HISTORY_DAYS)
        return GLOBAL_STATS
    except: return ({}, {}, {}, {}, {}, {})

def fmt_date(d): return d.strftime("%d%m%Y")

def get_pmu(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        return r.json()
    except: return {}

# ------------------------------------------------------------
#  ANALYSE COURSE (Données dynamiques réelles)
# ------------------------------------------------------------
def analyser_course_features(parts_data, perfs_data, b, dist, disc, hippo, corde_t):
    raw = (parts_data or {}).get("participants", [])
    parts = [p for p in raw if p.get("statut") == "PARTANT"]
    if not parts: return []
    perf_map = {p.get("numPmu"): p.get("coursesCourues", []) for p in (perfs_data or {}).get("participants", [])}
    all_h = [p.get("nom") for p in parts if p.get("nom")]
    all_g = [(p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0 for p in parts]
    
    # Calcul Proba Marché
    inv_c = [1.0/float(p.get("dernierRapportDirect",{}).get("rapport") or p.get("dernierRapportReference",{}).get("rapport") or 50) for p in parts]
    s_inv = sum(inv_c) or 1.0
    
    ans = []
    for i, p in enumerate(parts):
        num, ch, dr, en = p.get("numPmu", 0), p.get("nom", "Inconnu"), p.get("driver", "—"), p.get("entraineur", "—")
        gc = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0
        perfs = perf_map.get(num, [])
        
        scores = {
            "marche": round((inv_c[i]/s_inv)*100, 1),
            "forme": round(score_forme_pro(perfs), 1),
            "carriere": round(get_bucket_score(b[1].get("global", {}).get(ch)), 1),
            "gains": round(min(100, 15 * math.log10(max(gc/1000, 1) + 1)), 1),
            "driver": round(get_bucket_score(b[0].get("drivers", {}).get(dr)), 1),
            "entraineur": round(get_bucket_score(b[0].get("entraineurs", {}).get(en)), 1),
            "distance": round(score_dist_dyn(perfs, dist), 1),
            "cheval_stats": round(get_bucket_score(b[1].get("global", {}).get(ch)), 1),
            "elo": round(get_elo_dyn(ch, b[2], all_h), 1),
            "age_sexe": 65.0 if 3 <= (p.get("age") or 0) <= 6 else 45.0,
            "repos": 50.0, "elo_trend": 50.0, "confrontation": 50.0,
            "pedigree": round(get_pedigree_score(p.get("nomPere"), p.get("nomMere"), b[5].get("peres",{}), b[5].get("meres",{})), 1),
            "corde": round(get_corde_score(num, len(parts), corde_t, disc), 1),
            "equipment": round(get_equipment_score(p.get("oeilleres"), p.get("deferre")), 1),
            "profile_match": round(get_profile_match_score(detect_profile(perfs), dist, len(parts)), 1),
            "musique": round(get_musique_score(p.get("musique")), 1),
            "gains_relatifs": round(get_relative_gains_score(gc, all_g), 1),
            "form_ecurie": round(get_form_ecurie_score(en, b[0].get("entraineurs", {})), 1)
        }

        ans.append({
            "numPmu": num, "nom": ch, "age": p.get("age", 0), "sexe": p.get("sexe", "") or "—", "driver": dr, "entraineur": en,
            "musique": p.get("musique", ""), "nbCourses": p.get("nombreCourses", 0), "nbVictoires": p.get("nombreVictoires", 0), "nbPlaces": p.get("nombrePlaces", 0),
            "cote": round(1.0/max(0.001, inv_c[i]), 1), "probaMarche": round((inv_c[i]/s_inv)*100, 2), "gainsCarriere": gc//100,
            "ordreArrivee": p.get("ordreArrivee"), "profile": detect_profile(perfs), "scores": scores, "bonus": {"team": 0, "deferre": 0}
        })
    return ans

def analyser_course(parts_data, perfs_data, b, dist, disc, hippo, corde_t, capital=100):
    ans = analyser_course_features(parts_data, perfs_data, b, dist, disc, hippo, corde_t)
    if not ans: return []
    for a in ans:
        a["chance"] = a["chanceHeur"] = round(sum([a["scores"][k]*w for k,w in {"forme":0.2,"gains":0.1,"elo":0.25,"musique":0.15,"form_ecurie":0.15,"pedigree":0.07,"corde":0.08}.items()]), 2)
    
    t_h = sum(a["chanceHeur"] for a in ans) or 1
    for a in ans: a["chance"] = a["chanceHeur"] = round(a["chanceHeur"]/t_h*100, 2)
    ans.sort(key=lambda x: -x["chance"])
    for r, a in enumerate(ans, 1): a["rang"] = r
    
    pl3 = proba_place_simple([a["chance"] for a in ans], 3, len(ans))
    for i, a in enumerate(ans):
        a["chancePlace3"] = round(pl3[i], 2)
        a["edge"] = round(a["chance"] - a["probaMarche"], 2)
        is_v = a["edge"] > 4 and a["cote"] >= 4
        a["valueBet"], a["isGold"], a["isCoupSur"] = is_v, (is_v and a["scores"]["form_ecurie"] > 60), (pl3[i] >= 65 and a["rang"] == 1)
        a["kellyMise"] = kelly_amount(a["chance"]/100, a["cote"], capital, 0.25)
    return ans

# ------------------------------------------------------------
#  STATS COMPUTE (30j)
# ------------------------------------------------------------
def compute_all_stats(max_days):
    t_s, h_s = {"drivers": defaultdict(_empty_bucket), "entraineurs": defaultdict(_empty_bucket)}, {"global": defaultdict(_empty_bucket)}
    elo, ped_d = defaultdict(lambda: 1500.0), []
    today = datetime.now()
    for delta in range(1, max_days + 1):
        d_str = fmt_date(today - timedelta(days=delta))
        prog = get_pmu(f"{PMU_BASE}/{d_str}")
        if not prog or "programme" not in prog: continue
        for r in prog["programme"].get("reunions", []):
            for c in r.get("courses", []):
                if not c.get("arriveeDefinitive"): continue
                try:
                    parts = get_pmu(f"{PMU_BASE}/{d_str}/R{r['numOfficiel']}/C{c['numOrdre']}/participants")
                    for p in parts.get("participants", []):
                        ch, dr, en = p.get("nom"), p.get("driver"), p.get("entraineur")
                        won, pl = (1 if p.get("ordreArrivee") == 1 else 0), (1 if 1 <= (p.get("ordreArrivee") or 0) <= 3 else 0)
                        if ch: h_s["global"][ch]["c"] += 1; h_s["global"][ch]["v"] += won; h_s["global"][ch]["p"] += pl
                        if dr: t_s["drivers"][dr]["c"] += 1; t_s["drivers"][dr]["v"] += won; t_s["drivers"][dr]["p"] += pl
                        if en: t_s["entraineurs"][en]["c"] += 1; t_s["entraineurs"][en]["v"] += won; t_s["entraineurs"][en]["p"] += pl
                        ped_d.append({"pere": p.get("nomPere"), "mere": p.get("nomMere"), "place": p.get("ordreArrivee", 0)})
                except: continue
    p_s, m_s = build_pedigree_stats(ped_d)
    return (dict(t_s), dict(h_s), dict(elo), {}, {}, {"peres": p_s, "meres": m_s})

# ------------------------------------------------------------
#  ROUTES FLASK
# ------------------------------------------------------------
@app.route("/")
def home(): return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD: session["logged_in"] = True; return redirect(url_for("home"))
        return render_template("login.html", error="Incorrect")
    return render_template("login.html")

@app.route("/logout")
def logout(): session.pop("logged_in", None); return redirect(url_for("home"))

@app.route("/api/reunions")
def api_reunions():
    d = request.args.get("date") or fmt_date(datetime.now())
    prog = get_pmu(f"{PMU_BASE}/{d}")
    if not prog or "programme" not in prog: return jsonify({"reunions": []})
    out = []
    for r in prog["programme"].get("reunions", []):
        courses = []
        for c in r.get("courses", []):
            h = datetime.fromtimestamp(c["heureDepart"]/1000).strftime("%H:%M") if c.get("heureDepart") else "00:00"
            courses.append({"numCourse": c["numOrdre"], "libelle": c.get("libelle") or c.get("libelleCourt"), "heure": h, "nbPartants": c.get("nombreDeclaresPartants"), "arriveeDefinitive": c.get("arriveeDefinitive", False)})
        out.append({"numReunion": r["numOfficiel"], "hippodrome": r["hippodrome"]["libelleCourt"], "courses": courses})
    return jsonify({"date": d, "reunions": out})

@app.route("/api/course/<int:r_num>/<int:c_num>")
def api_course(r_num, c_num):
    d = request.args.get("date") or fmt_date(datetime.now())
    try:
        parts, perfs = get_pmu(f"{PMU_BASE}/{d}/R{r_num}/C{c_num}/participants"), get_pmu(f"{PMU_BASE}/{d}/R{r_num}/C{c_num}/performances-detaillees/pretty")
        prog = get_pmu(f"{PMU_BASE}/{d}")
        h, hippo, dist, disc, corde = "00:00", "Inconnu", 2000, "ATTELE", "GAUCHE"
        if "programme" in prog:
            for r in prog["programme"].get("reunions", []):
                if r["numOfficiel"] == r_num:
                    hippo = r["hippodrome"]["libelleCourt"]
                    for c in r["courses"]:
                        if c["numOrdre"] == c_num:
                            h = datetime.fromtimestamp(c["heureDepart"]/1000).strftime("%H:%M") if c.get("heureDepart") else "00:00"
                            dist, disc, corde = c.get("distance", 2000), c.get("discipline", "ATTELE"), c.get("corde", "GAUCHE")
        ans = analyser_course(parts, perfs, safe_stats(), dist, disc, hippo, corde, float(request.args.get("capital", 100)))
        return jsonify({"date": d, "reunion": {"hippodrome": hippo}, "course": {"libelle": f"R{r_num}C{c_num}", "heure": h, "distance": dist, "discipline": disc}, "analyses": ans, "ml_active": False, "timestamp": datetime.now().isoformat()})
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__": app.run(host="0.0.0.0", port=5000)
