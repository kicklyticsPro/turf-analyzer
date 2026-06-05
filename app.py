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
app.secret_key = "turf-secret-pro-v7.2-final-complete-ultra"
ADMIN_PASSWORD = "admin123"

HISTORY_DAYS = 30 
ML_BLEND_WEIGHT = 0.55
PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TurfAnalyzer/7.0)"}

# Pre-chargement du bundle de stats pour éviter de tout recalculer à chaque requête
GLOBAL_STATS_BUNDLE = None

# ------------------------------------------------------------
#  FONCTIONS DE CALCUL & SCORING
# ------------------------------------------------------------
def fmt_date(d): return d.strftime("%d%m%Y")

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"): return redirect(url_for("login_page", next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def score_forme_enrichi(perfs):
    if not perfs: return 50.0
    pts = 0
    v = perfs[:5]
    for p in v:
        pl = (p.get("place") or {}).get("place", 0)
        if pl == 1: pts += 100
        elif 1 <= pl <= 3: pts += 75
        else: pts += 30
    return float(pts / len(v))

def get_elo_score(ch, elo, all_h):
    if not elo: return 50.0
    my = elo.get(ch, 1500.0)
    elos = [elo.get(h, 1500.0) for h in all_h if h]
    if len(elos) < 2: return 50.0
    e_min, e_max = min(elos), max(elos)
    if e_max == e_min: return 50.0
    return float((my - e_min) / (e_max - e_min) * 100)

def safe_compute_stats():
    global GLOBAL_STATS_BUNDLE
    if GLOBAL_STATS_BUNDLE:
        return GLOBAL_STATS_BUNDLE
    try:
        GLOBAL_STATS_BUNDLE = compute_all_stats(HISTORY_DAYS)
        return GLOBAL_STATS_BUNDLE
    except:
        return ({}, {}, {}, {}, {}, {})

# ------------------------------------------------------------
#  ANALYSE COURSE (Zéro Undefined Garanti)
# ------------------------------------------------------------
def analyser_course_features(parts_data, perfs_data, b, dist, disc, hippo, corde_t):
    # Sécurisation des données d'entrée
    raw_participants = (parts_data or {}).get("participants", [])
    parts = [p for p in raw_participants if p.get("statut") == "PARTANT"]
    if not parts: return []
    
    perf_map = {p.get("numPmu"): p.get("coursesCourues", []) for p in (perfs_data or {}).get("participants", [])}
    all_h = [p.get("nom") for p in parts if p.get("nom")]
    all_g = [(p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0 for p in parts]
    
    ans = []
    for i, p in enumerate(parts):
        # Sécurisation de chaque variable pour éviter le 'undefined'
        num = p.get("numPmu", 0)
        ch = p.get("nom", "Inconnu")
        dr = p.get("driver", "—")
        en = p.get("entraineur", "—")
        gc = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0
        perfs = perf_map.get(num, [])
        musique = p.get("musique", "") or "—"
        
        # Calculs sécurisés
        s_forme = score_forme_enrichi(perfs)
        s_gains = min(100.0, 15 * math.log10(max(gc/1000, 1) + 1))
        s_elo = get_elo_score(ch, b[2] if len(b) > 2 else {}, all_h)
        s_musique = get_musique_score(musique)
        s_gains_rel = get_relative_gains_score(gc, all_g)
        s_form_ec = get_form_ecurie_score(en, (b[0] if b else {}).get("entraineurs", {}))
        
        # Valeurs par défaut pour les scores restants
        s_ped = get_pedigree_score(p.get("nomPere"), p.get("nomMere"), (b[5] if len(b)>5 else {}).get("peres",{}), (b[5] if len(b)>5 else {}).get("meres",{})) if b else 50.0
        s_corde = get_corde_score(num, len(parts), corde_t, disc)
        s_equip = get_equipment_score(p.get("oeilleres"), p.get("deferre"))
        prof = detect_profile(perfs)
        s_prof_match = get_profile_match_score(prof, dist, len(parts))

        ans.append({
            "numPmu": num,
            "nom": ch,
            "age": p.get("age", 0) or "—",
            "sexe": p.get("sexe", "") or "—",
            "driver": dr,
            "entraineur": en,
            "musique": musique,
            "nbCourses": p.get("nombreCourses", 0) or 0,
            "nbVictoires": p.get("nombreVictoires", 0) or 0,
            "nbPlaces": p.get("nombrePlaces", 0) or 0,
            "cote": float(p.get("dernierRapportDirect",{}).get("rapport") or p.get("dernierRapportReference",{}).get("rapport") or 10.0),
            "probaMarche": 10.0,
            "gainsCarriere": gc // 100,
            "ordreArrivee": p.get("ordreArrivee") or None,
            "profile": prof,
            "scores": {
                "marche": 10.0, "forme": round(float(s_forme), 1), "carriere": 50.0, "gains": round(float(s_gains), 1),
                "driver": 50.0, "entraineur": 50.0, "distance": 50.0, "cheval_stats": 50.0, "elo": round(float(s_elo), 1),
                "age_sexe": 50.0, "repos": 50.0, "elo_trend": 50.0, "confrontation": 50.0,
                "pedigree": round(float(s_ped), 1), "corde": round(float(s_corde), 1), "equipment": round(float(s_equip), 1),
                "profile_match": round(float(s_prof_match), 1), "musique": round(float(s_musique), 1),
                "gains_relatifs": round(float(s_gains_rel), 1), "form_ecurie": round(float(s_form_ec), 1)
            },
            "bonus": {"team": 0, "deferre": 0}
        })
    return ans

def analyser_course(parts_data, perfs_data, b, dist, disc, hippo, corde_t, capital=100):
    ans = analyser_course_features(parts_data, perfs_data, b, dist, disc, hippo, corde_t)
    if not ans: return []
    for a in ans:
        a["chance"] = a["chanceHeur"] = round(sum([a["scores"][k]*w for k,w in {"forme":0.2,"gains":0.1,"elo":0.2,"musique":0.15,"form_ecurie":0.15,"pedigree":0.1,"corde":0.1}.items()]), 2)
    
    t_h = sum(a["chanceHeur"] for a in ans) or 1
    for a in ans: a["chance"] = a["chanceHeur"] = round(a["chanceHeur"]/t_h*100, 2)
    ans.sort(key=lambda x: -x["chance"])
    for r, a in enumerate(ans, 1): a["rang"] = r
    
    pl3 = proba_place_simple([a["chance"] for a in ans], 3, len(ans))
    for i, a in enumerate(ans):
        a["chancePlace3"] = round(pl3[i], 2)
        is_val = a["chance"] > 15 and a["cote"] >= 4
        a["valueBet"] = is_val
        a["isGold"] = is_val and a["scores"]["form_ecurie"] > 60
        a["isCoupSur"] = a["chancePlace3"] >= 65 and a["rang"] == 1
        a["kellyMise"] = kelly_amount(a["chance"]/100, a["cote"], capital, 0.25)
    return ans

# ------------------------------------------------------------
#  API & STATS CORE
# ------------------------------------------------------------
def get_programme(d):
    try: return requests.get(f"{PMU_BASE}/{d}", headers=HEADERS, timeout=10).json()
    except: return None

def get_participants(d, r, c):
    try: return requests.get(f"{PMU_BASE}/{d}/R{r}/C{c}/participants", headers=HEADERS, timeout=10).json()
    except: return {"participants": []}

def get_performances(d, r, c):
    try: return requests.get(f"{PMU_BASE}/{d}/R{r}/C{c}/performances-detaillees/pretty", headers=HEADERS, timeout=10).json()
    except: return {"participants": []}

def compute_all_stats(max_days):
    t_s, h_s = {"drivers": defaultdict(lambda: {"c":0,"v":0,"p":0}), "entraineurs": defaultdict(lambda: {"c":0,"v":0,"p":0})}, {"global": defaultdict(lambda: {"c":0,"v":0,"p":0})}
    elo, ped_d = defaultdict(lambda: 1500.0), []
    today = datetime.now()
    for delta in range(1, max_days + 1):
        d_str = fmt_date(today - timedelta(days=delta))
        prog = get_programme(d_str)
        if not prog: continue
        for r in prog["programme"]["reunions"]:
            for c in r["courses"]:
                if not c.get("arriveeDefinitive"): continue
                try:
                    parts = get_participants(d_str, r["numOfficiel"], c["numOrdre"])
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
        if request.form.get("password") == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("home"))
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
        out = []
        for r in prog["programme"]["reunions"]:
            courses = [{"numCourse": c["numOrdre"], "libelle": c.get("libelle") or c.get("libelleCourt"), "heure": datetime.fromtimestamp(c["heureDepart"]/1000).strftime("%H:%M") if c.get("heureDepart") else "00:00", "nbPartants": c.get("nombreDeclaresPartants"), "arriveeDefinitive": c.get("arriveeDefinitive", False)} for c in r["courses"]]
            out.append({"numReunion": r["numOfficiel"], "hippodrome": r["hippodrome"]["libelleCourt"], "courses": courses})
        return jsonify({"date": d, "reunions": out})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/course/<int:r_num>/<int:c_num>")
def api_course(r_num, c_num):
    d = request.args.get("date") or fmt_date(datetime.now())
    cap = float(request.args.get("capital", 100))
    try:
        prog = get_programme(d)
        parts, perfs = get_participants(d, r_num, c_num), get_performances(d, r_num, c_num)
        h, hippo, dist, disc, corde = "00:00", "Inconnu", 2000, "ATTELE", "GAUCHE"
        if prog:
            for r in prog["programme"]["reunions"]:
                if r["numOfficiel"] == r_num:
                    hippo = r["hippodrome"]["libelleCourt"]
                    for c in r["courses"]:
                        if c["numOrdre"] == c_num:
                            h = datetime.fromtimestamp(c["heureDepart"]/1000).strftime("%H:%M") if c.get("heureDepart") else "00:00"
                            dist, disc, corde = c.get("distance", 2000), c.get("discipline", "ATTELE"), c.get("corde", "GAUCHE")
        
        ans = analyser_course(parts, perfs, safe_compute_stats(), dist, disc, hippo, corde, cap)
        return jsonify({"date": d, "reunion": {"hippodrome": hippo}, "course": {"libelle": f"R{r_num}C{c_num}", "heure": h, "distance": dist, "discipline": disc}, "analyses": ans, "ml_active": False, "timestamp": datetime.now().isoformat()})
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
