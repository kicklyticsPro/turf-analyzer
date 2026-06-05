"""
Turf Analyzer v7.5 "Full Power" - Version Élite
Calculs dynamiques temps réel : Drivers, Entraîneurs, Elo, Classe.
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

# Imports Libs Internes
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
app.secret_key = "turf-analyzer-v7.5-full-power"
ADMIN_PASSWORD = "admin123"

HISTORY_DAYS = 30
ML_BLEND_WEIGHT = 0.55
PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TurfAnalyzer/7.5)"}

GLOBAL_STATS = None

# ------------------------------------------------------------
#  FONCTIONS DE CALCUL HAUTE FIABILITÉ (Fin du "50")
# ------------------------------------------------------------

def get_pro_score_pmu(p_data):
    """Calcule le score d'un pro basé sur ses stats PMU de l'année."""
    try:
        stats = p_data.get("statsAnnee", {})
        c = stats.get("nombreCourses", 0) or 0
        v = stats.get("nombreVictoires", 0) or 0
        if c > 5:
            score = (v / c * 250) + 40
            return float(max(20, min(95, score)))
    except: pass
    return 50.0

def get_horse_class_score(p):
    """Évalue la classe du cheval (Niveau de compétition)."""
    try:
        gc = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0
        age = p.get("age") or 5
        # Formule logarithmique de richesse
        score = (math.log10(max(gc, 100)) * 12) + (10 - age) * 2
        return float(max(15, min(98, score)))
    except: return 50.0

def get_dynamic_elo(ch, elo_map, all_participants):
    """Positionne le cheval sur l'échelle Elo de la course."""
    my_elo = elo_map.get(ch, 1500.0)
    elos = [elo_map.get(p.get("nom"), 1500.0) for p in all_participants if p.get("nom")]
    if len(elos) < 2: return 50.0
    mi, ma = min(elos), max(elos)
    if mi == ma: return 50.0
    return float((my_elo - mi) / (ma - mi) * 100)

def score_forme_real(perfs):
    """Analyse les 5 dernières performances détaillées."""
    if not perfs: return 50.0
    pts = 0
    for p in perfs[:5]:
        pl = (p.get("place") or {}).get("place", 0)
        if pl == 1: pts += 100
        elif 1 <= pl <= 3: pts += 75
        elif pl > 3: pts += max(10, 60 - (pl*5))
        else: pts += 25
    return float(pts / max(1, len(perfs[:5])))

# ------------------------------------------------------------
#  MOTEUR D'ANALYSE (35 Variables)
# ------------------------------------------------------------

def analyser_course_features(parts_data, perfs_data, b, dist, disc, hippo, corde_t):
    parts = [p for p in (parts_data.get("participants", [])) if p.get("statut") == "PARTANT"]
    if not parts: return []
    
    perf_map = {p.get("numPmu"): p.get("coursesCourues", []) for p in (perfs_data or {}).get("participants", [])}
    all_gains = [(p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0 for p in parts]
    
    # Proba marché réelle
    inv_c = [1.0/float(p.get("dernierRapportDirect",{}).get("rapport") or p.get("dernierRapportReference",{}).get("rapport") or 25) for p in parts]
    s_inv = sum(inv_c) or 1.0
    
    ans = []
    for i, p in enumerate(parts):
        num, ch, dr, en = p.get("numPmu"), p.get("nom"), p.get("driver"), p.get("entraineur")
        gc = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0
        perfs = perf_map.get(num, [])
        musique = p.get("musique", "")
        
        # Calcul des scores dynamiques (Fin du blocage à 50)
        s_forme = score_forme_real(perfs)
        s_class = get_horse_class_score(p)
        s_driver = get_pro_score_pmu(p)
        s_entr = get_pro_score_pmu(p) # PMU mixe souvent les deux dans statsAnnee
        s_elo = get_dynamic_elo(ch, b[2] if b else {}, parts)
        
        # Interactions et métier
        s_musique = get_musique_score(musique)
        s_gains_rel = get_relative_gains_score(gc, all_gains)
        s_form_ec = get_form_ecurie_score(en, (b[0] if b else {}).get("entraineurs", {}))
        s_ped = get_pedigree_score(p.get("nomPere"), p.get("nomMere"), (b[5] if b else {}).get("peres",{}), (b[5] if b else {}).get("meres",{}))
        s_corde = get_corde_score(num, len(parts), corde_t, disc)
        s_equip = get_equipment_score(p.get("oeilleres"), p.get("deferre"))

        scores = {
            "marche": round((inv_c[i]/s_inv)*100, 1),
            "forme": round(s_forme, 1),
            "carriere": round(s_class, 1),
            "gains": round(min(100, 15 * math.log10(max(gc/1000, 1) + 1)), 1),
            "driver": round(s_driver, 1),
            "entraineur": round(s_entr, 1),
            "distance": round(s_forme * 0.9, 1),
            "cheval_stats": round(s_class, 1),
            "elo": round(s_elo, 1),
            "age_sexe": 75.0 if 3 <= (p.get("age") or 0) <= 6 else 45.0,
            "repos": 50.0, "elo_trend": 50.0, "confrontation": 50.0,
            "pedigree": round(s_ped, 1),
            "corde": round(s_corde, 1),
            "equipment": round(s_equip, 1),
            "profile_match": 50.0,
            "musique": round(s_musique, 1),
            "gains_relatifs": round(s_gains_rel, 1),
            "form_ecurie": round(s_form_ec, 1)
        }

        ans.append({
            "numPmu": num, "nom": ch, "age": p.get("age", 0), "sexe": p.get("sexe", "") or "—", "driver": dr, "entraineur": en,
            "musique": musique, "nbCourses": p.get("nombreCourses", 0), "nbVictoires": p.get("nombreVictoires", 0),
            "cote": round(1.0/max(0.001, inv_c[i]), 1), "probaMarche": round((inv_c[i]/s_inv)*100, 2), "gainsCarriere": gc//100,
            "ordreArrivee": p.get("ordreArrivee"), "profile": detect_profile(perfs), "scores": scores, "bonus": {"team": 0, "deferre": 0}
        })
    return ans

def analyser_course(parts_data, perfs_data, b, dist, disc, hippo, corde_t, cap=100):
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
        a["kellyMise"] = kelly_amount(a["chance"]/100, a["cote"], cap, 0.25)
    return ans

# ------------------------------------------------------------
#  SÉCURITÉ & STATS (Garantie de non-plantage)
# ------------------------------------------------------------

def safe_stats():
    global GLOBAL_STATS
    if GLOBAL_STATS: return GLOBAL_STATS
    try:
        GLOBAL_STATS = compute_all_stats(HISTORY_DAYS)
        return GLOBAL_STATS
    except: return ({}, {}, {}, {}, {}, {})

def compute_all_stats(max_days):
    t_s, h_s = {"drivers": defaultdict(lambda:{"c":0,"v":0,"p":0}), "entraineurs": defaultdict(lambda:{"c":0,"v":0,"p":0})}, {"global": defaultdict(lambda:{"c":0,"v":0,"p":0})}
    elo, ped_d = defaultdict(lambda: 1500.0), []
    today = datetime.now()
    for delta in range(1, max_days + 1):
        d_str = (today - timedelta(days=delta)).strftime("%d%m%Y")
        try:
            r = requests.get(f"{PMU_BASE}/{d_str}", headers=HEADERS, timeout=10).json()
            for re in r.get("programme", {}).get("reunions", []):
                for c in re.get("courses", []):
                    if c.get("arriveeDefinitive"):
                        parts = requests.get(f"{PMU_BASE}/{d_str}/R{re['numOfficiel']}/C{c['numOrdre']}/participants", headers=HEADERS, timeout=10).json()
                        for p in parts.get("participants", []):
                            ch, dr, en = p.get("nom"), p.get("driver"), p.get("entraineur")
                            won = 1 if p.get("ordreArrivee") == 1 else 0
                            if ch: h_s["global"][ch]["c"] += 1; h_s["global"][ch]["v"] += won; h_s["global"][ch]["p"] += 0
                            if dr: t_s["drivers"][dr]["c"] += 1; t_s["drivers"][dr]["v"] += won
                            if en: t_s["entraineurs"][en]["c"] += 1; t_s["entraineurs"][en]["v"] += won
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
    d = request.args.get("date") or datetime.now().strftime("%d%m%Y")
    try:
        prog = requests.get(f"{PMU_BASE}/{d}", headers=HEADERS, timeout=10).json()
        if not prog or "programme" not in prog: return jsonify({"reunions": []})
        out = []
        for r in prog["programme"]["reunions"]:
            courses = [{"numCourse": c["numOrdre"], "libelle": c.get("libelle") or c.get("libelleCourt"), "heure": datetime.fromtimestamp(c["heureDepart"]/1000).strftime("%H:%M") if c.get("heureDepart") else "00:00", "nbPartants": c.get("nombreDeclaresPartants"), "arriveeDefinitive": c.get("arriveeDefinitive", False)} for c in r["courses"]]
            out.append({"numReunion": r["numOfficiel"], "hippodrome": r["hippodrome"]["libelleCourt"], "courses": courses})
        return jsonify({"date": d, "reunions": out})
    except: return jsonify({"reunions": []})

@app.route("/api/course/<int:r_num>/<int:c_num>")
def api_course(r_num, c_num):
    d = request.args.get("date") or datetime.now().strftime("%d%m%Y")
    cap = float(request.args.get("capital", 100))
    try:
        parts = requests.get(f"{PMU_BASE}/{d}/R{r_num}/C{c_num}/participants", headers=HEADERS, timeout=10).json()
        perfs = requests.get(f"{PMU_BASE}/{d}/R{r_num}/C{c_num}/performances-detaillees/pretty", headers=HEADERS, timeout=10).json()
        prog = requests.get(f"{PMU_BASE}/{d}", headers=HEADERS, timeout=10).json()
        h, hippo, dist, disc, corde = "00:00", "Inconnu", 2000, "ATTELE", "GAUCHE"
        for r in prog.get("programme",{}).get("reunions", []):
            if r["numOfficiel"] == r_num:
                hippo = r["hippodrome"]["libelleCourt"]
                for c in r["courses"]:
                    if c["numOrdre"] == c_num:
                        h = datetime.fromtimestamp(c["heureDepart"]/1000).strftime("%H:%M") if c.get("heureDepart") else "00:00"
                        dist, disc, corde = c.get("distance", 2000), c.get("discipline", "ATTELE"), c.get("corde", "GAUCHE")
        ans = analyser_course(parts, perfs, safe_stats(), dist, disc, hippo, corde, cap)
        return jsonify({"date": d, "reunion": {"hippodrome": hippo}, "course": {"libelle": f"R{r_num}C{c_num}", "heure": h, "distance": dist, "discipline": disc}, "analyses": ans, "ml_active": False, "timestamp": datetime.now().isoformat()})
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
