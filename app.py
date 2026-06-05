from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from datetime import datetime, timedelta
import requests, math, os, pickle
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
app.secret_key = "turf-analyzer-pro-v7.8-ultimate-verified"
ADMIN_PASSWORD = "admin123"

HISTORY_DAYS = 30
ML_BLEND_WEIGHT = 0.55
PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

GLOBAL_STATS = None

def fmt_date(d): return d.strftime("%d%m%Y")

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"): return redirect(url_for("login_page", next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# --- CALCULS DE SCORING AVANCÉS ---
def score_forme_enrichi(perfs):
    if not perfs: return 50.0
    pts = 0
    valid_p = perfs[:5]
    for p in valid_p:
        pl = (p.get("place") or {}).get("place", 0)
        if pl == 1: pts += 100
        elif 1 <= pl <= 3: pts += 75
        elif pl > 3: pts += max(10, 60 - (pl*5))
        else: pts += 25
    return float(pts / len(valid_p))

def get_horse_class_pmu(p):
    try:
        gc = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 100
        age = p.get("age", 5)
        return float(max(15, min(98, (math.log10(max(1,gc))*12) + (10-age)*2)))
    except: return 50.0

def safe_compute_stats():
    global GLOBAL_STATS
    if GLOBAL_STATS: return GLOBAL_STATS
    try:
        GLOBAL_STATS = compute_all_stats(HISTORY_DAYS)
        return GLOBAL_STATS
    except: return ({}, {}, {}, {}, {}, {})

# --- MOTEUR D'ANALYSE (ZÉRO UNDEFINED) ---
def analyser_course_features(parts_data, perfs_data, b, dist, disc, hippo, corde_t):
    raw = (parts_data or {}).get("participants", [])
    parts = [p for p in raw if p.get("statut") == "PARTANT"]
    if not parts: return []
    perf_map = {p.get("numPmu"): p.get("coursesCourues", []) for p in (perfs_data or {}).get("participants", [])}
    all_h = [p.get("nom") for p in parts if p.get("nom")]
    all_g = [(p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0 for p in parts]
    
    inv_c = [1.0/float(p.get("dernierRapportDirect",{}).get("rapport") or p.get("dernierRapportReference",{}).get("rapport") or 20) for p in parts]
    s_inv = sum(inv_c) or 1.0
    
    ans = []
    for i, p in enumerate(parts):
        num, ch, en = p.get("numPmu"), p.get("nom"), p.get("entraineur")
        gc, perfs = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0), perf_map.get(num, [])
        s_forme = score_forme_enrichi(perfs)
        
        # Stats Annee Pro
        sa = p.get("statsAnnee", {})
        s_dr = float(max(20, min(95, (sa.get("nombreVictoires",0)/max(1,sa.get("nombreCourses",0))*250)+40)))

        sc = {
            "marche": round((inv_c[i]/s_inv)*100, 1), "forme": round(s_forme, 1),
            "carriere": round(get_horse_class_pmu(p), 1), "gains": round(min(100, 15*math.log10(max(1,gc/1000)+1)), 1),
            "driver": round(s_dr, 1),
            "entraineur": round(get_form_ecurie_score(en, b[0].get("entraineurs", {})), 1),
            "distance": round(s_forme * 0.9, 1), "cheval_stats": round(get_horse_class_pmu(p), 1),
            "elo": round((b[2].get(ch, 1500)-1400)/2, 1), "age_sexe": 75.0 if 3<=p.get("age",5)<=6 else 45.0,
            "repos": round(50 + (10 if "a" in p.get("musique","").lower() else -10), 1),
            "elo_trend": 50.0, "confrontation": 50.0,
            "pedigree": round(get_pedigree_score(p.get("nomPere"), p.get("nomMere"), b[5].get("peres",{}), b[5].get("meres",{})), 1),
            "corde": round(get_corde_score(num, len(parts), corde_t, disc), 1),
            "equipment": round(get_equipment_score(p.get("oeilleres"), p.get("deferre")), 1),
            "profile_match": round(get_profile_match_score(detect_profile(perfs), dist, len(parts)), 1),
            "musique": round(get_musique_score(p.get("musique")), 1),
            "gains_relatifs": round(get_relative_gains_score(gc, all_g), 1),
            "form_ecurie": round(get_form_ecurie_score(en, b[0].get("entraineurs", {})), 1)
        }
        ans.append({"numPmu": num, "nom": ch, "age": p.get("age",0), "sexe": p.get("sexe",""), "driver": p.get("driver","—"), "entraineur": en, "musique": p.get("musique",""), "nbCourses": p.get("nombreCourses",0), "nbVictoires": p.get("nombreVictoires",0), "nbPlaces": p.get("nombrePlaces",0), "cote": round(1.0/max(0.001, inv_c[i]), 1), "probaMarche": round((inv_c[i]/s_inv)*100, 2), "gainsCarriere": gc//100, "ordreArrivee": p.get("ordreArrivee"), "profile": detect_profile(perfs), "scores": sc, "bonus": {"team": 0, "deferre": 0}})

    for a in ans: a["chance"] = a["chanceHeur"] = round(sum([a["scores"][k]*w for k,w in {"forme":0.2,"gains":0.1,"elo":0.2,"musique":0.15,"form_ecurie":0.15,"pedigree":0.1,"corde":0.1}.items()]), 2)
    ans.sort(key=lambda x: -x["chance"])
    for r, a in enumerate(ans, 1): a["rang"] = r
    pl3 = proba_place_simple([a["chance"] for a in ans], 3, len(ans))
    for i, a in enumerate(ans):
        a["chancePlace3"] = round(pl3[i], 2)
        is_v = (a["chance"] - a["probaMarche"]) > 4 and a["cote"] >= 4
        a["valueBet"], a["isGold"], a["isCoupSur"] = is_v, (is_v and a["scores"]["form_ecurie"] > 60), (pl3[i] >= 65 and a["rang"] == 1)
        a["kellyMise"] = kelly_amount(a["chance"]/100, a["cote"], capital, 0.25)
    return ans

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
                    if not c.get("arriveeDefinitive"): continue
                    parts = requests.get(f"{PMU_BASE}/{d_str}/R{re['numOfficiel']}/C{c['numOrdre']}/participants", headers=HEADERS, timeout=10).json()
                    for p in parts.get("participants", []):
                        ch, dr, en = p.get("nom"), p.get("driver"), p.get("entraineur")
                        won = 1 if p.get("ordreArrivee") == 1 else 0
                        if ch: h_s["global"][ch]["c"] += 1; h_s["global"][ch]["v"] += won
                        if dr: t_s["drivers"][dr]["c"] += 1; t_s["drivers"][dr]["v"] += won
                        if en: t_s["entraineurs"][en]["c"] += 1; t_s["entraineurs"][en]["v"] += won
                        ped_d.append({"pere": p.get("nomPere"), "mere": p.get("nomMere"), "place": p.get("ordreArrivee", 0)})
        except: continue
    p_s, m_s = build_pedigree_stats(ped_d)
    return (dict(t_s), dict(h_s), dict(elo), {}, {}, {"peres": p_s, "meres": m_s})

# --- ROUTES ---
@app.route("/")
def home(): return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD: session["logged_in"] = True; return redirect(url_for("home"))
        return render_template("login.html", error="Incorrect")
    return render_template("login.html")

@app.route("/api/reunions")
def api_reunions():
    d = request.args.get("date") or datetime.now().strftime("%d%m%Y")
    try:
        r = requests.get(f"{PMU_BASE}/{d}", headers=HEADERS, timeout=10).json()
        out = []
        for re in r.get("programme", {}).get("reunions", []):
            courses = [{"numCourse": c["numOrdre"], "libelle": c.get("libelle") or c.get("libelleCourt"), "heure": datetime.fromtimestamp(c["heureDepart"]/1000).strftime("%H:%M") if c.get("heureDepart") else "00:00", "nbPartants": c.get("nombreDeclaresPartants"), "arriveeDefinitive": c.get("arriveeDefinitive", False)} for c in re.get("courses", [])]
            out.append({"numReunion": re["numOfficiel"], "hippodrome": re["hippodrome"]["libelleCourt"], "courses": courses})
        return jsonify({"date": d, "reunions": out})
    except: return jsonify({"reunions": []})

@app.route("/api/course/<int:rn>/<int:cn>")
def api_course(rn, cn):
    d = request.args.get("date") or datetime.now().strftime("%d%m%Y")
    cap = float(request.args.get("capital", 100))
    try:
        r = requests.get(f"{PMU_BASE}/{d}", headers=HEADERS, timeout=10).json()
        parts = requests.get(f"{PMU_BASE}/{d}/R{rn}/C{cn}/participants", headers=HEADERS, timeout=10).json()
        perfs = requests.get(f"{PMU_BASE}/{d}/R{rn}/C{cn}/performances-detaillees/pretty", headers=HEADERS, timeout=10).json()
        h, hippo, dist, disc, corde = "00:00", "Inconnu", 2000, "ATTELE", "GAUCHE"
        for re in r.get("programme",{}).get("reunions", []):
            if re["numOfficiel"] == rn:
                hippo = re["hippodrome"]["libelleCourt"]
                for co in re["courses"]:
                    if co["numOrdre"] == cn:
                        h = datetime.fromtimestamp(co["heureDepart"]/1000).strftime("%H:%M") if co.get("heureDepart") else "00:00"
                        dist, disc, corde = co.get("distance", 2000), co.get("discipline", "ATTELE"), co.get("corde", "GAUCHE")
        ans = analyser_course(parts, perfs, safe_compute_stats(), dist, disc, hippo, corde, cap)
        return jsonify({"date": d, "reunion": {"hippodrome": hippo}, "course": {"libelle": f"R{rn}C{cn}", "heure": h, "distance": dist, "discipline": disc}, "analyses": ans, "ml_active": False, "timestamp": datetime.now().isoformat()})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/logout")
def logout(): session.pop("logged_in", None); return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
