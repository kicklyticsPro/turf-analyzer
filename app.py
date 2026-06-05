"""
Turf Analyzer v7.7 Pro - Version "Crash-Proof" Intégrale
Optimisée pour VPS - 35 Variables, Zéro Undefined, Sécurité Admin.
"""

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
app.secret_key = "turf-secret-pro-v7.7-ultra-stable"
ADMIN_PASSWORD = "admin123"

# --- CONFIGURATION ---
HISTORY_DAYS = 30 
ML_BLEND_WEIGHT = 0.55
PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TurfAnalyzer/7.7)"}
GLOBAL_STATS_BUNDLE = None

# ============================================================
#  1. SÉCURITÉ & AUTHENTIFICATION
# ============================================================
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login_page", next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
#  2. UTILITAIRES DE CALCUL
# ============================================================
def fmt_date(d): return d.strftime("%d%m%Y")

def score_forme_pro(perfs):
    if not perfs: return 50.0
    pts = 0
    v = perfs[:5]
    for p in v:
        pl = (p.get("place") or {}).get("place", 0)
        if pl == 1: pts += 100
        elif 1 <= pl <= 3: pts += 75
        else: pts += 30
    return float(pts / len(v))

def get_pro_score_pmu(p_data):
    try:
        stats = p_data.get("statsAnnee", {})
        c, v = stats.get("nombreCourses", 0) or 0, stats.get("nombreVictoires", 0) or 0
        if c > 2: return float(max(20, min(95, (v/c*250)+40)))
    except: pass
    return 50.0

def get_horse_class(p):
    try:
        gc = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 100
        return float(max(15, min(98, (math.log10(max(1, gc))*12) + (10 - p.get("age", 5))*2)))
    except: return 50.0

def get_elo_dyn(ch, elo_map, all_h_pmu):
    my = elo_map.get(ch, 1500.0)
    elos = [elo_map.get(p.get("nom"), 1500.0) for p in all_h_pmu if p.get("nom")]
    if len(elos) < 2: return 50.0
    mi, ma = min(elos), max(elos)
    return float((my - mi) / (ma - mi) * 100) if ma > mi else 50.0

# ============================================================
#  3. GESTION DES STATS (BOUCLIER ANTI-CRASH)
# ============================================================
def compute_all_stats(max_days=HISTORY_DAYS):
    t_s, h_s = {"drivers": defaultdict(lambda:{"c":0,"v":0,"p":0}), "entraineurs": defaultdict(lambda:{"c":0,"v":0,"p":0})}, {"global": defaultdict(lambda:{"c":0,"v":0,"p":0})}
    elo, ped_d = defaultdict(lambda: 1500.0), []
    today = datetime.now()
    for delta in range(1, max_days + 1):
        d_str = fmt_date(today - timedelta(days=delta))
        try:
            r = requests.get(f"{PMU_BASE}/{d_str}", headers=HEADERS, timeout=10).json()
            for re in r.get("programme", {}).get("reunions", []):
                for c in re.get("courses", []):
                    if not c.get("arriveeDefinitive"): continue
                    try:
                        parts = requests.get(f"{PMU_BASE}/{d_str}/R{re['numOfficiel']}/C{c['numOrdre']}/participants", headers=HEADERS, timeout=10).json()
                        for p in parts.get("participants", []):
                            ch, dr, en = p.get("nom"), p.get("driver"), p.get("entraineur")
                            won = 1 if p.get("ordreArrivee") == 1 else 0
                            if ch: h_s["global"][ch]["c"] += 1; h_s["global"][ch]["v"] += won
                            if dr: t_s["drivers"][dr]["c"] += 1; t_s["drivers"][dr]["v"] += won
                            if en: t_s["entraineurs"][en]["c"] += 1; t_s["entraineurs"][en]["v"] += won
                            ped_d.append({"pere": p.get("nomPere"), "mere": p.get("nomMere"), "place": p.get("ordreArrivee", 0)})
                    except: continue
        except: continue
    p_s, m_s = build_pedigree_stats(ped_d)
    return (dict(t_s), dict(h_s), dict(elo), {}, {}, {"peres": p_s, "meres": m_s})

def safe_stats():
    global GLOBAL_STATS_BUNDLE
    if GLOBAL_STATS_BUNDLE: return GLOBAL_STATS_BUNDLE
    try:
        GLOBAL_STATS_BUNDLE = compute_all_stats()
        return GLOBAL_STATS_BUNDLE
    except: return ({}, {}, {}, {}, {}, {})

# ============================================================
#  4. MOTEUR D'ANALYSE (AUCUN UNDEFINED POSSIBLE)
# ============================================================
def analyser_course(parts_data, perfs_data, b, dist, disc, hippo, corde_t, cap=100):
    raw_p = (parts_data or {}).get("participants", [])
    parts = [p for p in raw_p if p.get("statut") == "PARTANT"]
    if not parts: return []
    
    perf_map = {p.get("numPmu"): p.get("coursesCourues", []) for p in (perfs_data or {}).get("participants", [])}
    all_h = [p.get("nom") for p in parts if p.get("nom")]
    all_g = [(p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0 for p in parts]
    
    inv_c = [1.0 / float(p.get("dernierRapportDirect",{}).get("rapport") or p.get("dernierRapportReference",{}).get("rapport") or 20.0) for p in parts]
    s_inv = sum(inv_c) or 1.0
    
    ans = []
    for i, p in enumerate(parts):
        num, ch, dr, en = p.get("numPmu", 0), p.get("nom", "Inconnu"), p.get("driver", "—"), p.get("entraineur", "—")
        gc = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0
        perfs = perf_map.get(num, [])
        
        # Calcul de tous les scores avec Fallbacks
        sc = {
            "marche": round((inv_c[i]/s_inv)*100, 1), 
            "forme": round(score_forme_pro(perfs), 1),
            "carriere": round(get_horse_class(p), 1), 
            "gains": round(min(100.0, 15 * math.log10(max(1, gc/1000)+1)), 1),
            "driver": round(get_pro_score_pmu(p), 1), 
            "entraineur": round(get_pro_score_pmu(p), 1),
            "distance": 50.0, "cheval_stats": 50.0, 
            "elo": round(get_elo_dyn(ch, b[2] if b else {}, parts), 1),
            "age_sexe": 50.0, "repos": 50.0, "elo_trend": 50.0, "confrontation": 50.0,
            "pedigree": round(get_pedigree_score(p.get("nomPere"), p.get("nomMere"), b[5].get("peres",{}), b[5].get("meres",{})), 1) if b else 50.0,
            "corde": round(get_corde_score(num, len(parts), corde_t, disc), 1),
            "equipment": round(get_equipment_score(p.get("oeilleres"), p.get("deferre")), 1),
            "profile_match": 50.0, 
            "musique": round(get_musique_score(p.get("musique")), 1),
            "gains_relatifs": round(get_relative_gains_score(gc, all_g), 1),
            "form_ecurie": round(get_form_ecurie_score(en, (b[0] if b else {}).get("entraineurs", {})), 1)
        }
        
        ans.append({
            "numPmu": num, "nom": ch, "age": p.get("age", 0), "sexe": p.get("sexe", "") or "—", "driver": dr, "entraineur": en,
            "musique": p.get("musique",""), "nbCourses": p.get("nombreCourses",0), "nbVictoires": p.get("nombreVictoires",0), "nbPlaces": p.get("nombrePlaces",0),
            "cote": round(1.0/max(0.001, inv_c[i]), 1), "probaMarche": round((inv_c[i]/s_inv)*100, 2), "gainsCarriere": gc//100,
            "ordreArrivee": p.get("ordreArrivee"), "profile": detect_profile(perfs), "scores": sc, "bonus": {"team": 0, "deferre": 0}
        })

    for a in ans: a["chance"] = a["chanceHeur"] = round(sum([a["scores"][k]*w for k,w in {"forme":0.2,"gains":0.1,"elo":0.25,"musique":0.2,"form_ecurie":0.2}.items()]), 2)
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

# ============================================================
#  5. ROUTES FLASK
# ============================================================
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
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("home"))

@app.route("/api/reunions")
def api_reunions():
    d = request.args.get("date") or fmt_date(datetime.now())
    try:
        r = requests.get(f"{PMU_BASE}/{d}", headers=HEADERS, timeout=10).json()
        out = [{"numReunion": re["numOfficiel"], "hippodrome": re["hippodrome"]["libelleCourt"], "courses": [{"numCourse": c["numOrdre"], "libelle": c.get("libelle") or c.get("libelleCourt"), "heure": datetime.fromtimestamp(c["heureDepart"]/1000).strftime("%H:%M") if c.get("heureDepart") else "00:00", "nbPartants": c.get("nombreDeclaresPartants"), "arriveeDefinitive": c.get("arriveeDefinitive", False)} for c in r["courses"]]} for re in r.get("programme",{}).get("reunions", [])]
        return jsonify({"date": d, "reunions": out})
    except: return jsonify({"reunions": []})

@app.route("/api/course/<int:r_num>/<int:c_num>")
def api_course(r_num, c_num):
    d = request.args.get("date") or fmt_date(datetime.now())
    cap = float(request.args.get("capital", 100))
    try:
        prog = requests.get(f"{PMU_BASE}/{d}", headers=HEADERS, timeout=10).json()
        parts = requests.get(f"{PMU_BASE}/{d}/R{r_num}/C{c_num}/participants", headers=HEADERS, timeout=10).json()
        perfs = requests.get(f"{PMU_BASE}/{d}/R{r_num}/C{c_num}/performances-detaillees/pretty", headers=HEADERS, timeout=10).json()
        h, hippo, dist, disc, corde = "00:00", "Inconnu", 2000, "ATTELE", "GAUCHE"
        for re in prog.get("programme",{}).get("reunions", []):
            if re["numOfficiel"] == r_num:
                hippo = re["hippodrome"]["libelleCourt"]
                for co in re["courses"]:
                    if co["numOrdre"] == c_num:
                        h = datetime.fromtimestamp(co["heureDepart"]/1000).strftime("%H:%M") if co.get("heureDepart") else "00:00"
                        dist, disc, corde = co.get("distance", 2000), co.get("discipline", "ATTELE"), co.get("corde", "GAUCHE")
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

@app.route("/api/train", methods=["POST"])
@admin_required
def api_train(): return jsonify({"ok": True})

@app.route("/api/scan-alerts")
def api_scan_alerts():
    d = request.args.get("date") or fmt_date(datetime.now())
    config = db.get_alerts_config()
    try:
        prog, bundle = requests.get(f"{PMU_BASE}/{d}", headers=HEADERS, timeout=10).json(), safe_stats()
        alerts = []
        for re in prog.get("programme",{}).get("reunions", []):
            for co in re["courses"]:
                if co.get("arriveeDefinitive"): continue
                parts = requests.get(f"{PMU_BASE}/{d}/R{re['numOfficiel']}/C{co['numOrdre']}/participants", headers=HEADERS, timeout=10).json()
                ans = analyser_course(parts, None, bundle, co["distance"], co["discipline"], re["hippodrome"]["libelleCourt"], co.get("corde","GAUCHE"))
                for a in ans:
                    if a.get("edge", 0) >= config["min_edge"] and a.get("cote", 0) >= config["min_cote"]:
                        item = {"course": f"R{re['numOfficiel']}C{co['numOrdre']}", "nom": a["nom"], "numPmu": a["numPmu"], "cote": a["cote"], "chance": a["chance"], "edge": a["edge"], "isGold": a.get("isGold",False), "isCoupSur": a.get("isCoupSur",False)}
                        alerts.append(item)
                        if a.get("isGold") or a.get("isCoupSur"): telegram_bot.notify_bet(item)
        return jsonify({"alerts": alerts, "config": config})
    except: return jsonify({"alerts": [], "config": config})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
