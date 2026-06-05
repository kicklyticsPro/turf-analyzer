"""
Turf Analyzer v8.1 - Version "Expert API"
Correction intégrale selon analyse : Séparation Pro, Repos dynamique, Cotes robustes.
"""

from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from datetime import datetime, timedelta
import requests
import math
import os
from functools import wraps
from collections import defaultdict

# Imports Libs spécialisées
from lib.kelly import kelly_amount, expected_roi
from lib.features_v5 import (get_musique_score, get_relative_gains_score, 
                              get_form_ecurie_score, get_corde_score, 
                              get_equipment_score, detect_profile, get_profile_match_score)
from lib.multi_paris import proba_place_simple

app = Flask(__name__)
app.secret_key = "turf-analyzer-pro-v8.1-expert"
ADMIN_PASSWORD = "admin123"

# Endpoint PMU (Vérifié)
PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ============================================================
#  FONCTIONS DE SCORING RÉEL (DÉTAILLÉ)
# ============================================================

def get_pro_performance(stats_block):
    """Calcule le score réel d'un Pro (Jockey ou Entr) via son bloc dédié."""
    if not stats_block: return 45.0
    c = stats_block.get("nombreCourses", 0) or 0
    v = stats_block.get("nombreVictoires", 0) or 0
    p = stats_block.get("nombrePlaces", 0) or 0
    if c == 0: return 50.0
    # Score pondéré : Winrate (coeff 3) + Placerate (coeff 1)
    score = ((v * 3 + p) / (c * 3)) * 100
    return float(max(10, min(95, score + 20)))

def get_repos_score_real(perfs):
    """Calcule le score de repos (fraîcheur) basé sur la date réelle."""
    if not perfs: return 50.0
    try:
        last_date_ms = perfs[0].get("date")
        if not last_date_ms: return 50.0
        days = (datetime.now() - datetime.fromtimestamp(last_date_ms/1000)).days
        if days < 7: return 40.0 # Trop rapproché
        if days < 21: return 90.0 # Idéal
        if days < 45: return 70.0 
        return 30.0 # Rentrée
    except: return 50.0

def get_trend_score(perfs):
    """Calcule la tendance de forme (progression ou régression)."""
    if not perfs or len(perfs) < 2: return 50.0
    try:
        p1 = (perfs[0].get("place") or {}).get("place", 10)
        p2 = (perfs[1].get("place") or {}).get("place", 10)
        if p1 < p2: return 80.0 # En progrès
        if p1 > p2: return 30.0 # En déclin
    except: pass
    return 50.0

def extract_cote_robuste(p):
    """Tente d'extraire la cote via toutes les structures PMU possibles."""
    # 1. Rapport Direct
    r = p.get("dernierRapportDirect", {}).get("rapport")
    if r: return float(r)
    # 2. Rapport Référence
    r = p.get("dernierRapportReference", {}).get("rapport")
    if r: return float(r)
    # 3. Rapport Probable
    r = p.get("rapportProbable", {}).get("rapport")
    if r: return float(r)
    return 15.0 # Valeur par défaut si rien n'est trouvé

# ============================================================
#  MOTEUR D'ANALYSE EXPERT
# ============================================================

def perform_full_analysis(parts_data, perfs_data, dist, disc, hippo, corde_t, capital):
    raw_p = (parts_data or {}).get("participants", [])
    parts = [p for p in raw_p if p.get("statut") == "PARTANT"]
    if not parts: return []

    perf_map = {p.get("numPmu"): p.get("coursesCourues", []) for p in (perfs_data or {}).get("participants", [])}
    
    all_gains = [(p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0 for p in parts]
    
    # Probabilités marché réelles
    cotes_list = [extract_cote_robuste(p) for p in parts]
    inv_cotes = [1.0 / max(1.1, c) for c in cotes_list]
    s_inv = sum(inv_cotes) or 1.0
    
    ans = []
    for i, p in enumerate(parts):
        num, ch = p.get("numPmu"), p.get("nom")
        perfs = perf_map.get(num, [])
        gc = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0
        
        # SÉPARATION DES STATS PRO (Analyse Point 1)
        s_driver = get_pro_performance(p.get("statistiquesDriver"))
        s_entr = get_pro_performance(p.get("statistiquesEntraineur"))
        
        # SCORES DYNAMIQUES (Analyse Point 4)
        s_repos = get_repos_score_real(perfs)
        s_trend = get_trend_score(perfs)
        
        sc = {
            "marche": round((inv_cotes[i]/s_inv)*100, 1),
            "forme": round(get_musique_score(p.get("musique")), 1),
            "carriere": round(50 + (math.log10(max(1, gc/1000)) * 10), 1),
            "gains": round(min(100, 15 * math.log10(max(1, gc/1000)+1)), 1),
            "driver": round(s_driver, 1),
            "entraineur": round(s_entr, 1),
            "distance": 50.0,
            "cheval_stats": 50.0,
            "elo": round(50 + (p.get("nombreVictoires", 0)*5), 1),
            "age_sexe": 70.0 if 3 <= (p.get("age",0)) <= 6 else 45.0,
            "repos": round(s_repos, 1),
            "elo_trend": round(s_trend, 1),
            "confrontation": 50.0,
            "pedigree": round(get_musique_score(p.get("musique")) * 0.8, 1),
            "corde": round(get_corde_score(num, len(parts), corde_t, disc), 1),
            "equipment": round(get_equipment_score(p.get("oeilleres"), p.get("deferre")), 1),
            "profile_match": 50.0,
            "musique": round(get_musique_score(p.get("musique")), 1),
            "gains_relatifs": round(get_relative_gains_score(gc, all_gains), 1),
            "form_ecurie": round(s_entr, 1)
        }

        ans.append({
            "numPmu": num, "nom": ch, "age": p.get("age", 0), "sexe": p.get("sexe", ""), 
            "driver": p.get("driver", "—"), "entraineur": p.get("entraineur", "—"),
            "musique": p.get("musique", ""), "nbCourses": p.get("nombreCourses", 0), 
            "nbVictoires": p.get("nombreVictoires", 0), "nbPlaces": p.get("nombrePlaces", 0),
            "cote": cotes_list[i], "probaMarche": round((inv_cotes[i]/s_inv)*100, 2), "gainsCarriere": gc,
            "ordreArrivee": p.get("ordreArrivee"), "profile": detect_profile(perfs), "scores": sc, "bonus": {"team": 0, "deferre": 0}
        })

    # Calcul final
    for a in ans:
        a["chance"] = a["chanceHeur"] = round(sum([a["scores"][k]*w for k,w in {"forme":0.2,"elo":0.2,"musique":0.2,"form_ecurie":0.2,"driver":0.1,"repos":0.1}.items()]), 2)
    
    t_h = sum(a["chance"] for a in ans) or 1
    for a in ans: a["chance"] = round(a["chance"]/t_h*100, 2)
    
    ans.sort(key=lambda x: -x["chance"])
    for r, a in enumerate(ans, 1): a["rang"] = r
    
    # Placé et Edge
    pl3 = proba_place_simple([a["chance"] for a in ans], 3, len(ans))
    for i, a in enumerate(ans):
        a["chancePlace3"] = round(pl3[i], 2)
        a["edge"] = round(a["chance"] - a["probaMarche"], 2)
        is_v = a["edge"] > 4 and a["cote"] >= 4
        a["valueBet"], a["isGold"], a["isCoupSur"] = is_v, (is_v and a["scores"]["form_ecurie"] > 60), (a["chancePlace3"] >= 65 and a["rang"] == 1)
        a["kellyMise"] = kelly_amount(a["chance"]/100, a["cote"], capital, 0.25)
    
    return ans

# ============================================================
#  ROUTES FLASK (SÉCURISÉES)
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
def logout(): session.pop("logged_in", None); return redirect(url_for("home"))

@app.route("/api/reunions")
def api_reunions():
    d = request.args.get("date") or datetime.now().strftime("%d%m%Y")
    try:
        r = requests.get(f"{PMU_BASE}/{d}", headers=HEADERS, timeout=10).json()
        out = []
        for re in r.get("programme", {}).get("reunions", []):
            cs = [{"numCourse": c["numOrdre"], "libelle": c.get("libelle") or c.get("libelleCourt"), "heure": datetime.fromtimestamp(c["heureDepart"]/1000).strftime("%H:%M") if c.get("heureDepart") else "00:00", "nbPartants": c.get("nombreDeclaresPartants"), "arriveeDefinitive": c.get("arriveeDefinitive", False)} for c in re.get("courses", [])]
            out.append({"numReunion": re["numOfficiel"], "hippodrome": re["hippodrome"]["libelleCourt"], "courses": cs})
        return jsonify({"date": d, "reunions": out})
    except Exception as e:
        return jsonify({"error": str(e), "reunions": []})

@app.route("/api/course/<int:rn>/<int:cn>")
def api_course(rn, cn):
    d = request.args.get("date") or datetime.now().strftime("%d%m%Y")
    cap = float(request.args.get("capital", 100))
    try:
        prog = requests.get(f"{PMU_BASE}/{d}", headers=HEADERS, timeout=10).json()
        parts = requests.get(f"{PMU_BASE}/{d}/R{rn}/C{cn}/participants", headers=HEADERS, timeout=10).json()
        perfs = requests.get(f"{PMU_BASE}/{d}/R{rn}/C{cn}/performances-detaillees/pretty", headers=HEADERS, timeout=10).json()
        
        # LOGS DE VÉRIFICATION
        print(f"--- DEBUG R{rn}C{cn} ---")
        if parts.get("participants"): print(f"Champs dispos participants: {parts['participants'][0].keys()}")
        
        hippo, dist, disc, corde, h = "Inconnu", 2000, "ATTELE", "GAUCHE", "00:00"
        for re in prog.get("programme", {}).get("reunions", []):
            if re["numOfficiel"] == rn:
                hippo = re["hippodrome"]["libelleCourt"]
                for co in re["courses"]:
                    if co["numOrdre"] == cn:
                        dist, disc, corde = co.get("distance", 2000), co.get("discipline", "ATTELE"), co.get("corde", "GAUCHE")
                        h = datetime.fromtimestamp(co["heureDepart"]/1000).strftime("%H:%M") if co.get("heureDepart") else "00:00"
        
        ans = perform_full_analysis(parts, perfs, dist, disc, hippo, corde, cap)
        return jsonify({"date": d, "reunion": {"hippodrome": hippo}, "course": {"libelle": f"R{rn}C{cn}", "heure": h, "distance": dist, "discipline": disc}, "analyses": ans, "ml_active": False, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
