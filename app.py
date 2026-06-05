"""
Turf Analyzer v9.0 - Version "Réalité PMU"
Zéro Donnée Fictive - Poids dynamiques - Analyse granulaire Driver/Entraîneur.
"""

from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from datetime import datetime, timedelta
import requests
import math
import os
import traceback
from functools import wraps
from collections import defaultdict

# Imports Libs spécialisées
from lib.kelly import kelly_amount, expected_roi
from lib.features_v5 import (get_musique_score, get_relative_gains_score, 
                              get_form_ecurie_score, get_corde_score, 
                              get_equipment_score, detect_profile, get_profile_match_score)
from lib.multi_paris import proba_place_simple

app = Flask(__name__)
app.secret_key = "turf-analyzer-pro-v9-reality"
ADMIN_PASSWORD = "admin123"

# Endpoint PMU (Vérifié)
PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ============================================================
#  1. SÉCURITÉ
# ============================================================
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login_page", next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
#  2. FONCTIONS DE SCORING RÉEL (SANS FICTION)
# ============================================================

def get_pro_performance(stats_block):
    """Calcule le score réel d'un Pro via son bloc dédié (Driver vs Entraîneur)."""
    if not stats_block: return None
    c = stats_block.get("nombreCourses", 0) or 0
    v = stats_block.get("nombreVictoires", 0) or 0
    p = stats_block.get("nombrePlaces", 0) or 0
    if c < 3: return None # Pas assez significatif
    
    score = ((v * 3 + p) / (c * 3)) * 100
    return float(max(0, min(100, score + 20)))

def get_repos_score_real(perfs):
    """Calcule la fraîcheur basée sur la date de la dernière course."""
    if not perfs: return None
    try:
        last_date_ms = perfs[0].get("date")
        if not last_date_ms: return None
        days = (datetime.now() - datetime.fromtimestamp(last_date_ms/1000)).days
        if days < 5: return 20.0 # Épuisé
        if days < 25: return 95.0 # Optimal
        if days < 50: return 60.0 # Correct
        return 40.0 # Rentrée
    except: return None

def get_trend_score_real(perfs):
    """Tendance de forme basée sur les classements réels."""
    if not perfs or len(perfs) < 2: return None
    try:
        # On regarde si les places s'améliorent (ex: 5e puis 2e)
        p1 = (perfs[0].get("place") or {}).get("place", 15)
        p2 = (perfs[1].get("place") or {}).get("place", 15)
        if p1 == 0 or p2 == 0: return None # Donnée incomplète (DAI...)
        if p1 < p2: return 85.0 # En progrès
        if p1 > p2: return 35.0 # En déclin
        return 50.0
    except: return None

def get_distance_score_real(perfs, target_dist):
    """Score d'aptitude à la distance basé sur le passé sur +/- 200m."""
    if not perfs or not target_dist: return None
    similar = []
    for p in perfs:
        d = p.get("distance", 0)
        if d > 0 and abs(d - target_dist) <= 200:
            pl = (p.get("place") or {}).get("place", 0)
            if pl > 0: similar.append(pl)
    
    if not similar: return None
    # Plus la place est basse (1er, 2e), plus le score est haut
    avg_place = sum(similar) / len(similar)
    score = 100 - (avg_place * 7)
    return float(max(10, min(100, score)))

def extract_cote_robuste(p):
    """Tente d'extraire la cote via toutes les structures PMU."""
    sources = [
        p.get("dernierRapportDirect", {}).get("rapport"),
        p.get("dernierRapportReference", {}).get("rapport"),
        p.get("rapportProbable", {}).get("rapport"),
        p.get("dernierRapportProbable", {}).get("rapport")
    ]
    for r in sources:
        if r: return float(r)
    return None

# ============================================================
#  3. MOTEUR D'ANALYSE DYNAMIQUE (POIDS REDISTRIBUÉS)
# ============================================================

def perform_full_analysis(parts_data, perfs_data, dist, disc, hippo, corde_t, capital):
    raw_p = (parts_data or {}).get("participants", [])
    parts = [p for p in raw_p if p.get("statut") == "PARTANT"]
    if not parts: return []

    # Vérification endpoint perfs (Analyse Point 2)
    perf_map = {}
    if perfs_data and "participants" in perfs_data:
        for p in perfs_data["participants"]:
            num = p.get("numPmu")
            if num:
                perf_map[num] = p.get("coursesCourues", [])
    
    all_gains = [(p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0 for p in parts]
    avg_gains = sum(all_gains) / len(all_gains) if all_gains else 1000
    
    ans = []
    for i, p in enumerate(parts):
        num, ch = p.get("numPmu"), p.get("nom")
        perfs = perf_map.get(num, [])
        gc = (p.get("gainsParticipant") or {}).get("gainsCarriere", 0) or 0
        
        # SCORES RÉELS (Analyse Point 1, 4, 5)
        # On remplace 50 par None quand la donnée manque
        scores = {
            "forme": get_musique_score(p.get("musique")),
            "gains": round(get_relative_gains_score(gc, all_gains), 1),
            "driver": get_pro_performance(p.get("statistiquesDriver")),
            "entraineur": get_pro_performance(p.get("statistiquesEntraineur")),
            "repos": get_repos_score_real(perfs),
            "trend": get_trend_score_real(perfs),
            "distance": get_distance_score_real(perfs, dist),
            "pedigree": round(get_musique_score(p.get("musique")) * 0.8, 1),
            "corde": get_corde_score(num, len(parts), corde_t, disc),
            "equipement": get_equipment_score(p.get("oeilleres"), p.get("deferre")),
            "form_ecurie": get_form_ecurie_score(p.get("entraineur"), {}) # nécessite stats locales idéalement
        }

        # Cote robuste (Analyse Point 3)
        cote = extract_cote_robuste(p)

        ans.append({
            "numPmu": num, "nom": ch, "age": p.get("age", 0), "sexe": p.get("sexe", ""), 
            "driver": p.get("driver", "—"), "entraineur": p.get("entraineur", "—"),
            "musique": p.get("musique", ""), "nbCourses": p.get("nombreCourses", 0), 
            "nbVictoires": p.get("nombreVictoires", 0), "nbPlaces": p.get("nombrePlaces", 0),
            "cote": cote, "gainsCarriere": gc, # Pas de division par 100
            "ordreArrivee": p.get("ordreArrivee"), "scores": scores, "bonus": {"team": 0, "deferre": 0}
        })

    # CALCUL FINAL AVEC POIDS DYNAMIQUES (Analyse Point: Ce que je ferais)
    base_weights = {
        "forme": 0.25, "gains": 0.15, "driver": 0.15, "entraineur": 0.15, 
        "repos": 0.10, "distance": 0.10, "corde": 0.05, "musique": 0.05
    }

    for a in ans:
        valid_scores = {k: v for k, v in a["scores"].items() if v is not None and k in base_weights}
        
        if not valid_scores:
            a["chanceHeur"] = 1.0 / len(ans) * 100
        else:
            # Redistribution des poids
            current_total_weight = sum(base_weights[k] for k in valid_scores.keys())
            if current_total_weight > 0:
                raw_chance = sum(valid_scores[k] * (base_weights[k] / current_total_weight) for k in valid_scores.keys())
                a["chanceHeur"] = raw_chance
            else:
                a["chanceHeur"] = 50.0

    # Normalisation
    t_h = sum(a["chanceHeur"] for a in ans) or 1
    for a in ans: a["chance"] = round(a["chanceHeur"]/t_h*100, 2)
    
    ans.sort(key=lambda x: -x["chance"])
    for r, a in enumerate(ans, 1): a["rang"] = r
    
    # Placé et Edge
    # Note: On calcule la proba marché ici (Analyse Point 3)
    cotes_valides = [a["cote"] for a in ans if a["cote"] is not None]
    inv_sum = sum(1.0/c for c in cotes_valides) if cotes_valides else 1.0
    
    pl3 = proba_place_simple([a["chance"] for a in ans], 3, len(ans))
    for i, a in enumerate(ans):
        a["chancePlace3"] = round(pl3[i], 2)
        if a["cote"]:
            a["probaMarche"] = round((1.0/a["cote"])/inv_sum*100, 2)
            a["edge"] = round(a["chance"] - a["probaMarche"], 2)
        else:
            a["probaMarche"] = 0
            a["edge"] = 0
            
        is_v = a["edge"] > 4 and (a["cote"] or 0) >= 4
        a["valueBet"] = is_v
        a["isCoupSur"] = a["chancePlace3"] >= 65 and a["rang"] == 1
        a["kellyMise"] = kelly_amount(a["chance"]/100, a["cote"] or 10, capital, 0.25)
    
    return ans

# ============================================================
#  4. ROUTES FLASK
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
    date_str = request.args.get("date") or datetime.now().strftime("%d%m%Y")
    try:
        r = requests.get(f"{PMU_BASE}/{date_str}", headers=HEADERS, timeout=10)
        r.raise_for_status()
        prog = r.json()
        out = []
        for re in prog.get("programme", {}).get("reunions", []):
            cs = []
            for c in re.get("courses", []):
                h = datetime.fromtimestamp(c["heureDepart"]/1000).strftime("%H:%M") if c.get("heureDepart") else "00:00"
                cs.append({
                    "numCourse": c["numOrdre"], "libelle": c.get("libelle") or c.get("libelleCourt"), 
                    "heure": h, "nbPartants": c.get("nombreDeclaresPartants"), 
                    "arriveeDefinitive": c.get("arriveeDefinitive", False)
                })
            out.append({"numReunion": re["numOfficiel"], "hippodrome": re["hippodrome"]["libelleCourt"], "courses": cs})
        return jsonify({"date": date_str, "reunions": out})
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
        
        # LOGS DE VÉRIFICATION (Analyse Point 8)
        if parts.get("participants"):
            p0 = parts['participants'][0]
            print(f"DEBUG: Champs disponibles pour P1: {p0.keys()}")
            if "statistiquesDriver" in p0: print("OK: statistiquessDriver trouvé")
            if "statistiquesEntraineur" in p0: print("OK: statistiquesEntraineur trouvé")
        
        h, hippo, dist, disc, corde = "00:00", "Inconnu", 2000, "ATTELE", "GAUCHE"
        for re in prog.get("programme", {}).get("reunions", []):
            if re["numOfficiel"] == rn:
                hippo = re["hippodrome"]["libelleCourt"]
                for co in re["courses"]:
                    if co["numOrdre"] == cn:
                        dist, disc, corde = co.get("distance", 2000), co.get("discipline", "ATTELE"), co.get("corde", "GAUCHE")
                        h = datetime.fromtimestamp(co["heureDepart"]/1000).strftime("%H:%M") if co.get("heureDepart") else "00:00"
        
        ans = perform_full_analysis(parts, perfs, dist, disc, hippo, corde, cap)
        return jsonify({
            "date": d, "reunion": {"hippodrome": hippo}, 
            "course": {"libelle": f"R{rn}C{cn}", "heure": h, "distance": dist, "discipline": disc}, 
            "analyses": ans, "ml_active": False, "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        traceback.print_exc()
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
