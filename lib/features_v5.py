"""
v5 - Nouvelles features : analyse musique, gains relatifs, fraicheur brute, forme écurie.
"""
import re
from collections import defaultdict
import math

def empty_bucket():
    return {"c": 0, "v": 0, "p": 0}

# ============================================================
#  Pedigree (v4)
# ============================================================
def build_pedigree_stats(all_horses_data):
    pere_stats = defaultdict(empty_bucket)
    mere_stats = defaultdict(empty_bucket)
    for h in all_horses_data:
        pere = h.get("pere")
        mere = h.get("mere")
        place = h.get("place", 0) or 0
        won = 1 if place == 1 else 0
        placed = 1 if 1 <= place <= 3 else 0
        if pere:
            pere_stats[pere]["c"] += 1
            pere_stats[pere]["v"] += won
            pere_stats[pere]["p"] += placed
        if mere:
            mere_stats[mere]["c"] += 1
            mere_stats[mere]["v"] += won
            mere_stats[mere]["p"] += placed
    return dict(pere_stats), dict(mere_stats)

def get_pedigree_score(pere, mere, pere_stats, mere_stats):
    def bucket_score(bucket, min_count=10):
        if not bucket or bucket["c"] < min_count:
            return None
        c, v, p = bucket["c"], bucket["v"], bucket["p"]
        tv = v / c
        tp = p / c
        confiance = min(1.0, c / 100)
        raw = tv * 250 + tp * 60
        return min(100, raw * confiance + 40 * (1 - confiance))

    s_p = bucket_score(pere_stats.get(pere))
    s_m = bucket_score(mere_stats.get(mere))
    if s_p is None and s_m is None: return 50
    if s_p is None: return s_m
    if s_m is None: return s_p
    return s_p * 0.70 + s_m * 0.30

# ============================================================
#  Corde (v4)
# ============================================================
def get_corde_score(num_pmu, nb_partants, type_corde, discipline):
    if not num_pmu or not nb_partants: return 50
    rel = (num_pmu - 1) / max(nb_partants - 1, 1)
    if discipline == "ATTELE":
        score = 65 - rel * 25
    elif discipline == "PLAT":
        score = 70 - rel * 35
    elif discipline in ("MONTE", "HAIES", "STEEPLE-CHASE", "CROSS"):
        score = 55 - rel * 10
    else:
        score = 50
    if type_corde == "CORDE_AUCUNE" or not type_corde:
        score = 50
    return max(0, min(100, score))

# ============================================================
#  Équipements (v4 amélioré)
# ============================================================
def get_equipment_score(oeilleres, deferre, prev_oeilleres=None, prev_deferre=None):
    score = 50
    if oeilleres and oeilleres != "SANS_OEILLERES":
        score += 5
        if prev_oeilleres == "SANS_OEILLERES": score += 10
    if deferre == "DEFERRE_DES_4":
        score += 12
    elif deferre in ("DEFERRE_ANTERIEURS", "DEFERRE_POSTERIEURS"):
        score += 5
    if prev_deferre and prev_deferre != deferre and "DEFERRE" in (deferre or ""):
        score += 5
    return min(100, max(0, score))

# ============================================================
#  Profil & Commentaires (v4)
# ============================================================
PROFIL_KEYWORDS = {
    "attaquant": ["s'est élancé", "a pris la tête", "en tête", "a mené", "tenu la corde", "a impulsé", "à l'aise en tête"],
    "finisseur": ["dans la ligne droite", "fini fort", "a remonté", "dans les derniers mètres", "ligne d'arrivée", "a coiffé"],
    "fragile": ["s'est galopé", "fauté", "disqualifié", "a faibli", "ne s'est pas employé", "distancé", "a chuté"],
    "regulier": ["dans le peloton", "à mi-parcours", "a suivi", "régulier", "honorable"],
}

def detect_profile(perfs_detail, comment_text=None):
    """
    Analyse les commentaires des courses passées pour détecter le profil du cheval.
    """
    counters = defaultdict(int)
    total = 0
    
    # Analyse des 5 dernières courses
    for course in (perfs_detail or [])[:5]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                comment = (p.get("commentaire") or {}).get("texte", "")
                if comment:
                    total += 1
                    cl = comment.lower()
                    for profil, keywords in PROFIL_KEYWORDS.items():
                        for kw in keywords:
                            if kw in cl:
                                counters[profil] += 1
                                break

    if total == 0: 
        return {p: 50 for p in PROFIL_KEYWORDS}

    # Normalisation 0-100
    res = {}
    for p in PROFIL_KEYWORDS:
        res[p] = min(100, (counters[p] / total) * 100 * 2.5)
    
    return res

def get_profile_match_score(profil, distance, nb_partants):
    if not profil: return 50
    score = 50
    
    # Tactique selon distance
    if distance and distance < 2000:
        score += (profil.get("attaquant", 50) - 50) * 0.5
    elif distance and distance > 2700:
        score += (profil.get("finisseur", 50) - 50) * 0.5
    else:
        score += (profil.get("regulier", 50) - 50) * 0.3
        
    score -= (profil.get("fragile", 0) * 0.15)
    return max(0, min(100, score))

# ============================================================
#  NOUVEAUTÉS v5
# ============================================================

def parse_musique(musique_str):
    """
    Analyse une musique (ex: '1a 2a Da (25) 4a') et extrait les derniers rangs.
    """
    if not musique_str:
        return []
    # Nettoie les parenthèses (dates)
    m = re.sub(r'\(.*?\)', '', musique_str)
    # Trouve les chiffres ou D/A/T/S
    ranks = re.findall(r'([0-9DATS])', m)
    
    val_ranks = []
    for r in ranks:
        if r.isdigit():
            val_ranks.append(int(r))
        elif r == '0':
            val_ranks.append(10)
        elif r in ('D', 'A', 'T', 'S'):
            val_ranks.append(11) # Disqualifié ou autre échec
    return val_ranks

def get_musique_score(musique_str):
    ranks = parse_musique(musique_str)
    if not ranks:
        return 50
    
    score = 0
    weights = [1.0, 0.8, 0.6, 0.4, 0.2]
    total_w = 0
    
    for i, r in enumerate(ranks[:5]):
        w = weights[i]
        total_w += w
        if r == 1: pts = 100
        elif r == 2: pts = 85
        elif r == 3: pts = 70
        elif r <= 5: pts = 50
        elif r <= 9: pts = 30
        else: pts = 10
        score += pts * w
    
    return score / total_w if total_w > 0 else 50

def get_relative_gains_score(gains_carriere, race_gains_list):
    """
    Positionne les gains du cheval par rapport aux autres partants.
    """
    if not race_gains_list or gains_carriere is None:
        return 50
    
    gains_list = sorted([g for g in race_gains_list if g is not None])
    if not gains_list:
        return 50
        
    rank = 0
    for g in gains_list:
        if gains_carriere > g:
            rank += 1
    
    score = (rank / len(gains_list)) * 100
    return score

def get_form_ecurie_score(trainer_name, all_trainer_stats):
    """
    Score de forme de l'écurie (tous les chevaux de l'entraîneur confondus).
    """
    stats = all_trainer_stats.get(trainer_name)
    if not stats or stats['c'] < 5:
        return 50
    
    win_rate = stats['v'] / stats['c']
    place_rate = stats['p'] / stats['c']
    
    score = (win_rate * 200 + place_rate * 100)
    return min(100, max(10, score + 30))

def get_days_since_last_race(musique_str, perfs_detail, today_ts):
    """
    Retourne le nombre de jours bruts depuis la dernière course.
    """
    if not perfs_detail:
        return 60 # Défaut
    
    last_date_ms = perfs_detail[0].get("date")
    if not last_date_ms:
        return 60
        
    days = (today_ts - (last_date_ms / 1000)) / 86400
    return max(0, days)
