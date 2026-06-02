"""
v5 - Scraping Geny pour données enrichies (terrain, météo, pronostics presse).
Cache mémoire pour limiter les requêtes.
"""
import re
import requests
from functools import lru_cache

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
}


@lru_cache(maxsize=128)
def fetch_url(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        print(f"[Geny] Erreur fetch {url}: {e}")
    return None


def parse_terrain_meteo(html):
    """
    Extrait le terrain et la météo de Geny depuis le HTML.
    Cherche des patterns comme "Terrain : souple", "Temps : ensoleillé".
    """
    if not html:
        return {"terrain": None, "meteo": None, "temperature": None}

    result = {"terrain": None, "meteo": None, "temperature": None}

    # Terrain
    m = re.search(r"[Tt]errain\s*:?\s*([a-zéèà]+)", html)
    if m:
        terrain = m.group(1).lower().strip()
        if terrain in ("bon", "souple", "lourd", "très lourd", "collant",
                       "sec", "léger", "psf"):
            result["terrain"] = terrain

    # Météo (mots simples)
    for meteo_keyword in ["ensoleillé", "nuageux", "pluvieux", "couvert",
                          "orageux", "brumeux", "neigeux"]:
        if meteo_keyword in html.lower():
            result["meteo"] = meteo_keyword
            break

    # Température
    m = re.search(r"(\d{1,2})°?\s*C?", html)
    if m:
        try:
            result["temperature"] = int(m.group(1))
        except Exception:
            pass

    return result


def get_terrain_score(terrain, perfs_detail=None):
    """
    Score 0-100 selon la perf passée du cheval sur ce type de terrain.
    Pour l'instant : heuristique basique (étendre quand on a l'historique perfs/terrain).
    """
    if not terrain:
        return 50
    # Score neutre, à enrichir
    return 50


# ============================================================
#  Pronostics presse (Geny)
# ============================================================
def parse_pronostic_presse(html):
    """
    Cherche dans le HTML Geny la liste des pronostics presse.
    Format typique : "Tiercé Magazine : 5 - 12 - 3 - 8 - 1"
    """
    if not html:
        return {}

    pronos = {}
    patterns = [
        (r"Tiercé\s+Magazine\s*:?\s*([\d\s\-,]+)", "tierce_mag"),
        (r"Le\s+Parisien\s*:?\s*([\d\s\-,]+)", "parisien"),
        (r"Equidia\s*:?\s*([\d\s\-,]+)", "equidia"),
        (r"Pronostic\s+Geny\s*:?\s*([\d\s\-,]+)", "geny"),
    ]
    for pat, key in patterns:
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            nums_str = m.group(1)
            nums = [int(n) for n in re.findall(r"\d+", nums_str)]
            if nums:
                pronos[key] = nums[:7]
    return pronos


def get_geny_data(date_str, r_num, c_num, libelle=""):
    """
    Tente de récupérer les données enrichies depuis Geny.
    """
    if len(date_str) == 8:
        d, m, y = date_str[:2], date_str[2:4], date_str[4:]
        date_iso = f"{y}-{m}-{d}"
    else:
        return {"terrain": None, "meteo": None, "pronostics_presse": {}}

    # Heuristique pour trouver le lien de la course
    # On commence par charger la page de la réunion
    base_url = f"https://www.geny.com/reunions-courses-pmu/_d-{date_iso}"
    html_reunion = fetch_url(base_url)
    
    # Si on trouve un lien vers la course spécifique R{X}C{Y}
    course_path = None
    if html_reunion:
        # Recherche d'un lien type /partants-pmu/2026-06-02-hippo-prix-nom_c123456
        # qui correspondrait à notre R et C
        pattern = rf'href="(/partants-pmu/{date_iso}-[^"]+_c\d+)"[^>]*>C{c_num}'
        m = re.search(pattern, html_reunion)
        if m:
            course_path = m.group(1)

    if course_path:
        url_course = f"https://www.geny.com{course_path}"
        html_course = fetch_url(url_course)
        if html_course:
            terrain_data = parse_terrain_meteo(html_course)
            pronostics = parse_pronostic_presse(html_course)
            return {**terrain_data, "pronostics_presse": pronostics}

    # Fallback sur la page réunion si pas de lien spécifique trouvé
    if html_reunion:
        terrain_data = parse_terrain_meteo(html_reunion)
        pronostics = parse_pronostic_presse(html_reunion)
        return {**terrain_data, "pronostics_presse": pronostics}

    return {"terrain": None, "meteo": None, "pronostics_presse": {}}


def score_concordance_presse(num_pmu, pronostics_presse):
    """
    Score selon la fréquence d'apparition du cheval dans les pronos presse.
    Plus c'est cité haut, plus le score est élevé.
    """
    if not pronostics_presse or not num_pmu:
        return 50

    total_score = 0
    total_weight = 0
    for source, nums in pronostics_presse.items():
        if num_pmu in nums:
            position = nums.index(num_pmu) + 1
            # Position 1 = 100 pts, position 2 = 80, ..., position 7 = 20
            pts = max(20, 100 - (position - 1) * 15)
            total_score += pts
            total_weight += 1

    if total_weight == 0:
        return 50
    return total_score / total_weight
