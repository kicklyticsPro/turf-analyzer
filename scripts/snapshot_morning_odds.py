"""
scripts/snapshot_morning_odds.py — Capture les cotes à 8h pour l'analyse Smart Money.
Lancer ce script chaque matin via Cron (ex: 0 8 * * *).
"""
import os
import sys
from datetime import datetime

# Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app
from lib import db

def snapshot():
    date_str = app.fmt_date(datetime.now())
    print(f"--- SNAPSHOT DES COTES : {date_str} ---")
    
    try:
        prog = app.get_programme(date_str)
    except Exception as e:
        print(f"Erreur programme: {e}")
        return

    count = 0
    for r in prog["programme"]["reunions"]:
        r_num = r["numOfficiel"]
        hippo = r["hippodrome"]["libelleCourt"]
        for c in r["courses"]:
            c_num = c["numOrdre"]
            course_id = f"R{r_num}C{c_num}"
            try:
                # Récupère les participants pour avoir les cotes actuelles (du matin)
                parts_data = app.get_participants(date_str, r_num, c_num)
                for p in parts_data.get("participants", []):
                    num = p.get("numPmu")
                    rap = p.get("dernierRapportDirect") or p.get("dernierRapportReference")
                    if rap and rap.get("rapport"):
                        odd = float(rap["rapport"])
                        db.save_morning_odd(date_str, hippo, num, odd)
                        count += 1
            except Exception:
                continue
    
    print(f"✅ {count} cotes mémorisées pour la journée.")

if __name__ == "__main__":
    snapshot()
