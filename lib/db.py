"""
v5 - Base SQLite pour persistance des paris et historique d'analyses.
"""
import sqlite3
import os
import json
from datetime import datetime, timedelta
from contextlib import contextmanager


DB_PATH = os.environ.get("DB_PATH", "/tmp/turf_cache/turf_v5.db")


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                date_course TEXT NOT NULL,
                course TEXT,
                hippodrome TEXT,
                discipline TEXT,
                cheval TEXT NOT NULL,
                num_pmu INTEGER,
                type_pari TEXT DEFAULT 'simple_gagnant',
                type_detection TEXT DEFAULT 'value', -- NEW: value, gold, coup_sur
                cote REAL NOT NULL,
                mise REAL NOT NULL,
                edge REAL DEFAULT 0,
                chance_calculee REAL DEFAULT 0,
                statut TEXT DEFAULT 'EN_ATTENTE',
                gain REAL DEFAULT 0,
                place INTEGER,
                notes TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_bets_statut ON bets(statut);
            CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(date_course);
            CREATE INDEX IF NOT EXISTS idx_bets_hippo ON bets(hippodrome);

            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cheval TEXT NOT NULL UNIQUE,
                notes TEXT,
                added_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alerts_config (
                id INTEGER PRIMARY KEY,
                min_edge REAL DEFAULT 5.0,
                min_cote REAL DEFAULT 4.0,
                max_cote REAL DEFAULT 50.0,
                enabled INTEGER DEFAULT 1,
                updated_at TEXT
            );

            INSERT OR IGNORE INTO alerts_config (id) VALUES (1);

            CREATE TABLE IF NOT EXISTS odd_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_course TEXT NOT NULL,
                course_id TEXT NOT NULL, -- R1C1
                num_pmu INTEGER NOT NULL,
                morning_odd REAL,
                captured_at TEXT,
                UNIQUE(date_course, course_id, num_pmu)
            );
        """)


# ============================================================
#  Bets
# ============================================================
def add_bet(bet):
    init_db()
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO bets (created_at, date_course, course, hippodrome, discipline,
                              cheval, num_pmu, type_pari, type_detection, cote, mise, edge, chance_calculee, statut)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'EN_ATTENTE')
        """, (datetime.now().isoformat(),
              bet.get("date_course", datetime.now().strftime("%d/%m/%Y")),
              bet.get("course", ""),
              bet.get("hippodrome", ""),
              bet.get("discipline", ""),
              bet["cheval"],
              bet.get("num_pmu"),
              bet.get("type_pari", "simple_gagnant"),
              bet.get("type_detection", "value"),
              float(bet["cote"]),
              float(bet["mise"]),
              float(bet.get("edge", 0)),
              float(bet.get("chance_calculee", 0))))
        bet_id = cur.lastrowid
    return get_bet(bet_id)


def get_bet(bet_id):
    init_db()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
        return dict(row) if row else None


def list_bets(statut=None, limit=200):
    init_db()
    with get_db() as conn:
        if statut:
            rows = conn.execute(
                "SELECT * FROM bets WHERE statut = ? ORDER BY id DESC LIMIT ?",
                (statut, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM bets ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def update_bet_result(bet_id, gagne, place=None):
    init_db()
    with get_db() as conn:
        row = conn.execute("SELECT cote, mise FROM bets WHERE id = ?", (bet_id,)).fetchone()
        if not row:
            return False
        gain = round(row["mise"] * row["cote"], 2) if gagne else 0
        conn.execute("""
            UPDATE bets SET statut = ?, gain = ?, place = ?, resolved_at = ?
            WHERE id = ?
        """, ("GAGNE" if gagne else "PERDU", gain, place,
              datetime.now().isoformat(), bet_id))
    return True


def delete_bet(bet_id):
    init_db()
    with get_db() as conn:
        conn.execute("DELETE FROM bets WHERE id = ?", (bet_id,))


def compute_stats(days=None):
    """Stats globales, optionnellement filtrées sur les N derniers jours."""
    init_db()
    with get_db() as conn:
        query = "SELECT statut, COUNT(*) as c, SUM(mise) as m, SUM(gain) as g FROM bets"
        params = []
        if days:
            since = (datetime.now() - timedelta(days=days)).isoformat()
            query += " WHERE created_at >= ?"
            params.append(since)
        query += " GROUP BY statut"
        rows = {r["statut"]: dict(r) for r in conn.execute(query, params).fetchall()}

    wins = rows.get("GAGNE") or {}
    losses = rows.get("PERDU") or {}
    pending = rows.get("EN_ATTENTE") or {}
    nb_wins = wins.get("c", 0) or 0
    nb_losses = losses.get("c", 0) or 0
    nb_pending = pending.get("c", 0) or 0
    total_resolved = nb_wins + nb_losses
    total_bets = total_resolved + nb_pending
    mise_resolved = (wins.get("m", 0) or 0) + (losses.get("m", 0) or 0)
    gain_total = wins.get("g", 0) or 0
    profit = gain_total - mise_resolved
    roi = (profit / mise_resolved * 100) if mise_resolved else 0

    return {
        "total_bets": total_bets,
        "resolved": total_resolved,
        "wins": nb_wins,
        "losses": nb_losses,
        "en_attente": nb_pending,
        "winrate": round(nb_wins / total_resolved * 100, 2) if total_resolved else 0,
        "total_mise": round(mise_resolved, 2),
        "total_gain": round(gain_total, 2),
        "profit": round(profit, 2),
        "roi": round(roi, 2),
    }


def stats_by_dimension(dimension="hippodrome"):
    """Stats agrégées par hippodrome / discipline / type_pari."""
    init_db()
    if dimension not in ("hippodrome", "discipline", "type_pari"):
        return []
    with get_db() as conn:
        rows = conn.execute(f"""
            SELECT {dimension} as dim,
                   COUNT(*) as nb,
                   SUM(CASE WHEN statut = 'GAGNE' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN statut IN ('GAGNE','PERDU') THEN mise ELSE 0 END) as mise,
                   SUM(gain) as gain
            FROM bets
            WHERE {dimension} IS NOT NULL AND {dimension} != ''
            GROUP BY {dimension}
            HAVING nb > 0
            ORDER BY gain - mise DESC
        """).fetchall()
        result = []
        for r in rows:
            mise = r["mise"] or 0
            gain = r["gain"] or 0
            profit = gain - mise
            roi = (profit / mise * 100) if mise else 0
            result.append({
                "dimension": r["dim"],
                "nb_bets": r["nb"],
                "wins": r["wins"],
                "winrate": round(r["wins"] / r["nb"] * 100, 1),
                "mise": round(mise, 2),
                "gain": round(gain, 2),
                "profit": round(profit, 2),
                "roi": round(roi, 2),
            })
        return result


def cumulative_profit():
    """Évolution du profit cumulé dans le temps (pour graphique)."""
    init_db()
    with get_db() as conn:
        rows = conn.execute("""
            SELECT resolved_at, mise, gain, statut
            FROM bets
            WHERE statut IN ('GAGNE', 'PERDU')
            ORDER BY resolved_at
        """).fetchall()
    cumul = 0
    result = []
    for r in rows:
        if not r["resolved_at"]:
            continue
        delta = (r["gain"] or 0) - r["mise"]
        cumul += delta
        result.append({
            "date": r["resolved_at"][:10],
            "profit_cumule": round(cumul, 2),
            "delta": round(delta, 2),
        })
    return result


# ============================================================
#  Watchlist
# ============================================================
def add_to_watchlist(cheval, notes=""):
    init_db()
    with get_db() as conn:
        try:
            conn.execute("INSERT INTO watchlist (cheval, notes, added_at) VALUES (?, ?, ?)",
                         (cheval, notes, datetime.now().isoformat()))
        except sqlite3.IntegrityError:
            pass


def remove_from_watchlist(cheval):
    init_db()
    with get_db() as conn:
        conn.execute("DELETE FROM watchlist WHERE cheval = ?", (cheval,))


def get_watchlist():
    init_db()
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM watchlist ORDER BY added_at DESC").fetchall()
        return [dict(r) for r in rows]


# ============================================================
#  Alerts config
# ============================================================
def get_alerts_config():
    init_db()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM alerts_config WHERE id = 1").fetchone()
        return dict(row) if row else {"min_edge": 5.0, "min_cote": 4.0, "max_cote": 50.0, "enabled": 1}


def update_alerts_config(min_edge, min_cote, max_cote, enabled):
    init_db()
    with get_db() as conn:
        conn.execute("""
            UPDATE alerts_config
            SET min_edge = ?, min_cote = ?, max_cote = ?, enabled = ?, updated_at = ?
            WHERE id = 1
        """, (min_edge, min_cote, max_cote, int(enabled), datetime.now().isoformat()))


# ============================================================
#  Odd Snapshots (Smart Money)
# ============================================================
def save_morning_odd(date_course, course_id, num_pmu, odd):
    init_db()
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO odd_snapshots (date_course, course_id, num_pmu, morning_odd, captured_at)
            VALUES (?, ?, ?, ?, ?)
        """, (date_course, course_id, num_pmu, float(odd), datetime.now().isoformat()))

def get_morning_odd(date_course, course_id, num_pmu):
    init_db()
    with get_db() as conn:
        row = conn.execute("""
            SELECT morning_odd FROM odd_snapshots 
            WHERE date_course = ? AND course_id = ? AND num_pmu = ?
        """, (date_course, course_id, num_pmu)).fetchone()
        return row["morning_odd"] if row else None


# ============================================================
#  Migration depuis le JSON v4 (si présent)
# ============================================================
def migrate_from_json(json_path):
    if not os.path.exists(json_path):
        return 0
    try:
        with open(json_path, "r") as f:
            old_bets = json.load(f)
    except Exception:
        return 0

    init_db()
    n_migrated = 0
    with get_db() as conn:
        for b in old_bets:
            # Évite les doublons
            existing = conn.execute(
                "SELECT 1 FROM bets WHERE cheval = ? AND cote = ? AND mise = ? AND date_course = ?",
                (b.get("cheval"), b.get("cote"), b.get("mise"), b.get("date"))).fetchone()
            if existing:
                continue
            conn.execute("""
                INSERT INTO bets (created_at, date_course, course, cheval, cote, mise,
                                  edge, statut, gain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (b.get("created_at", datetime.now().isoformat()),
                  b.get("date", ""), b.get("course", ""),
                  b.get("cheval", "?"), float(b.get("cote", 0)),
                  float(b.get("mise", 0)), float(b.get("edge", 0)),
                  b.get("statut", "EN_ATTENTE"), float(b.get("gain", 0))))
            n_migrated += 1
    return n_migrated
