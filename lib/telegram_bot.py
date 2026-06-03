"""
lib/telegram_bot.py — Notifications pour GOLD et COUP SUR.
"""
import requests
import os

# À configurer dans vos variables d'environnement ou directement ici
TOKEN = os.environ.get("TELEGRAM_TOKEN", "VOTRE_TOKEN_ICI")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "VOTRE_CHAT_ID_ICI")

def send_message(text):
    if not TOKEN or "VOTRE" in TOKEN:
        return False
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"[Telegram] Erreur: {e}")
        return False

def notify_bet(bet_data):
    """
    Formatte et envoie une alerte pour un pari GOLD ou COUP SUR.
    """
    is_gold = bet_data.get("isGold")
    is_coup_sur = bet_data.get("isCoupSur")
    
    if not is_gold and not is_coup_sur:
        return
    
    icon = "✅ COUP SÛR" if is_coup_sur else "⭐ IA GOLD"
    
    msg = f"<b>{icon} DÉTECTÉ !</b>\n\n"
    msg += f"🐎 <b>{bet_data['nom']}</b> (#{bet_data['numPmu']})\n"
    msg += f"🏁 Course: {bet_data.get('course_id', 'N/A')}\n"
    msg += f"🕒 Départ: {bet_data.get('heure', '--:--')}\n\n"
    msg += f"📈 Cote: <b>{bet_data['cote']}</b>\n"
    msg += f"🎯 Chance: {bet_data['chance']}%\n"
    msg += f"💰 Mise Kelly: <b>{bet_data.get('kellyMise', 0)}€</b>\n"
    
    if bet_data.get("drop_pct", 0) > 10:
        msg += f"\n⚠️ <b>SMART MONEY : -{bet_data['drop_pct']}% de chute !</b>"
        
    return send_message(msg)
