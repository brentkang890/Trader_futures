"""
🤖 Pro Trader AI Telegram Bot — Final Premium Version
✨ Fitur Lengkap:
- /signal, /pro, /scalp, /stats, /status, /mode, /autotune, /retrain, /profile
- 🔔 Auto Alert Sinyal baru (confidence ≥ threshold)
- 🚀 Emoji arah sinyal (LONG/SHORT)
- 🔉 Notifikasi suara otomatis (Telegram audio ping)
- HTML aman & stabil untuk Railway
"""

import os
import time
import requests

# === KONFIGURASI DASAR ===
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL", "https://web-production-af34.up.railway.app")
ALERT_INTERVAL = int(os.environ.get("ALERT_INTERVAL", 30))  # detik antar pengecekan sinyal
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.8))  # ambang confidence minimal

if not BOT_TOKEN or not CHAT_ID or not APP_URL:
    raise ValueError("❌ Environment BOT_TOKEN, CHAT_ID, APP_URL wajib diset!")

# === TELEGRAM API HELPER ===
def send_message(text, parse_mode="HTML", sound=True):
    """Kirim pesan ke Telegram (dengan/ tanpa suara notifikasi)"""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": text[:4096],
            "parse_mode": parse_mode,
            "disable_notification": not sound
        }
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("[ERROR send_message]", e)

def get_updates(offset=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params = {"timeout": 100, "offset": offset}
        return requests.get(url, params=params, timeout=120).json()
    except Exception as e:
        print("[ERROR get_updates]", e)
        return {}

# === AUTO ALERT (CEK SINYAL BARU) ===
last_signal_id = None

def check_new_signal():
    """Pantau endpoint /logs_summary untuk sinyal baru"""
    global last_signal_id
    try:
        r = requests.get(f"{APP_URL}/logs_summary", timeout=15)
        if r.status_code != 200:
            return
        d = r.json()

        sig_id = f"{d.get('pair')}-{d.get('signal_type')}-{d.get('entry')}"
        conf = float(d.get("confidence", 0))
        ml_conf = d.get("ml_prob", 0) or 0

        if sig_id != last_signal_id and conf >= CONF_THRESHOLD:
            last_signal_id = sig_id
            signal_type = d.get("signal_type", "")
            emoji = "🚀" if signal_type.upper() == "LONG" else "🩸" if signal_type.upper() == "SHORT" else "⚙️"
            alert_emoji = "🔔" if conf >= 0.9 else "📡"

            msg = (
                f"{alert_emoji} <b>NEW SIGNAL DETECTED</b><br>"
                f"{emoji} <b>{d.get('pair')} ({d.get('timeframe')})</b><br>"
                f"💡 Signal: {signal_type}<br>"
                f"🎯 Entry: {d.get('entry')}<br>"
                f"🏁 TP1: {d.get('tp1')} | 🛑 SL: {d.get('sl')}<br>"
                f"📊 Confidence: {conf:.2f}<br>"
                f"🤖 ML Confidence: {ml_conf * 100:.1f}%<br>"
                f"🧠 {d.get('reasoning', '')[:250]}..."
            )
            send_message(msg, sound=True)
    except Exception as e:
        print("[ERROR auto alert]", e)

# === COMMAND HANDLER ===
def handle_command(text):
    t = text.strip().lower()

    if t in ("/start", "start"):
        return (
            "🤖 <b>Pro Trader AI (SMC Pro)</b> aktif & siap digunakan!<br><br>"
            "📈 Command utama:<br>"
            "/signal BTCUSDT 15m — AI Signal + ML Confidence<br>"
            "/pro BTCUSDT 15m — Analisis penuh SMC<br>"
            "/scalp BTCUSDT — Scalping cepat (3m)<br><br>"
            "📊 Perintah lainnya:<br>"
            "/stats /status /log /mode /autotune /retrain /profile<br><br>"
            "🔔 Auto Alert aktif setiap 30 detik (Confidence ≥ 0.8)"
        )

    if t.startswith("/signal"):
        try:
            parts = t.split()
            pair = parts[1].upper()
            tf = parts[2] if len(parts) > 2 else "15m"
            d = requests.get(f"{APP_URL}/pro_signal?pair={pair}&tf_entry={tf}&auto_log=true", timeout=30).json()
            ml = d.get("ml_prob", 0)
            emoji = "🚀" if d.get("signal_type") == "LONG" else "🩸"
            return (
                f"{emoji} <b>{pair} ({tf})</b><br>"
                f"💡 Signal: {d.get('signal_type')}<br>"
                f"🎯 Entry: {d.get('entry')}<br>"
                f"🏁 TP1: {d.get('tp1')} | 🛑 SL: {d.get('sl')}<br>"
                f"📊 Confidence: {d.get('confidence')}<br>"
                f"🤖 ML Confidence: {ml * 100:.2f}%<br>"
                f"🧠 {d.get('reasoning', '')[:400]}"
            )
        except Exception as e:
            return f"⚠️ Error /signal: {e}"

    if t.startswith("/pro"):
        try:
            parts = t.split()
            pair = parts[1].upper()
            tf = parts[2] if len(parts) > 2 else "15m"
            d = requests.get(f"{APP_URL}/pro_signal?pair={pair}&tf_entry={tf}&auto_log=true", timeout=25).json()
            emoji = "🚀" if d.get("signal_type") == "LONG" else "🩸"
            return (
                f"{emoji} <b>{pair} ({tf})</b><br>"
                f"💡 Signal: {d.get('signal_type')}<br>"
                f"🎯 Entry: {d.get('entry')}<br>"
                f"🏁 TP1: {d.get('tp1')} | 🛑 SL: {d.get('sl')}<br>"
                f"📊 Confidence: {d.get('confidence')}"
            )
        except Exception as e:
            return f"⚠️ Error /pro: {e}"

    if t.startswith("/scalp"):
        try:
            pair = t.split()[1].upper()
            d = requests.get(f"{APP_URL}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=20).json()
            emoji = "🚀" if d.get("signal_type") == "LONG" else "🩸"
            return (
                f"⚡ <b>Scalp {pair}</b> {emoji}<br>"
                f"💡 {d.get('signal_type')}<br>"
                f"🎯 {d.get('entry')} | 🏁 TP1: {d.get('tp1')} | 🛑 SL: {d.get('sl')}<br>"
                f"📊 Confidence: {d.get('confidence')}"
            )
        except Exception as e:
            return f"⚠️ Error /scalp: {e}"

    if t == "/stats":
        try:
            d = requests.get(f"{APP_URL}/ai_performance", timeout=15).json()
            return (
                f"📈 <b>AI Stats</b><br>"
                f"📊 Total: {d.get('total_signals')} sinyal<br>"
                f"✅ Winrate: {d.get('winrate')}%<br>"
                f"💹 Profit Factor: {d.get('profit_factor')}<br>"
                f"🤖 Model: {d.get('model_status')}"
            )
        except Exception as e:
            return f"⚠️ Error /stats: {e}"

    if t == "/autotune":
        requests.get(f"{APP_URL}/force_autotune", timeout=60)
        return "⚙️ Auto-Tune selesai ✅"

    if t == "/retrain":
        send_message("🧠 Melatih ulang model AI...", sound=True)
        requests.post(f"{APP_URL}/retrain_learning", timeout=120)
        return "✅ Retrain selesai!"

    if t.startswith("/mode"):
        mode = t.split()[1]
        d = requests.get(f"{APP_URL}/set_mode?mode={mode}").json()
        return f"✅ Mode diubah ke <b>{d.get('mode')}</b>"

    if t == "/profile":
        d = requests.get(f"{APP_URL}/smc_profiles", timeout=20).json()
        return f"🧩 <b>Profil aktif:</b> {', '.join(d.keys())}"

    return "⚠️ Perintah tidak dikenal. Gunakan /start untuk daftar perintah."

# === LOOP UTAMA ===
def main():
    offset = None
    send_message("🤖 Pro Trader AI Bot aktif! /start untuk bantuan.", sound=False)
    last_alert_check = 0
    while True:
        try:
            # 1️⃣ Cek perintah manual
            upd = get_updates(offset)
            if "result" in upd:
                for u in upd["result"]:
                    offset = u["update_id"] + 1
                    msg = u.get("message", {})
                    if "text" in msg:
                        reply = handle_command(msg["text"])
                        if reply:
                            send_message(reply)

            # 2️⃣ Jalankan auto alert
            if time.time() - last_alert_check > ALERT_INTERVAL:
                check_new_signal()
                last_alert_check = time.time()

            time.sleep(1.5)
        except Exception as e:
            print("[ERROR loop]", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
