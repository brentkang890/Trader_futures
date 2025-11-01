"""
telegram_bot.py
Pro Trader AI Hybrid (SMC + ICT PRO + XGBoost)
Final Version — Smart Context + Natural Input + Auto Notify + Professional Greeting
"""

import os
import re
import time
import requests
import threading

# ===============================
# CONFIG
# ===============================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
APP_URL = os.getenv("APP_URL", "https://your-app-name.up.railway.app")

AUTO_MODE = os.getenv("AUTO_MODE", "HYBRID")  # SIGNAL_UPDATE / STATUS_UPDATE / HYBRID
AUTO_INTERVAL_MIN = int(os.getenv("AUTO_INTERVAL_MIN", 60))

if not BOT_TOKEN or not CHAT_ID or not APP_URL:
    raise ValueError("❌ Env BOT_TOKEN, CHAT_ID, dan APP_URL harus diset di Railway.")

# ===============================
# CORE TELEGRAM FUNCTIONS
# ===============================
def send_message(text, parse_mode="HTML"):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text[:4096], "parse_mode": parse_mode}
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("[ERROR] send_message:", e)

def get_updates(offset=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params = {"timeout": 100, "offset": offset}
        return requests.get(url, params=params, timeout=120).json()
    except Exception as e:
        print("[ERROR] get_updates:", e)
        return {}

def download_file(file_id):
    try:
        info = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}").json()
        path = info["result"]["file_path"]
        url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{path}"
        r = requests.get(url, timeout=60)
        return r.content
    except Exception as e:
        print("download_file error:", e)
        return None

# ===============================
# SMART CONTEXT + AUTO TRANSLATE
# ===============================
last_pair_context = None

def detect_pair_and_tf(text: str):
    global last_pair_context

    translate_map = {
        "menit": "m", "jam": "h", "hari": "d",
        "m15": "15m", "h1": "1h", "h4": "4h",
        "1jam": "1h", "4jam": "4h",
        "gold": "XAUUSD", "emas": "XAUUSD",
        "bitcoin": "BTCUSDT", "btc": "BTCUSDT",
        "eth": "ETHUSDT", "sol": "SOLUSDT",
        "euro": "EURUSD"
    }
    for k, v in translate_map.items():
        text = re.sub(k, v, text, flags=re.IGNORECASE)

    pair_match = re.search(r"\b([A-Z]{3,6}(USDT|USD|EUR|JPY|GBP|IDR))\b", text.upper())
    tf_match = re.search(r"(\d+[mhHdD])", text)

    pair = pair_match.group(1) if pair_match else last_pair_context
    tf = tf_match.group(1).lower() if tf_match else "15m"

    if pair:
        last_pair_context = pair
    return pair, tf

def handle_smart_message(text: str):
    pair, tf = detect_pair_and_tf(text)
    if not pair:
        return "⚠️ Tidak menemukan pair dalam pesan kamu. Contoh: <code>BTCUSDT 15m</code>"
    try:
        send_message(f"🤖 Menganalisis {pair} ({tf}) berdasarkan perintah kamu...")
        r = requests.post(f"{APP_URL.rstrip('/')}/signal", json={"pair": pair, "timeframe": tf}, timeout=60)
        d = r.json()
        if "error" in d:
            return f"⚠️ {d['error']}"
        return (
            f"📊 <b>{d.get('pair')} ({d.get('timeframe')})</b>\n"
            f"💡 <b>{d.get('signal_type')}</b>\n"
            f"🎯 Entry: {d.get('entry')}\n"
            f"🏁 TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
            f"🛑 SL: {d.get('sl')}\n"
            f"📊 Confidence: {d.get('confidence')}\n"
            f"🧠 Reasoning: {d.get('reasoning')}"
        )
    except Exception as e:
        return f"⚠️ Error analisis: {e}"

# ===============================
# COMMAND HANDLER
# ===============================
def handle_command(text):
    t = text.strip().lower()

    # START (Professional Greeting)
    if t in ("/start", "start"):
        return (
            "🤖 <b>Selamat datang, Trader!</b>\n\n"
            "Saya <b>Pro Trader AI</b> — asisten pribadi kamu di dunia Forex & Crypto.\n"
            "Saya menggunakan strategi <b>ICT (Inner Circle Trader)</b> dan <b>Smart Money Concepts</b>, "
            "serta didukung oleh <b>Machine Learning XGBoost</b> yang terus belajar dari hasil trading kamu.\n\n"
            "📈 <b>Saya bisa bantu kamu:</b>\n"
            "• Analisis otomatis: <code>BTCUSDT 15m</code>\n"
            "• Prediksi arah pasar multi-timeframe\n"
            "• Baca file CSV dan chart untuk deteksi pola\n"
            "• Kirim sinyal otomatis tiap jam (AI Mode)\n"
            "• Menjelaskan reasoning di balik setiap sinyal\n\n"
            "💡 <b>Contoh:</b>\n"
            "<code>analisa XAUUSD 1h</code> atau <code>ETHUSDT 4h</code>\n\n"
            "🚀 Ayo kita mulai perjalanan trading profesional bersama!"
        )

    # SIGNAL
    if t.startswith("/signal"):
        parts = t.split()
        if len(parts) < 2:
            return "⚙️ Format: <code>/signal BTCUSDT 15m</code>"
        pair = parts[1].upper()
        tf = parts[2] if len(parts) > 2 else "15m"
        send_message(f"⏳ Menganalisis {pair} ({tf}) ...")
        r = requests.post(f"{APP_URL.rstrip('/')}/signal", json={"pair": pair, "timeframe": tf}, timeout=60)
        d = r.json()
        if "error" in d:
            return f"⚠️ {d['error']}"
        return (
            f"📊 <b>{d.get('pair')} ({d.get('timeframe')})</b>\n"
            f"💡 <b>{d.get('signal_type')}</b>\n"
            f"🎯 Entry: {d.get('entry')}\n"
            f"🏁 TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
            f"🛑 SL: {d.get('sl')}\n"
            f"📊 Confidence: {d.get('confidence')}\n"
            f"🧠 Reasoning: {d.get('reasoning')}"
        )

    # SCALP
    if t.startswith("/scalp"):
        parts = t.split()
        pair = parts[1].upper() if len(parts) > 1 else "BTCUSDT"
        r = requests.get(f"{APP_URL.rstrip('/')}/scalp_signal?pair={pair}&tf=3m", timeout=40)
        d = r.json()
        return (
            f"⚡ <b>Scalp {pair}</b>\n"
            f"💡 {d.get('signal_type')} | Conf: {d.get('confidence')}\n"
            f"🎯 Entry: {d.get('entry')}\n"
            f"🏁 TP1: {d.get('tp1')} | 🛑 SL: {d.get('sl')}\n"
            f"🧠 {d.get('reasoning')}"
        )

    # STATUS
    if t == "/status":
        r1 = requests.get(f"{APP_URL.rstrip('/')}/learning_status", timeout=10).json()
        r2 = requests.get(f"{APP_URL.rstrip('/')}/ai_performance", timeout=10).json()
        return (
            f"🤖 <b>Status Model AI</b>\n"
            f"📦 Model: {'✅ Ada' if r1.get('model_exists') else '❌ Tidak ada'}\n"
            f"🧮 Log: {r1.get('trade_log_count', 0)} data\n"
            f"🧠 Algo: {r1.get('algo')}\n\n"
            f"📈 <b>Performa:</b>\n"
            f"✅ Winrate: {r2.get('winrate')}%\n"
            f"💹 Profit Factor: {r2.get('profit_factor')}"
        )

    # RETRAIN
    if t == "/retrain":
        send_message("🔁 Melatih ulang model AI...")
        r = requests.post(f"{APP_URL.rstrip('/')}/retrain_learning", timeout=120)
        d = r.json()
        return f"✅ Model retrain selesai!\n📊 Sample: {d.get('samples')}"

    # NATURAL INPUT
    if any(k in t for k in ["analisa", "cek", "lihat", "prediksi", "signal", "sinyal", "buy", "sell", "tolong"]):
        return handle_smart_message(text)

    # PAIR ONLY
    parts = t.split()
    if len(parts) >= 1:
        pair = parts[0].upper()
        tf = parts[1] if len(parts) > 1 else "15m"
        if re.match(r"^[A-Z0-9]+(USDT|USD|EUR|JPY|GBP|IDR)?$", pair):
            return handle_smart_message(text)

    return "❌ Perintah tidak dikenal. Ketik /start untuk daftar lengkap."

# ===============================
# AUTO NOTIFICATION
# ===============================
def send_auto_update():
    try:
        if AUTO_MODE in ("SIGNAL_UPDATE", "HYBRID"):
            r = requests.post(f"{APP_URL.rstrip('/')}/signal", json={"pair": "BTCUSDT", "timeframe": "15m"}, timeout=60)
            d = r.json()
            msg = (
                f"📡 <b>Auto Signal Update</b>\n"
                f"📊 {d.get('pair')} ({d.get('timeframe')})\n"
                f"💡 {d.get('signal_type')} | Conf: {d.get('confidence')}\n"
                f"🎯 Entry: {d.get('entry')} | 🛑 SL: {d.get('sl')}\n"
                f"🧠 {d.get('reasoning')}"
            )
            send_message(msg)

        if AUTO_MODE in ("STATUS_UPDATE", "HYBRID"):
            r = requests.get(f"{APP_URL.rstrip('/')}/ai_performance", timeout=20)
            d = r.json()
            msg = (
                f"📈 <b>AI Performance Update</b>\n"
                f"✅ Winrate: {d.get('winrate')}%\n"
                f"💹 Profit Factor: {d.get('profit_factor')}\n"
                f"📊 Total Signals: {d.get('total_signals')}"
            )
            send_message(msg)
    except Exception as e:
        print("[AUTO UPDATE ERROR]", e)

def auto_scheduler():
    send_message(f"🔔 Auto notification aktif (interval {AUTO_INTERVAL_MIN} menit, mode: {AUTO_MODE})")
    while True:
        send_auto_update()
        time.sleep(AUTO_INTERVAL_MIN * 60)

# ===============================
# MAIN LOOP
# ===============================
def main():
    offset = None
    send_message("🤖 Pro Trader AI Hybrid aktif!\nKetik /start untuk daftar perintah.")
    while True:
        try:
            updates = get_updates(offset)
            if "result" in updates:
                for u in updates["result"]:
                    offset = u["update_id"] + 1
                    msg = u.get("message", {})

                    if "text" in msg:
                        reply = handle_command(msg["text"])
                        send_message(reply)

                    elif "photo" in msg:
                        photo = msg["photo"][-1]
                        file_data = download_file(photo["file_id"])
                        if not file_data:
                            send_message("⚠️ Gagal unduh gambar.")
                            continue
                        send_message("🖼️ Menganalisis chart...")
                        files = {"file": ("chart.jpg", file_data, "image/jpeg")}
                        r = requests.post(f"{APP_URL.rstrip('/')}/analyze_chart", files=files, timeout=60)
                        d = r.json()
                        send_message(
                            f"📊 Chart Analysis\n"
                            f"💡 {d.get('signal_type')}\n"
                            f"🎯 Entry: {d.get('entry')}\n"
                            f"🏁 TP1: {d.get('tp1')} | 🛑 SL: {d.get('sl')}\n"
                            f"🧠 {d.get('reasoning')}"
                        )

                    elif "document" in msg:
                        doc = msg["document"]
                        fname = doc.get("file_name", "")
                        file_data = download_file(doc["file_id"])
                        if not file_data:
                            send_message("⚠️ Gagal unduh file.")
                            continue
                        if not fname.lower().endswith(".csv"):
                            send_message("⚠️ Hanya file CSV yang didukung.")
                            continue
                        send_message("📄 CSV diterima, sedang dianalisis...")
                        files = {"file": (fname, file_data, "text/csv")}
                        r = requests.post(f"{APP_URL.rstrip('/')}/analyze_csv", files=files, timeout=60)
                        d = r.json()
                        send_message(
                            f"✅ CSV Analysis\n"
                            f"📊 {d.get('pair')} ({d.get('timeframe')})\n"
                            f"💡 {d.get('signal_type')}\n"
                            f"🎯 Entry: {d.get('entry')}\n"
                            f"🏁 TP1: {d.get('tp1')} | 🛑 SL: {d.get('sl')}\n"
                            f"📊 Confidence: {d.get('confidence')}"
                        )
            time.sleep(1.5)
        except Exception as e:
            print("Loop error:", e)
            time.sleep(5)

# ===============================
# RUN BOT + AUTO UPDATER
# ===============================
if __name__ == "__main__":
    t_bot = threading.Thread(target=main)
    t_auto = threading.Thread(target=auto_scheduler)
    t_bot.start()
    t_auto.start()
