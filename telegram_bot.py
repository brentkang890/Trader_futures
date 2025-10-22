# ======================================================
# 🤖 PRO TRADER AI - TELEGRAM BOT PRO
# Auto-Analyze Chart + AI Signal Generator (Crypto & Forex)
# ======================================================

import os
import time
import requests
from io import BytesIO

# ---------------- KONFIG ----------------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL")

if not BOT_TOKEN or not CHAT_ID or not APP_URL:
    raise ValueError("❌ BOT_TOKEN, CHAT_ID, atau APP_URL belum diatur di Railway.")

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ---------------- UTILITAS ----------------
def send_message(text, chat_id=None, parse_mode="HTML"):
    """Kirim pesan ke Telegram"""
    chat_id = chat_id or CHAT_ID
    if not text:
        return
    try:
        requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": chat_id,
            "text": text[:4096],
            "parse_mode": parse_mode
        }, timeout=10)
    except Exception as e:
        print("[ERROR] Gagal kirim pesan:", e)

def get_updates(offset=None):
    """Ambil pesan terbaru"""
    try:
        res = requests.get(f"{BASE_URL}/getUpdates", params={
            "timeout": 60, "offset": offset
        }, timeout=90)
        return res.json()
    except Exception as e:
        print("[ERROR] Gagal ambil update:", e)
        return {}

def analyze_pair(pair, tf):
    """Minta sinyal AI Agent"""
    url = f"{APP_URL}/pro_signal?pair={pair}&tf_main=1h&tf_entry={tf}&auto_log=true"
    try:
        r = requests.get(url, timeout=30)
        d = r.json()
    except Exception as e:
        return f"❌ Error fetch AI Agent: {e}"
    if "error" in d:
        return f"⚠️ {d['error']}"
    return format_signal(d)

def format_signal(d):
    """Format hasil analisis AI"""
    return (
        f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
        f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
        f"🎯 Entry: {d.get('entry')}\n"
        f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
        f"🛡 SL: {d.get('sl')}\n"
        f"📈 Confidence: {d.get('confidence')}\n"
        f"🧠 {d.get('reasoning')}"
    )

# ---------------- HANDLER ----------------
def handle_command(text):
    text = text.strip().lower()

    if text in ["/start", "start"]:
        return (
            "🤖 <b>Selamat datang di Pro Trader AI Bot</b>\n"
            "Saya bisa membantu analisis Crypto & Forex secara otomatis.\n\n"
            "🧭 <b>Contoh perintah:</b>\n"
            "- <code>BTCUSDT 15m</code>\n"
            "- <code>XAUUSD 1h</code>\n"
            "- <code>scalp BTCUSDT</code>\n"
            "- <code>status</code> / <code>stats</code> / <code>log</code>\n"
            "- <code>/train</code> untuk retrain model AI\n\n"
            "📷 Kirim <b>gambar chart</b> untuk analisis otomatis."
        )

    if text == "status":
        r = requests.get(f"{APP_URL}/learning_status", timeout=15).json()
        return (
            "📦 <b>Status AI Agent</b>\n"
            f"Model: {'✅ Ada' if r.get('model_exists') else '❌ Tidak ada'}\n"
            f"Data Log: {r.get('trade_log_count', 0)} sinyal\n"
            f"Fitur: {', '.join(r.get('features', [])) if r.get('features') else '-'}"
        )

    if text == "stats":
        r = requests.get(f"{APP_URL}/ai_performance", timeout=20).json()
        if "error" in r:
            return f"⚠️ {r['error']}"
        return (
            "📈 <b>Statistik AI</b>\n"
            f"Total sinyal: {r['total_signals']}\n"
            f"✅ Winrate: {r['winrate']}%\n"
            f"💰 Profit Factor: {r.get('profit_factor', 'N/A')}\n"
            f"📉 Max Drawdown: {r.get('max_drawdown', 'N/A')}"
        )

    if text == "log":
        r = requests.get(f"{APP_URL}/logs_summary", timeout=15).json()
        if "error" in r:
            return f"⚠️ {r['error']}"
        return format_signal(r)

    if text == "/train":
        send_message("🔄 Melatih ulang model AI, mohon tunggu...")
        r = requests.get(f"{APP_URL}/retrain_learning", timeout=60).json()
        return f"✅ Model retrain selesai.\n📊 Samples: {r.get('samples')} | Status: {r.get('status')}"

    if text.startswith("scalp "):
        pair = text.split()[1].upper()
        r = requests.get(f"{APP_URL}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=25).json()
        return format_signal(r)

    parts = text.split()
    if len(parts) >= 1:
        pair = parts[0].upper()
        tf = parts[1] if len(parts) > 1 else "15m"
        return analyze_pair(pair, tf)

    return "⚠️ Perintah tidak dikenali."

# ---------------- PROGRAM UTAMA ----------------
def main():
    offset = None
    print(f"🤖 PRO TRADER AI BOT PRO — Terhubung ke {APP_URL}")
    send_message("🤖 <b>Pro Trader AI Bot PRO Aktif!</b>\nKirim <code>/start</code> untuk bantuan.")

    while True:
        try:
            updates = get_updates(offset)
            if "result" in updates:
                for update in updates["result"]:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    chat_id = msg.get("chat", {}).get("id", CHAT_ID)

                    # 🔹 Teks (perintah manual)
                    if "text" in msg:
                        text = msg["text"]
                        reply = handle_command(text)
                        send_message(reply, chat_id)

                    # 🔹 Gambar chart (auto-analyze)
                    elif "photo" in msg:
                        photo = msg["photo"][-1]
                        file_id = photo["file_id"]
                        file_info = requests.get(f"{BASE_URL}/getFile?file_id={file_id}").json()
                        file_path = file_info["result"]["file_path"]
                        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
                        image_data = requests.get(file_url).content

                        send_message("📷 Menganalisis gambar chart otomatis...", chat_id)
                        files = {"file": ("chart.jpg", image_data, "image/jpeg")}
                        resp = requests.post(f"{APP_URL}/analyze_chart", files=files, timeout=90)

                        if resp.status_code == 200:
                            d = resp.json()
                            send_message(format_signal(d), chat_id)
                        else:
                            send_message(f"⚠️ Gagal analisis chart: {resp.text}", chat_id)

            time.sleep(2)
        except Exception as e:
            print("[ERROR LOOP]", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
