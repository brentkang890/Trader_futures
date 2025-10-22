# telegram_bot.py
"""
🤖 Pro Trader AI Telegram Bot (Polling Mode)
Terhubung ke FastAPI (main_protrader.py)
Perintah yang tersedia:
- status → info model AI
- stats → performa AI
- log → sinyal terakhir
- scalp <PAIR> → sinyal scalping cepat
- <PAIR> [TF] → sinyal analisis normal (contoh: BTCUSDT 15m)
- kirim gambar chart → bot akan analisis chart otomatis
"""

import os
import time
import requests
from io import BytesIO

# ---------------- KONFIGURASI ----------------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL")

if not BOT_TOKEN or not CHAT_ID or not APP_URL:
    raise ValueError("❌ Environment variable BOT_TOKEN, CHAT_ID, APP_URL belum diset di Railway!")

# ---------------- UTILITAS ----------------
def send_message(text, parse_mode="HTML"):
    """Kirim pesan teks ke Telegram"""
    if not text:
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": str(text)[:4096], "parse_mode": parse_mode}
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("[ERROR] send_message:", e)

def get_updates(offset=None):
    """Ambil update terbaru dari Telegram"""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params = {"timeout": 100, "offset": offset}
        return requests.get(url, params=params, timeout=120).json()
    except Exception as e:
        print("[ERROR] get_updates:", e)
        return {}

# ---------------- HANDLER PERINTAH ----------------
def handle_command(text: str) -> str:
    if not text:
        return "⚠️ Pesan kosong."
    text = text.strip().lower()

    # 🔹 Status model AI
    if text == "status":
        try:
            r = requests.get(f"{APP_URL}/learning_status", timeout=20)
            d = r.json()
            msg = (
                "🤖 <b>Status Model AI</b>\n"
                f"📦 Model: {'✅ Ada' if d.get('model_exists') else '❌ Tidak ada'}\n"
                f"📊 Data Log: {d.get('trade_log_count', 0)} sinyal\n"
                f"🧩 Fitur: {', '.join(d.get('features', [])) if d.get('features') else '-'}"
            )
            return msg
        except Exception as e:
            return f"⚠️ Gagal ambil status: {e}"

    # 🔹 Statistik performa AI
    if text == "stats":
        try:
            r = requests.get(f"{APP_URL}/ai_performance", timeout=25)
            d = r.json()
            if "error" in d:
                return f"⚠️ {d['error']}"
            msg = (
                "📈 <b>Statistik Performa AI</b>\n"
                f"📊 Total sinyal: {d.get('total_signals')}\n"
                f"✅ Winrate: {d.get('winrate')}%\n"
                f"💰 Profit Factor: {d.get('profit_factor', 'N/A')}\n"
                f"📉 Max Drawdown: {d.get('max_drawdown', 'N/A')}\n"
                f"⚙️ Model: {d.get('model_status')}"
            )
            return msg
        except Exception as e:
            return f"⚠️ Gagal ambil statistik: {e}"

    # 🔹 Log sinyal terakhir
    if text == "log":
        try:
            r = requests.get(f"{APP_URL}/logs_summary", timeout=20)
            d = r.json()
            if "detail" in d:
                return d["detail"]
            msg = (
                f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
                f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
                f"🎯 Entry: {d.get('entry')}\n"
                f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
                f"🛡 SL: {d.get('sl')}\n"
                f"📈 Confidence: {d.get('confidence')}\n"
                f"🧠 {d.get('reasoning')}"
            )
            return msg
        except Exception as e:
            return f"⚠️ Tidak bisa ambil log: {e}"

    # 🔹 Scalping cepat
    if text.startswith("scalp "):
        try:
            pair = text.split()[1].upper()
            r = requests.get(f"{APP_URL}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=25)
            d = r.json()
            if "error" in d:
                return f"⚠️ {d['error']}"
            msg = (
                f"⚡️ <b>Scalp {d.get('pair')}</b> ({d.get('timeframe')})\n"
                f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
                f"🎯 Entry: {d.get('entry')}\n"
                f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
                f"🛡 SL: {d.get('sl')}\n"
                f"📈 Confidence: {d.get('confidence')}\n"
                f"🧠 {d.get('reasoning')}"
            )
            return msg
        except Exception as e:
            return f"⚠️ Gagal ambil scalp signal: {e}"

    # 🔹 Prediksi normal
    parts = text.split()
    if len(parts) == 0:
        return "⚠️ Format salah. Contoh: <code>BTCUSDT 15m</code>"
    pair = parts[0].upper()
    tf = parts[1] if len(parts) > 1 else "15m"
    try:
        url = f"{APP_URL}/pro_signal?pair={pair}&tf_main=1h&tf_entry={tf}&auto_log=true"
        r = requests.get(url, timeout=30)
        d = r.json()
        if "error" in d:
            return f"⚠️ {d['error']}"
        msg = (
            f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
            f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
            f"🎯 Entry: {d.get('entry')}\n"
            f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
            f"🛡 SL: {d.get('sl')}\n"
            f"📈 Confidence: {d.get('confidence')}\n"
            f"🧠 {d.get('reasoning')}"
        )
        return msg
    except Exception as e:
        return f"❌ Error: {e}"

# ---------------- UTAMA ----------------
def main():
    offset = None
    print(f"🤖 BOT AKTIF — Hubung ke: {APP_URL}")
    send_message(
        "🤖 <b>Pro Trader AI Bot Aktif!</b>\n"
        "Perintah:\n"
        "- status\n- stats\n- log\n- scalp BTCUSDT\n\n"
        "Contoh: <code>XAUUSD 15m</code>\n"
        "Kirim gambar chart untuk analisis otomatis 📷"
    )
    while True:
        try:
            updates = get_updates(offset)
            if "result" in updates:
                for u in updates["result"]:
                    offset = u["update_id"] + 1
                    msg = u.get("message", {})
                    # teks command
                    if "text" in msg:
                        reply = handle_command(msg["text"])
                        send_message(reply)
                    # gambar chart
                    elif "photo" in msg:
                        photo = msg["photo"][-1]
                        file_id = photo["file_id"]
                        file_info = requests.get(
                            f"https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}"
                        ).json()
                        file_path = file_info["result"]["file_path"]
                        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
                        image_data = requests.get(file_url).content

                        send_message("📷 Menganalisis gambar chart, mohon tunggu...")
                        files = {"file": ("chart.jpg", image_data, "image/jpeg")}
                        try:
                            resp = requests.post(f"{APP_URL}/analyze_chart", files=files, timeout=80)
                            if resp.status_code == 200:
                                d = resp.json()
                                msg = (
                                    f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
                                    f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
                                    f"🎯 Entry: {d.get('entry')}\n"
                                    f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
                                    f"🛡 SL: {d.get('sl')}\n"
                                    f"📈 Confidence: {d.get('confidence')}\n"
                                    f"🧠 {d.get('reasoning')}"
                                )
                                send_message(msg)
                            else:
                                send_message(f"⚠️ Gagal analisis gambar: {resp.text}")
                        except Exception as e:
                            send_message(f"⚠️ Error analisis gambar: {e}")
            time.sleep(2)
        except Exception as e:
            print("[ERROR LOOP]", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
