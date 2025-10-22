# telegram_bot.py
"""
Telegram bot untuk Pro Trader AI
Perintah:
- status          -> info model
- stats           -> performa AI
- log             -> sinyal terakhir
- scalp <PAIR>    -> scalp signal (3m)
- <PAIR> [TF]     -> pro_signal (contoh: BTCUSDT 15m atau XAUUSD 1h)
- /sentiment      -> ringkasan sentiment (crypto+macro)
- /mode <PAIR>    -> tampilkan mode (crypto/forex) untuk pair
- /context <PAIR> -> konteks analisis (last price + sentiment)
Bot juga bisa menerima gambar chart (kirim foto) dan akan mem-forward ke /analyze_chart
"""

import os
import time
import requests
from io import BytesIO

# ---------------- KONFIG ----------------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL")

if not BOT_TOKEN or not CHAT_ID or not APP_URL:
    raise ValueError("Environment variable BOT_TOKEN, CHAT_ID, atau APP_URL belum diatur.")

# ---------------- UTILITAS ----------------
def send_message(text, parse_mode="HTML"):
    if not text:
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text[:4096], "parse_mode": parse_mode}
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("[ERROR] Gagal kirim pesan:", e)

def get_updates(offset=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params = {"timeout": 100, "offset": offset}
        return requests.get(url, params=params, timeout=120).json()
    except Exception as e:
        print("[ERROR] Gagal ambil update:", e)
        return {}

# ---------------- HANDLER ----------------
def handle_command(text):
    if not text:
        return "Pesan kosong."
    text = text.strip()
    low = text.lower()

    # slash commands
    if low.startswith("/sentiment"):
        try:
            r = requests.get(f"{APP_URL}/sentiment", timeout=12)
            d = r.json()
            return f"📡 <b>Sentiment</b>\nCrypto: {d.get('crypto_sentiment')}\nMacro: {d.get('macro_sentiment')}"
        except Exception as e:
            return f"⚠️ Gagal ambil sentiment: {e}"

    if low.startswith("/mode"):
        parts = text.split()
        if len(parts) < 2:
            return "Format: /mode <PAIR>  (contoh: /mode BTCUSDT)"
        pair = parts[1].upper()
        try:
            r = requests.get(f"{APP_URL}/mode?pair={pair}", timeout=12)
            d = r.json()
            return f"🔎 Mode for {pair}: {d.get('mode')}\nSumber: {', '.join(d.get('data_sources') or [])}"
        except Exception as e:
            return f"⚠️ Gagal ambil mode: {e}"

    if low.startswith("/context"):
        parts = text.split()
        if len(parts) < 2:
            return "Format: /context <PAIR>"
        pair = parts[1].upper()
        try:
            r = requests.get(f"{APP_URL}/context?pair={pair}", timeout=12)
            d = r.json()
            return f"🧭 Context {pair}\nMode: {d.get('mode')}\nLast price: {d.get('last_price')}\nSentiment: {d.get('crypto_sentiment') or d.get('macro_sentiment')}"
        except Exception as e:
            return f"⚠️ Gagal ambil context: {e}"

    # legacy text commands (no slash)
    low = text.lower()
    if low == "status":
        try:
            r = requests.get(f"{APP_URL}/learning_status", timeout=12)
            d = r.json()
            return (
                "🤖 <b>Status Model AI</b>\n"
                f"📦 Model: {'✅ Ada' if d.get('model_exists') else '❌ Tidak ada'}\n"
                f"📊 Data Log: {d.get('trade_log_count', 0)} sinyal\n"
                f"🧩 Fitur: {', '.join(d.get('features', [])) if d.get('features') else '-'}"
            )
        except Exception as e:
            return f"⚠️ Gagal ambil status: {e}"

    if low == "stats":
        try:
            r = requests.get(f"{APP_URL}/ai_performance", timeout=12)
            d = r.json()
            if "error" in d:
                return f"⚠️ {d['error']}"
            return (
                "📈 <b>Statistik Performa AI</b>\n"
                f"📊 Total sinyal: {d.get('total_signals')}\n"
                f"✅ Winrate: {d.get('winrate')}%\n"
                f"⚙️ Model: {d.get('model_status')}"
            )
        except Exception as e:
            return f"⚠️ Gagal ambil statistik: {e}"

    if low == "log":
        try:
            r = requests.get(f"{APP_URL}/logs_summary", timeout=12)
            d = r.json()
            if "detail" in d:
                return d["detail"]
            return (
                f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
                f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
                f"🎯 Entry: {d.get('entry')}\n"
                f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
                f"🛡 SL: {d.get('sl')}\n"
                f"📈 Confidence: {d.get('confidence')}\n"
                f"🧠 {d.get('reasoning')}"
            )
        except Exception as e:
            return f"⚠️ Gagal ambil log: {e}"

    if low.startswith("scalp "):
        try:
            pair = text.split()[1].upper()
            r = requests.get(f"{APP_URL}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=18)
            d = r.json()
            return (
                f"⚡️ <b>Scalp {d.get('pair')}</b> ({d.get('timeframe')})\n"
                f"💡 {d.get('signal_type')} — Entry {d.get('entry')} | TP1 {d.get('tp1')} | SL {d.get('sl')}\n"
                f"Confidence: {d.get('confidence')}"
            )
        except Exception as e:
            return f"⚠️ Gagal ambil scalp: {e}"

    # default: assume "<PAIR> [TF]"
    parts = text.split()
    if len(parts) == 0:
        return "Format salah. Contoh: BTCUSDT 15m"
    pair = parts[0].upper()
    tf = parts[1] if len(parts) > 1 else "15m"
    try:
        r = requests.get(f"{APP_URL}/pro_signal?pair={pair}&tf_main=1h&tf_entry={tf}&auto_log=true", timeout=25)
        d = r.json()
        if "error" in d:
            return f"⚠️ {d['error']}"
        return (
            f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
            f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
            f"🎯 Entry: {d.get('entry')}\n"
            f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
            f"🛡 SL: {d.get('sl')}\n"
            f"📈 Confidence: {d.get('confidence')}\n"
            f"🧠 {d.get('reasoning')}"
        )
    except Exception as e:
        return f"❌ Error: {e}"

# ---------------- PROGRAM UTAMA ----------------
def main():
    offset = None
    print(f"🤖 BOT AKTIF — Hubung ke: {APP_URL}")
    send_message("🤖 <b>Pro Trader AI Bot Aktif!</b>\nKetik contoh:\n<code>BTCUSDT 15m</code> atau <code>XAUUSD 1h</code>\nPerintah tambahan: /sentiment, /mode <PAIR>, /context <PAIR>")

    while True:
        try:
            updates = get_updates(offset)
            if "result" in updates:
                for update in updates["result"]:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    if not msg:
                        continue
                    # text
                    if "text" in msg:
                        reply = handle_command(msg["text"])
                        send_message(reply)
                    # photo
                    elif "photo" in msg:
                        photo = msg["photo"][-1]
                        file_id = photo["file_id"]
                        file_info = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}").json()
                        file_path = file_info["result"]["file_path"]
                        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
                        image_data = requests.get(file_url).content
                        send_message("📷 Menganalisis gambar chart, mohon tunggu...")
                        files = {"file": ("chart.jpg", image_data, "image/jpeg")}
                        resp = requests.post(f"{APP_URL}/analyze_chart", files=files, timeout=60)
                        if resp.status_code == 200:
                            d = resp.json()
                            caption = (
                                f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
                                f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
                                f"🎯 Entry: {d.get('entry')}\n"
                                f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
                                f"🛡 SL: {d.get('sl')}\n"
                                f"📈 Confidence: {d.get('confidence')}\n"
                                f"🧠 {d.get('reasoning')}"
                            )
                            send_message(caption)
                        else:
                            send_message(f"⚠️ Gagal analisis gambar: {resp.text}")
            time.sleep(2)
        except Exception as e:
            print("[ERROR LOOP]", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
