import os
import time
import requests
from io import BytesIO

# ---------------- KONFIG ----------------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL")

if not BOT_TOKEN or not CHAT_ID or not APP_URL:
    raise ValueError("❌ Environment variable BOT_TOKEN, CHAT_ID, atau APP_URL belum diatur di Railway.")

# ---------------- UTILITAS ----------------
def send_message(text, parse_mode="HTML"):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text[:4096], "parse_mode": parse_mode}
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("[ERROR] Kirim pesan:", e)

def get_updates(offset=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params = {"timeout": 100, "offset": offset}
        return requests.get(url, params=params, timeout=120).json()
    except Exception as e:
        print("[ERROR] Gagal ambil update:", e)
        return {}

def send_photo(image_bytes, caption="📈 Hasil Analisis Chart", parse_mode="HTML"):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        files = {"photo": ("chart.jpg", image_bytes)}
        data = {"chat_id": CHAT_ID, "caption": caption, "parse_mode": parse_mode}
        requests.post(url, files=files, data=data, timeout=20)
    except Exception as e:
        print("[ERROR] Kirim foto gagal:", e)

# ---------------- HANDLER ----------------
def handle_command(text):
    if not text:
        return "⚠️ Pesan kosong."
    text = text.strip().lower()

    # Perintah umum
    if text == "status":
        r = requests.get(f"{APP_URL}/learning_status", timeout=20).json()
        return (
            "🤖 <b>Status Model AI</b>\n"
            f"📦 Model: {'✅ Ada' if r.get('model_exists') else '❌ Tidak ada'}\n"
            f"📊 Data Log: {r.get('trade_log_count', 0)} sinyal\n"
            f"🧩 Fitur: {', '.join(r.get('features', [])) if r.get('features') else '-'}"
        )

    if text == "stats":
        r = requests.get(f"{APP_URL}/ai_performance", timeout=20).json()
        if "error" in r:
            return f"⚠️ {r['error']}"
        return (
            "📈 <b>Statistik Performa AI</b>\n"
            f"📊 Total sinyal: {r['total_signals']}\n"
            f"✅ Winrate: {r['winrate']}%\n"
            f"💰 Profit Factor: {r.get('profit_factor', 'N/A')}\n"
            f"📉 Max Drawdown: {r.get('max_drawdown', 'N/A')}\n"
            f"⚙️ Model: {r['model_status']}"
        )

    if text == "log":
        r = requests.get(f"{APP_URL}/logs_summary", timeout=15).json()
        if "detail" in r:
            return r["detail"]
        return (
            f"📊 <b>{r.get('pair')}</b> ({r.get('timeframe')})\n"
            f"💡 Signal: <b>{r.get('signal_type')}</b>\n"
            f"🎯 Entry: {r.get('entry')}\n"
            f"TP1: {r.get('tp1')} | TP2: {r.get('tp2')}\n"
            f"🛡 SL: {r.get('sl')}\n"
            f"📈 Confidence: {r.get('confidence')}\n"
            f"🧠 {r.get('reasoning')}"
        )

    # Perintah khusus (pair & timeframe)
    parts = text.split()
    if len(parts) == 0:
        return "⚠️ Format salah. Contoh: <code>BTCUSDT 15m</code>"
    elif len(parts) == 1:
        pair, tf = parts[0], "15m"
    else:
        pair, tf = parts[0], parts[1]

    url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_main=1h&tf_entry={tf}&auto_log=true"
    r = requests.get(url, timeout=30).json()
    if "error" in r:
        return f"⚠️ {r['error']}"
    return (
        f"📊 <b>{r.get('pair')}</b> ({r.get('timeframe')})\n"
        f"💡 Signal: <b>{r.get('signal_type')}</b>\n"
        f"🎯 Entry: {r.get('entry')}\n"
        f"TP1: {r.get('tp1')} | TP2: {r.get('tp2')}\n"
        f"🛡 SL: {r.get('sl')}\n"
        f"📈 Confidence: {r.get('confidence')}\n"
        f"🧠 {r.get('reasoning')}"
    )

# ---------------- PROGRAM UTAMA ----------------
def main():
    offset = None
    print(f"🤖 BOT AKTIF — Hubung ke: {APP_URL}")
    send_message(
        "🤖 <b>Pro Trader AI Bot Aktif!</b>\n"
        "Ketik contoh:\n<code>BTCUSDT 15m</code> atau <code>XAUUSD 1h</code>\n\n"
        "Perintah lain:\n- status\n- stats\n- log"
    )

    while True:
        try:
            updates = get_updates(offset)
            if "result" in updates:
                for update in updates["result"]:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    if "text" in msg:
                        reply = handle_command(msg["text"])
                        send_message(reply)
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
