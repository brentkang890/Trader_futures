# telegram_bot.py
"""
Telegram polling bot sederhana untuk Pro Trader AI.
Environment variables required:
- BOT_TOKEN
- CHAT_ID
- APP_URL (URL FastAPI service)
"""
import os
import time
import requests

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL")

if not BOT_TOKEN or not CHAT_ID or not APP_URL:
    raise ValueError("Environment variables BOT_TOKEN, CHAT_ID, APP_URL harus diset.")

def send_message(text, parse_mode="HTML"):
    if not text:
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text[:4096], "parse_mode": parse_mode}
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("[ERROR] send_message:", e)

def get_updates(offset=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params = {"timeout": 90, "offset": offset}
        return requests.get(url, params=params, timeout=120).json()
    except Exception as e:
        print("[ERROR] get_updates:", e)
        return {}

def handle_command(text):
    if not text:
        return "Pesan kosong."
    t = text.strip().lower()
    if t == "status":
        try:
            r = requests.get(f"{APP_URL}/learning_status", timeout=15); d = r.json()
            return ("🤖 Status Model\n"
                    f"Model: {'✅ Ada' if d.get('model_exists') else '❌ Tidak ada'}\n"
                    f"Data log: {d.get('trade_log_count',0)} sinyal\n"
                    f"Fitur: {', '.join(d.get('features',[])) if d.get('features') else '-'}")
        except Exception as e:
            return f"Error ambil status: {e}"
    if t == "stats":
        try:
            r = requests.get(f"{APP_URL}/ai_performance", timeout=20); d = r.json()
            if "error" in d:
                return f"⚠️ {d['error']}"
            return (f"📈 Statistik\nTotal sinyal: {d.get('total_signals')}\nWinrate: {d.get('winrate')}%\nModel: {d.get('model_status')}")
        except Exception as e:
            return f"Error ambil stats: {e}"
    if t == "log":
        try:
            r = requests.get(f"{APP_URL}/logs_summary", timeout=15); d = r.json()
            if "detail" in d:
                return d["detail"]
            return (f"{d.get('pair')} ({d.get('timeframe')})\nSignal: {d.get('signal_type')}\nEntry: {d.get('entry')}\nTP1: {d.get('tp1')} | TP2: {d.get('tp2')}\nSL: {d.get('sl')}\nConfidence: {d.get('confidence')}")
        except Exception as e:
            return f"Error ambil log: {e}"
    if t.startswith("scalp "):
        try:
            pair = t.split()[1].upper()
            r = requests.get(f"{APP_URL}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=20)
            d = r.json()
            return (f"Scalp {d.get('pair')} ({d.get('timeframe')})\nSignal: {d.get('signal_type')}\nEntry: {d.get('entry')}\nTP1: {d.get('tp1')}\nSL: {d.get('sl')}\nConfidence: {d.get('confidence')}")
        except Exception as e:
            return f"Error scalp: {e}"
    # default: pair [tf]
    parts = t.split()
    if len(parts) == 0:
        return "Format salah. Contoh: <code>BTCUSDT 15m</code>"
    pair = parts[0].upper()
    tf = parts[1] if len(parts) > 1 else "15m"
    try:
        r = requests.get(f"{APP_URL}/pro_signal?pair={pair}&tf_main=1h&tf_entry={tf}&auto_log=true", timeout=25)
        d = r.json()
        if "error" in d:
            return f"⚠️ {d['error']}"
        return (f"{d.get('pair')} ({d.get('timeframe')})\nSignal: {d.get('signal_type')}\nEntry: {d.get('entry')}\nTP1: {d.get('tp1')}\nSL: {d.get('sl')}\nConfidence: {d.get('confidence')}")
    except Exception as e:
        return f"Error pro_signal: {e}"

def main():
    offset = None
    send_message("🤖 Pro Trader Bot aktif. Perintah: status, stats, log, scalp <PAIR>, <PAIR> [TF]")
    while True:
        try:
            upd = get_updates(offset)
            if "result" in upd:
                for u in upd["result"]:
                    offset = u["update_id"] + 1
                    msg = u.get("message", {})
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
                        send_message("Menganalisis chart...")
                        files = {"file": ("chart.jpg", image_data, "image/jpeg")}
                        try:
                            r = requests.post(f"{APP_URL}/analyze_chart", files=files, timeout=60)
                            if r.status_code == 200:
                                d = r.json()
                                send_message(f"{d.get('pair')} ({d.get('timeframe')})\nSignal: {d.get('signal_type')}\nEntry: {d.get('entry')}\nTP1:{d.get('tp1')} SL:{d.get('sl')}\nConfidence:{d.get('confidence')}")
                            else:
                                send_message(f"Gagal analisis gambar: {r.text}")
                        except Exception as e:
                            send_message(f"Error analisis gambar: {e}")
            time.sleep(2)
        except Exception as e:
            print("Loop error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
