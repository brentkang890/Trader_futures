# telegram_bot.py (updated)
"""
Telegram polling bot untuk Pro Trader AI
- Menambah dukungan upload CSV (dokumen)
- Menambah perintah `backtest <PAIR>`
- Tetap kompatibel dengan service utama (APP_URL)
"""
import os
import time
import requests

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL")     # contoh: https://your-ai-app.rly.app
BACKTEST_URL = os.environ.get("BACKTEST_URL")  # optional

if not BOT_TOKEN or not CHAT_ID or not APP_URL:
    raise ValueError("Environment variables BOT_TOKEN, CHAT_ID, APP_URL harus diset.")

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

def handle_command(text):
    if not text:
        return "Pesan kosong."
    t = text.strip().lower()

    # BACKTEST
    if t.startswith("backtest"):
        try:
            parts = t.split()
            if len(parts) < 2:
                return "Format: <code>backtest BTCUSDT</code> atau <code>backtest XAUUSD</code>"
            pair = parts[1].upper()
            payload = {"pair": pair, "side": "LONG", "entry": 30000, "tp1": 31000, "sl": 29500, "timeframe": "15m"}
            url = BACKTEST_URL or f"{APP_URL.rstrip('/')}/backtest"
            r = requests.post(url, json=payload, timeout=30)
            d = r.json()
            if "error" in d:
                return f"тЪая╕П Backtest error: {d.get('error')}"
            return (f"ЁЯзо Backtest {d.get('pair')}\nHit: {d.get('hit')}\nPnL: {d.get('pnl_total')}")
        except Exception as e:
            return f"тЪая╕П Gagal backtest: {e}"

    if t == "status":
        try:
            r = requests.get(f"{APP_URL.rstrip('/')}/learning_status", timeout=15)
            d = r.json()
            return ("ЁЯдЦ Status Model\n"
                    f"Model: {'тЬЕ Ada' if d.get('model_exists') else 'тЭМ Tidak ada'}\n"
                    f"Data log: {d.get('trade_log_count',0)} sinyal\n"
                    f"Fitur: {', '.join(d.get('features',[])) if d.get('features') else '-'}")
        except Exception as e:
            return f"Error ambil status: {e}"

    if t == "stats":
        try:
            r = requests.get(f"{APP_URL.rstrip('/')}/ai_performance", timeout=20); d = r.json()
            if "error" in d:
                return f"тЪая╕П {d['error']}"
            return (f"ЁЯУИ Statistik\nTotal: {d.get('total_signals')}\nWinrate: {d.get('winrate')}%\nModel: {d.get('model_status')}")
        except Exception as e:
            return f"Error ambil stats: {e}"

    if t == "log":
        try:
            r = requests.get(f"{APP_URL.rstrip('/')}/logs_summary", timeout=15); d = r.json()
            if "detail" in d:
                return d["detail"]
            return (f"{d.get('pair')} ({d.get('timeframe')})\nSignal: {d.get('signal_type')}\nEntry: {d.get('entry')}\nTP1:{d.get('tp1')} TP2:{d.get('tp2')}\nSL:{d.get('sl')}\nConf:{d.get('confidence')}")
        except Exception as e:
            return f"Error ambil log: {e}"

    if t.startswith("scalp "):
        try:
            pair = t.split()[1].upper()
            r = requests.get(f"{APP_URL.rstrip('/')}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=20)
            d = r.json()
            return (f"Scalp {d.get('pair')} ({d.get('timeframe')})\nSignal: {d.get('signal_type')}\nEntry: {d.get('entry')}\nTP1:{d.get('tp1')} SL:{d.get('sl')}\nConf:{d.get('confidence')}")
        except Exception as e:
            return f"Error scalp: {e}"

    # default: pro_signal <PAIR> [TF]
    parts = t.split()
    if len(parts) == 0:
        return "Format salah. Contoh: <code>BTCUSDT 15m</code>"
    pair = parts[0].upper()
    tf = parts[1] if len(parts) > 1 else "15m"
    try:
        r = requests.get(f"{APP_URL.rstrip('/')}/pro_signal?pair={pair}&tf_main=1h&tf_entry={tf}&auto_log=true", timeout=25)
        d = r.json()
        if "error" in d:
            return f"тЪая╕П {d['error']}"
        return (f"{d.get('pair')} ({d.get('timeframe')})\nSignal: {d.get('signal_type')}\nEntry:{d.get('entry')}\nTP1:{d.get('tp1')} SL:{d.get('sl')}\nConf:{d.get('confidence')}")
    except Exception as e:
        return f"Error pro_signal: {e}"

def main():
    offset = None
    send_message("ЁЯдЦ Pro Trader Bot aktif. Commands: status, stats, log, scalp <PAIR>, backtest <PAIR>, <PAIR> [TF]. Kirim CSV sebagai Dokumen untuk analisis.")
    while True:
        try:
            upd = get_updates(offset)
            if "result" in upd:
                for u in upd["result"]:
                    offset = u["update_id"] + 1
                    msg = u.get("message", {})
                    # text messages
                    if "text" in msg:
                        reply = handle_command(msg["text"])
                        send_message(reply)
                    # photos -> analyze_chart (existing behavior)
                    elif "photo" in msg:
                        photo = msg["photo"][-1]
                        file_data = download_file(photo["file_id"])
                        if not file_data:
                            send_message("Gagal download gambar.")
                            continue
                        send_message("Menganalisis chart (photo)...")
                        files = {"file": ("chart.jpg", file_data, "image/jpeg")}
                        try:
                            r = requests.post(f"{APP_URL.rstrip('/')}/analyze_chart", files=files, timeout=60)
                            if r.status_code == 200:
                                d = r.json()
                                send_message(f"{d.get('pair')} ({d.get('timeframe')})\nSignal: {d.get('signal_type')}\nEntry:{d.get('entry')}\nTP1:{d.get('tp1')} SL:{d.get('sl')}\nConf:{d.get('confidence')}")
                            else:
                                send_message(f"Gagal analisis gambar: {r.text}")
                        except Exception as e:
                            send_message(f"Error analisis gambar: {e}")

                    # documents -> support CSV upload
                    elif "document" in msg:
                        doc = msg["document"]
                        fname = doc.get("file_name","")
                        mime = doc.get("mime_type","")
                        file_data = download_file(doc["file_id"])
                        if not file_data:
                            send_message("Gagal download file.")
                            continue

                        # handle csv files
                        if fname.lower().endswith(".csv") or mime in ("text/csv","application/vnd.ms-excel"):
                            send_message("Menerima CSV. Mengirim ke AI service untuk analisis...")
                            files = {"file": (fname, file_data, "text/csv")}
                            try:
                                r = requests.post(f"{APP_URL.rstrip('/')}/analyze_csv", files=files, timeout=60)
                                if r.status_code == 200:
                                    d = r.json()
                                    send_message(f"Analisis CSV untuk {d.get('pair','CSV')}:\nSignal: {d.get('signal_type')}\nEntry:{d.get('entry')}\nTP1:{d.get('tp1')} SL:{d.get('sl')}\nConf:{d.get('confidence')}")
                                    # optional: if you want bot to also ask AI service to append to trade_log,
                                    # we can POST the resulting signal to a dedicated endpoint (needs to exist).
                                else:
                                    send_message(f"Gagal analisis CSV: {r.text}")
                            except Exception as e:
                                send_message(f"Error saat analisis CSV: {e}")
                        else:
                            send_message("Maaf, hanya file CSV didukung untuk saat ini.")
            time.sleep(1.5)
        except Exception as e:
            print("Loop error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
