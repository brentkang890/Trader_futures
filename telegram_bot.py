import requests
import time
import json
import os

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL")

# === Fungsi Telegram ===
def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[ERROR] Gagal kirim pesan Telegram: {e}")

def get_updates(offset=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {"timeout": 100, "offset": offset}
    try:
        return requests.get(url, params=params, timeout=120).json()
    except Exception as e:
        print(f"[ERROR] Gagal ambil update: {e}")
        return {}

def download_file(file_id):
    """Ambil gambar chart dari Telegram"""
    try:
        file_info = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}").json()
        file_path = file_info["result"]["file_path"]
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        return requests.get(file_url).content
    except Exception as e:
        print(f"[ERROR] Gagal download file: {e}")
        return None

# === Fungsi Command ===
def handle_command(command):
    cmd = command.strip().lower()
    try:
        if cmd == "status":
            r = requests.get(f"{APP_URL}/learning_status_summary", timeout=15)
            d = r.json()
            return (
                f"🤖 <b>Status Model AI</b>\n"
                f"📦 Model: {d['model_status']}\n"
                f"📊 Log Data: {d['log_count']} sinyal\n"
                f"🧠 Kesiapan Retrain: {'✅ Siap' if d['learning_ready'] else '❌ Belum cukup data'}\n"
                f"📋 {d['description']}"
            )

        elif cmd == "log":
            r = requests.get(f"{APP_URL}/logs_summary", timeout=15)
            d = r.json()
            if "detail" in d:
                return f"ℹ️ {d['detail']}"
            return (
                f"📊 <b>{d['pair']} ({d['timeframe']})</b>\n"
                f"💡 Signal: <b>{d['signal_type']}</b>\n"
                f"🎯 Entry: {d['entry']}\n"
                f"🎯 TP1: {d['tp1']}\n"
                f"🎯 TP2: {d['tp2']}\n"
                f"🛡 SL: {d['sl']}\n"
                f"📈 Confidence: {d['confidence']}\n"
                f"🧠 Reasoning: {d['reasoning']}"
            )

        elif cmd.startswith("scalp "):
            pair = cmd.split()[1].upper()
            r = requests.get(f"{APP_URL}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=25)
            d = r.json()
            return (
                f"📊 <b>{d['pair']} ({d['timeframe']})</b>\n"
                f"💡 Signal: <b>{d['signal_type']}</b>\n"
                f"🎯 Entry: {d['entry']}\n"
                f"🎯 TP1: {d['tp1']}\n"
                f"🎯 TP2: {d['tp2']}\n"
                f"🛡 SL: {d['sl']}\n"
                f"📈 Confidence: {d['confidence']}\n"
                f"🧠 Reasoning: {d['reasoning']}"
            )

        else:
            parts = cmd.split()
            if len(parts) == 2:
                pair, tf = parts
            else:
                pair, tf = parts[0], "15m"
            r = requests.get(f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_entry={tf}&auto_log=true", timeout=25)
            d = r.json()
            return (
                f"📊 <b>{d['pair']} ({d['timeframe']})</b>\n"
                f"💡 Signal: <b>{d['signal_type']}</b>\n"
                f"🎯 Entry: {d['entry']}\n"
                f"🎯 TP1: {d['tp1']}\n"
                f"🎯 TP2: {d['tp2']}\n"
                f"🛡 SL: {d['sl']}\n"
                f"📈 Confidence: {d['confidence']}\n"
                f"🧠 Reasoning: {d['reasoning']}"
            )
    except Exception as e:
        return f"❌ Error: {e}"

def handle_photo(file_id):
    """Analisis chart dari gambar dan auto-log hasilnya"""
    send_message("📸 Gambar diterima. Sedang dianalisis oleh AI...")
    img_data = download_file(file_id)
    if not img_data:
        return send_message("⚠️ Gagal mengunduh gambar dari Telegram.")

    try:
        url = f"{APP_URL}/analyze_chart"
        files = {"file": ("chart.jpg", img_data, "image/jpeg")}
        data = {"auto_backtest": "true"}
        r = requests.post(url, files=files, data=data, timeout=90)
        if r.status_code != 200:
            return send_message(f"⚠️ Gagal analisis chart: {r.text}")

        d = r.json()

        # === kirim hasil analisis ke Telegram ===
        msg = (
            f"📊 <b>{d.get('pair','IMG')} ({d.get('timeframe','chart')})</b>\n"
            f"💡 Signal: <b>{d.get('signal_type','')}</b>\n"
            f"🎯 Entry: {d.get('entry','')}\n"
            f"🎯 TP1: {d.get('tp1','')}\n"
            f"🎯 TP2: {d.get('tp2','')}\n"
            f"🛡 SL: {d.get('sl','')}\n"
            f"📈 Confidence: {d.get('confidence','')}\n"
            f"🧠 Reasoning: {d.get('reasoning','')}"
        )
        send_message(msg)

        # === auto log hasil ke AI Agent ===
        payload = {
            "pair": d.get("pair", "IMG"),
            "timeframe": d.get("timeframe", "chart"),
            "signal_type": d.get("signal_type"),
            "entry": d.get("entry"),
            "tp1": d.get("tp1"),
            "tp2": d.get("tp2"),
            "sl": d.get("sl"),
            "confidence": d.get("confidence"),
            "reasoning": d.get("reasoning")
        }
        try:
            requests.post(f"{APP_URL}/pro_signal?auto_log=true", json=payload, timeout=10)
        except:
            pass

        send_message("✅ Hasil analisis gambar telah disimpan ke AI untuk pembelajaran otomatis.")

    except Exception as e:
        send_message(f"❌ Error analisis chart: {e}")

def main():
    offset = None
    send_message("🤖 <b>Pro Trader AI Bot aktif!</b>\n"
                 "Kirim pair (contoh: <b>BTCUSDT 15m</b>) atau kirim <b>gambar chart</b> untuk analisis visual.")
    while True:
        updates = get_updates(offset)
        if "result" in updates:
            for update in updates["result"]:
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                if "text" in msg:
                    response = handle_command(msg["text"])
                    send_message(response)
                elif "photo" in msg:
                    file_id = msg["photo"][-1]["file_id"]
                    handle_photo(file_id)
        time.sleep(2)

if __name__ == "__main__":
    main()
