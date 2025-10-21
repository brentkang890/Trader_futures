import requests
import time
import os

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
APP_URL = os.environ.get("APP_URL", "")

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

def handle_text_command(command):
    try:
        cmd = command.lower().strip()

        if cmd.startswith("scalp "):
            pair = cmd.split()[1].upper()
            url = f"{APP_URL}/scalp_signal?pair={pair}&tf=3m"
        elif cmd.startswith("log"):
            url = f"{APP_URL}/logs"
        elif cmd.startswith("status"):
            url = f"{APP_URL}/learning_status"
        elif len(cmd.split()) == 2:
            pair, tf = cmd.split()
            url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_entry={tf}"
        elif len(cmd.split()) == 1:
            pair = cmd.split()[0]
            url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_entry=15m"
        else:
            return "⚠️ Format salah!\nGunakan: <b>BTCUSDT 15m</b> atau <b>scalp BTCUSDT</b>"

        print(f"[INFO] Fetching {url}")
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return f"⚠️ Gagal ambil sinyal: {r.text}"

        d = r.json()
        msg = (
            f"📊 <b>{d.get('pair', '')} ({d.get('timeframe', '')})</b>\n"
            f"💡 Signal: <b>{d.get('signal_type', '')}</b>\n"
            f"🎯 Entry: {d.get('entry', '')}\n"
            f"🎯 TP1: {d.get('tp1', '')}\n"
            f"🎯 TP2: {d.get('tp2', '')}\n"
            f"🛡 SL: {d.get('sl', '')}\n"
            f"📈 Confidence: {d.get('confidence', '')}\n\n"
            f"🧠 Reasoning: {d.get('reasoning', '')}"
        )
        return msg

    except Exception as e:
        print(f"[ERROR] {e}")
        return f"❌ Error: {e}"

def handle_photo(photo_id):
    """Analisis gambar chart"""
    try:
        info = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={photo_id}").json()
        file_path = info["result"]["file_path"]
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        img = requests.get(file_url).content

        files = {"file": ("chart.png", img, "image/png")}
        r = requests.post(f"{APP_URL}/analyze_chart", files=files, timeout=60)
        if r.status_code != 200:
            send_message(f"⚠️ Gagal analisis chart: {r.text}")
            return
        d = r.json()
        msg = (
            f"📉 <b>Analisis Gambar Chart</b>\n"
            f"💡 Signal: <b>{d['signal_type']}</b>\n"
            f"🎯 Entry: {d['entry']}\n"
            f"🎯 TP1: {d['tp1']}\n"
            f"🎯 TP2: {d['tp2']}\n"
            f"🛡 SL: {d['sl']}\n"
            f"📈 Confidence: {d['confidence']}\n\n"
            f"🧠 Reasoning: {d['reasoning']}"
        )
        send_message(msg)
    except Exception as e:
        send_message(f"❌ Error analisis gambar: {e}")

def main():
    offset = None
    send_message("🤖 <b>Pro Trader AI Bot aktif & terhubung ke AI Agent!</b>\nKirim: <b>BTCUSDT 15m</b> atau gambar chart.")

    while True:
        updates = get_updates(offset)
        if "result" in updates:
            for upd in updates["result"]:
                offset = upd["update_id"] + 1
                msg = upd.get("message", {})

                if "text" in msg:
                    text = msg["text"].strip()
                    if text.startswith("/start"):
                        send_message("👋 Kirim pair + timeframe (contoh: <b>BTCUSDT 15m</b>)\nAtau kirim gambar chart.")
                    else:
                        send_message(handle_text_command(text))
                elif "photo" in msg:
                    photo_id = msg["photo"][-1]["file_id"]
                    handle_photo(photo_id)
        time.sleep(2)

if __name__ == "__main__":
    main()
