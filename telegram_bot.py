import requests
import time
import json
import os

BOT_TOKEN = os.environ.get("BOT_TOKEN", "8483103988:AAHeHGuHA6T0rx6nRN-w5bgGrYlf0kbmgHs")
CHAT_ID = os.environ.get("CHAT_ID", "6123645566")
APP_URL = os.environ.get("APP_URL", "https://web-production-af34.up.railway.app")

def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    requests.post(url, json=payload)

def get_updates(offset=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {"timeout": 100, "offset": offset}
    return requests.get(url, params=params).json()

def handle_command(command):
    try:
        parts = command.strip().split()
        if len(parts) == 2:
            pair, tf = parts
            url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_entry={tf}&auto_log=true"
        elif len(parts) == 1:
            pair = parts[0]
            url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_entry=15m&auto_log=true"
        else:
            return "Format: <b>BTCUSDT 15m</b> atau <b>ETHUSDT</b>"
        
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            msg = f"📊 <b>{data['pair']} ({data['timeframe']})</b>\n"
            msg += f"💡 Signal: <b>{data['signal_type']}</b>\n"
            msg += f"🎯 Entry: {data['entry']}\n"
            msg += f"🎯 TP1: {data['tp1']}\n"
            msg += f"🎯 TP2: {data['tp2']}\n"
            msg += f"🛡 SL: {data['sl']}\n"
            msg += f"📈 Confidence: {data['confidence']}\n\n"
            msg += f"🧠 Reasoning: {data['reasoning']}"
            return msg
        else:
            return f"⚠️ Gagal fetch data: {r.text}"
    except Exception as e:
        return f"❌ Error: {e}"

def main():
    offset = None
    send_message("🤖 Pro Trader AI Bot Aktif dan Siap Membantu!")
    while True:
        updates = get_updates(offset)
        if "result" in updates:
            for update in updates["result"]:
                offset = update["update_id"] + 1
                if "message" in update and "text" in update["message"]:
                    text = update["message"]["text"]
                    if text.startswith("/start"):
                        send_message("Selamat datang! Kirim perintah seperti:\n\n<b>BTCUSDT 15m</b> atau <b>ETHUSDT</b>")
                    else:
                        response = handle_command(text)
                        send_message(response)
        time.sleep(2)

if __name__ == "__main__":
    main()
