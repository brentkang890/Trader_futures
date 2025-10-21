import requests
import time
import json
import os

# === KONFIGURASI DARI RAILWAY VARIABLE ===
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8483103988:AAHeHGuHA6T0rx6nRN-w5bgGrYlf0kbmgHs")
CHAT_ID = os.environ.get("CHAT_ID", "6123645566")
APP_URL = os.environ.get("APP_URL", "https://web-production-af34.up.railway.app")

# === FUNGSI KIRIM PESAN TELEGRAM ===
def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"[WARN] Gagal kirim pesan: {response.text}")
    except Exception as e:
        print(f"[ERROR] Gagal kirim pesan Telegram: {e}")

# === FUNGSI MENGAMBIL PESAN BARU ===
def get_updates(offset=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {"timeout": 60, "offset": offset}
    try:
        return requests.get(url, params=params, timeout=65).json()
    except Exception as e:
        print(f"[ERROR] Gagal ambil update: {e}")
        return {}

# === FUNGSI HANDLE COMMAND USER ===
def handle_command(command):
    try:
        parts = command.strip().split()
        if len(parts) == 2:
            pair, tf = parts
            url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_main=tf_{tf}"
        elif len(parts) == 1:
            pair = parts[0]
            url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_main=tf_15m"
        else:
            return "⚠️ Format salah!\n\nGunakan format:\n<b>BTCUSDT 15m</b> atau <b>ETHUSDT</b>"

        print(f"[INFO] Fetching signal dari: {url}")
        r = requests.get(url, timeout=25)

        if r.status_code == 200:
            data = r.json()
            msg = (
                f"📊 <b>{data['pair']} ({data['timeframe']})</b>\n"
                f"💡 Signal: <b>{data['signal_type']}</b>\n"
                f"🎯 Entry: {data['entry']}\n"
                f"🎯 TP1: {data['tp1']}\n"
                f"🎯 TP2: {data['tp2']}\n"
                f"🛡 SL: {data['sl']}\n"
                f"📈 Confidence: {data['confidence']}\n\n"
                f"🧠 Reasoning: {data['reasoning']}"
            )
            return msg
        else:
            return f"⚠️ Gagal ambil sinyal: {r.text}"

    except Exception as e:
        print(f"[ERROR] Saat handle command: {e}")
        return f"❌ Terjadi error: {e}"

# === PROGRAM UTAMA ===
def main():
    offset = None
    print(f"🤖 BOT AKTIF | Terhubung ke: {APP_URL}")
    send_message("🤖 <b>Pro Trader AI Bot Aktif dan Siap Membantu!</b>")

    while True:
        try:
            updates = get_updates(offset)
            if "result" in updates:
                for update in updates["result"]:
                    offset = update["update_id"] + 1
                    if "message" in update and "text" in update["message"]:
                        text = update["message"]["text"].strip()
                        if text.startswith("/start"):
                            send_message(
                                "👋 Halo! Kirim pair + timeframe (contoh: <b>BTCUSDT 15m</b>)\n"
                                "Atau kirim <b>BTCUSDT</b> untuk default timeframe 15m."
                            )
                        else:
                            response = handle_command(text)
                            send_message(response)
            time.sleep(2)  # Lebih responsif

        except Exception as e:
            print(f"[ERROR LOOP] {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
