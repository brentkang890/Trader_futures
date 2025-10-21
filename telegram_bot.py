import requests
import time
import os

# === KONFIGURASI DARI RAILWAY ENV ===
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
APP_URL = os.environ.get("APP_URL", "")

# === KIRIM PESAN TELEGRAM ===
def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[ERROR] Gagal kirim pesan Telegram: {e}")

# === AMBIL PESAN DARI USER ===
def get_updates(offset=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {"timeout": 100, "offset": offset}
    try:
        return requests.get(url, params=params, timeout=120).json()
    except Exception as e:
        print(f"[ERROR] Gagal ambil update: {e}")
        return {}

# === HANDLE COMMAND TEXT ===
def handle_text_command(command):
    try:
        cmd = command.strip().lower()

        # --- Auto Mapping untuk Forex / Pair umum ---
        symbol_map = {
            "XAUUSD": "XAUUSDT",
            "EURUSD": "EURUSDT",
            "GBPUSD": "GBPUSDT",
            "USDJPY": "USDJPY",
            "AUDUSD": "AUDUSDT",
            "BTCUSD": "BTCUSDT",
            "ETHUSD": "ETHUSDT"
        }

        parts = cmd.split()
        if parts:
            base = parts[0].upper()
            if base in symbol_map:
                parts[0] = symbol_map[base]
                cmd = " ".join(parts)

        # === MODE STATUS MODEL ===
        if cmd == "status":
            url = f"{APP_URL}/learning_status_summary"
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                return f"⚠️ Gagal ambil status model: {r.text}"
            d = r.json()
            return (
                f"🧠 <b>Status Model AI</b>\n"
                f"📦 Model: {d.get('model_status','❓')}\n"
                f"📊 Log Data: {d.get('log_count',0)} sinyal\n"
                f"🧩 Kesiapan Retrain: {'✅ Siap' if d.get('learning_ready') else '❌ Belum cukup data'}\n"
                f"📋 {d.get('description','')}"
            )

        # === MODE SCALPING ===
        elif cmd.startswith("scalp "):
            pair = cmd.split()[1].upper()
            url = f"{APP_URL}/scalp_signal?pair={pair}&tf=3m&auto_log=true"

        # === MODE LIHAT LOG ===
        elif cmd.startswith("log"):
            url = f"{APP_URL}/logs_summary"

        # === SIGNAL NORMAL ===
        elif len(parts) == 2:
            pair, tf = parts
            url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_entry={tf}&auto_log=true"
        elif len(parts) == 1:
            pair = parts[0]
            url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_entry=15m&auto_log=true"
        else:
            return (
                "⚠️ Format salah!\n\n"
                "Gunakan format seperti:\n"
                "<b>BTCUSDT 15m</b> — sinyal normal\n"
                "<b>scalp BTCUSDT</b> — mode scalping\n"
                "<b>status</b> — status model AI\n"
                "<b>log</b> — sinyal terakhir"
            )

        print(f"[INFO] Fetching data dari: {url}")
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return f"⚠️ Gagal ambil sinyal: {r.text}"

        d = r.json()

        # === HANDLE ERROR DARI AI AGENT ===
        if "error" in d:
            return f"⚠️ {d['error']}"

        # === FORMAT PESAN HASIL ===
        msg = (
            f"📊 <b>{d.get('pair','?')} ({d.get('timeframe','?')})</b>\n"
            f"💡 Signal: <b>{d.get('signal_type','?')}</b>\n"
            f"🎯 Entry: {d.get('entry','?')}\n"
            f"🎯 TP1: {d.get('tp1','?')}\n"
            f"🎯 TP2: {d.get('tp2','?')}\n"
            f"🛡 SL: {d.get('sl','?')}\n"
            f"📈 Confidence: {d.get('confidence','?')}\n"
            f"🤖 Probabilitas Model: {d.get('model_prob','?')}\n\n"
            f"🧠 Reasoning: {d.get('reasoning','?')}"
        )
        return msg

    except Exception as e:
        print(f"[ERROR HANDLE] {e}")
        return f"❌ Error internal: {e}"

# === HANDLE FOTO CHART (sementara dinonaktifkan kalau OCR belum aktif) ===
def handle_photo(photo_id):
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

# === MAIN LOOP ===
def main():
    offset = None
    send_message("🤖 <b>Pro Trader AI Bot Aktif & Terhubung ke AI Agent!</b>\n\n"
                 "Kirim perintah seperti:\n"
                 "• <b>BTCUSDT 15m</b>\n"
                 "• <b>XAUUSD 5m</b>\n"
                 "• <b>scalp BTCUSDT</b>\n"
                 "• <b>status</b> atau <b>log</b>")

    while True:
        updates = get_updates(offset)
        if "result" in updates:
            for upd in updates["result"]:
                offset = upd["update_id"] + 1
                msg = upd.get("message", {})

                if "text" in msg:
                    text = msg["text"].strip()
                    if text.startswith("/start"):
                        send_message(
                            "👋 Selamat datang di <b>Pro Trader AI</b>!\n"
                            "Kirim perintah seperti:\n"
                            "• <b>BTCUSDT 15m</b>\n"
                            "• <b>XAUUSD 5m</b>\n"
                            "• <b>scalp BTCUSDT</b>\n"
                            "• <b>status</b> atau <b>log</b>"
                        )
                    else:
                        response = handle_text_command(text)
                        send_message(response)

                elif "photo" in msg:
                    photo_id = msg["photo"][-1]["file_id"]
                    handle_photo(photo_id)

        time.sleep(3)

if __name__ == "__main__":
    main()
