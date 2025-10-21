import requests
import time
import json
import os

# === KONFIGURASI ENV (Railway Variables) ===
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL")  # URL dari app AI kamu, misal: https://web-production-af34.up.railway.app

# === Fungsi kirim pesan ke Telegram ===
def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[ERROR] Gagal kirim pesan Telegram: {e}")

# === Fungsi ambil pesan baru dari Telegram ===
def get_updates(offset=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {"timeout": 100, "offset": offset}
    try:
        return requests.get(url, params=params, timeout=120).json()
    except Exception as e:
        print(f"[ERROR] Gagal ambil update: {e}")
        return {}

# === Fungsi utama untuk handle command ===
def handle_command(command):
    cmd = command.strip().lower()
    try:
        # === STATUS MODEL ===
        if cmd == "status":
            r = requests.get(f"{APP_URL}/learning_status_summary", timeout=15)
            if r.status_code != 200:
                return f"⚠️ Gagal ambil status: {r.text}"
            d = r.json()
            return (
                f"🤖 <b>Status Model AI</b>\n"
                f"📦 Model: {d.get('model_status','❌')}\n"
                f"📊 Log Data: {d.get('log_count',0)} sinyal\n"
                f"🧠 Kesiapan Retrain: {'✅ Siap' if d.get('learning_ready') else '❌ Belum cukup data'}\n"
                f"📋 {d.get('description','')}"
            )

        # === LIHAT LOG TERBARU ===
        elif cmd == "log":
            r = requests.get(f"{APP_URL}/logs_summary", timeout=15)
            if r.status_code != 200:
                return f"⚠️ Gagal ambil log: {r.text}"
            d = r.json()
            if "detail" in d:
                return f"ℹ️ {d['detail']}"
            return (
                f"📊 <b>{d.get('pair','?')} ({d.get('timeframe','?')})</b>\n"
                f"💡 Signal: <b>{d.get('signal_type','?')}</b>\n"
                f"🎯 Entry: {d.get('entry','?')}\n"
                f"🎯 TP1: {d.get('tp1','?')}\n"
                f"🎯 TP2: {d.get('tp2','?')}\n"
                f"🛡 SL: {d.get('sl','?')}\n"
                f"📈 Confidence: {d.get('confidence','?')}\n"
                f"🧠 Reasoning: {d.get('reasoning','')}"
            )

        # === STATISTIK PERFORMA AI ===
        elif cmd == "stats":
            r = requests.get(f"{APP_URL}/ai_performance", timeout=20)
            if r.status_code != 200:
                return f"⚠️ Gagal ambil statistik: {r.text}"
            d = r.json()
            if "error" in d:
                return f"ℹ️ {d['error']}"
            return (
                f"📈 <b>Statistik Performa AI</b>\n"
                f"============================\n"
                f"💹 Total Sinyal: {d.get('total_signals',0)}\n"
                f"✅ Winrate: {d.get('winrate',0)}%\n"
                f"💰 Profit Factor: {d.get('profit_factor','N/A')}\n"
                f"⚙️ Model: {d.get('model_status','❌')}"
            )

        # === SCALPING MODE (contoh: scalp btcusdt) ===
        elif cmd.startswith("scalp "):
            pair = cmd.split()[1].upper()
            r = requests.get(f"{APP_URL}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=25)
            if r.status_code != 200:
                return f"⚠️ Gagal ambil scalp signal: {r.text}"
            d = r.json()
            return (
                f"⚡ <b>Scalp Signal {d.get('pair','')}</b> ({d.get('timeframe','')})\n"
                f"💡 <b>{d.get('signal_type','')}</b>\n"
                f"🎯 Entry: {d.get('entry','')}\n"
                f"🎯 TP1: {d.get('tp1','')}\n"
                f"🎯 TP2: {d.get('tp2','')}\n"
                f"🛡 SL: {d.get('sl','')}\n"
                f"📈 Confidence: {d.get('confidence','')}\n"
                f"🧠 Reasoning: {d.get('reasoning','')}"
            )

        # === SIGNAL BIASA (contoh: BTCUSDT 15m) ===
        else:
            parts = cmd.split()
            if len(parts) == 2:
                pair, tf = parts
            elif len(parts) == 1:
                pair, tf = parts[0], "15m"
            else:
                return "⚠️ Format salah!\nGunakan: <b>BTCUSDT 15m</b> atau <b>ETHUSDT</b>"

            url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_main=1h&tf_entry={tf}&auto_log=true"
            print(f"[INFO] Fetching: {url}")
            r = requests.get(url, timeout=25)

            if r.status_code != 200:
                return f"⚠️ Gagal ambil sinyal: {r.text}"

            try:
                d = r.json()
            except Exception:
                return f"⚠️ Gagal parsing respon dari AI: {r.text[:150]}"

            if "error" in d:
                return f"⚠️ {d['error']}"

            return (
                f"📊 <b>{d.get('pair','?')} ({d.get('timeframe','?')})</b>\n"
                f"💡 Signal: <b>{d.get('signal_type','?')}</b>\n"
                f"🎯 Entry: {d.get('entry','?')}\n"
                f"🎯 TP1: {d.get('tp1','?')}\n"
                f"🎯 TP2: {d.get('tp2','?')}\n"
                f"🛡 SL: {d.get('sl','?')}\n"
                f"📈 Confidence: {d.get('confidence','?')}\n"
                f"🧠 Reasoning: {d.get('reasoning','')}"
            )

    except Exception as e:
        import traceback
        print("[ERROR CMD]", traceback.format_exc())
        return f"❌ Terjadi error internal: {e}"

# === MAIN LOOP ===
def main():
    offset = None
    print("🤖 Bot aktif dan tersambung ke:", APP_URL)
    send_message("🤖 <b>Pro Trader AI Bot siap digunakan!</b>\nKirim <b>BTCUSDT 15m</b> untuk sinyal cepat.\nAtau kirim <b>status</b> untuk lihat kondisi model.")

    while True:
        try:
            updates = get_updates(offset)
            if "result" in updates:
                for update in updates["result"]:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    if not msg: continue

                    text = msg.get("text", "").strip()
                    if not text: continue

                    print(f"[MSG] {text}")
                    if text.startswith("/start"):
                        send_message(
                            "👋 Halo! Kirim pair dan timeframe (contoh: <b>BTCUSDT 15m</b>).\n"
                            "Ketik <b>status</b> untuk cek kondisi model.\n"
                            "Ketik <b>stats</b> untuk performa AI."
                        )
                    else:
                        response = handle_command(text)
                        send_message(response)

            time.sleep(2)
        except Exception as e:
            print(f"[ERROR LOOP] {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
