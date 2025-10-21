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
        # === STATUS MODEL ===
        if cmd == "status":
            r = requests.get(f"{APP_URL}/learning_status_summary", timeout=15)
            if r.status_code != 200:
                return f"⚠️ Gagal ambil status: {r.text}"
            d = r.json()
            return (
                f"🤖 <b>Status Model AI</b>\n"
                f"📦 Model: {d.get('model_status', '❓')}\n"
                f"📊 Log Data: {d.get('log_count', 0)} sinyal\n"
                f"🧠 Kesiapan Retrain: {'✅ Siap' if d.get('learning_ready') else '❌ Belum cukup data'}\n"
                f"📋 {d.get('description', '')}"
            )

        # === LIHAT SINYAL TERAKHIR ===
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
            r = requests.get(f"{APP_URL}/ai_performance", timeout=25)
            if r.status_code != 200:
                return f"⚠️ Gagal ambil statistik: {r.text}"
            d = r.json()
            if "error" in d:
                return f"⚠️ {d['error']}"

            msg = (
                f"📈 <b>Statistik Performa AI</b>\n"
                f"========================\n"
                f"💹 Total Sinyal: {d.get('total_signals', 0)}\n"
                f"✅ Winrate Keseluruhan: {d.get('winrate', 0)}%\n"
                f"💰 Profit Factor: {d.get('profit_factor', 'N/A')}\n"
                f"📉 Max Drawdown: {d.get('max_drawdown', 'N/A')}\n"
                f"📊 Avg Confidence: {d.get('avg_confidence', 0)}\n"
                f"⚙️ Model: {d.get('model_status', '❌ Belum Ada')}\n\n"
            )

            if d.get("pair_stats"):
                msg += "📊 <b>Berdasarkan Pair:</b>\n"
                for p in d["pair_stats"][:5]:
                    msg += f"- {p['pair']} → {p['winrate']}% winrate ({p['signals']} sinyal)\n"
                msg += "\n"

            if d.get("tf_stats"):
                msg += "🕒 <b>Berdasarkan Timeframe:</b>\n"
                for t in d["tf_stats"]:
                    msg += f"- {t['timeframe']} → {t['winrate']}% winrate ({t['signals']} sinyal)\n"

            return msg

        # === SCALPING MODE ===
        elif cmd.startswith("scalp "):
            pair = cmd.split()[1].upper()
            r = requests.get(f"{APP_URL}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=25)
            if r.status_code != 200:
                return f"⚠️ Gagal ambil scalp signal: {r.text}"
            d = r.json()
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

        # === SIGNAL BIASA ===
        else:
            parts = cmd.split()
            if len(parts) == 2:
                pair, tf = parts
            else:
                pair, tf = parts[0], "15m"

            # ✅ PERBAIKAN UTAMA: tambah tf_main=1h agar endpoint tidak error
            url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_main=1h&tf_entry={tf}&auto_log=true"
            r = requests.get(url, timeout=25)

            if r.status_code != 200:
                return f"⚠️ Gagal ambil sinyal dari AI: {r.text}"

            try:
                d = r.json()
            except Exception:
                return f"⚠️ Gagal parsing respon AI: {r.text[:200]}"

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
        err = traceback.format_exc()
        print(f"[ERROR CMD] {err}")
        return f"❌ Error internal: {str(e)}"
