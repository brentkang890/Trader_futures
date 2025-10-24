# telegram_bot.py
"""
🤖 Pro Trader AI Telegram Bot
- Versi emoji (UTF-8 aman untuk Railway)
- Support: sinyal, backtest, status, stats, log, scalp, upload CSV & chart
"""

import os
import time
import requests

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL", "https://web-production-af34.up.railway.app")
BACKTEST_URL = os.environ.get("BACKTEST_URL")

if not BOT_TOKEN or not CHAT_ID or not APP_URL:
    raise ValueError("❌ Environment variables BOT_TOKEN, CHAT_ID, APP_URL harus diset.")

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
        return "⚠️ Pesan kosong."
    t = text.strip().lower()

    # 🧠 START
    if t in ("start", "/start"):
        return (
            "🤖 <b>Pro Trader AI Bot Aktif!</b>\n\n"
            "📈 Contoh:\n"
            "- BTCUSDT 15m atau XAUUSD 1h\n\n"
            "🧩 Perintah lain:\n"
            "- backtest BTCUSDT\n"
            "- status\n"
            "- stats\n"
            "- log\n"
            "- scalp BTCUSDT\n\n"
            "📄 Kirim file CSV untuk analisis otomatis."
        )

    # 🧪 BACKTEST
    if t.startswith("backtest"):
        try:
            parts = t.split()
            if len(parts) < 2:
                return "⚙️ Format: <code>backtest BTCUSDT</code>"
            pair = parts[1].upper()
            payload = {"pair": pair, "side": "LONG", "entry": 30000, "tp1": 31000, "sl": 29500, "timeframe": "15m"}
            url = BACKTEST_URL or f"{APP_URL.rstrip('/')}/backtest"
            r = requests.post(url, json=payload, timeout=30)
            d = r.json()
            if "error" in d:
                return f"⚠️ Backtest error: {d.get('error')}"
            return (
                f"📊 <b>Backtest {d.get('pair')}</b>\n"
                f"🎯 Hit: {d.get('hit')}\n"
                f"💰 PnL: {d.get('pnl_total')}\n"
            )
        except Exception as e:
            return f"⚠️ Gagal backtest: {e}"

    # 📊 STATUS MODEL
    if t == "status":
        try:
            r = requests.get(f"{APP_URL.rstrip('/')}/learning_status", timeout=15)
            d = r.json()
            return (
                f"🤖 <b>Status Model</b>\n"
                f"📦 Model: {'✅ Ada' if d.get('model_exists') else '❌ Tidak ada'}\n"
                f"🧮 Data log: {d.get('trade_log_count', 0)} sinyal\n"
                f"🧠 Fitur: {', '.join(d.get('features', [])) if d.get('features') else '-'}"
            )
        except Exception as e:
            return f"⚠️ Error ambil status: {e}"

    # 📈 STATISTIK
    if t == "stats":
        try:
            r = requests.get(f"{APP_URL.rstrip('/')}/ai_performance", timeout=20)
            d = r.json()
            if "error" in d:
                return f"⚠️ {d['error']}"
            return (
                f"📊 <b>Statistik AI</b>\n"
                f"📈 Total sinyal: {d.get('total_signals')}\n"
                f"✅ Winrate: {d.get('winrate')}%\n"
                f"💹 Profit factor: {d.get('profit_factor')}\n"
                f"🤖 Model: {d.get('model_status')}"
            )
        except Exception as e:
            return f"⚠️ Error ambil stats: {e}"

    # 🧾 LOG
    if t == "log":
        try:
            r = requests.get(f"{APP_URL.rstrip('/')}/logs_summary", timeout=15)
            d = r.json()
            if "detail" in d:
                return d["detail"]
            return (
                f"📜 <b>Log Terakhir</b>\n"
                f"📊 {d.get('pair')} ({d.get('timeframe')})\n"
                f"💡 Signal: {d.get('signal_type')}\n"
                f"🎯 Entry: {d.get('entry')}\n"
                f"🏁 TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
                f"🛑 SL: {d.get('sl')}\n"
                f"📊 Confidence: {d.get('confidence')}\n"
                f"🧠 {d.get('reasoning')}"
            )
        except Exception as e:
            return f"⚠️ Error ambil log: {e}"

    # ⚡ SCALPING
    if t.startswith("scalp "):
        try:
            pair = t.split()[1].upper()
            r = requests.get(f"{APP_URL.rstrip('/')}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=20)
            d = r.json()
            return (
                f"⚡ <b>Scalp {d.get('pair')}</b> ({d.get('timeframe')})\n"
                f"💡 Signal: {d.get('signal_type')}\n"
                f"🎯 Entry: {d.get('entry')}\n"
                f"🏁 TP1: {d.get('tp1')} | 🛑 SL: {d.get('sl')}\n"
                f"📊 Confidence: {d.get('confidence')}"
            )
        except Exception as e:
            return f"⚠️ Error scalp: {e}"

    # Default: PRO SIGNAL
    parts = t.split()
    if len(parts) == 0:
        return "❌ Format salah. Contoh: <code>BTCUSDT 15m</code>"
    pair = parts[0].upper()
    tf = parts[1] if len(parts) > 1 else "15m"
    try:
        r = requests.get(f"{APP_URL.rstrip('/')}/pro_signal?pair={pair}&tf_main=1h&tf_entry={tf}&auto_log=true", timeout=25)
        d = r.json()
        if "error" in d:
            return f"⚠️ {d['error']}"
        return (
            f"📊 <b>{d.get('pair')} ({d.get('timeframe')})</b>\n"
            f"💡 Signal: {d.get('signal_type')}\n"
            f"🎯 Entry: {d.get('entry')}\n"
            f"🏁 TP1: {d.get('tp1')} | 🛑 SL: {d.get('sl')}\n"
            f"📊 Confidence: {d.get('confidence')}\n"
            f"🧠 {d.get('reasoning', '')}"
        )
    except Exception as e:
        return f"⚠️ Error pro_signal: {e}"

def main():
    offset = None
    send_message("🤖 Pro Trader AI Bot aktif!\nKetik /start untuk daftar perintah.")
    while True:
        try:
            upd = get_updates(offset)
            if "result" in upd:
                for u in upd["result"]:
                    offset = u["update_id"] + 1
                    msg = u.get("message", {})

                    # text
                    if "text" in msg:
                        reply = handle_command(msg["text"])
                        send_message(reply)

                    # photo
                    elif "photo" in msg:
                        photo = msg["photo"][-1]
                        file_data = download_file(photo["file_id"])
                        if not file_data:
                            send_message("⚠️ Gagal download gambar.")
                            continue
                        send_message("🖼️ Menganalisis chart, mohon tunggu...")
                        files = {"file": ("chart.jpg", file_data, "image/jpeg")}
                        try:
                            r = requests.post(f"{APP_URL.rstrip('/')}/analyze_chart", files=files, timeout=60)
                            if r.status_code == 200:
                                d = r.json()
                                send_message(
                                    f"📊 {d.get('pair')} ({d.get('timeframe')})\n"
                                    f"💡 Signal: {d.get('signal_type')}\n"
                                    f"🎯 Entry: {d.get('entry')}\n"
                                    f"🏁 TP1: {d.get('tp1')} | 🛑 SL: {d.get('sl')}\n"
                                    f"📊 Confidence: {d.get('confidence')}"
                                )
                            else:
                                send_message(f"⚠️ Gagal analisis gambar: {r.text}")
                        except Exception as e:
                            send_message(f"⚠️ Error analisis gambar: {e}")

                    # document (CSV)
                    elif "document" in msg:
                        doc = msg["document"]
                        fname = doc.get("file_name", "")
                        mime = doc.get("mime_type", "")
                        file_data = download_file(doc["file_id"])
                        if not file_data:
                            send_message("⚠️ Gagal download file.")
                            continue
                        if fname.lower().endswith(".csv") or mime in ("text/csv", "application/vnd.ms-excel"):
                            send_message("📄 CSV diterima, sedang dianalisis oleh AI...")
                            files = {"file": (fname, file_data, "text/csv")}
                            try:
                                r = requests.post(f"{APP_URL.rstrip('/')}/analyze_csv", files=files, timeout=60)
                                if r.status_code == 200:
                                    d = r.json()
                                    send_message(
                                        f"✅ Hasil analisis CSV:\n"
                                        f"📊 {d.get('pair', 'CSV')} ({d.get('timeframe', '')})\n"
                                        f"💡 Signal: {d.get('signal_type')}\n"
                                        f"🎯 Entry: {d.get('entry')}\n"
                                        f"🏁 TP1: {d.get('tp1')} | 🛑 SL: {d.get('sl')}\n"
                                        f"📊 Confidence: {d.get('confidence')}"
                                    )
                                else:
                                    send_message(f"⚠️ Gagal analisis CSV: {r.text}")
                            except Exception as e:
                                send_message(f"⚠️ Error analisis CSV: {e}")
                        else:
                            send_message("⚠️ Hanya file CSV yang didukung.")
            time.sleep(1.5)
        except Exception as e:
            print("Loop error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
