# telegram_bot_auto_signal.py
# Auto Signal Master Bot
# - Auto-scan crypto + forex (selected major pairs) every 1 hour
# - Send only strong signals (confidence >= 0.8)
# - Support manual commands, CSV & chart upload, scalp, and /force to bypass filter

import os
import re
import time
import json
import requests
from datetime import datetime
from threading import Event

from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")  # your personal chat id where auto signals are sent
APP_URL = os.getenv("APP_URL", "https://web-production-af34.up.railway.app").rstrip("/")
if not APP_URL.startswith("http"):
    APP_URL = "https://" + APP_URL

API_TIMEOUT = int(os.getenv("API_TIMEOUT", "25"))
STRONG_SIGNAL_THRESHOLD = float(os.getenv("STRONG_SIGNAL_THRESHOLD", "0.8"))
AUTO_SCAN_HOURS = int(os.getenv("AUTO_SCAN_HOURS", "1"))  # every N hours; user wanted every 1 hour

# You can override pair lists via environment variable AUTO_PAIRS (comma separated)
AUTO_PAIRS_CRYPTO = os.getenv("AUTO_PAIRS_CRYPTO", "").strip()
if AUTO_PAIRS_CRYPTO:
    AUTO_PAIRS_CRYPTO = [p.strip().upper() for p in AUTO_PAIRS_CRYPTO.split(",")]
else:
    AUTO_PAIRS_CRYPTO = [
        "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","LTCUSDT","DOGEUSDT","MATICUSDT","DOTUSDT",
        "AVAXUSDT","LINKUSDT","TRXUSDT","ATOMUSDT","XLMUSDT","UNIUSDT","SANDUSDT","AXSUSDT","EOSUSDT","FTMUSDT"
    ]

AUTO_PAIRS_FOREX = os.getenv("AUTO_PAIRS_FOREX", "").strip()
if AUTO_PAIRS_FOREX:
    AUTO_PAIRS_FOREX = [p.strip().upper() for p in AUTO_PAIRS_FOREX.split(",")]
else:
    AUTO_PAIRS_FOREX = [
        "XAUUSD","EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF","EURJPY","GBPJPY"
    ]

# Timeframes to check (we'll request /pro_signal with tf_entry)
AUTO_TIMEFRAMES = os.getenv("AUTO_TIMEFRAMES", "").strip()
if AUTO_TIMEFRAMES:
    AUTO_TIMEFRAMES = [t.strip().lower() for t in AUTO_TIMEFRAMES.split(",")]
else:
    AUTO_TIMEFRAMES = ["15m", "1h", "4h"]

# Merge full scan list (crypto + forex)
AUTO_PAIRS = list(dict.fromkeys(AUTO_PAIRS_CRYPTO + AUTO_PAIRS_FOREX))

# ---------------- HELPERS ----------------
def format_signal(result: dict) -> str:
    """Pretty-format a signal dict into Telegram message (HTML)."""
    if not isinstance(result, dict):
        return "⚠️ Tidak bisa membaca hasil sinyal."
    if "error" in result:
        return f"❌ Error: {result.get('error')}"
    try:
        lines = []
        lines.append(f"📊 <b>{result.get('pair','?')}</b> ({result.get('timeframe','?')})")
        lines.append(f"💡 <b>{result.get('signal_type','?')}</b>")
        lines.append(f"🎯 Entry: <code>{result.get('entry')}</code>")
        lines.append(f"🎯 TP1: <code>{result.get('tp1')}</code> | TP2: <code>{result.get('tp2')}</code>")
        lines.append(f"🛑 SL: <code>{result.get('sl')}</code>")
        if result.get("confidence") is not None:
            lines.append(f"📊 Confidence: {result.get('confidence')}")
        if result.get("position_size"):
            lines.append(f"📈 Position: {result.get('position_size')}")
        if result.get("market_mode"):
            lines.append(f"🪙 Market: {result.get('market_mode')}")
        if result.get("reasoning"):
            # shorten reasoning if too long
            reasoning = str(result.get("reasoning"))[:800]
            lines.append(f"🧠 Reasoning: {reasoning}")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Format error: {e}"

def parse_pair_tf(text: str):
    """
    Robust pair+tf parser for manual messages.
    Accept forms: 'BTCUSDT 15m', 'btc/usdt 15M', 'analisa btcusdt 1h', 'gold 1h'
    """
    if not text:
        return None, "15m"
    t = text.upper().replace("/", " ").replace("_", " ").strip()
    # try find timeframe like 15M 1H 4H etc
    tf_match = re.search(r"(\d+\s*[MHWD])", t)
    tf = tf_match.group(1).replace(" ", "").lower() if tf_match else "15m"
    # remove known verbs
    t_clean = re.sub(r"\b(ANALISA|ANALYZE|ANALYSE|CHECK|FORCE|SCALP)\b", " ", t, flags=re.IGNORECASE).strip()
    # common aliases
    aliases = {
        "GOLD": "XAUUSD", "EMAS": "XAUUSD",
        "BITCOIN": "BTCUSDT", "BTC": "BTCUSDT",
        "ETH": "ETHUSDT", "SOL": "SOLUSDT", "EUR": "EURUSD"
    }
    for a,v in aliases.items():
        if a in t_clean:
            return v, tf
    # try token pattern
    m = re.search(r"([A-Z0-9]{3,6})\s*([A-Z]{3,4})", t_clean)
    if m:
        pair = (m.group(1) + m.group(2)).upper()
        return pair, tf
    # contiguous pattern like BTCUSDT15M
    m2 = re.search(r"([A-Z]{3,6}(?:USDT|USD|EUR|JPY|GBP|IDR|BTC|ETH))", t_clean)
    if m2:
        return m2.group(1).upper(), tf
    # fallback: try first token
    token = t_clean.split()[0] if t_clean.split() else None
    if token:
        return token.replace(" ", "").upper(), tf
    return None, tf

def send_request_get(endpoint: str, params: dict = None, timeout: int = API_TIMEOUT):
    url = f"{APP_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        try:
            return r.json()
        except Exception:
            return {"error": f"invalid_json_response: {r.text}"}
    except Exception as e:
        return {"error": str(e)}

# ---------------- AUTO-SCAN ----------------
def auto_check_and_send(app):
    """
    Iterate AUTO_PAIRS and AUTO_TIMEFRAMES; call /pro_signal; send to CHAT_ID if confidence >= threshold.
    """
    bot = app.bot
    print(f"[AUTO] Auto-scan started at {datetime.utcnow().isoformat()} - pairs={len(AUTO_PAIRS)} timeframes={AUTO_TIMEFRAMES}")
    # iterate pairs
    for pair in AUTO_PAIRS:
        for tf in AUTO_TIMEFRAMES:
            try:
                params = {"pair": pair, "tf_entry": tf}
                res = send_request_get("pro_signal", params=params)
                if not isinstance(res, dict):
                    print(f"[AUTO WARN] non-dict response for {pair} {tf}: {res}")
                    continue
                if "error" in res:
                    # ignore non-critical errors
                    print(f"[AUTO] {pair} {tf} -> error: {res.get('error')}")
                    continue
                conf = float(res.get("confidence", 0) or 0)
                if conf >= STRONG_SIGNAL_THRESHOLD and res.get("signal_type") and res.get("signal_type") != "WAIT":
                    msg = format_signal(res)
                    try:
                        bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML")
                        print(f"[AUTO] Sent strong signal {pair} {tf} (conf={conf})")
                    except Exception as e:
                        print(f"[AUTO ERROR] send_message failed for {pair} {tf}: {e}")
                else:
                    print(f"[AUTO] {pair} {tf} no strong signal (conf={conf})")
                # small sleep to be gentle on API rate limits
                time.sleep(0.8)
            except Exception as e:
                print(f"[AUTO EXC] {pair} {tf}: {e}")
                time.sleep(0.5)
    print(f"[AUTO] Auto-scan finished at {datetime.utcnow().isoformat()}")

# ---------------- TELEGRAM HANDLERS ----------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 <b>AI Trader Auto Signal</b>\n\n"
        "Saya akan memantau semua pair crypto & forex dan mengirim SINYAL KUAT (confidence ≥ 0.8) otomatis tiap 1 jam.\n\n"
        "Perintah manual:\n"
        "- <code>BTCUSDT 15m</code>\n"
        "- <code>scalp BTCUSDT</code>\n"
        "- <code>force BTCUSDT 15m</code> (tampilkan semua sinyal tanpa filter)\n"
        "Kirim CSV atau gambar chart untuk analisis juga."
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def manual_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip() if update.message.text else ""
    if not text:
        return
    t_low = text.lower()
    is_force = t_low.startswith("force")
    if is_force:
        text = text.replace("force", "", 1).strip()

    if t_low.startswith("scalp"):
        pair, _ = parse_pair_tf(text)
        if not pair:
            await update.message.reply_text("❌ Tidak bisa mendeteksi pair.")
            return
        await update.message.reply_text(f"⚡ Scalp {pair} ...")
        res = send_request_get("scalp_signal", params={"pair": pair, "tf": "3m"})
        await update.message.reply_text(format_signal(res), parse_mode="HTML")
        return

    pair, tf = parse_pair_tf(text)
    if not pair:
        await update.message.reply_text("❌ Tidak bisa mendeteksi pair dari pesan itu.")
        return

    await update.message.reply_text(f"🔍 Menganalisis {pair} ({tf}) ...")
    res = send_request_get("pro_signal", params={"pair": pair, "tf_entry": tf})
    if "error" in res:
        await update.message.reply_text(f"❌ Error: {res['error']}")
        return

    conf = float(res.get("confidence", 0) or 0)
    # if not forced and below threshold, inform user (we chose option 2 previously)
    if (not is_force) and conf < STRONG_SIGNAL_THRESHOLD:
        await update.message.reply_text(
            f"⚠️ Tidak ada sinyal kuat untuk {pair} ({tf}).\nConfidence saat ini: {conf}",
            parse_mode="HTML"
        )
        return

    # send formatted
    await update.message.reply_text(format_signal(res), parse_mode="HTML")

async def handle_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        return
    await update.message.reply_text("📄 Menerima CSV, mengirim ke AI untuk analisis...")
    file = await doc.get_file()
    try:
        content = requests.get(file.file_path, timeout=30).content
        files = {"file": ("upload.csv", content)}
        url = f"{APP_URL}/analyze_csv"
        r = requests.post(url, files=files, timeout=60)
        try:
            d = r.json()
        except Exception:
            d = {"error": r.text}
        await update.message.reply_text(format_signal(d), parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Gagal analisis CSV: {e}")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # image chart
    photo = update.message.photo[-1] if update.message.photo else None
    if not photo:
        await update.message.reply_text("⚠️ Tidak ada gambar.")
        return
    await update.message.reply_text("📷 Menganalisis chart (OCR + heuristics)...")
    file = await photo.get_file()
    try:
        content = requests.get(file.file_path, timeout=60).content
        files = {"file": ("chart.jpg", content)}
        url = f"{APP_URL}/analyze_chart"
        r = requests.post(url, files=files, timeout=60)
        try:
            d = r.json()
        except Exception:
            d = {"error": r.text}
        await update.message.reply_text(format_signal(d), parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Gagal analisis gambar: {e}")

# ---------------- MAIN RUN ----------------
def main():
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN belum diset di environment.")
        return
    if not CHAT_ID:
        print("❌ CHAT_ID belum diset di environment. Auto signals akan gagal dikirim.")
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_csv))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manual_message))

    # scheduler for auto-scan
    scheduler = BackgroundScheduler()
    # schedule every AUTO_SCAN_HOURS hours
    scheduler.add_job(lambda: auto_check_and_send(app), 'interval', hours=AUTO_SCAN_HOURS, next_run_time=None)
    scheduler.start()
    print(f"[STARTUP] Auto-scan scheduled every {AUTO_SCAN_HOURS} hour(s). Pairs monitored: {len(AUTO_PAIRS)} TF: {AUTO_TIMEFRAMES}")

    # graceful stop signal
    stop_event = Event()
    try:
        print("🤖 Telegram bot running (auto-signal mode). Press Ctrl+C to stop.")
        app.run_polling()
    finally:
        stop_event.set()
        scheduler.shutdown(wait=False)
        print("[SHUTDOWN] Scheduler stopped.")

if __name__ == "__main__":
    main()
