# ==========================================================
# ProTraderAI - Telegram Bot Final (Fully Stable)
# Features:
# - Manual /force & /scalp analysis
# - Auto-scan (crypto + forex)
# - Retrain AI model (GET version)
# - CSV + Chart analysis
# - Logs, status, performance
# - Compatible with Hybrid PRO AI + Backtester
# ==========================================================

import os
import re
import time
import threading
import requests
from datetime import datetime
from threading import Event, Lock

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
)
from apscheduler.schedulers.background import BackgroundScheduler

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
APP_URL = os.getenv("APP_URL", "").rstrip("/")
if APP_URL and not APP_URL.startswith("http"):
    APP_URL = "https://" + APP_URL

API_TIMEOUT = int(os.getenv("API_TIMEOUT", "25"))
STRONG_SIGNAL_THRESHOLD = float(os.getenv("STRONG_SIGNAL_THRESHOLD", "0.8"))
AUTO_SCAN_HOURS = int(os.getenv("AUTO_SCAN_HOURS", "1"))
AUTO_TIMEFRAMES = os.getenv("AUTO_TIMEFRAMES", "15m,1h,4h").split(",")

AUTO_PAIRS_CRYPTO = os.getenv("AUTO_PAIRS_CRYPTO", "").strip()
AUTO_PAIRS_FOREX = os.getenv("AUTO_PAIRS_FOREX", "").strip()

if AUTO_PAIRS_CRYPTO:
    AUTO_PAIRS_CRYPTO = [p.strip().upper() for p in AUTO_PAIRS_CRYPTO.split(",")]
else:
    AUTO_PAIRS_CRYPTO = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "LTCUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT"
    ]

if AUTO_PAIRS_FOREX:
    AUTO_PAIRS_FOREX = [p.strip().upper() for p in AUTO_PAIRS_FOREX.split(",")]
else:
    AUTO_PAIRS_FOREX = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD"]

AUTO_PAIRS = list(dict.fromkeys(AUTO_PAIRS_CRYPTO + AUTO_PAIRS_FOREX))

VERBOSE = True
scheduler = BackgroundScheduler()
auto_job = None
auto_job_lock = Lock()
stop_event = Event()

# ---------------- LOG ----------------
def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# ---------------- HELPERS ----------------
def format_signal(result: dict) -> str:
    """Format sinyal ke pesan Telegram (HTML)"""
    if not isinstance(result, dict):
        return "⚠️ Tidak bisa membaca hasil sinyal."
    if "error" in result:
        return f"❌ Error: {result.get('error')}"
    try:
        lines = []
        pair = result.get("pair", "?")
        tf = result.get("timeframe", "?")
        lines.append(f"📊 <b>{pair}</b> ({tf})")
        lines.append(f"💡 <b>{result.get('signal_type','?')}</b>")

        engine = result.get("engine_mode", "")
        if engine.upper() == "SMC":
            lines.append("🧩 Engine: SMC / ICT Smart Money")
        elif engine.upper() == "HYBRID":
            lines.append("🧩 Engine: Hybrid Technical Fallback")
        elif engine:
            lines.append(f"🧩 Engine: {engine}")

        lines.append(f"🎯 Entry: <code>{result.get('entry')}</code>")
        lines.append(f"🎯 TP1: <code>{result.get('tp1')}</code> | TP2: <code>{result.get('tp2')}</code>")
        lines.append(f"🛑 SL: <code>{result.get('sl')}</code>")
        if result.get("confidence") is not None:
            lines.append(f"📊 Confidence: {result.get('confidence')}")
        if result.get("reasoning"):
            lines.append(f"🧠 Reasoning: {result.get('reasoning')}")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Format error: {e}"

def parse_pair_tf(text: str):
    """Parse 'BTCUSDT 15m' -> (BTCUSDT, 15m)"""
    if not text:
        return None, "15m"
    t = text.upper().replace("/", " ").replace("_", " ").strip()
    tf_match = re.search(r"(\d+\s*[MHDW])", t)
    tf = tf_match.group(1).replace(" ", "").lower() if tf_match else "15m"
    t_clean = re.sub(r"\b(ANALISA|FORCE|SCALP|INFO|CHECK)\b", " ", t, flags=re.IGNORECASE).strip()
    aliases = {
        "GOLD": "XAUUSD", "EMAS": "XAUUSD",
        "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"
    }
    for a, v in aliases.items():
        if a in t_clean:
            return v, tf
    m = re.search(r"([A-Z]{3,6}(?:USDT|USD|JPY|EUR|GBP|BTC|ETH))", t_clean)
    if m:
        return m.group(1).upper(), tf
    return t_clean.split()[0].upper() if t_clean else None, tf

def send_request_get(endpoint: str, params: dict = None, timeout: int = API_TIMEOUT):
    """GET ke backend"""
    if not APP_URL:
        return {"error": "APP_URL belum dikonfigurasikan"}
    url = f"{APP_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def send_request_post(endpoint: str, files: dict = None, data: dict = None, timeout: int = API_TIMEOUT):
    """POST ke backend"""
    if not APP_URL:
        return {"error": "APP_URL belum dikonfigurasikan"}
    url = f"{APP_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        r = requests.post(url, files=files, data=data, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ---------------- AUTO SCAN ----------------
def auto_check_and_send(app):
    now = datetime.utcnow().isoformat()
    vprint(f"[AUTO] Start scan {now}")
    for pair in AUTO_PAIRS:
        for tf in AUTO_TIMEFRAMES:
            try:
                res = send_request_get("pro_signal", {"pair": pair, "tf_entry": tf})
                if not isinstance(res, dict) or "error" in res:
                    vprint(f"[AUTO] {pair} {tf} -> error")
                    continue
                conf = float(res.get("confidence", 0) or 0)
                if conf >= STRONG_SIGNAL_THRESHOLD and res.get("signal_type") != "WAIT":
                    msg = format_signal(res)
                    tg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                    try:
                        requests.post(tg_url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
                        vprint(f"[AUTO] Sent {pair} {tf} conf={conf}")
                    except Exception as e:
                        vprint(f"[AUTO ERROR] send fail {pair} {tf}: {e}")
                time.sleep(0.6)
            except Exception as e:
                vprint(f"[AUTO EXC] {pair} {tf}: {e}")
                time.sleep(0.3)
    vprint(f"[AUTO] Done scan {datetime.utcnow().isoformat()}")

def start_auto_job(app):
    global auto_job
    with auto_job_lock:
        if auto_job is None:
            threading.Thread(target=auto_check_and_send, args=(app,), daemon=True).start()
            auto_job = scheduler.add_job(lambda: auto_check_and_send(app), 'interval', hours=AUTO_SCAN_HOURS)
            vprint("[AUTO] Scheduled every", AUTO_SCAN_HOURS, "hours")

def stop_auto_job():
    global auto_job
    with auto_job_lock:
        if auto_job:
            try: auto_job.remove()
            except Exception: pass
            auto_job = None
            vprint("[AUTO] Job removed")

# ---------------- COMMAND HANDLERS ----------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 <b>ProTraderAI Assistant</b>\n\n"
        "Gunakan perintah:\n"
        "- <code>BTCUSDT 15m</code>\n"
        "- <code>force BTCUSDT 15m</code>\n"
        "- <code>scalp BTCUSDT</code>\n\n"
        "Upload CSV atau chart untuk analisis otomatis.\n\n"
        "Command:\n"
        "/status • /logs • /performance • /retrain • /auto_on • /auto_off"
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    res = send_request_get("learning_status")
    msg = (
        "📊 <b>Status AI</b>\n\n"
        f"📁 RF model: {'✅' if res.get('rf_model_exists') else '❌'}\n"
        f"📁 XGB model: {'✅' if res.get('xgb_model_exists') else '❌'}\n"
        f"📈 Log data: {res.get('trade_log_count', 0)}"
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def retrain_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧠 Melatih ulang model AI... harap tunggu ⏳")
    res = send_request_get("retrain_learning")
    rf, xgb = res.get("rf", {}), res.get("xgb", {})
    msg = f"✅ <b>Retrain Selesai</b>\n\nRF: {rf}\nXGB: {xgb}"
    await update.message.reply_text(msg, parse_mode="HTML")

async def manual_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text: return
    t_low = text.lower()
    is_force = t_low.startswith("force")
    if is_force: text = text.replace("force", "", 1).strip()
    if t_low.startswith("scalp"):
        pair, _ = parse_pair_tf(text)
        res = send_request_get("scalp_signal", {"pair": pair, "tf": "3m"})
        await update.message.reply_text(format_signal(res), parse_mode="HTML")
        return
    pair, tf = parse_pair_tf(text)
    if not pair:
        await update.message.reply_text("❌ Pair tidak dikenali.")
        return
    await update.message.reply_text(f"🔍 Analisis {pair} ({tf}) ...")
    res = send_request_get("pro_signal", {"pair": pair, "tf_entry": tf})
    conf = float(res.get("confidence", 0) or 0)
    if (not is_force) and conf < STRONG_SIGNAL_THRESHOLD:
        await update.message.reply_text(f"⚠️ Tidak ada sinyal kuat ({conf})")
        return
    await update.message.reply_text(format_signal(res), parse_mode="HTML")

async def handle_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    await update.message.reply_text("📄 CSV diterima, menganalisis...")
    file = await doc.get_file()
    content = requests.get(file.file_path).content
    files = {"file": ("upload.csv", content)}
    res = send_request_post("analyze_csv", files=files)
    body = res.get("result") if isinstance(res, dict) and "result" in res else res
    await update.message.reply_text(format_signal(body), parse_mode="HTML")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    await update.message.reply_text("📷 Analisis chart...")
    file = await photo.get_file()
    content = requests.get(file.file_path).content
    files = {"file": ("chart.jpg", content)}
    res = send_request_post("analyze_chart", files=files)
    body = res.get("result") if isinstance(res, dict) and "result" in res else res
    await update.message.reply_text(format_signal(body), parse_mode="HTML")

# ---------------- RUN ----------------
def main():
    vprint("[STARTUP] Telegram Bot running...")
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN belum di set")
        return
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("retrain", retrain_cmd))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_csv))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manual_message))

    scheduler.start()
    start_auto_job(app)

    app.run_polling()

if __name__ == "__main__":
    main()
