# telegram_bot.py
# ProTraderAI - Telegram Bot with Auto-scan & Backtest integration
# Env required: BOT_TOKEN, CHAT_ID, APP_URL
# Optional envs: API_TIMEOUT, STRONG_SIGNAL_THRESHOLD, AUTO_SCAN_HOURS, AUTO_TIMEFRAMES,
#               ENABLE_BACKTEST, ENABLE_CHART_IMAGE, ENABLE_VOICE_ALERT, DELAY_BETWEEN_REQUESTS
#
# Dependencies: python-telegram-bot==20.3, requests, apscheduler

import os
import re
import time
import requests
from datetime import datetime
from threading import Event, Lock

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
)

from apscheduler.schedulers.background import BackgroundScheduler

# ---------------- CONFIG (from env) ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()  # destination chat id for auto signals
APP_URL = os.getenv("APP_URL", "").rstrip("/")
if APP_URL and not APP_URL.startswith("http"):
    APP_URL = "https://" + APP_URL

API_TIMEOUT = int(os.getenv("API_TIMEOUT", "25"))
STRONG_SIGNAL_THRESHOLD = float(os.getenv("STRONG_SIGNAL_THRESHOLD", "0.75"))
AUTO_SCAN_HOURS = int(os.getenv("AUTO_SCAN_HOURS", "1"))
AUTO_TIMEFRAMES = os.getenv("AUTO_TIMEFRAMES", "3m,5m,15m,1h,4h,1d").split(",")
DELAY_BETWEEN_REQUESTS = float(os.getenv("DELAY_BETWEEN_REQUESTS", "0.6"))

ENABLE_BACKTEST = os.getenv("ENABLE_BACKTEST", "true").lower() in ("1", "true", "yes")
ENABLE_CHART_IMAGE = os.getenv("ENABLE_CHART_IMAGE", "true").lower() in ("1", "true", "yes")
ENABLE_VOICE_ALERT = os.getenv("ENABLE_VOICE_ALERT", "false").lower() in ("1", "true", "yes")

# Pairs: try to respect user's env or fallback to full list (crypto + forex)
PAIR_LIST_ENV = os.getenv("PAIR_LIST", "").strip()
if PAIR_LIST_ENV:
    AUTO_PAIRS = [p.strip().upper() for p in PAIR_LIST_ENV.split(",") if p.strip()]
else:
    AUTO_PAIRS = [
        "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","LTCUSDT","DOGEUSDT",
        "MATICUSDT","DOTUSDT","AVAXUSDT","LINKUSDT",
        "XAUUSD","EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF"
    ]

# ---------------- STATE ----------------
scheduler = BackgroundScheduler()
auto_job = None
auto_job_lock = Lock()
stop_event = Event()

# ---------------- HELPERS ----------------
def format_signal(result: dict) -> str:
    """Pretty-format a signal dict into Telegram message (HTML)."""
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
            reasoning = str(result.get("reasoning"))[:800]
            lines.append(f"🧠 Reasoning: {reasoning}")
        # backtest summary
        bt = result.get("backtest_raw") or result.get("backtest")
        if isinstance(bt, dict):
            lines.append("")  # blank line
            lines.append("📋 Backtest summary:")
            hit = bt.get("hit")
            pnl = bt.get("pnl_total") or bt.get("pnl")
            if hit is not None:
                lines.append(f"  • Hit: {hit}")
            if pnl is not None:
                lines.append(f"  • PnL total: {pnl}")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Format error: {e}"

def parse_pair_tf(text: str):
    """
    Parse many input formats into (PAIR, TF).
    Accepts: 'BTCUSDT 15m', 'btc/usd 1H', 'analisa XAUUSD 1h', 'force btc 15m'
    """
    if not text:
        return None, "15m"
    t = text.upper().replace("/", " ").replace("_", " ").strip()
    # timeframe
    tf_match = re.search(r"(\d+\s*[MHDW])", t)
    tf = tf_match.group(1).replace(" ", "").lower() if tf_match else "15m"
    # remove verbs
    t_clean = re.sub(r"\b(ANALISA|ANALYZE|ANALYSE|CHECK|FORCE|SCALP|INFO|BACKTEST)\b", " ", t, flags=re.IGNORECASE).strip()
    # aliases
    aliases = {
        "GOLD": "XAUUSD", "EMAS": "XAUUSD",
        "BITCOIN": "BTCUSDT", "BTC": "BTCUSDT",
        "ETH": "ETHUSDT", "SOL": "SOLUSDT", "EUR": "EURUSD"
    }
    for a, v in aliases.items():
        if a in t_clean:
            return v, tf
    # find pair tokens like BTC USDT or BTCUSDT
    m = re.search(r"([A-Z0-9]{3,6})\s*([A-Z]{3,4})", t_clean)
    if m:
        pair = (m.group(1) + m.group(2)).upper()
        return pair, tf
    m2 = re.search(r"([A-Z]{3,6}(?:USDT|USD|EUR|JPY|GBP|IDR|BTC|ETH))", t_clean)
    if m2:
        return m2.group(1).upper(), tf
    # fallback: first token
    token = t_clean.split()[0] if t_clean.split() else None
    if token:
        return token.replace(" ", "").upper(), tf
    return None, tf

def send_request_get(endpoint: str, params: dict = None, timeout: int = API_TIMEOUT):
    """Send GET request to backend APP_URL with error wrapper."""
    if not APP_URL:
        return {"error": "APP_URL not configured"}
    url = f"{APP_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        try:
            return r.json()
        except Exception:
            return {"error": f"invalid_json_response: {r.text}"}
    except Exception as e:
        return {"error": str(e)}

def send_request_post(endpoint: str, files: dict = None, data: dict = None, timeout: int = API_TIMEOUT):
    """Send POST request with files/data to backend APP_URL."""
    if not APP_URL:
        return {"error": "APP_URL not configured"}
    url = f"{APP_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        r = requests.post(url, files=files, data=data, timeout=timeout)
        try:
            return r.json()
        except Exception:
            return {"error": f"invalid_json_response: {r.text}"}
    except Exception as e:
        return {"error": str(e)}

# ---------------- AUTO-SCAN LOGIC ----------------
def process_signal_and_send(bot, res):
    """Helper to send message (and optionally chart/voice) for one result dict."""
    if not isinstance(res, dict):
        return
    try:
        msg = format_signal(res)
        if not CHAT_ID:
            print("[AUTO] CHAT_ID not configured; skipping send.")
            return
        # send main text
        bot.send_message(chat_id=int(CHAT_ID), text=msg, parse_mode="HTML")
        # optionally request & send chart image from backend (endpoint: /signal_chart?pair=...&tf_entry=...)
        if ENABLE_CHART_IMAGE:
            try:
                chart = send_request_get("signal_chart", params={"pair": res.get("pair"), "tf_entry": res.get("timeframe")}, timeout=30)
                # backend may return {'chart_url': '...'} or binary; we handle chart_url
                if isinstance(chart, dict) and chart.get("chart_url"):
                    bot.send_photo(chat_id=int(CHAT_ID), photo=chart.get("chart_url"))
            except Exception as e:
                print(f"[AUTO] chart send failed: {e}")
        # optionally send voice alert (voice file url from backend /voice_alert?pair=...&type=buy/sell)
        if ENABLE_VOICE_ALERT and res.get("signal_type"):
            try:
                vtype = "buy" if res.get("signal_type", "").upper() in ("LONG","BUY") else "sell"
                voice = send_request_get("voice_alert", params={"pair": res.get("pair"), "type": vtype}, timeout=20)
                if isinstance(voice, dict) and voice.get("voice_url"):
                    bot.send_audio(chat_id=int(CHAT_ID), audio=voice.get("voice_url"))
            except Exception as e:
                print(f"[AUTO] voice send failed: {e}")
    except Exception as e:
        print(f"[AUTO] Error sending signal message: {e}")

def auto_check_and_send(app):
    """
    Iterate AUTO_PAIRS and AUTO_TIMEFRAMES; call /pro_signal; send to CHAT_ID if strong.
    """
    bot = app.bot
    now = datetime.utcnow().isoformat()
    print(f"[AUTO] Auto-scan start {now} - pairs {len(AUTO_PAIRS)} TF {AUTO_TIMEFRAMES}")
    for pair in AUTO_PAIRS:
        for tf in AUTO_TIMEFRAMES:
            try:
                params = {"pair": pair, "tf_entry": tf}
                res = send_request_get("pro_signal", params=params)
                if not isinstance(res, dict):
                    print(f"[AUTO] non-dict response for {pair} {tf}: {res}")
                    continue
                if "error" in res:
                    print(f"[AUTO] {pair} {tf} -> error: {res.get('error')}")
                    continue
                conf = float(res.get("confidence", 0) or 0)
                # send always if confidence >= threshold and not WAIT
                if conf >= STRONG_SIGNAL_THRESHOLD and res.get("signal_type") and res.get("signal_type") != "WAIT":
                    process_signal_and_send(bot, res)
                    print(f"[AUTO] Sent strong signal {pair} {tf} (conf={conf})")
                else:
                    print(f"[AUTO] {pair} {tf} no strong signal (conf={conf})")
                # optionally call backtest for logging or attach result to the message
                if ENABLE_BACKTEST:
                    try:
                        bt = send_request_post("backtest", data={"pair": pair, "tf_entry": tf}, timeout=60)
                        # if backend returns backtest summary, we can send condensed info
                        if isinstance(bt, dict) and bt.get("summary"):
                            # attach small message with summary
                            summary_text = f"📋 Backtest {pair} {tf} -> {bt.get('summary')}"
                            bot.send_message(chat_id=int(CHAT_ID), text=summary_text)
                    except Exception as e:
                        print(f"[AUTO] backtest call failed for {pair} {tf}: {e}")
                time.sleep(DELAY_BETWEEN_REQUESTS)
            except Exception as e:
                print(f"[AUTO EXC] {pair} {tf}: {e}")
                time.sleep(0.3)
    print(f"[AUTO] Auto-scan finished at {datetime.utcnow().isoformat()}")

def start_auto_job(app):
    global auto_job
    with auto_job_lock:
        if auto_job is None:
            # schedule immediately first run
            auto_job = scheduler.add_job(lambda: auto_check_and_send(app), 'interval', hours=AUTO_SCAN_HOURS, next_run_time=None)
            print("[AUTO] Job scheduled.")
        else:
            print("[AUTO] Job already running.")

def stop_auto_job():
    global auto_job
    with auto_job_lock:
        if auto_job is not None:
            try:
                auto_job.remove()
            except Exception:
                pass
            auto_job = None
            print("[AUTO] Job removed.")
        else:
            print("[AUTO] No auto job to remove.")

# ---------------- TELEGRAM HANDLERS ----------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 <b>ProTraderAI - Assistant</b>\n\n"
        "Kirim perintah seperti:\n"
        "- <code>BTCUSDT 15m</code> atau <code>analisa BTCUSDT 15m</code>\n"
        "- <code>force BTCUSDT 15m</code> (tampilkan semua sinyal)\n"
        "- <code>scalp BTCUSDT</code>\n\n"
        "Command:\n"
        "/status - status model\n"
        "/logs - sinyal terakhir\n"
        "/performance - performa AI\n"
        "/retrain - retrain model ML\n"
        "/auto_on - aktifkan auto-scan\n"
        "/auto_off - nonaktifkan auto-scan\n"
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        res = send_request_get("learning_status")
        if "error" in res:
            await update.message.reply_text(f"⚠️ {res.get('error')}")
            return
        msg = (
            "📊 <b>Status Model AI</b>\n\n"
            f"📁 Model file: {'✅ Ada' if res.get('model_exists') else '❌ Tidak ada'}\n"
            f"📈 Jumlah data log: {res.get('trade_log_count', 0)}\n"
            f"🧠 Algoritma: {res.get('algo', 'Unknown')}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal mengambil status model.\nError: {e}")

async def logs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        res = send_request_get("logs_summary")
        if "error" in res or res is None:
            await update.message.reply_text("⚠️ Belum ada log yang tersimpan.")
            return
        msg = (
            "📋 <b>Log Terakhir AI Agent</b>\n\n"
            f"🪙 Pair: {res.get('pair')}\n"
            f"🕒 Timeframe: {res.get('timeframe')}\n"
            f"💡 Sinyal: {res.get('signal_type')}\n"
            f"🎯 Entry: {res.get('entry')}\n"
            f"🎯 TP1: {res.get('tp1')} | TP2: {res.get('tp2')}\n"
            f"🛑 SL: {res.get('sl')}\n"
            f"📊 Confidence: {res.get('confidence')}\n\n"
            f"🧠 Reasoning:\n{res.get('reasoning', '-')}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal mengambil log terakhir.\nError: {e}")

async def performance_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        res = send_request_get("ai_performance")
        if "error" in res:
            await update.message.reply_text("⚠️ Belum ada data performa AI.")
            return
        msg = (
            "📈 <b>AI Performance Report</b>\n\n"
            f"📊 Total sinyal: {res.get('total_signals', 0)}\n"
            f"🏆 Winrate: {res.get('winrate', 0)}%\n"
            f"💰 Profit Factor: {res.get('profit_factor', 0)}\n"
        )
        await update.message.reply_text(msg, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal mengambil data performa.\nError: {e}")

async def auto_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        app = context.application
        start_auto_job(app)
        await update.message.reply_text("✅ Auto-scan diaktifkan.")
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal mengaktifkan auto-scan: {e}")

async def auto_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        stop_auto_job()
        await update.message.reply_text("⛔ Auto-scan dinonaktifkan.")
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal menonaktifkan auto-scan: {e}")

async def retrain_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Perintah retrain model AI langsung dari Telegram."""
    try:
        await update.message.reply_text("🧠 Melatih ulang model AI... harap tunggu (proses bisa memakan waktu) ⏳")

        if not APP_URL:
            await update.message.reply_text("❌ APP_URL belum dikonfigurasikan di environment.")
            return

        url = f"{APP_URL.rstrip('/')}/retrain_learning"
        r = requests.post(url, timeout=300)
        try:
            res = r.json()
        except Exception:
            await update.message.reply_text(f"⚠️ Retrain selesai tapi response tidak JSON: {r.text}")
            return

        if "error" in res:
            await update.message.reply_text(f"❌ Gagal retrain model.\nError: {res.get('error')}")
            return

        algo = res.get("algo", "XGBoost")
        samples = res.get("samples", res.get("sample_count", "N/A"))
        msg = (
            "✅ <b>Model retrained successfully!</b>\n\n"
            f"🧠 Algorithm: {algo}\n"
            f"📈 Samples used: {samples}\n"
            f"📂 Model path: {res.get('model_path', 'xgb_model.json')}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    except Exception as e:
        await update.message.reply_text(f"❌ Retrain gagal.\nError: {e}")

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
        await update.message.reply_text(f"❌ Error: {res.get('error')}")
        return

    # optionally request backtest for manual check
    if ENABLE_BACKTEST:
        try:
            bt = send_request_post("backtest", data={"pair": pair, "tf_entry": tf}, timeout=60)
            if isinstance(bt, dict):
                # attach backtest result into response dict
                res["backtest"] = bt
        except Exception as e:
            print(f"[MANUAL] backtest error: {e}")

    conf = float(res.get("confidence", 0) or 0)
    if (not is_force) and conf < STRONG_SIGNAL_THRESHOLD:
        await update.message.reply_text(
            f"⚠️ Tidak ada sinyal kuat untuk {pair} ({tf}).\nConfidence saat ini: {conf}",
            parse_mode="HTML"
        )
        return

    await update.message.reply_text(format_signal(res), parse_mode="HTML")

# ---------------- FILE / IMG / CSV handlers ----------------
async def handle_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        await update.message.reply_text("⚠️ Tidak ada file terdeteksi.")
        return
    await update.message.reply_text("📄 Menerima CSV, mengirim ke AI untuk analisis...")
    file = await doc.get_file()
    try:
        content = requests.get(file.file_path, timeout=30).content
        files = {"file": ("upload.csv", content)}
        res = send_request_post("analyze_csv", files=files, timeout=60)
        await update.message.reply_text(format_signal(res), parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Gagal analisis CSV: {e}")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1] if update.message.photo else None
    if not photo:
        await update.message.reply_text("⚠️ Tidak ada gambar.")
        return
    await update.message.reply_text("📷 Menganalisis chart (OCR + heuristics)...")
    file = await photo.get_file()
    try:
        content = requests.get(file.file_path, timeout=60).content
        files = {"file": ("chart.jpg", content)}
        res = send_request_post("analyze_chart", files=files, timeout=60)
        await update.message.reply_text(format_signal(res), parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Gagal analisis gambar: {e}")

# ---------------- SETUP & RUN ----------------
def main():
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN belum di set pada environment.")
        return
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))
    app.add_handler(CommandHandler("performance", performance_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on_cmd))
    app.add_handler(CommandHandler("auto_off", auto_off_cmd))
    app.add_handler(CommandHandler("retrain", retrain_cmd))

    # Message handlers
    app.add_handler(MessageHandler(filters.Document.ALL, handle_csv))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manual_message))

    # Start scheduler and auto job
    scheduler.start()
    start_auto_job(app)

    print(f"[STARTUP] Bot running. Auto-scan every {AUTO_SCAN_HOURS} hour(s). Pairs: {len(AUTO_PAIRS)} TF: {AUTO_TIMEFRAMES}")
    try:
        app.run_polling()
    finally:
        stop_event.set()
        try:
            stop_auto_job()
            scheduler.shutdown(wait=False)
        except Exception:
            pass
        print("[SHUTDOWN] Bot stopped, scheduler shutdown.")

if __name__ == "__main__":
    main()
