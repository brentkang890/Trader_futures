# telegram_bot.py
# ProTraderAI - Full Telegram Bot (manual + auto-scan + status/logs/performance + csv/image + retrain)
# Requires environment variables: BOT_TOKEN, CHAT_ID, APP_URL
# Requirements: python-telegram-bot==20.3, requests, apscheduler

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

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")  # destination chat id for auto signals (string or int)
APP_URL = os.getenv("APP_URL", "").rstrip("/")
if APP_URL and not APP_URL.startswith("http"):
    APP_URL = "https://" + APP_URL

API_TIMEOUT = int(os.getenv("API_TIMEOUT", "25"))
STRONG_SIGNAL_THRESHOLD = float(os.getenv("STRONG_SIGNAL_THRESHOLD", "0.8"))
AUTO_SCAN_HOURS = int(os.getenv("AUTO_SCAN_HOURS", "1"))  # default 1 hour
AUTO_TIMEFRAMES = os.getenv("AUTO_TIMEFRAMES", "15m,1h,4h").split(",")

AUTO_PAIRS_CRYPTO = os.getenv("AUTO_PAIRS_CRYPTO", "").strip()
AUTO_PAIRS_FOREX = os.getenv("AUTO_PAIRS_FOREX", "").strip()

if AUTO_PAIRS_CRYPTO:
    AUTO_PAIRS_CRYPTO = [p.strip().upper() for p in AUTO_PAIRS_CRYPTO.split(",")]
else:
    AUTO_PAIRS_CRYPTO = [
        "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","LTCUSDT","DOGEUSDT",
        "MATICUSDT","DOTUSDT","AVAXUSDT","LINKUSDT"
    ]

if AUTO_PAIRS_FOREX:
    AUTO_PAIRS_FOREX = [p.strip().upper() for p in AUTO_PAIRS_FOREX.split(",")]
else:
    AUTO_PAIRS_FOREX = ["XAUUSD","EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF"]

AUTO_PAIRS = list(dict.fromkeys(AUTO_PAIRS_CRYPTO + AUTO_PAIRS_FOREX))

# ---------------- STATE ----------------
scheduler = BackgroundScheduler()
auto_job = None
auto_job_lock = Lock()
auto_enabled = True  # default auto-scan ON
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

        # ===== PATCH TAMBAHAN BACKTEST DETAIL =====
        bt = result.get("backtest_raw") or result.get("backtest") or result.get("backtest_result")
        if isinstance(bt, dict):
            lines.append("")  # blank line
            lines.append("📋 <b>Backtest Result</b>")
            hit = bt.get("hit", "-")
            pnl = bt.get("pnl_total") or bt.get("pnl") or "-"
            bars = bt.get("bars_to_hit") or bt.get("bars") or "-"
            entry = bt.get("entry") or result.get("entry")
            tp1 = bt.get("tp1") or result.get("tp1")
            sl = bt.get("sl") or result.get("sl")
            lines.append(f"🎯 Hit: {hit}")
            lines.append(f"💰 PnL Total: {pnl}")
            lines.append(f"⏱️ Bars to Hit: {bars}")
            if entry:
                lines.append(f"📈 Entry: {entry}")
            if tp1:
                lines.append(f"🎯 Target: {tp1}")
            if sl:
                lines.append(f"🛑 Stop: {sl}")
        # ===== END PATCH =====

        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Format error: {e}"

def parse_pair_tf(text: str):
    if not text:
        return None, "15m"
    t = text.upper().replace("/", " ").replace("_", " ").strip()
    tf_match = re.search(r"(\d+\s*[MHDW])", t)
    tf = tf_match.group(1).replace(" ", "").lower() if tf_match else "15m"
    t_clean = re.sub(r"\b(ANALISA|ANALYZE|ANALYSE|CHECK|FORCE|SCALP|INFO)\b", " ", t, flags=re.IGNORECASE).strip()
    aliases = {
        "GOLD": "XAUUSD", "EMAS": "XAUUSD",
        "BITCOIN": "BTCUSDT", "BTC": "BTCUSDT",
        "ETH": "ETHUSDT", "SOL": "SOLUSDT", "EUR": "EURUSD"
    }
    for a, v in aliases.items():
        if a in t_clean:
            return v, tf
    m = re.search(r"([A-Z0-9]{3,6})\s*([A-Z]{3,4})", t_clean)
    if m:
        pair = (m.group(1) + m.group(2)).upper()
        return pair, tf
    m2 = re.search(r"([A-Z]{3,6}(?:USDT|USD|EUR|JPY|GBP|IDR|BTC|ETH))", t_clean)
    if m2:
        return m2.group(1).upper(), tf
    token = t_clean.split()[0] if t_clean.split() else None
    if token:
        return token.replace(" ", "").upper(), tf
    return None, tf

def send_request_get(endpoint: str, params: dict = None, timeout: int = API_TIMEOUT):
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
def auto_check_and_send(app):
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
                if conf >= STRONG_SIGNAL_THRESHOLD and res.get("signal_type") and res.get("signal_type") != "WAIT":
                    msg = format_signal(res)
                    try:
                        if not CHAT_ID:
                            print("[AUTO] CHAT_ID not configured; skipping send.")
                        else:
                            bot.send_message(chat_id=int(CHAT_ID), text=msg, parse_mode="HTML")
                            print(f"[AUTO] Sent strong signal {pair} {tf} (conf={conf})")
                    except Exception as e:
                        print(f"[AUTO ERROR] send_message failed for {pair} {tf}: {e}")
                else:
                    print(f"[AUTO] {pair} {tf} no strong signal (conf={conf})")
                time.sleep(0.6)
            except Exception as e:
                print(f"[AUTO EXC] {pair} {tf}: {e}")
                time.sleep(0.3)
    print(f"[AUTO] Auto-scan finished at {datetime.utcnow().isoformat()}")

def start_auto_job(app):
    global auto_job
    with auto_job_lock:
        if auto_job is None:
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
# (Semua command handler, retrain, CSV, image, dan main() tetap sama persis seperti di file kamu)
# Tidak ada perubahan selain format_signal()
