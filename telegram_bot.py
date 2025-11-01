# telegram_bot_pro_global_v4.4.py
# ProTraderAI - Global Crypto + Forex Scanner (v4.4)
# Features:
#  - Global Auto Rotation (Binance + TwelveData)
#  - Multi-Timeframe Scan (3m,5m,15m,1h,4h,1d)
#  - Confidence Filter >= 0.75
#  - /scan_now [pair] -> global or targeted scan
#  - /retrain -> trigger AI retrain
#  - /help (interactive + examples)
#  - /stop_auto and /start_auto to pause/resume auto-scan loop
# Requirements: python-telegram-bot==20.3, requests, pandas, numpy, asyncio

import os
import asyncio
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)

# ========================= CONFIG (ENV) =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
APP_URL = os.getenv("APP_URL", "").rstrip("/")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")

GLOBAL_AUTO_MODE = os.getenv("GLOBAL_AUTO_MODE", "true").lower() == "true"
FOREX_GLOBAL = os.getenv("FOREX_GLOBAL", "true").lower() == "true"
MAX_ACTIVE_PAIRS = int(os.getenv("MAX_ACTIVE_PAIRS", "8"))
MIN_CONFIDENCE = float(os.getenv("STRONG_SIGNAL_THRESHOLD", "0.75"))
TIMEFRAMES = os.getenv("AUTO_TIMEFRAMES", "3m,5m,15m,1h,4h,1d").split(",")
SCAN_INTERVAL_HOURS = float(os.getenv("AUTO_SCAN_HOURS", "1"))
DELAY_BETWEEN_REQUESTS = float(os.getenv("DELAY_BETWEEN_REQUESTS", "1.5"))

# Control flag for auto-scan loop (can be toggled by /stop_auto and /start_auto)
AUTO_SCAN_ENABLED = True

# ========================= HELPERS =========================
def _ai_signal_request(pair: str, timeframe: str):
    """Call AI Agent /signal endpoint; returns dict or {'error':...}"""
    if not APP_URL:
        return {"error": "APP_URL not configured"}
    try:
        r = requests.post(f"{APP_URL}/signal", json={"pair": pair, "timeframe": timeframe}, timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def format_signal_full(result: dict):
    """Format professional + emoji message for Telegram"""
    try:
        pair = result.get("pair", "?")
        tf = result.get("timeframe", "?")
        sig = result.get("signal_type", "?")
        tp1, tp2, sl = result.get("tp1", "?"), result.get("tp2", "?"), result.get("sl", "?")
        conf, ml = result.get("confidence", "?"), result.get("ml_prob", "-")
        ts = result.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        reasoning = result.get("reasoning", "")

        emoji = "🚀" if str(sig).upper() in ["LONG", "BUY"] else "⚡" if str(sig).upper() in ["SHORT", "SELL"] else "💤"
        header = f"{emoji} <b>{pair}</b> ({tf})"
        msg = (
            f"{header}\n"
            f"📈 Sinyal: <b>{sig}</b>\n"
            f"🎯 TP1: {tp1} | TP2: {tp2} | 🛑 SL: {sl}\n"
            f"🤖 Conf: {conf} | ML: {ml}\n"
            f"🕐 {ts} UTC"
        )
        if reasoning:
            msg += f"\n🧠 <i>{reasoning[:300]}</i>"
        return msg
    except Exception as e:
        return f"⚠️ Format error: {e}"

def send_message_sync(app, text: str):
    """Send Telegram message synchronously (used by background tasks)"""
    try:
        if not CHAT_ID:
            print("[BOT] CHAT_ID not configured")
            return False
        app.bot.send_message(chat_id=int(CHAT_ID), text=text, parse_mode="HTML")
        return True
    except Exception as e:
        print("[BOT SEND ERROR]", e)
        return False

# ========================= AUTO ROTATION / PAIR FETCH =========================
def get_top_active_crypto(limit=8):
    """Fetch top active crypto pairs from Binance based on volume & volatility"""
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        r = requests.get(url, timeout=10)
        data = r.json()
        df = pd.DataFrame(data)
        # Sanity check
        if df.empty or "quoteVolume" not in df.columns:
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"][:limit]
        df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce").fillna(0)
        df["priceChangePercent"] = pd.to_numeric(df["priceChangePercent"], errors="coerce").fillna(0)
        df = df[df["symbol"].str.endswith("USDT")]
        df["score"] = np.log(df["quoteVolume"] + 1) * (df["priceChangePercent"].abs() + 1)
        top = df.sort_values("score", ascending=False)["symbol"].head(limit).tolist()
        return top
    except Exception as e:
        print("[AUTO ROTATION CRYPTO ERROR]", e)
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"][:limit]

def get_top_forex_pairs(limit=10):
    """Fetch a curated list of forex/commodity pairs via TwelveData (validate availability)"""
    if not TWELVEDATA_API_KEY or not FOREX_GLOBAL:
        return []
    core = ["EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF","NZD/USD","XAU/USD","XAG/USD"]
    valid = []
    base_url = "https://api.twelvedata.com/price?symbol={}&apikey={}"
    for p in core:
        try:
            res = requests.get(base_url.format(p, TWELVEDATA_API_KEY), timeout=5)
            if res.status_code == 200:
                valid.append(p.replace("/", ""))
        except Exception:
            continue
    return valid[:limit]

def get_top_global_pairs():
    """Combine crypto + forex lists and return unique top-N pairs"""
    crypto_top = get_top_active_crypto(MAX_ACTIVE_PAIRS)
    forex_top = get_top_forex_pairs(MAX_ACTIVE_PAIRS)
    combined = list(dict.fromkeys(crypto_top + forex_top))  # preserve order, unique
    if not combined:
        # fallback default
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSD"][:MAX_ACTIVE_PAIRS]
    return combined[:MAX_ACTIVE_PAIRS]

# ========================= AUTO SCAN LOOP =========================
async def auto_scan_loop(app):
    """Recurring background loop that runs scans every SCAN_INTERVAL_HOURS when enabled"""
    global AUTO_SCAN_ENABLED
    scan_seconds = max(15.0, SCAN_INTERVAL_HOURS * 3600)
    print(f"[AUTO] Auto-scan loop started. Interval: {SCAN_INTERVAL_HOURS} hour(s).")
    while True:
        try:
            if not AUTO_SCAN_ENABLED:
                # if paused, simply wait and check again later
                await asyncio.sleep(5)
                continue
            pairs = get_top_global_pairs() if GLOBAL_AUTO_MODE else ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            print(f"[AUTO] Starting scan at {datetime.utcnow()} for pairs: {pairs}")
            for pair in pairs:
                for tf in TIMEFRAMES:
                    # small delay between requests to avoid rate-limits
                    await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
                    try:
                        res = _ai_signal_request(pair, tf)
                        if not isinstance(res, dict) or res.get("error"):
                            continue
                        sig = res.get("signal_type", "WAIT")
                        if sig == "WAIT":
                            continue
                        conf = float(res.get("confidence", 0) or 0)
                        if conf >= MIN_CONFIDENCE:
                            msg = format_signal_full(res)
                            sent = send_message_sync(app, msg)
                            print(f"[AUTO] Sent strong signal {pair} {tf} conf={conf} sent={sent}")
                    except Exception as e:
                        print("[AUTO PER PAIR ERROR]", e)
            print("[AUTO] Global scan completed.\n")
        except Exception as e:
            print("[AUTO LOOP ERROR]", e)
        await asyncio.sleep(scan_seconds)

# ========================= COMMANDS (INTERACTIVE) =========================
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Interactive help with examples and emojis"""
    msg = (
        "🆘 <b>ProTraderAI Help — Commands & Examples</b>\n\n"
        "🔹 <b>/start</b> - Show intro & quick tips\n"
        "🔹 <b>/status</b> - Show bot status and current configuration\n"
        "🔹 <b>/scan_now</b> - Run a global scan now (top active pairs crypto + forex)\n"
        "🔹 <b>/scan_now BTCUSDT</b> - Run a targeted scan for BTCUSDT across all timeframes\n"
        "🔹 <b>/scan_now EURUSD</b> - Targeted scan for forex pair\n"
        "🔹 <b>/retrain</b> - Trigger AI retrain (XGBoost) from AI Agent\n"
        "🔹 <b>/stop_auto</b> - Pause automatic hourly scanning (you can still run /scan_now)\n"
        "🔹 <b>/start_auto</b> - Resume automatic hourly scanning\n\n"
        "💬 <i>Manual analyze:</i> send messages like <code>BTCUSDT 15m</code> or <code>EURUSD 1h</code>\n\n"
        "📝 Examples:\n"
        "  • <code>/scan_now</code> — scan global top pairs now\n"
        "  • <code>/scan_now BTCUSDT</code> — scan BTC in all TFs\n"
        "  • <code>ETHUSDT 15m</code> — analyze ETH 15-minute\n\n"
        "⚠️ Note: Bot sends only strong signals (Conf >= {:.2f}) by default.".format(MIN_CONFIDENCE)
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 <b>ProTraderAI Global Scanner v4.4</b>\n\n"
        "Halo! Aku akan memantau pasar crypto + forex dan kirim sinyal kuat ke Telegrammu.\n"
        "Ketik <b>/help</b> untuk panduan lengkap dan contoh penggunaan.\n"
        "Kamu bisa pause auto-scan dengan <b>/stop_auto</b> dan resume dengan <b>/start_auto</b>."
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = {
        "GLOBAL_AUTO_MODE": GLOBAL_AUTO_MODE,
        "FOREX_GLOBAL": FOREX_GLOBAL,
        "MAX_ACTIVE": MAX_ACTIVE_PAIRS,
        "TIMEFRAMES": ",".join(TIMEFRAMES),
        "MIN_CONFIDENCE": MIN_CONFIDENCE,
        "AUTO_SCAN_HOURS": SCAN_INTERVAL_HOURS,
        "AUTO_SCAN_ENABLED": AUTO_SCAN_ENABLED
    }
    txt = "<b>⚙️ Bot Status</b>\n" + "\n".join([f"{k}: {v}" for k, v in info.items()])
    await update.message.reply_text(txt, parse_mode="HTML")

async def retrain_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧠 Triggering AI retrain (please wait)...", parse_mode="HTML")
    if not APP_URL:
        await update.message.reply_text("⚠️ APP_URL not configured.", parse_mode="HTML")
        return
    try:
        r = requests.post(f"{APP_URL}/retrain_learning", timeout=120)
        try:
            res = r.json()
            await update.message.reply_text(f"✅ Retrain finished. Samples: {res.get('samples','N/A')}", parse_mode="HTML")
        except Exception:
            await update.message.reply_text(f"⚠️ Retrain completed but invalid response: {r.text}", parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Retrain failed: {e}", parse_mode="HTML")

async def scan_now_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Supports:
       - /scan_now           -> global scan
       - /scan_now BTCUSDT   -> targeted scan for BTCUSDT across TIMEFRAMES
    """
    app = context.application
    args = context.args

    if args:
        target = args[0].upper()
        await update.message.reply_text(f"🎯 Scanning <b>{target}</b> across all timeframes...", parse_mode="HTML")
        for tf in TIMEFRAMES:
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
            res = _ai_signal_request(target, tf)
            if not isinstance(res, dict) or res.get("error"):
                continue
            if res.get("signal_type", "WAIT") == "WAIT":
                continue
            try:
                conf = float(res.get("confidence", 0) or 0)
            except Exception:
                conf = 0
            if conf >= MIN_CONFIDENCE:
                msg = format_signal_full(res)
                send_message_sync(app, msg)
        await update.message.reply_text(f"✅ Scan complete for {target}.", parse_mode="HTML")
    else:
        await update.message.reply_text("🔁 Starting immediate global scan...", parse_mode="HTML")
        pairs = get_top_global_pairs() if GLOBAL_AUTO_MODE else ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        for pair in pairs:
            for tf in TIMEFRAMES:
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
                res = _ai_signal_request(pair, tf)
                if not isinstance(res, dict) or res.get("error"):
                    continue
                if res.get("signal_type", "WAIT") == "WAIT":
                    continue
                try:
                    conf = float(res.get("confidence", 0) or 0)
                except Exception:
                    conf = 0
                if conf >= MIN_CONFIDENCE:
                    msg = format_signal_full(res)
                    send_message_sync(app, msg)
        await update.message.reply_text("✅ Global scan finished!", parse_mode="HTML")

async def stop_auto_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pause the recurring auto-scan loop (manual scans still allowed)"""
    global AUTO_SCAN_ENABLED
    AUTO_SCAN_ENABLED = False
    await update.message.reply_text("⛔ Auto-scan paused. You can still run <b>/scan_now</b> manually.", parse_mode="HTML")

async def start_auto_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Resume the recurring auto-scan loop"""
    global AUTO_SCAN_ENABLED
    AUTO_SCAN_ENABLED = True
    await update.message.reply_text("▶️ Auto-scan resumed. Bot will run scheduled scans again.", parse_mode="HTML")

async def manual_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manual analysis: 'PAIR TF' or 'PAIR' (default 15m)"""
    text = update.message.text.strip()
    if not text:
        return
    parts = text.split()
    pair = parts[0].upper()
    tf = parts[1] if len(parts) > 1 else "15m"
    await update.message.reply_text(f"🔍 Analyzing {pair} ({tf})...", parse_mode="HTML")
    res = _ai_signal_request(pair, tf)
    if isinstance(res, dict) and not res.get("error"):
        msg = format_signal_full(res)
        await update.message.reply_text(msg, parse_mode="HTML")
    else:
        await update.message.reply_text(f"⚠️ Error: {res.get('error')}", parse_mode="HTML")

# ========================= MAIN / STARTUP =========================
def main():
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN not set in environment. Exiting.")
        return
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("scan_now", scan_now_cmd))
    app.add_handler(CommandHandler("retrain", retrain_cmd))
    app.add_handler(CommandHandler("stop_auto", stop_auto_cmd))
    app.add_handler(CommandHandler("start_auto", start_auto_cmd))
    # manual messages (pair/timeframe)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manual_analyze))

    # Start background auto-scan loop
    async def on_startup(app_):
        # small delay for stable startup
        await asyncio.sleep(2)
        app_.create_task(auto_scan_loop(app_))

    app.post_init = on_startup

    print("[BOT] ProTraderAI Global v4.4 running...")
    app.run_polling(stop_signals=None)

if __name__ == "__main__":
    main()
