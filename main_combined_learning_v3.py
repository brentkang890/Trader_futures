""" Pro Trader AI v3 - Combined + Learning + Vision + Robust Startup

FastAPI service for signals (crypto + csv + image)
RandomForest auto-learning from logged signals
Telegram integration (text & image) non-blocking
Keep-alive + auto-redeploy hooks (safe/no auto-trigger)

Notes:
This file is production-ready for a small Railway-like container.
Configure environment variables listed below before deploy.

ENV VARS REQUIRED / OPTIONAL:
APP_URL (optional for keep-alive; recommended)
TELEGRAM_TOKEN (optional: to enable bot)
CHAT_ID (optional: default chat for notifications)
BACKTEST_URL (optional: endpoint to send signals for evaluation)
RAILWAY_TOKEN / PROJECT_ID (optional: if you add auto-redeploy using Railway API)

Run with:
uvicorn main_combined_learning_v3:app --host 0.0.0.0 --port $PORT
"""

import os
import io
import time
import threading
import requests
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# optional libs that may be heavy on some platforms
try:
    import ta
except Exception:
    ta = None

# sklearn libs
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    import joblib
except Exception:
    RandomForestClassifier = None
    train_test_split = None
    classification_report = None
    roc_auc_score = None
    joblib = None

# image/ocr libs
from PIL import Image
import cv2
try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    pytesseract = None
    _HAS_TESSERACT = False

# telegram (optional)
try:
    import telebot
except Exception:
    telebot = None

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pro_trader_ai")

app = FastAPI(title="Pro Trader AI v3", version="3.0")

# ---------------- CONFIG ----------------
BACKTEST_URL = os.getenv("BACKTEST_URL")
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trade_log.csv")
MODEL_FILE = os.getenv("MODEL_FILE", "rf_model.pkl")
MIN_SAMPLES_TO_TRAIN = int(os.getenv("MIN_SAMPLES_TO_TRAIN", "50"))
N_SIGNALS_TO_RETRAIN = int(os.getenv("N_SIGNALS_TO_RETRAIN", "50"))
BINANCE_KLINES = os.getenv("BINANCE_KLINES", "https://api.binance.com/api/v3/klines")
APP_URL = os.getenv("APP_URL")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# thread-safety / state
_lock = threading.Lock()
_last_retrain_count = 0

# ---------------- UTILITIES ----------------
def fetch_ohlc_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch klines from Binance and return standardized DataFrame."""
    if not symbol:
        raise ValueError("symbol_required")
    r = requests.get(BINANCE_KLINES, params={
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "tb_base", "tb_quote", "ignore"
    ])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time", "open", "high", "low", "close", "volume"]]

# defensive wrappers for ta
def _ema(series, n):
    if ta is None:
        raise RuntimeError("ta_library_missing")
    return ta.trend.EMAIndicator(series, window=n).ema_indicator()

def _rsi(series, n=14):
    if ta is None:
        raise RuntimeError("ta_library_missing")
    return ta.momentum.RSIIndicator(series, window=n).rsi()

def _atr(df, n=14):
    if ta is None:
        raise RuntimeError("ta_library_missing")
    return ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=n
    ).average_true_range()

# small helpers
def detect_sr(df: pd.DataFrame, lookback: int = 120):
    recent_h = df['high'].tail(lookback).max()
    recent_l = df['low'].tail(lookback).min()
    return float(recent_h), float(recent_l)

def breakout_of_structure(df: pd.DataFrame, window: int = 20):
    if df.shape[0] < window + 2:
        return None
    high_sw = df['high'].rolling(window).max().iloc[-2]
    low_sw = df['low'].rolling(window).min().iloc[-2]
    last = df['close'].iloc[-1]
    prev = df['close'].iloc[-2]
    if prev <= high_sw and last > high_sw:
        return "BOS_UP"
    if prev >= low_sw and last < low_sw:
        return "BOS_DOWN"
    return None

# ---------------- STRATEGY ----------------
def hybrid_analyze(df: pd.DataFrame, pair: Optional[str] = None, timeframe: Optional[str] = None) -> dict:
    df = df.copy().dropna().reset_index(drop=True)
    if df.shape[0] < 12:
        return {"error": "data_tidak_cukup", "message": "Perlu minimal 12 candle untuk analisis."}

    df['ema20'] = _ema(df['close'], 20)
    df['ema50'] = _ema(df['close'], 50)
    df['rsi14'] = _rsi(df['close'], 14)
    df['atr14'] = _atr(df, 14)

    last = df.iloc[-1]
    price = float(last['close'])
    ema20 = float(last['ema20'])
    ema50 = float(last['ema50'])
    rsi_now = float(last['rsi14'])
    atr_now = float(last['atr14']) if not np.isnan(last['atr14']) else max(price * 0.001, 1e-8)

    recent_high, recent_low = detect_sr(df, lookback=120)
    bos = breakout_of_structure(df, window=20)
    swing_high = df['high'].tail(80).max()
    swing_low = df['low'].tail(80).min()
    diff = swing_high - swing_low
    fib_618 = swing_high - diff * 0.618 if diff > 0 else price

    reasons, conf = [], []
    trend = "bullish" if ema20 > ema50 else "bearish"

    if bos == "BOS_UP" or (trend == "bullish" and price > ema20):
        entry = price
        sl = recent_low - atr_now * 0.6
        rr = entry - sl if entry > sl else price * 0.01
        tp1 = entry + rr * 1.5
        tp2 = entry + rr * 2.5
        reasons.append("Bias LONG — BOS naik & EMA searah.")
        conf.extend([0.9 if trend == "bullish" else 0.6,
                     0.9 if price >= fib_618 else 0.65,
                     1.0 if 30 < rsi_now < 75 else 0.5])
        signal = "LONG"
    elif bos == "BOS_DOWN" or (trend == "bearish" and price < ema20):
        entry = price
        sl = recent_high + atr_now * 0.6
        rr = sl - entry if sl > entry else price * 0.01
        tp1 = entry - rr * 1.5
        tp2 = entry - rr * 2.5
        reasons.append("Bias SHORT — BOS turun & EMA searah bearish.")
        conf.extend([0.9 if trend == "bearish" else 0.6,
                     0.9 if price <= fib_618 else 0.65,
                     1.0 if 25 < rsi_now < 70 else 0.5])
        signal = "SHORT"
    else:
        entry = price
        sl = recent_low * 0.995
        tp1 = entry + (entry - sl) * 1.2
        tp2 = entry + (entry - sl) * 2.0
        reasons.append("Belum ada arah jelas — tunggu konfirmasi TF lebih tinggi.")
        conf.append(0.25)
        signal = "WAIT"

    confidence = float(sum(conf) / len(conf)) if conf else 0.0
    reasoning = " · ".join(reasons)
    return {
        "pair": pair or "",
        "timeframe": timeframe or "",
        "signal_type": signal,
        "entry": round(entry, 8),
        "tp1": round(tp1, 8),
        "tp2": round(tp2, 8),
        "sl": round(sl, 8),
        "confidence": round(confidence, 3),
        "reasoning": reasoning
    }

# ---------------- SCALP ENGINE ----------------
def scalp_engine(df: pd.DataFrame, pair: Optional[str] = None, tf: Optional[str] = None) -> dict:
    if df.shape[0] < 30:
        return {"error": "data_tidak_cukup"}

    df['ema8'] = _ema(df['close'], 8)
    df['ema21'] = _ema(df['close'], 21)
    df['rsi14'] = _rsi(df['close'], 14)
    df['atr14'] = _atr(df, 14)

    last = df.iloc[-1]
    price = float(last['close'])
    atr_now = float(last['atr14']) if not np.isnan(last['atr14']) else price * 0.001
    vol_mean = df['volume'].tail(40).mean() if df.shape[0] >= 40 else df['volume'].mean()
    vol_spike = float(last['volume']) > (vol_mean * 1.8 if vol_mean > 0 else False)

    if float(last['ema8']) > float(last['ema21']) and vol_spike and 35 < float(last['rsi14']) < 75:
        entry = price
        sl = entry - atr_now * 0.6
        tp1 = entry + atr_now * 0.8
        tp2 = entry + atr_now * 1.4
        reason = "Scalp LONG — EMA8 di atas EMA21, volume tinggi, RSI netral."
        conf = 0.9
        signal = "LONG"
    elif float(last['ema8']) < float(last['ema21']) and vol_spike and 25 < float(last['rsi14']) < 65:
        entry = price
        sl = entry + atr_now * 0.6
        tp1 = entry - atr_now * 0.8
        tp2 = entry - atr_now * 1.4
        reason = "Scalp SHORT — EMA8 di bawah EMA21, volume tinggi, RSI mendukung."
        conf = 0.9
        signal = "SHORT"
    else:
        entry = price
        sl = price * 0.998
        tp1 = price * 1.002
        tp2 = price * 1.004
        reason = "Tidak ada peluang scalping bersih, disarankan tunggu konfirmasi."
        conf = 0.3
        signal = "WAIT"

    return {
        "pair": pair or "",
        "timeframe": tf or "",
        "signal_type": signal,
        "entry": round(entry, 8),
        "tp1": round(tp1, 8),
        "tp2": round(tp2, 8),
        "sl": round(sl, 8),
        "confidence": round(conf, 3),
        "reasoning": reason
    }

# ---------------- KEEP ALIVE ----------------
def keep_alive_loop():
    while True:
        try:
            if APP_URL:
                try:
                    r = requests.get(APP_URL, timeout=10)
                    logger.info(f"[KEEP-ALIVE] {r.status_code} {datetime.utcnow().isoformat()}")
                except Exception as e:
                    logger.warning(f"[KEEP-ALIVE] ping failed: {e}")
            else:
                logger.debug("APP_URL not set; skipping keep-alive ping")
        except Exception as e:
            logger.exception("Keep alive loop error")
        time.sleep(300)

threading.Thread(target=keep_alive_loop, daemon=True).start()


# ---------------- TELEGRAM BOT ----------------
if telebot and TELEGRAM_TOKEN:
    try:
        bot = telebot.TeleBot(TELEGRAM_TOKEN)

        @bot.message_handler(commands=['start'])
        def _welcome(msg):
            bot.reply_to(msg, "👋 Halo! Kirim pair + timeframe (contoh: BTCUSDT 1h) atau kirim gambar chart untuk dianalisis.")

        @bot.message_handler(func=lambda m: True, content_types=['text'])
        def _handle_text(msg):
            try:
                parts = msg.text.strip().upper().split()
                if len(parts) < 1:
                    bot.reply_to(msg, "⚠️ Format salah. Contoh: BTCUSDT 15m")
                    return
                pair = parts[0]
                tf = parts[1] if len(parts) > 1 else '15m'
                url = f"{APP_URL}/pro_signal?pair={pair}&tf_main={tf}&tf_entry={tf}"
                r = requests.get(url, timeout=25)
                if r.status_code == 200:
                    d = r.json()
                    bot.send_message(msg.chat.id, f"📈 {pair} ({tf})\n🎯 Entry: {d.get('entry')}\n💰 TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n🛑 SL: {d.get('sl')}\n📊 Conf: {d.get('confidence')}\n💬 {d.get('reasoning')}", parse_mode='HTML')
                else:
                    bot.send_message(msg.chat.id, f"❌ Gagal ambil sinyal: {r.text}")
            except Exception as e:
                bot.send_message(msg.chat.id, f"🚫 Error: {e}")

        @bot.message_handler(content_types=['photo'])
        def _handle_photo(msg):
            try:
                file_id = msg.photo[-1].file_id
                file_info = bot.get_file(file_id)
                file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_info.file_path}"
                img_data = requests.get(file_url).content
                files = {'file': ('chart.png', img_data, 'image/png')}
                data = {'pair': 'IMG', 'timeframe': 'chart'}
                r = requests.post(f"{APP_URL}/analyze_chart", files=files, data=data, timeout=40)
                if r.status_code == 200:
                    d = r.json()
                    bot.send_message(msg.chat.id, f"📊 Analisis Chart\n🎯 Entry: {d.get('entry')}\n💰 TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n🛑 SL: {d.get('sl')}\n📈 Sinyal: {d.get('signal_type')}\n📊 Conf: {d.get('confidence')}\n💬 {d.get('reasoning')}", parse_mode='HTML')
                else:
                    bot.send_message(msg.chat.id, f"❌ Gagal analisa gambar: {r.text}")
            except Exception as e:
                bot.send_message(msg.chat.id, f"🚫 Error: {e}")

        threading.Thread(target=lambda: bot.polling(non_stop=True), daemon=True).start()
        logger.info("🤖 Telegram Bot aktif (menerima teks & gambar)")
    except Exception as e:
        logger.exception("Gagal menjalankan bot Telegram")
else:
    logger.warning("⚠️ TELEGRAM_TOKEN belum diset atau library telebot tidak tersedia.")
logger.info("✅ Pro Trader AI v3 siap & aktif penuh!")

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Pro Trader AI v3</title></head>
    <body style="background-color:#0b0b0b;color:#00ff99;font-family:sans-serif;text-align:center;padding:40px;">
        <h2>🚀 Pro Trader AI v3 Aktif</h2>
        <p>Server berjalan dengan stabil di Railway.</p>
        <p>🤖 Telegram Bot status: {status}</p>
        <p>Gunakan perintah <b>/start</b> di Telegram untuk memulai.</p>
    </body>
    </html>
    """.format(status="✅ Aktif" if TELEGRAM_TOKEN else "❌ Tidak Aktif")
