# main_combined_learning_v2.py
"""
Pro Trader AI v2 - Vision Mode (Bahasa Indonesia)
- Analisa Crypto, Forex, dan Chart Gambar (Vision)
- Telegram Integration (teks & gambar)
- Auto-Learning + Keep Alive
"""

import os, io, time, threading, requests, cv2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.responses import JSONResponse
import ta
from sklearn.ensemble import RandomForestClassifier
import joblib
from PIL import Image
try:
    import pytesseract
    _HAS_TESSERACT = True
except:
    _HAS_TESSERACT = False

app = FastAPI(title="Pro Trader AI v2", version="2.0")

TRADE_LOG_FILE = "trade_log.csv"
MODEL_FILE = "rf_model.pkl"

# ================= UTILITAS =================
def fetch_ohlc_binance(symbol: str, interval: str, limit: int = 500):
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": limit}, timeout=10)
    data = r.json()
    df = pd.DataFrame(data, columns=["t","o","h","l","c","v","ct","qa","n","tb","tq","i"])
    df = df.astype(float, errors="ignore")
    return df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})

def ema(series, n): return ta.trend.EMAIndicator(series, window=n).ema_indicator()
def rsi(series, n=14): return ta.momentum.RSIIndicator(series, window=n).rsi()
def atr(df, n=14): return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=n).average_true_range()

def detect_sr(df, lookback=100):
    return float(df['high'].tail(lookback).max()), float(df['low'].tail(lookback).min())

def hybrid_analyze(df, pair=None, tf=None):
    if len(df) < 20: return {"error":"data_tidak_cukup"}
    df['ema20'], df['ema50'], df['rsi'], df['atr'] = ema(df['close'],20), ema(df['close'],50), rsi(df['close']), atr(df)
    last = df.iloc[-1]
    price, ema20, ema50, rsi_now, atr_now = float(last['close']), float(last['ema20']), float(last['ema50']), float(last['rsi']), float(last['atr'])
    high, low = detect_sr(df)
    trend = "bullish" if ema20 > ema50 else "bearish"
    reasons, conf = [], []
    if trend=="bullish" and price>ema20 and 35<rsi_now<75:
        entry, sl, tp1, tp2 = price, low-atr_now*0.5, price+atr_now*1.5, price+atr_now*2.5
        signal="LONG"; reasons.append("BOS naik & EMA searah.")
        conf.append(0.9)
    elif trend=="bearish" and price<ema20 and 25<rsi_now<70:
        entry, sl, tp1, tp2 = price, high+atr_now*0.5, price-atr_now*1.5, price-atr_now*2.5
        signal="SHORT"; reasons.append("BOS turun & EMA searah.")
        conf.append(0.9)
    else:
        entry, sl, tp1, tp2 = price, price*0.995, price*1.002, price*1.004
        signal="WAIT"; reasons.append("Belum ada arah jelas.")
        conf.append(0.3)
    return {"pair":pair or "","timeframe":tf or "","signal_type":signal,"entry":round(entry,8),"tp1":round(tp1,8),"tp2":round(tp2,8),"sl":round(sl,8),"confidence":float(np.mean(conf)),"reasoning":" · ".join(reasons)}

# ================= OCR & VISION =================
def ocr_y_axis_prices(img):
    """Deteksi angka harga di sisi kanan chart"""
    if not _HAS_TESSERACT: return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    y_map = {}
    for i, text in enumerate(data["text"]):
        if text.strip().replace(".", "", 1).isdigit():
            y = data["top"][i]
            val = float(text)
            y_map[y] = val
    return y_map

def detect_candles_from_plot(img, y_map, max_bars=200):
    """Deteksi bentuk candlestick sederhana"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candles = []
    if len(y_map)<2: return pd.DataFrame(columns=["open","high","low","close"])
    ys = sorted(y_map.keys()); y0, y1 = ys[0], ys[-1]
    v0, v1 = y_map[y0], y_map[y1]
    def px_to_price(y): return v0 + (v1 - v0) * ((y - y0) / (y1 - y0))
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if 5<w<25 and 10<h<150:
            high, low = px_to_price(y), px_to_price(y+h)
            open_ = (high+low)/2; close_ = open_
            candles.append([open_,high,low,close_])
    return pd.DataFrame(candles[-max_bars:],columns=["open","high","low","close"])

# ================= ENDPOINT ANALISA =================
@app.post("/analyze_chart")
def analyze_chart(file: UploadFile = File(...), pair: Optional[str]=Form(None), timeframe: Optional[str]=Form(None)):
    """Analisis chart dari gambar"""
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        y_map = ocr_y_axis_prices(img_cv)
        df = detect_candles_from_plot(img_cv, y_map)
        if df.empty: raise HTTPException(status_code=400, detail="chart_tidak_terbaca")
        res = hybrid_analyze(df, pair, timeframe)
        return JSONResponse(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gagal_menganalisa_gambar: {e}")

# ================= TELEGRAM BOT =================
import telebot
APP_URL = os.getenv("APP_URL")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if TELEGRAM_TOKEN:
    bot = telebot.TeleBot(TELEGRAM_TOKEN)

    @bot.message_handler(commands=['start'])
    def start_msg(msg):
        bot.reply_to(msg,"👋 Hai! Kirim pair (contoh: BTCUSDT 1h) atau kirim gambar chart.")

    @bot.message_handler(content_types=['text'])
    def handle_text(msg):
        try:
            pair, tf = msg.text.strip().upper().split()
            r = requests.get(f"{APP_URL}/pro_signal?pair={pair}&tf={tf}", timeout=25)
            data = r.json()
            bot.send_message(msg.chat.id,
                f"📈 <b>{pair}</b> ({tf})\n🎯 Entry: {data['entry']}\n💰 TP1: {data['tp1']} | TP2: {data['tp2']}\n🛑 SL: {data['sl']}\n📊 Conf: {data['confidence']}\n💬 {data['reasoning']}",
                parse_mode="HTML")
        except Exception as e:
            bot.send_message(msg.chat.id, f"❌ Error: {e}")

    @bot.message_handler(content_types=['photo'])
    def handle_image(msg):
        try:
            file_id = msg.photo[-1].file_id
            file_info = bot.get_file(file_id)
            file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_info.file_path}"
            img_data = requests.get(file_url).content
            files = {'file': ('chart.png', img_data, 'image/png')}
            data = {'pair': 'IMG', 'timeframe': 'chart'}
            r = requests.post(f"{APP_URL}/analyze_chart", files=files, data=data, timeout=40)
            if r.status_code == 200:
                data = r.json()
                bot.send_message(msg.chat.id,
                    f"📊 <b>Analisis Chart</b>\n🎯 Entry: {data['entry']}\n💰 TP1: {data['tp1']} | TP2: {data['tp2']}\n🛑 SL: {data['sl']}\n📈 Sinyal: {data['signal_type']}\n📉 Conf: {data['confidence']}\n💬 {data['reasoning']}",
                    parse_mode="HTML")
            else:
                bot.send_message(msg.chat.id, f"❌ Gagal analisa gambar: {r.text}")
        except Exception as e:
            bot.send_message(msg.chat.id, f"🚫 Error: {e}")

    threading.Thread(target=lambda: bot.polling(non_stop=True), daemon=True).start()
    print("🤖 Telegram Bot aktif untuk teks & gambar!")
else:
    print("⚠️ TELEGRAM_TOKEN belum diset.")

# ================= KEEP ALIVE =================
def keep_alive_loop():
    while True:
        try:
            if APP_URL:
                r = requests.get(APP_URL)
                print(f"[KEEP-ALIVE] {r.status_code} {datetime.utcnow().isoformat()}")
        except Exception as e:
            print("Keep alive error:", e)
        time.sleep(300)
threading.Thread(target=keep_alive_loop, daemon=True).start()

print("✅ Pro Trader AI v2 aktif penuh!")
