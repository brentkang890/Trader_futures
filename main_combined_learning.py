# main_combined_learning.py
"""
Pro Trader AI - Combined + Learning (Bahasa Indonesia)
- Analisis Crypto & Forex (Binance / TwelveData)
- Auto-logging sinyal ke trade_log.csv
- Integrated Learning (RandomForest) + model cache + retrain background
- Chart OCR + candle detection (best-effort)
- Sentiment fusion (best-effort)
- Endpoints: /pro_signal, /scalp_signal, /analyze_chart, /analyze_csv, /learning_status, /retrain_learning, /ai_performance, /logs, /logs_summary, /download_logs, /model_debug, /sentiment, /mode, /context
"""
import os
import io
import re
import csv
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any

import requests
import pandas as pd
import numpy as np

from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.encoders import jsonable_encoder

# technical libs
import ta

# learning libs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# image libs (optional)
from PIL import Image
import cv2
try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    pytesseract = None
    _HAS_TESSERACT = False

app = FastAPI(
    title="Pro Trader AI - Combined + Learning (ID)",
    description="Analisis Crypto (Binance) & Forex (TwelveData) + Pembelajaran Terintegrasi",
    version="1.0"
)

# ---------------- KONFIG ----------------
BACKTEST_URL = os.environ.get("BACKTEST_URL")  # optional evaluator endpoint
TRADE_LOG_FILE = os.environ.get("TRADE_LOG_FILE", "trade_log.csv")
MODEL_FILE = os.environ.get("MODEL_FILE", "rf_model.pkl")
MIN_SAMPLES_TO_TRAIN = int(os.environ.get("MIN_SAMPLES_TO_TRAIN", 50))
N_SIGNALS_TO_RETRAIN = int(os.environ.get("N_SIGNALS_TO_RETRAIN", 50))
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
TWELVEDATA_URL = "https://api.twelvedata.com/time_series"

ALPHA_API_KEY = os.environ.get("ALPHA_API_KEY", "")  # kept if needed elsewhere
ALPHA_URL = "https://www.alphavantage.co/query"

# sentiment endpoints (best-effort)
FNG_URL = "https://api.alternative.me/fng/"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

# thread-safety + caching
_lock = threading.Lock()
_last_retrain_count = 0
_cached_model = None  # cache untuk model yang dimuat dari disk

# ---------------- HELPERS ----------------
def respond(obj: Any, status_code: int = 200):
    """Safe JSON response — always return valid JSON, handle NaN/inf/None gracefully."""
    import json

    def clean_value(v):
        """Convert NaN/Inf and unsupported types to safe JSON values"""
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return 0.0
        if isinstance(v, (np.int64, np.int32)):
            return int(v)
        if isinstance(v, (np.float32, np.float64)):
            return float(v)
        if isinstance(v, dict):
            return {str(k): clean_value(val) for k, val in v.items()}
        if isinstance(v, list):
            return [clean_value(val) for val in v]
        return v

    try:
        encoded = jsonable_encoder(obj)
        cleaned = clean_value(encoded)
        return JSONResponse(content=cleaned, status_code=status_code)
    except Exception as e:
        print("⚠️ respond() fallback:", e)
        try:
            safe_str = json.dumps(str(obj))
            return JSONResponse(content={"fallback": safe_str}, status_code=status_code)
        except Exception:
            return PlainTextResponse(str(obj), status_code=status_code)

def ensure_trade_log():
    """Pastikan file trade_log.csv ada."""
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "pair", "timeframe", "signal_type",
                "entry", "tp1", "tp2", "sl", "confidence",
                "reasoning", "backtest_hit", "backtest_pnl"
            ])

def append_trade_log(logrec: Dict[str, Any]):
    """Tambahkan 1 baris log hasil sinyal ke trade_log.csv"""
    ensure_trade_log()
    with open(TRADE_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            logrec.get("pair"),
            logrec.get("timeframe"),
            logrec.get("signal_type"),
            logrec.get("entry"),
            logrec.get("tp1"),
            logrec.get("tp2"),
            logrec.get("sl"),
            logrec.get("confidence"),
            logrec.get("reasoning"),
            logrec.get("backtest_hit"),
            logrec.get("backtest_pnl")
        ])

def detect_market(pair: str) -> str:
    """Tentukan apakah pair crypto atau forex (best-effort)."""
    p = (pair or "").upper()
    if any(x in p for x in ["USDT", "BUSD", "BTC", "ETH", "SOL", "BNB", "ADA", "DOGE"]):
        return "crypto"
    # untuk simbol seperti EURUSD, XAUUSD, GBPJPY, dll.
    if len(p) >= 6 and p[-3:].isalpha() and p[:-3].isalpha():
        return "forex"
    return "crypto"

# ---------------- FETCH OHLC ----------------
def fetch_ohlc_twelvedata(symbol: str, interval: str="15m", limit: int=500) -> pd.DataFrame:
    """Ambil data time series dari TwelveData untuk forex (XAUUSD, EURUSD, dsb)."""
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY_not_set")
    mapping = {
        "1m": "1min", "3m": "1min", "5m": "5min", "15m": "15min",
        "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1day"
    }
    iv = mapping.get(interval, interval)
    params = {
        "symbol": symbol,
        "interval": iv,
        "outputsize": limit,
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON"
    }
    r = requests.get(TWELVEDATA_URL, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    if "status" in j and j.get("status") == "error":
        raise RuntimeError(f"TwelveData error: {j.get('message')}")
    values = j.get("values")
    if not values:
        raise RuntimeError(f"TwelveData no data for {symbol}: {j}")
    df = pd.DataFrame(values)
    # ensure required cols
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = 0.0
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
        df.insert(0, "open_time", pd.RangeIndex(start=0, stop=len(df)))
    else:
        df.insert(0, "open_time", pd.RangeIndex(start=0, stop=len(df)))
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["open_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    return df.tail(limit).reset_index(drop=True)

def fetch_ohlc_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Try Binance (crypto). If fail, fallback to TwelveData for forex."""
    symbol = symbol.upper()
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(BINANCE_KLINES, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data, columns=[
                "open_time","open","high","low","close","volume","close_time",
                "qav","num_trades","tb_base","tb_quote","ignore"
            ])
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[["open_time","open","high","low","close","volume"]].reset_index(drop=True)
            return df
    except Exception:
        pass

    # Fallback to TwelveData for forex or any non-Binance symbol
    try:
        return fetch_ohlc_twelvedata(symbol, interval, limit)
    except Exception as e:
        # last fallback: AlphaVantage if TwelveData not configured
        try:
            return fetch_ohlc_alpha_forex(symbol, interval, limit)
        except Exception:
            raise RuntimeError(f"fetch_ohlc_failed_for_{symbol}: {e}")

def fetch_ohlc_alpha_forex(symbol: str, interval: str="15m", limit: int=500) -> pd.DataFrame:
    """Fallback lama: AlphaVantage FX_INTRADAY — tetap tersedia jika TwelveData tidak dipakai."""
    if not ALPHA_API_KEY:
        raise RuntimeError("ALPHA_API_KEY_not_set")
    symbol = symbol.upper()
    from_sym = symbol[:3]
    to_sym = symbol[3:]
    mapping = {"1m":"1min","3m":"5min","5m":"5min","15m":"15min","30m":"30min","1h":"60min","4h":"60min","1d":"daily"}
    iv = mapping.get(interval, "15min")
    params = {"function": "FX_INTRADAY", "from_symbol": from_sym, "to_symbol": to_sym,
              "interval": iv, "apikey": ALPHA_API_KEY, "outputsize":"compact"}
    r = requests.get(ALPHA_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    possible_keys = [k for k in data.keys() if "Time Series" in k]
    if not possible_keys:
        raise ValueError(f"AlphaVantage no data: {data}")
    key = possible_keys[0]
    ts = data[key]
    df = pd.DataFrame(ts).T
    df.columns = [c.split('. ')[-1] for c in df.columns]
    df = df.rename(columns=lambda c: c.strip())
    cols_need = [c for c in ["open","high","low","close"] if c in df.columns]
    df = df[cols_need].astype(float)
    df = df.sort_index().tail(limit).reset_index(drop=True)
    df["volume"] = 0.0
    df.insert(0, "open_time", pd.RangeIndex(start=0, stop=len(df)))
    return df[["open_time","open","high","low","close","volume"]]

# ---------------- INDICATORS ----------------
def ema(series: pd.Series, n: int):
    return ta.trend.EMAIndicator(series, window=n).ema_indicator()

def rsi(series: pd.Series, n: int=14):
    return ta.momentum.RSIIndicator(series, window=n).rsi()

def atr(df: pd.DataFrame, n: int=14):
    return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=n).average_true_range()

def detect_sr(df: pd.DataFrame, lookback:int=120):
    recent_h = df['high'].tail(lookback).max()
    recent_l = df['low'].tail(lookback).min()
    return float(recent_h), float(recent_l)

def breakout_of_structure(df: pd.DataFrame, window:int=20):
    if df.shape[0] < window+2: return None
    high_sw = df['high'].rolling(window).max().iloc[-2]
    low_sw = df['low'].rolling(window).min().iloc[-2]
    last = df['close'].iloc[-1]
    prev = df['close'].iloc[-2]
    if prev <= high_sw and last > high_sw: return "BOS_UP"
    if prev >= low_sw and last < low_sw: return "BOS_DOWN"
    return None
    
# ---------------- SMC HELPERS (POI / OB / FVG / LIQ / PATTERN / APEX) ----------------
def _rolling_argrelextrema(series: pd.Series, order: int = 3, mode: str = "max"):
    idxs = []
    if len(series) < order*2+1: 
        return idxs
    for i in range(order, len(series)-order):
        window = series.iloc[i-order:i+order+1]
        center = series.iloc[i]
        if mode == "max" and center == window.max():
            idxs.append(i)
        if mode == "min" and center == window.min():
            idxs.append(i)
    return idxs

def find_swings(df: pd.DataFrame, order:int=3):
    highs = _rolling_argrelextrema(df['high'], order, "max")
    lows  = _rolling_argrelextrema(df['low'],  order, "min")
    return highs, lows

def detect_supply_demand_zones(df: pd.DataFrame, lookback:int=200, min_touches:int=2, tolerance:float=0.002):
    """
    Heuristik: zona supply ~ dekat swing-high signifikan, demand ~ dekat swing-low signifikan.
    tolerance = 0.2% default dari harga untuk 'sentuhan' ulang
    """
    d = df.tail(lookback).reset_index(drop=True)
    highs, lows = find_swings(d, order=3)
    zones = {"supply": [], "demand": []}
    # supply
    for i in highs:
        level = float(d['high'].iloc[i])
        touches = (np.abs(d['high']-level)/level < tolerance).sum()
        if touches >= min_touches:
            zones["supply"].append({"level": round(level,8), "touches": int(touches)})
    # demand
    for i in lows:
        level = float(d['low'].iloc[i])
        touches = (np.abs(d['low']-level)/level < tolerance).sum()
        if touches >= min_touches:
            zones["demand"].append({"level": round(level,8), "touches": int(touches)})
    # sort by kedekatan ke harga sekarang
    px = float(d['close'].iloc[-1])
    zones["supply"] = sorted(zones["supply"], key=lambda z: abs(z["level"]-px))[:5]
    zones["demand"] = sorted(zones["demand"], key=lambda z: abs(z["level"]-px))[:5]
    return zones

def detect_order_blocks(df: pd.DataFrame, window:int=20):
    """
    OB sederhana: Bullish OB = bearish candle terakhir sebelum BOS naik; Bearish OB sebaliknya.
    Kita aproksimasi: cari candle dengan body besar yang langsung diikuti move continuation.
    """
    d = df.tail(max(window+5, 40)).reset_index(drop=True)
    res = {"bullish_ob": None, "bearish_ob": None}
    rng = range(2, len(d)-2)
    for i in rng:
        o,c,prev_c = float(d['open'].iloc[i]), float(d['close'].iloc[i]), float(d['close'].iloc[i-1])
        hi,lo = float(d['high'].iloc[i]), float(d['low'].iloc[i])
        body = abs(c-o)
        atrv = (d['high']-d['low']).rolling(14).mean().iloc[i] if i>=14 else (d['high']-d['low']).mean()
        if atrv == 0 or pd.isna(atrv): atrv = (d['high']-d['low']).mean()
        big_body = body > 0.6*atrv

        # Bearish OB kandidat -> diikuti 2 close turun
        if o < c and big_body and float(d['close'].iloc[i+1]) > c and float(d['close'].iloc[i+2]) > c:
            # sebenarnya ini bullish momentum bar -> OB bearish cari kebalikannya
            pass
        # Bullish OB: bearish candle besar, lalu 2 close naik (break away)
        if o > c and big_body and float(d['close'].iloc[i+1]) > c and float(d['close'].iloc[i+2]) > d['close'].iloc[i+1]:
            res["bullish_ob"] = {"open": round(o,8), "close": round(c,8)}
        # Bearish OB: bullish candle besar, lalu 2 close turun
        if o < c and big_body and float(d['close'].iloc[i+1]) < c and float(d['close'].iloc[i+2]) < d['close'].iloc[i+1]:
            res["bearish_ob"] = {"open": round(o,8), "close": round(c,8)}
    return res

def detect_fvg(df: pd.DataFrame, lookback:int=120):
    """
    Fair Value Gap (FVG) klasik: celah antara high candle n-1 dengan low candle n+1 (bullish gap) atau sebaliknya.
    """
    d = df.tail(lookback).reset_index(drop=True)
    gaps = {"bullish_fvg": [], "bearish_fvg": []}
    for i in range(1, len(d)-1):
        hi_prev = float(d['high'].iloc[i-1]); lo_prev = float(d['low'].iloc[i-1])
        hi_next = float(d['high'].iloc[i+1]); lo_next = float(d['low'].iloc[i+1])
        # bullish FVG jika low(i+1) > high(i-1)
        if lo_next > hi_prev:
            gaps["bullish_fvg"].append({"from": round(hi_prev,8), "to": round(lo_next,8)})
        # bearish FVG jika high(i+1) < low(i-1)
        if hi_next < lo_prev:
            gaps["bearish_fvg"].append({"from": round(hi_next,8), "to": round(lo_prev,8)})
    # ambil yang paling dekat harga
    px = float(d['close'].iloc[-1])
    for k in gaps:
        gaps[k] = sorted(gaps[k], key=lambda g: min(abs(g["from"]-px), abs(g["to"]-px)))[:3]
    return gaps

def detect_liquidity_grab(df: pd.DataFrame, lookback:int=100, wick_ratio:float=0.6):
    """
    Liquidity grab: candle yang tusuk level ekstrem lalu close balik (long wick).
    """
    d = df.tail(lookback).reset_index(drop=True)
    events = []
    swingH = d['high'].rolling(20).max()
    swingL = d['low'].rolling(20).min()
    for i in range(20, len(d)):
        o,c,h,l = map(float, d[['open','close','high','low']].iloc[i])
        body = abs(c-o)
        upper_wick = h - max(o,c)
        lower_wick = min(o,c) - l
        # ambil grab atas / bawah
        grab_up = h > float(swingH.iloc[i-1]) and upper_wick > wick_ratio*(upper_wick+lower_wick+body)
        grab_dn = l < float(swingL.iloc[i-1]) and lower_wick > wick_ratio*(upper_wick+lower_wick+body)
        if grab_up:
            events.append({"type":"liq_grab_up", "level": round(h,8), "index": int(i)})
        if grab_dn:
            events.append({"type":"liq_grab_down", "level": round(l,8), "index": int(i)})
    return events[-3:]  # terakhir-terakhir saja

def detect_head_and_shoulders(df: pd.DataFrame, lookback:int=150, tol:float=0.015):
    """
    Deteksi H&S sederhana berbasis swing high (toleransi tinggi kiri/kanan mirip).
    """
    d = df.tail(lookback).reset_index(drop=True)
    highs, lows = find_swings(d, order=3)
    if len(highs) < 3:
        return None
    # ambil tiga high terakhir
    hs = highs[-3:]
    hL, hH, hR = [float(d['high'].iloc[i]) for i in hs]
    cond_lr_mirror = abs(hL - hR)/max(hL,hR) < tol
    cond_head_higher = hH > hL and hH > hR
    if cond_lr_mirror and cond_head_higher:
        return {"pattern":"H&S_Top", "indices": hs, "left": round(hL,8), "head": round(hH,8), "right": round(hR,8)}
    # inverse H&S pakai swing low
    if len(lows) >= 3:
        ls = lows[-3:]
        lL, lH, lR = [float(d['low'].iloc[i]) for i in ls]
        cond_lr_mirror2 = abs(lL - lR)/max(lL,lR) < tol
        cond_head_lower = lH < lL and lH < lR
        if cond_lr_mirror2 and cond_head_lower:
            return {"pattern":"Inv_H&S", "indices": ls, "left": round(lL,8), "head": round(lH,8), "right": round(lR,8)}
    return None

def detect_apex_compression(df: pd.DataFrame, window:int=50, slope_tol:float=0.0005):
    """
    Apex = range menyempit (ATR menurun) + dua trendline konvergen (pakai high/low linear fit).
    """
    d = df.tail(max(window, 50)).reset_index(drop=True)
    if d.shape[0] < 20:
        return None
    x = np.arange(len(d))
    hi = d['high'].astype(float).values
    lo = d['low'].astype(float).values
    # regresi linear
    m_hi, b_hi = np.polyfit(x, hi, 1)
    m_lo, b_lo = np.polyfit(x, lo, 1)
    # konvergen jika m_hi < 0 dan m_lo > 0 atau jarak hi-lo menyempit
    rng = (hi - lo)
    narrowing = (pd.Series(rng).rolling(10).mean().iloc[-1] < pd.Series(rng).rolling(10).mean().iloc[0])
    if narrowing and (m_hi < -slope_tol and m_lo > slope_tol):
        apx_top = float(m_hi*x[-1] + b_hi)
        apx_bot = float(m_lo*x[-1] + b_lo)
        return {"apex_top": round(apx_top,8), "apex_bot": round(apx_bot,8)}
    return None

# ---------------- STRATEGI HYBRID ----------------
def hybrid_analyze(df: pd.DataFrame, pair:Optional[str]=None, timeframe:Optional[str]=None) -> dict:
    df = df.copy().dropna().reset_index(drop=True)
    if df.shape[0] < 12:
        return {"error":"data_tidak_cukup", "message":"Perlu minimal 12 candle untuk analisis."}

    df['ema20'] = ema(df['close'],20)
    df['ema50'] = ema(df['close'],50)
    df['rsi14'] = rsi(df['close'],14)
    df['atr14'] = atr(df,14)

    last = df.iloc[-1]
    price = float(last['close'])
    ema20 = float(last['ema20'])
    ema50 = float(last['ema50'])
    rsi_now = float(last['rsi14'])
    atr_now = float(last['atr14']) if not np.isnan(last['atr14']) else price*0.001

    recent_high, recent_low = detect_sr(df, lookback=120)
    bos = breakout_of_structure(df, window=20)
    swing_high = df['high'].tail(80).max()
    swing_low  = df['low'].tail(80).min()
    diff = swing_high - swing_low
    fib_618 = swing_high - diff*0.618 if diff>0 else price

    reasons, conf = [], []
    trend = "bullish" if ema20 > ema50 else "bearish"

    if bos == "BOS_UP" or (trend == "bullish" and price > ema20):
        entry = price
        sl = recent_low - atr_now*0.6
        rr = entry - sl if entry>sl else price*0.01
        tp1 = entry + rr*1.5
        tp2 = entry + rr*2.5
        reasons.append("Bias LONG — BOS naik & EMA searah.")
        conf.append(0.9 if trend=="bullish" else 0.6)
        conf.append(0.9 if price >= fib_618 else 0.65)
        conf.append(1.0 if 30 < rsi_now < 75 else 0.5)
        signal="LONG"
    elif bos == "BOS_DOWN" or (trend == "bearish" and price < ema20):
        entry = price
        sl = recent_high + atr_now*0.6
        rr = sl - entry if sl>entry else price*0.01
        tp1 = entry - rr*1.5
        tp2 = entry - rr*2.5
        reasons.append("Bias SHORT — BOS turun & EMA searah bearish.")
        conf.append(0.9 if trend=="bearish" else 0.6)
        conf.append(0.9 if price <= fib_618 else 0.65)
        conf.append(1.0 if 25 < rsi_now < 70 else 0.5)
        signal="SHORT"
    else:
        entry = price
        sl = recent_low * 0.995
        tp1 = entry + (entry-sl)*1.2
        tp2 = entry + (entry-sl)*2.0
        reasons.append("Belum ada arah jelas — tunggu konfirmasi TF lebih tinggi.")
        conf.append(0.25)
        signal="WAIT"

    confidence = float(sum(conf)/len(conf))
    reasoning = " · ".join(reasons)
# --- SMC enrichment ---
    try:
        zones = detect_supply_demand_zones(df, lookback=180)
        ob    = detect_order_blocks(df, window=30)
        fvg   = detect_fvg(df, lookback=150)
        liq   = detect_liquidity_grab(df, lookback=120)
        hs    = detect_head_and_shoulders(df, lookback=160)
        apex  = detect_apex_compression(df, window=60)

        # Perkuat confidence bila entry dekat demand (untuk LONG) atau dekat supply (untuk SHORT)
        near_boost = 0.0
        px = price
        if signal == "LONG" and zones.get("demand"):
            nearest_demand = min(zones["demand"], key=lambda z: abs(z["level"]-px))
            if abs(nearest_demand["level"]-px)/px < 0.004:  # ±0.4%
                near_boost += 0.05
                reasons.append(f"Dekat Demand POI (~{nearest_demand['level']}).")
        if signal == "SHORT" and zones.get("supply"):
            nearest_supply = min(zones["supply"], key=lambda z: abs(z["level"]-px))
            if abs(nearest_supply["level"]-px)/px < 0.004:
                near_boost += 0.05
                reasons.append(f"Dekat Supply POI (~{nearest_supply['level']}).")

        # FVG magnet: jika ada FVG terdekat searah, tambah keyakinan kecil
        if signal == "LONG" and fvg["bullish_fvg"]:
            reasons.append("Bullish FVG terdekat berpotensi menjadi magnet harga.")
            near_boost += 0.03
        if signal == "SHORT" and fvg["bearish_fvg"]:
            reasons.append("Bearish FVG terdekat berpotensi menjadi magnet harga.")
            near_boost += 0.03

        # Liquidity grab baru-baru ini -> tambah konfluensi pembalikan
        if liq:
            last_liq = liq[-1]["type"]
            reasons.append(f"Liquidity event: {last_liq}.")
            near_boost += 0.02

        # Head & Shoulder jadi peringatan kebalikan trend
        if hs:
            reasons.append(f"Pattern terdeteksi: {hs['pattern']}.")
            # jika H&S top tapi sinyal LONG -> kurangi sedikit
            if hs["pattern"] == "H&S_Top" and signal == "LONG":
                near_boost -= 0.05
            if hs["pattern"] == "Inv_H&S" and signal == "SHORT":
                near_boost -= 0.05

        # Apex compression -> breakout bias: jika RSI netral, tambah sedikit
        if apex and 35 < rsi_now < 70:
            reasons.append("Apex/volatility compression — potensi break.")
            near_boost += 0.02

        confidence = float(sum(conf)/len(conf)) + near_boost
        confidence = max(0.0, min(1.0, confidence))
    except Exception as _e:
        # kalau enrichment gagal, pakai confidence awal
        confidence = float(sum(conf)/len(conf))
    return {
        "pair": pair or "",
        "timeframe": timeframe or "",
        "signal_type": signal,
        "entry": round(entry,8),
        "tp1": round(tp1,8),
        "tp2": round(tp2,8),
        "sl": round(sl,8),
        "confidence": round(confidence,3),
        "reasoning": reasoning
        "poi": zones if 'zones' in locals() else None,
        "order_blocks": ob if 'ob' in locals() else None,
        "fvg": fvg if 'fvg' in locals() else None,
        "liquidity_events": liq if 'liq' in locals() else None,
        "patterns": hs if 'hs' in locals() else None,
        "apex": apex if 'apex' in locals() else None
    }

# ---------------- SCALPING ENGINE ----------------
def scalp_engine(df: pd.DataFrame, pair:Optional[str]=None, tf:Optional[str]=None) -> dict:
    if df.shape[0] < 30:
        return {"error": "data_tidak_cukup"}

    df['ema8'] = ema(df['close'], 8)
    df['ema21'] = ema(df['close'], 21)
    df['rsi14'] = rsi(df['close'], 14)
    df['atr14'] = atr(df, 14)

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

# ---------------- IMAGE OCR HELPERS ----------------
def ocr_y_axis_prices(img_cv):
    """Extract numeric labels from right-side y-axis using tesseract (best-effort)."""
    if not _HAS_TESSERACT:
        return {}
    h, w = img_cv.shape[:2]
    right_x = int(w*0.76)
    crops = [
        img_cv[int(h*0.02):int(h*0.98), right_x:w].copy(),
        img_cv[int(h*0.06):int(h*0.94), int(w*0.7):w].copy()
    ]
    vals = None
    for crop in crops:
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th = cv2.medianBlur(th, 3)
            config = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.,"
            txt = pytesseract.image_to_string(th, config=config)
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            parsed = []
            for ln in lines:
                s = ln.replace(",", ".").replace(" ", "")
                m = re.findall(r"[\d\.]+", s)
                if m:
                    try:
                        parsed.append(float(m[0]))
                    except:
                        pass
            if parsed:
                vals = parsed
                break
        except Exception:
            continue
    if not vals:
        return {}
    ys = np.linspace(int(h*0.02), int(h*0.98), num=len(vals)).astype(int).tolist()
    prices = vals
    return {int(y): float(p) for y,p in zip(ys, prices)}

def detect_candles_from_plot(img_cv, y_map, max_bars=200):
    """Heuristic detection: HSV detection for colored candles. Returns DataFrame open/high/low/close."""
    h,w,_ = img_cv.shape
    top = int(h*0.06); bottom = int(h*0.94)
    left = int(w*0.06); right = int(w*0.94)
    if bottom <= top or right <= left:
        return pd.DataFrame(columns=["open","high","low","close"])
    plot = img_cv[top:bottom, left:right].copy()
    try:
        hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
    except Exception:
        return pd.DataFrame(columns=["open","high","low","close"])

    lower_red1 = np.array([0, 40, 40]); upper_red1 = np.array([12,255,255])
    lower_red2 = np.array([160,40,40]); upper_red2 = np.array([180,255,255])
    lower_green = np.array([35,40,40]); upper_green = np.array([95,255,255])

    mask_r = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                            cv2.inRange(hsv, lower_red2, upper_red2))
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(mask_r, mask_g)
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candles = []
    ys = sorted(y_map.keys()) if y_map else None
    ps = [y_map[y] for y in ys] if ys else None
    for c in contours:
        x,y,cw,ch = cv2.boundingRect(c)
        if cw < 4 or ch < 6:
            continue
        global_y_top = y + top
        global_y_bot = y + ch + top
        if ys and len(ys) > 1:
            high = float(np.interp(global_y_top, ys, ps))
            low = float(np.interp(global_y_bot, ys, ps))
        else:
            rel_top = (y) / plot.shape[0]
            rel_bottom = (y+ch) / plot.shape[0]
            high = 100*(1-rel_top)
            low = 100*(1-rel_bottom)
        roi_g = mask_g[y:y+ch, x:x+cw]
        roi_r = mask_r[y:y+ch, x:x+cw]
        bullish = int(np.mean(roi_g) > np.mean(roi_r))
        if bullish:
            open_p = low + (high-low)*0.25
            close_p = high - (high-low)*0.12
        else:
            open_p = high - (high-low)*0.25
            close_p = low + (high-low)*0.12
        candles.append({"x": x, "open": open_p, "high": high, "low": low, "close": close_p})

    if not candles:
        return pd.DataFrame(columns=["open","high","low","close"])
    df = pd.DataFrame(sorted(candles, key=lambda r: r["x"]))[["open","high","low","close"]]
    if not ys or len(ys) <= 1:
        lastc = df["close"].iloc[-1] if not df["close"].isna().all() else 1.0
        df = df / lastc
    return df.tail(max_bars).reset_index(drop=True)

# ---------------- FEATURE ENGINEERING ----------------
def compute_features_for_row(pair: str, timeframe: str, entry: float, tp: Optional[float], sl: float) -> Optional[Dict[str, float]]:
    try:
        kdf = fetch_ohlc_binance(pair, timeframe, limit=200)
    except Exception:
        return None
    kdf = kdf.tail(60).reset_index(drop=True)
    for col in ['open','high','low','close','volume']:
        kdf[col] = pd.to_numeric(kdf[col], errors='coerce').fillna(0.0)
    close = kdf['close'].astype(float)
    high = kdf['high'].astype(float)
    low = kdf['low'].astype(float)
    vol = kdf['volume'].astype(float) if 'volume' in kdf.columns else pd.Series([0.0]*len(kdf))

    ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    rsi14 = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    atr14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    vol_mean = vol.tail(40).mean() if len(vol) >= 40 else vol.mean()
    vol_now = vol.iloc[-1] if len(vol) > 0 else 0.0
    vol_spike = 1.0 if vol_now > vol_mean * 1.8 else 0.0

    recent_high = high.tail(80).max()
    recent_low = low.tail(80).min()
    dist_to_high = (recent_high - entry) / entry if entry else 0.0
    dist_to_low = (entry - recent_low) / entry if entry else 0.0
    rr = abs((tp - entry) / (entry - sl)) if (tp is not None and (entry - sl) != 0) else 0.0

    return {
        "ema8_21_diff": (ema8 - ema21) / (entry if entry != 0 else 1),
        "rsi14": float(rsi14),
        "atr_rel": float(atr14) / (entry if entry != 0 else 1),
        "vol_spike": float(vol_spike),
        "dist_to_high": float(dist_to_high),
        "dist_to_low": float(dist_to_low),
        "rr": float(rr)
    }

# ---------------- LEARNING SYSTEM ----------------
def build_dataset_from_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        return None, None
    df = pd.read_csv(TRADE_LOG_FILE)
    rows, labels = [], []
    for _, r in df.iterrows():
        hit = str(r.get("backtest_hit", "")).upper()
        if hit == "" or hit == "NO_HIT":
            label = 0
        elif hit.startswith("TP"):
            label = 1
        elif hit.startswith("SL"):
            label = 0
        else:
            label = 0
        try:
            feats = compute_features_for_row(
                str(r['pair']),
                str(r.get('timeframe', "15m")),
                float(r['entry']),
                r.get('tp1', None),
                float(r['sl'])
            )
        except Exception:
            feats = None
        if feats is None:
            continue
        rows.append(feats)
        labels.append(label)
    if not rows:
        return None, None
    X = pd.DataFrame(rows)
    y = pd.Series(labels)
    return X, y

def train_and_save_model():
    """Melatih model RandomForest baru dan update cache model."""
    global _last_retrain_count, _cached_model
    X, y = build_dataset_from_trade_log()
    if X is None or len(y) < MIN_SAMPLES_TO_TRAIN:
        return {"status": "data_tidak_cukup", "samples": 0 if y is None else len(y)}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    yprob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, yprob) if yprob is not None else None

    joblib.dump({"clf": clf, "features": list(X.columns)}, MODEL_FILE)

    _cached_model = {"clf": clf, "features": list(X.columns)}

    try:
        df = pd.read_csv(TRADE_LOG_FILE)
        _last_retrain_count = len(df)
    except:
        _last_retrain_count = 0

    return {"status": "trained", "samples": len(y), "auc": auc, "report": report}

def predict_with_model(payload: Dict[str, Any]):
    """Prediksi probabilitas sukses sinyal menggunakan model pembelajaran (pakai cache jika ada)."""
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(MODEL_FILE):
            raise RuntimeError("model_belum_dilatih")
        _cached_model = joblib.load(MODEL_FILE)

    mod = _cached_model
    clf = mod["clf"]

    pair = payload.get("pair")
    timeframe = payload.get("timeframe") or "15m"
    entry = float(payload.get("entry"))
    tp = payload.get("tp") or payload.get("tp1")
    sl = float(payload.get("sl"))

    feats = compute_features_for_row(pair, timeframe, entry, tp, sl)
    if feats is None:
        raise RuntimeError("gagal_menghitung_fitur")

    X = pd.DataFrame([feats])
    prob = float(clf.predict_proba(X)[:, 1][0]) if hasattr(clf, "predict_proba") else float(clf.predict(X)[0])
    return {"prob": prob, "features": feats}

def maybe_trigger_retrain_background():
    """Jalankan retrain model di background (daemon thread)."""
    def worker():
        try:
            res = train_and_save_model()
            print("Retrain result:", res)
        except Exception as e:
            print("Retrain error:", e)
    t = threading.Thread(target=worker, daemon=True)
    t.start()

def check_and_trigger_retrain_if_needed():
    """Cek apakah sudah waktunya retrain model berdasarkan jumlah sinyal baru."""
    global _last_retrain_count
    with _lock:
        try:
            df = pd.read_csv(TRADE_LOG_FILE)
            total = len(df)
            if _last_retrain_count == 0:
                _last_retrain_count = total
            if total - _last_retrain_count >= N_SIGNALS_TO_RETRAIN:
                _last_retrain_count = total
                maybe_trigger_retrain_background()
        except Exception as e:
            print("check_retrain error", e)

# ---------------- SENTIMENT HELPERS ----------------
def get_crypto_sentiment():
    """Ambil Fear & Greed + BTC dominance (best-effort)."""
    out = {"fear_greed": None, "fng_value": None, "btc_dominance": None, "source": []}
    try:
        r = requests.get(FNG_URL, params={"limit": 1}, timeout=8)
        j = r.json()
        if "data" in j and len(j["data"])>0:
            d = j["data"][0]
            out["fear_greed"] = d.get("value_classification")
            out["fng_value"] = int(d.get("value")) if d.get("value") else None
            out["source"].append("alternative.me/fng")
    except Exception:
        pass
    try:
        r = requests.get(COINGECKO_GLOBAL, timeout=8)
        j = r.json()
        mp = j.get("data", {}).get("market_cap_percentage", {})
        btc_dom = mp.get("btc") or mp.get("btc_dominance")
        if btc_dom is not None:
            out["btc_dominance"] = round(float(btc_dom), 2)
            out["source"].append("coingecko/global")
    except Exception:
        pass
    return out

def get_macro_sentiment():
    """Ambil beberapa indikator makro sederhana (best-effort)."""
    out = {"dxy": None, "vix": None, "snp_change": None, "source": []}
    if ALPHA_API_KEY:
        try:
            out["source"].append("alphavantage/FX (placeholder)")
        except Exception:
            pass
    return out

def fuse_confidence(tech_conf: float, market: str, crypto_sent: dict=None, macro_sent: dict=None) -> float:
    tech = float(tech_conf or 0.5)
    if market == "crypto":
        sent_score = 0.5
        if crypto_sent:
            fng = crypto_sent.get("fng_value")
            btc_dom = crypto_sent.get("btc_dominance")
            if fng is not None:
                sent_score = min(1.0, max(0.0, (fng/100.0)))
            if btc_dom is not None:
                sent_score = (sent_score + (btc_dom/100.0))/2.0
        final = 0.65 * tech + 0.35 * sent_score
    else:
        macro_score = 0.5
        if macro_sent:
            macro_score = 0.5
        final = 0.7 * tech + 0.3 * macro_score
    return round(max(0.0, min(1.0, final)), 3)

# ---------------- POSTPROCESS + ENDPOINTS ----------------
def _postprocess_with_learning(signal: Dict[str, Any]) -> Dict[str, Any]:
    try:
        market = detect_market(signal.get("pair", ""))
        crypto_sent = get_crypto_sentiment() if market == "crypto" else None
        macro_sent = get_macro_sentiment() if market == "forex" else None

        model_prob = None
        if os.path.exists(MODEL_FILE):
            try:
                pred = predict_with_model({
                    "pair": signal.get("pair"),
                    "timeframe": signal.get("timeframe"),
                    "entry": signal.get("entry"),
                    "tp": signal.get("tp1"),
                    "sl": signal.get("sl")
                })
                model_prob = pred.get("prob", None)
                signal["model_prob"] = round(model_prob, 3) if model_prob is not None else None
            except Exception as e:
                signal["model_error"] = str(e)

        orig = float(signal.get("confidence", 0.5))
        fused = fuse_confidence(orig, market, crypto_sent, macro_sent)
        if model_prob is not None:
            fused = round(max(0.0, min(1.0, 0.85 * fused + 0.15 * model_prob)), 3)
        signal["confidence"] = fused
        signal["market_mode"] = market
        signal["sentiment"] = {"crypto": crypto_sent, "macro": macro_sent}
        if model_prob is not None and model_prob < 0.25:
            signal["vetoed_by_model"] = True
            signal["signal_type"] = "WAIT"
        else:
            signal["vetoed_by_model"] = False

    except Exception as e:
        signal["postprocess_error"] = str(e)
    return signal

@app.get("/health")
def health():
    return respond({"status": "ok", "service": "Pro Trader AI - Learning (ID)"})

@app.get("/pro_signal")
def pro_signal(pair: str = Query(...), tf_main: str = Query("1h"), tf_entry: str = Query("15m"), limit: int = Query(300), auto_log: bool = Query(False)):
    try:
        df_entry = fetch_ohlc_binance(pair, tf_entry, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch_failed: {e}")

    res = hybrid_analyze(df_entry, pair=pair, timeframe=tf_entry)
    try:
        df_main = fetch_ohlc_binance(pair, tf_main, limit=200)
        ema20_main = float(ema(df_main['close'], 20).iloc[-1])
        ema50_main = float(ema(df_main['close'], 50).iloc[-1])
        res['context_main_trend'] = "bullish" if ema20_main > ema50_main else "bearish"
    except:
        pass

    res = _postprocess_with_learning(res)

    if auto_log:
        payload_bt = {
            "pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"],
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"),
            "sl": res["sl"], "confidence": res["confidence"], "reason": res["reasoning"]
        }
        bt_res = post_to_backtester(payload_bt)
        res["backtest_raw"] = bt_res

        logrec = {
            "pair": res["pair"], "timeframe": res["timeframe"], "signal_type": res["signal_type"],
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"),
            "sl": res["sl"], "confidence": res["confidence"], "reasoning": res["reasoning"],
            "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None,
            "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None
        }
        append_trade_log(logrec)
        check_and_trigger_retrain_if_needed()

    return respond(res)

@app.get("/scalp_signal")
def scalp_signal(pair: str = Query(...), tf: str = Query("3m"), limit: int = Query(300), auto_log: bool = Query(False)):
    try:
        df = fetch_ohlc_binance(pair, tf, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch_failed: {e}")

    res = scalp_engine(df, pair=pair, tf=tf)
    res = _postprocess_with_learning(res)

    if auto_log:
        payload_bt = {
            "pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"],
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"),
            "sl": res["sl"], "confidence": res["confidence"], "reason": res["reasoning"]
        }
        bt_res = post_to_backtester(payload_bt)
        res["backtest_raw"] = bt_res

        logrec = {
            "pair": res["pair"], "timeframe": res["timeframe"], "signal_type": res["signal_type"],
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"),
            "sl": res["sl"], "confidence": res["confidence"], "reasoning": res["reasoning"],
            "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None,
            "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None
        }
        append_trade_log(logrec)
        check_and_trigger_retrain_if_needed()

    return respond(res)

@app.post("/analyze_chart")
def analyze_chart(file: UploadFile = File(...), pair: Optional[str] = Form(None), timeframe: Optional[str] = Form(None), auto_backtest: Optional[str] = Form("true")):
    auto_flag = auto_backtest.lower() != "false"
    try:
        contents = file.file.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"gambar_tidak_valid: {e}")

    if not _HAS_TESSERACT:
        raise HTTPException(status_code=400, detail="tesseract_ocr_tidak_tersedia")

    y_map = {}
    try:
        y_map = ocr_y_axis_prices(img_cv)
    except:
        y_map = {}

    df_ohlc = detect_candles_from_plot(img_cv, y_map, max_bars=200)
    if df_ohlc.empty:
        raise HTTPException(status_code=400, detail="gagal_membaca_chart")

    for col in ['open', 'high', 'low', 'close']:
        df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors='coerce')
    df_ohlc = df_ohlc.dropna().reset_index(drop=True)

    res = hybrid_analyze(df_ohlc, pair=pair or "IMG", timeframe=timeframe or "img")
    res = _postprocess_with_learning(res)

    if auto_flag:
        bt_res = post_to_backtester({
            "pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"],
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"),
            "sl": res["sl"], "confidence": res["confidence"], "reason": res["reasoning"]
        })
        res["backtest_raw"] = bt_res
        append_trade_log({
            "pair": res["pair"], "timeframe": res["timeframe"], "signal_type": res["signal_type"],
            "entry": res["entry"], "tp1": res["tp1"], "tp2": res["tp2"], "sl": res["sl"],
            "confidence": res["confidence"], "reasoning": res["reasoning"],
            "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None,
            "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None
        })
        check_and_trigger_retrain_if_needed()

    res['bars_used'] = int(df_ohlc.shape[0])
    return respond(res)

# ==========================================================
# ENHANCED ENDPOINT: /analyze_csv
# - Auto log ke trade_log.csv
# - Auto trigger retrain learning
# - Auto backtest integration
# ==========================================================
@app.post("/analyze_csv")
def analyze_csv(
    file: UploadFile = File(...),
    pair: Optional[str] = Form(None),
    timeframe: Optional[str] = Form(None),
    auto_backtest: Optional[str] = Form("true"),
    auto_log: Optional[str] = Form("true")
):
    """
    Analisis file CSV candlestick (open, high, low, close)
    dan otomatis simpan hasil ke trade_log.csv + retrain model jika perlu.
    """
    auto_bt = auto_backtest.lower() != "false"
    auto_lg = auto_log.lower() != "false"

    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_csv: {e}")

    # Pastikan kolom valid
    df.columns = [c.strip().lower() for c in df.columns]
    def find_col(k): return next((c for c in df.columns if k in c), None)
    o, h, l, ccol = find_col('open'), find_col('high'), find_col('low'), find_col('close')
    if not all([o, h, l, ccol]):
        raise HTTPException(status_code=400, detail="kolom_tidak_lengkap (butuh open, high, low, close)")

    # Ubah kolom ke format standar
    df2 = df[[o, h, l, ccol]].rename(columns={o: 'open', h: 'high', l: 'low', ccol: 'close'})
    for col in ['open', 'high', 'low', 'close']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2 = df2.dropna().reset_index(drop=True)

    # Analisis teknikal
    res = hybrid_analyze(df2, pair=pair or "CSV", timeframe=timeframe or "csv")
    res = _postprocess_with_learning(res)

    # Backtest otomatis (kalau BACKTEST_URL aktif)
    bt_res = {}
    if auto_bt and res.get("signal_type") != "WAIT":
        bt_payload = {
            "pair": res["pair"],
            "timeframe": res["timeframe"],
            "side": res["signal_type"],
            "entry": res["entry"],
            "tp1": res.get("tp1"),
            "tp2": res.get("tp2"),
            "sl": res["sl"],
            "confidence": res["confidence"],
            "reason": res["reasoning"]
        }
        bt_res = post_to_backtester(bt_payload)
        res["backtest_raw"] = bt_res

    # Simpan otomatis ke trade_log.csv
    if auto_lg:
        backtest_hit = bt_res.get("hit") or bt_res.get("result") or bt_res.get("outcome")
        backtest_pnl = bt_res.get("pnl_total") or bt_res.get("pnl") or bt_res.get("profit")
        append_trade_log({
            "pair": res["pair"],
            "timeframe": res["timeframe"],
            "signal_type": res["signal_type"],
            "entry": res["entry"],
            "tp1": res.get("tp1"),
            "tp2": res.get("tp2"),
            "sl": res["sl"],
            "confidence": res["confidence"],
            "reasoning": res["reasoning"],
            "backtest_hit": backtest_hit,
            "backtest_pnl": backtest_pnl
        })

        # Retrain otomatis kalau cukup sinyal
        check_and_trigger_retrain_if_needed()

    # Tambahkan info tambahan
    res["bars_used"] = int(df2.shape[0])
    res["auto_logged"] = auto_lg
    res["auto_retrain_triggered"] = auto_lg
    return respond(res)

@app.get("/sentiment")
def sentiment():
    c = get_crypto_sentiment()
    m = get_macro_sentiment()
    return respond({"crypto_sentiment": c, "macro_sentiment": m})

@app.get("/mode")
def mode(pair: str = Query(...)):
    p = pair.upper()
    m = detect_market(p)
    sources = {"crypto": ["binance", "coingecko", "alternative.me"], "forex": ["twelvedata (fallback: alphavantage)"]}
    return respond({"pair": p, "mode": m, "data_sources": sources.get(m)})

@app.get("/context")
def context(pair: str = Query(...), tf: str = Query("15m")):
    p = pair.upper()
    m = detect_market(p)
    c = get_crypto_sentiment() if m == "crypto" else None
    macro = get_macro_sentiment() if m == "forex" else None
    last_price = None
    try:
        df = fetch_ohlc_binance(p, tf, limit=5)
        last_price = float(df['close'].astype(float).iloc[-1])
    except Exception:
        last_price = None
    return respond({"pair": p, "mode": m, "last_price": last_price, "crypto_sentiment": c, "macro_sentiment": macro})

@app.get("/learning_status")
def learning_status():
    info = {"model_exists": os.path.exists(MODEL_FILE)}
    if info["model_exists"]:
        try:
            mod = joblib.load(MODEL_FILE)
            info["features"] = mod.get("features")
        except:
            info["features"] = None
    try:
        df = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else pd.DataFrame()
        info["trade_log_count"] = len(df)
    except:
        info["trade_log_count"] = 0
    return respond(info)

@app.get("/model_debug")
def model_debug():
    info = {"model_exists": os.path.exists(MODEL_FILE), "last_trained": None, "features": None, "feature_importance": None}
    if os.path.exists(MODEL_FILE):
        try:
            mod = joblib.load(MODEL_FILE)
            clf = mod.get("clf")
            features = mod.get("features")
            info["features"] = features
            info["last_trained"] = datetime.fromtimestamp(os.path.getmtime(MODEL_FILE)).isoformat()
            if hasattr(clf, "feature_importances_"):
                imps = clf.feature_importances_.tolist()
                info["feature_importance"] = dict(zip(features, [round(float(x),6) for x in imps]))
        except Exception as e:
            info["error_loading_model"] = str(e)
    return respond(info)

@app.get("/retrain_learning")
def retrain_learning():
    res = train_and_save_model()
    return respond(res)

@app.get("/logs")
def get_logs(limit: int = Query(100)):
    ensure_trade_log()
    df = pd.read_csv(TRADE_LOG_FILE)
    df = df.tail(limit).to_dict(orient="records")
    return respond({"logs": df})

@app.get("/logs_summary")
def logs_summary():
    try:
        if not os.path.exists(TRADE_LOG_FILE):
            return respond({"detail": "Belum ada log sinyal tersimpan."})
        df = pd.read_csv(TRADE_LOG_FILE)
        if df.empty:
            return respond({"detail": "Belum ada data sinyal terbaru."})
        last = df.iloc[-1]
        data = {
            "pair": last.get("pair", ""),
            "timeframe": last.get("timeframe", ""),
            "signal_type": last.get("signal_type", ""),
            "entry": last.get("entry", ""),
            "tp1": last.get("tp1", ""),
            "tp2": last.get("tp2", ""),
            "sl": last.get("sl", ""),
            "confidence": last.get("confidence", ""),
            "reasoning": last.get("reasoning", "")
        }
        return respond(data)
    except Exception as e:
        return respond({"error": str(e)})

@app.get("/download_logs")
def download_logs():
    ensure_trade_log()
    return FileResponse(TRADE_LOG_FILE, media_type="text/csv", filename="trade_log.csv")

@app.get("/ai_performance")
def ai_performance():
    try:
        ensure_trade_log()
        pd.options.mode.use_inf_as_na = True
        df = pd.read_csv(TRADE_LOG_FILE)
        if df.empty:
            return respond({"error": "Belum ada data sinyal untuk dianalisis."})

        total = len(df)
        tp_hits = df["backtest_hit"].astype(str).str.upper().str.startswith("TP").sum()
        sl_hits = df["backtest_hit"].astype(str).str.upper().str.startswith("SL").sum()
        winrate = round((tp_hits / total) * 100, 2) if total > 0 else 0

        avg_conf = pd.to_numeric(df.get("confidence", pd.Series([], dtype=float)), errors="coerce").mean()
        if pd.isna(avg_conf) or np.isinf(avg_conf):
            avg_conf = 0.0
        avg_conf = round(float(avg_conf), 3)

        pnl_values = pd.to_numeric(df.get("backtest_pnl", pd.Series([], dtype=float)), errors="coerce").dropna()
        total_pnl = float(pnl_values.sum()) if not pnl_values.empty else 0.0

        profit_factor = None
        if not pnl_values.empty and (pnl_values < 0).any():
            prof = pnl_values[pnl_values > 0].sum()
            loss = abs(pnl_values[pnl_values < 0].sum())
            profit_factor = round(prof / loss, 2) if loss != 0 else None

        max_drawdown = float(pnl_values.min()) if not pnl_values.empty else 0.0

        pair_stats = []
        for pair, group in df.groupby("pair"):
            tp_pair = group["backtest_hit"].astype(str).str.upper().str.startswith("TP").sum()
            wr_pair = round((tp_pair / len(group)) * 100, 2)
            pair_stats.append({"pair": pair, "signals": len(group), "winrate": wr_pair})
        pair_stats = sorted(pair_stats, key=lambda x: x["signals"], reverse=True)

        tf_stats = []
        for tf, group in df.groupby("timeframe"):
            tp_tf = group["backtest_hit"].astype(str).str.upper().str.startswith("TP").sum()
            wr_tf = round((tp_tf / len(group)) * 100, 2)
            tf_stats.append({"timeframe": tf, "signals": len(group), "winrate": wr_tf})
        tf_stats = sorted(tf_stats, key=lambda x: x["signals"], reverse=True)

        model_exists = os.path.exists(MODEL_FILE)
        data = {
            "total_signals": total,
            "tp_hits": int(tp_hits),
            "sl_hits": int(sl_hits),
            "winrate": winrate,
            "avg_confidence": avg_conf,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "pair_stats": pair_stats,
            "tf_stats": tf_stats,
            "model_status": "✅ Sudah Dilatih" if model_exists else "❌ Belum Ada Model"
        }
        return respond(data)
    except Exception as e:
        return respond({"error": str(e)})

# ---------------- BACKTEST COMM ----------------
def post_to_backtester(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not BACKTEST_URL:
        return {"error": "BACKTEST_URL_not_configured"}
    try:
        r = requests.post(BACKTEST_URL, json=payload, timeout=15)
        try:
            return r.json()
        except:
            return {"status_code": r.status_code, "text": r.text}
    except Exception as e:
        return {"error": "backtester_unreachable", "detail": str(e)}

# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup_event():
    ensure_trade_log()
    global _cached_model
    if os.path.exists(MODEL_FILE):
        try:
            _cached_model = joblib.load(MODEL_FILE)
            print("Loaded cached model on startup.")
        except Exception as e:
            print("Failed load cached model on startup:", e)

# Run server:
# uvicorn main_combined_learning:app --host 0.0.0.0 --port $PORT
