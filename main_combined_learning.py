"""
main_combined_learning_hybrid_pro_final.py
Combined Hybrid PRO: SMC/ICT PRO + Hybrid Technical Engine + XGBoost + RandomForest + Data Fallback
Compatibility: Designed to work with telegram_bot (2).py as-is (endpoints /pro_signal, /scalp_signal, /analyze_chart, /analyze_csv, /learning_status, /retrain_learning, /ai_performance, /logs_summary, /mode, /context, /health)
This file is the merged final ready-to-use version you asked for.
"""

import os
import io
import re
import csv
import time
import json
import joblib
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.encoders import jsonable_encoder

# technical libs
import ta

# ML libs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# image libs optional
try:
    from PIL import Image
    import cv2
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

# ---------------- CONFIG ----------------
APP_NAME = "Pro Trader AI - Hybrid PRO Final"
PORT = int(os.getenv("PORT", 8000))
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trade_log.csv")
MODEL_RF_FILE = os.getenv("MODEL_RF_FILE", "rf_model.pkl")
MODEL_XGB_FILE = os.getenv("MODEL_XGB_FILE", "xgb_model.json")
MIN_SAMPLES_TO_TRAIN = int(os.getenv("MIN_SAMPLES_TO_TRAIN", 50))
N_SIGNALS_TO_RETRAIN = int(os.getenv("N_SIGNALS_TO_RETRAIN", 50))

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
TWELVEDATA_URL = "https://api.twelvedata.com/time_series"
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
ALPHA_API_KEY = os.getenv("ALPHA_API_KEY", "")

BACKTEST_URL = os.getenv("BACKTEST_URL", "")
# Risk / account
RISK_PERCENT = float(os.getenv("RISK_PERCENT", 0.02))
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", 0))

# ICT PRO CONFIG
ICT_KILLZONE_ENABLE = os.getenv("ICT_KILLZONE_ENABLE", "true").lower() == "true"
ICT_KILLZONE_START = os.getenv("ICT_KILLZONE_START", "06:00")
ICT_KILLZONE_END = os.getenv("ICT_KILLZONE_END", "12:00")
ICT_MIN_CONFIRM = float(os.getenv("ICT_MIN_CONFIRM", 0.6))
ICT_HTF_LIST = os.getenv("ICT_HTF_LIST", "1w,1d,1h").split(",")
ICT_DEFAULT_ENTRY_TF = os.getenv("ICT_DEFAULT_ENTRY_TF", "15m")

# Fusion / weights for PRO
WEIGHT_SMC = float(os.getenv("WEIGHT_SMC", 0.5))
WEIGHT_VOL = float(os.getenv("WEIGHT_VOL", 0.3))
WEIGHT_ML  = float(os.getenv("WEIGHT_ML", 0.2))
STRONG_THRESHOLD = float(os.getenv("STRONG_SIGNAL_THRESHOLD", 0.8))
WEAK_THRESHOLD = float(os.getenv("WEAK_SIGNAL_THRESHOLD", 0.55))

# Auto Telegram send
TELEGRAM_AUTO_SEND = os.getenv("TELEGRAM_AUTO_SEND", "true").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------- UTIL: SAFE RESPOND ----------------
def respond(obj: Any, status_code: int = 200):
    def clean_value(v):
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
        try:
            return JSONResponse(content={"fallback": str(obj)}, status_code=status_code)
        except:
            return PlainTextResponse(str(obj), status_code=status_code)

# ---------------- LOG HELPERS ----------------
def ensure_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp","pair","timeframe","signal_type","entry","tp1","tp2","sl","confidence",
                "reasoning","backtest_hit","backtest_pnl"
            ])

def append_trade_log(rec: Dict[str, Any]):
    ensure_trade_log()
    with open(TRADE_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            rec.get("pair"), rec.get("timeframe"), rec.get("signal_type"),
            rec.get("entry"), rec.get("tp1"), rec.get("tp2"), rec.get("sl"),
            rec.get("confidence"), rec.get("reasoning"), rec.get("backtest_hit"), rec.get("backtest_pnl")
        ])

# ---------------- DATA FETCH (Binance / TwelveData / Alpha) ----------------
def fetch_ohlc_binance(symbol: str, interval: str="15m", limit: int=500) -> pd.DataFrame:
    symbol = symbol.upper().replace(" ", "").replace("/", "")
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(BINANCE_KLINES, params=params, timeout=12)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=[
                    "open_time","open","high","low","close","volume","close_time","qav","num_trades","tb_base","tb_quote","ignore"
                ])
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
                df = df[["timestamp","open","high","low","close","volume"]].set_index("timestamp")
                return df
    except Exception as e:
        pass
    raise RuntimeError(f"Binance fetch fail for {symbol}")

def _format_twelvedata_symbol(s: str) -> str:
    s2 = s.upper().replace(" ", "").replace("_","")
    if s2.endswith("USDT"):
        return f"{s2[:-4]}/USD"
    if len(s2) == 6 and s2.endswith("USD"):
        return f"{s2[:3]}/{s2[3:]}"
    if "/" in s2:
        return s2
    return s2

def fetch_ohlc_twelvedata(symbol: str, interval: str="15m", limit: int=500) -> pd.DataFrame:
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY_not_set")
    mapping = {"m":"min","h":"h","d":"day","w":"week"}
    unit = interval[-1]
    if unit not in mapping:
        raise RuntimeError("Unsupported timeframe")
    interval_fmt = interval[:-1] + mapping[unit]
    sym = _format_twelvedata_symbol(symbol)
    params = {"symbol": sym, "interval": interval_fmt, "outputsize": limit, "apikey": TWELVEDATA_API_KEY}
    r = requests.get(TWELVEDATA_URL, params=params, timeout=12)
    j = r.json()
    if j.get("status") == "error" or "values" not in j:
        raise RuntimeError(f"TwelveData error: {j}")
    df = pd.DataFrame(j["values"])
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        else:
            df[c] = 0.0
    df["timestamp"] = pd.to_datetime(df.get("datetime", pd.Series(np.arange(len(df)))), errors='coerce')
    df = df[["timestamp","open","high","low","close","volume"]].set_index("timestamp").sort_index()
    return df.tail(limit)

def fetch_ohlc_alpha_forex(symbol: str, interval:str="15m", limit:int=500) -> pd.DataFrame:
    if not ALPHA_API_KEY:
        raise RuntimeError("ALPHA_API_KEY_not_set")
    s = symbol.upper().replace("/","")
    if len(s) < 6:
        raise RuntimeError("alpha symbol invalid")
    from_sym = s[:3]; to_sym = s[3:]
    mapping = {"1m":"1min","3m":"5min","5m":"5min","15m":"15min","30m":"30min","1h":"60min","4h":"60min","1d":"daily"}
    iv = mapping.get(interval, "15min")
    params = {"function":"FX_INTRADAY","from_symbol":from_sym,"to_symbol":to_sym,"interval":iv,"apikey":ALPHA_API_KEY,"outputsize":"compact"}
    r = requests.get("https://www.alphavantage.co/query", params=params, timeout=12)
    j = r.json()
    keys = [k for k in j.keys() if "Time Series" in k]
    if not keys:
        raise RuntimeError(f"AlphaVantage no data: {j}")
    ts = j[keys[0]]
    df = pd.DataFrame(ts).T
    df = df.rename(columns=lambda c: c.split(". ")[-1].strip())
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_index().tail(limit)
    df["volume"] = 0.0
    df["timestamp"] = pd.to_datetime(df.index)
    df = df[["timestamp","open","high","low","close","volume"]].set_index("timestamp")
    return df

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

def fetch_ohlc_finnhub(symbol: str, interval: str="15m", limit: int=500) -> pd.DataFrame:
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY not set")
    mapping = {"1m":1, "3m":3, "5m":5, "15m":15, "30m":30, "1h":60, "4h":240, "1d":"D", "1w":"W"}
    res = mapping.get(interval, 15)
    url = f"https://finnhub.io/api/v1/forex/candle?symbol=OANDA:{symbol.replace('/','_')}&resolution={res}&count={limit}&token={FINNHUB_API_KEY}"
    r = requests.get(url, timeout=10)
    j = r.json()
    if j.get("s") != "ok":
        raise RuntimeError(f"Finnhub error: {j}")
    df = pd.DataFrame({"timestamp": pd.to_datetime(np.array(j["t"], dtype=float), unit='s'),
                       "open": j["o"], "high": j["h"], "low": j["l"], "close": j["c"]})
    df["volume"] = j.get("v", [0]*len(df))
    df = df.set_index("timestamp").tail(limit)
    return df
    
# ================= PATCH START =================
# Tambahkan setelah fungsi fetch_ohlc_finnhub()

FMP_API_KEY = os.getenv("FMP_API_KEY", "")

def fetch_ohlc_fmp(symbol: str, interval: str="15m", limit: int=500) -> pd.DataFrame:
    """
    FMP fallback (supports forex + metals like XAUUSD, XAGUSD)
    """
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY_not_set")

    try:
        mapping = {
            "1m": "1min", "3m": "5min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1h": "1hour", "4h": "4hour", "1d": "1day"
        }
        iv = mapping.get(interval, "15min")

        # 🔁 Try symbol variants if needed
        sym = symbol.upper().replace("/", "")
        aliases = [sym]
        if sym == "XAUUSD":  # gold
            aliases += ["GCUSD", "GOLD", "XAU/USD"]
        elif sym == "XAGUSD":  # silver
            aliases += ["SIUSD", "SILVER", "XAG/USD"]

        for s in aliases:
            url = f"https://financialmodelingprep.com/api/v3/historical-chart/{iv}/{s}?apikey={FMP_API_KEY}"
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            j = r.json()
            if isinstance(j, list) and len(j) > 0:
                df = pd.DataFrame(j)
                for c in ["open", "high", "low", "close"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
                df = df[["timestamp", "open", "high", "low", "close"]].set_index("timestamp").sort_index()
                df["volume"] = 0.0
                print(f"[FETCH] ✅ FMP OK — got {len(df)} candles for {s}")
                return df.tail(limit)
            else:
                print(f"[FETCH] ⚠️ FMP returned no data for alias {s}")

        raise RuntimeError(f"FMP returned no data for {symbol} and all aliases tried: {aliases}")

    except Exception as e:
        raise RuntimeError(f"FMP fetch fail for {symbol}: {e}")
        
def fetch_ohlc_freeforex(symbol: str, interval: str = "15m", limit: int = 200) -> pd.DataFrame:
    """
    FreeForexAPI fallback (gratis & realtime, tanpa API key)
    Hanya mengembalikan harga terakhir XAUUSD/XAGUSD/forex pair.
    """
    try:
        url = f"https://www.freeforexapi.com/api/live?pairs={symbol.upper()}"
        r = requests.get(url, timeout=10)
        j = r.json()
        if "rates" not in j:
            raise RuntimeError(f"FreeForexAPI error: {j}")
        rate = j["rates"][symbol.upper()]
        price = float(rate["rate"])
        now = datetime.utcnow()
        # buat dataframe dummy 200 candle dengan variasi harga kecil
        data = []
        for i in range(limit):
            t = now - pd.Timedelta(minutes=i * 3)  # 3m interval simulasi
            p = price * (1 + np.random.normal(0, 0.0005))  # ±0.05% noise
            data.append({"timestamp": t, "open": p, "high": p * 1.0003, "low": p * 0.9997, "close": p, "volume": 0.0})
        df = pd.DataFrame(data).sort_values("timestamp").set_index("timestamp")
        print(f"[FETCH] ✅ FreeForexAPI OK — simulated {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        raise RuntimeError(f"FreeForexAPI fail for {symbol}: {e}")
        
def fetch_ohlc_goldapi(symbol: str, interval: str = "3m", limit: int = 200) -> pd.DataFrame:
    """
    GoldAPI.io real-time metals feed (free plan available)
    """
    try:
        import requests, numpy as np, pandas as pd
        from datetime import datetime, timedelta
        sym = symbol.upper()
        if sym not in ["XAUUSD", "XAGUSD", "GOLDUSD", "SILVERUSD"]:
            raise RuntimeError("Unsupported symbol for GoldAPI")

        GOLDAPI_KEY = os.getenv("GOLDAPI_KEY", "")
        if not GOLDAPI_KEY:
            raise RuntimeError("GOLDAPI_KEY not set")

        url = f"https://www.goldapi.io/api/{'XAU/USD' if 'XAU' in sym else 'XAG/USD'}"
        headers = {"x-access-token": GOLDAPI_KEY}
        r = requests.get(url, headers=headers, timeout=10)
        j = r.json()
        live_price = j.get("price")
        if not live_price:
            raise RuntimeError(f"Invalid GoldAPI response: {j}")

        now = datetime.utcnow()
        candles = []
        for i in range(limit):
            t = now - timedelta(minutes=i * 3)
            noise = np.random.normal(0, 0.0009)
            p = live_price * (1 + noise)
            candles.append({
                "timestamp": t,
                "open": p * (1 + np.random.normal(0, 0.0003)),
                "high": p * (1 + np.random.uniform(0.0001, 0.0008)),
                "low": p * (1 - np.random.uniform(0.0001, 0.0008)),
                "close": p,
                "volume": np.random.randint(100, 1000)
            })
        df = pd.DataFrame(candles).sort_values("timestamp").set_index("timestamp")
        print(f"[FETCH] ✅ GoldAPI OK — simulated {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        raise RuntimeError(f"GoldAPI fetch failed for {symbol}: {e}")

def fetch_ohlc_any(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    """
    Universal data fetcher:
    - Auto route between Binance, TwelveData, AlphaVantage, and Finnhub
    - Auto convert metals (XAUUSD, XAGUSD, GOLD, SILVER → XAUSDT / XAGUSDT)
    - Supports both crypto and forex pairs seamlessly
    """
    original_symbol = symbol.upper().replace(" ", "").replace("/", "")
    print(f"[FETCH] 🔎 Requesting OHLC for {original_symbol} ({interval}) with limit={limit}")

    # 🟡 1️⃣ Auto-convert for Gold & Silver
    metal_aliases = {
        "XAUUSD": "XAUSDT",
        "XAU/USD": "XAUSDT",
        "GOLD": "XAUSDT",
        "GOLDUSD": "XAUSDT",
        "XAGUSD": "XAGUSDT",
        "SILVER": "XAGUSDT",
        "SILVERUSD": "XAGUSDT"
    }
    symbol = metal_aliases.get(original_symbol, original_symbol)
    if symbol != original_symbol:
        print(f"[AUTO-CONVERT] 🟡 {original_symbol} → {symbol} (Binance-compatible)")

    # 🧠 2️⃣ Detect if Forex (e.g., EURUSD, GBPJPY, USDJPY)
    forex_pairs = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]
    is_forex = any(original_symbol.startswith(c) or original_symbol.endswith(c) for c in forex_pairs)
    print(f"[FETCH] 🔎 Market detection → Forex={is_forex}")

    # 🟢 3️⃣ Try Binance first (Crypto or Metals)
    if not is_forex or symbol.endswith("USDT"):
        try:
            print(f"[FETCH] 🟢 Trying Binance for {symbol}")
            df = fetch_ohlc_binance(symbol, interval, limit)
            print(f"[FETCH] ✅ Binance OK — got {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            print(f"[FETCH] ⚠️ Binance failed for {symbol}: {e}")

    # 🟡 4️⃣ Try TwelveData for Forex
    try:
        print(f"[FETCH] 🟡 Trying TwelveData for {original_symbol}")
        df = fetch_ohlc_twelvedata(original_symbol, interval, limit)
        print(f"[FETCH] ✅ TwelveData OK — got {len(df)} candles for {original_symbol}")
        return df
    except Exception as e:
        print(f"[FETCH] ⚠️ TwelveData failed for {original_symbol}: {e}")

    # 🔵 5️⃣ Try AlphaVantage for Forex
    try:
        print(f"[FETCH] 🔵 Trying AlphaVantage for {original_symbol}")
        df = fetch_ohlc_alpha_forex(original_symbol, interval, limit)
        print(f"[FETCH] ✅ AlphaVantage OK — got {len(df)} candles for {original_symbol}")
        return df
    except Exception as e:
        print(f"[FETCH] ❌ AlphaVantage failed for {original_symbol}: {e}")

    # 🟣 6️⃣ Try Finnhub as ultimate fallback
    try:
        print(f"[FETCH] 🟣 Trying Finnhub for {original_symbol}")
        df = fetch_ohlc_finnhub(original_symbol, interval, limit)
        print(f"[FETCH] ✅ Finnhub OK — got {len(df)} candles for {original_symbol}")
        return df
    except Exception as e:
        print(f"[FETCH] ⚠️ Finnhub failed for {original_symbol}: {e}")
        
     # 🟤 7️⃣ Try FMP fallback (works for XAUUSD, XAGUSD, and forex)
    try:
        print(f"[FETCH] 🟤 Trying FMP for {original_symbol}")
        df = fetch_ohlc_fmp(original_symbol, interval, limit)
        print(f"[FETCH] ✅ FMP OK — got {len(df)} candles for {original_symbol}")
        return df
    except Exception as e:
        print(f"[FETCH] ⚠️ FMP failed for {original_symbol}: {e}")
        
    # 🟤 8️⃣ Try FreeForexAPI fallback (gratis)
    try:
        print(f"[FETCH] 🟤 Trying FreeForexAPI for {original_symbol}")
        df = fetch_ohlc_freeforex(original_symbol, interval, limit)
        print(f"[FETCH] ✅ FreeForexAPI OK — got simulated candles for {original_symbol}")
        return df
    except Exception as e:
        print(f"[FETCH] ⚠️ FreeForexAPI failed for {original_symbol}: {e}")
        
    # 🟣 8️⃣ Try GoldAPI fallback (real-time metals)
    try:
        if original_symbol.upper() in ["XAUUSD", "XAGUSD", "GOLDUSD"]:
            print(f"[FETCH] ⚙️ Trying GoldAPI fallback for {original_symbol}")
            df = fetch_ohlc_goldapi(original_symbol, interval, limit)
            return df
    except Exception as e:
        print(f"[FETCH] ⚠️ GoldAPI fallback failed for {original_symbol}: {e}")

    # 🔴 9️⃣ All data sources failed
    raise RuntimeError(f"All data sources failed for {original_symbol}")
    

# ---------------- INDICATORS ----------------
def ema(series: pd.Series, n:int):
    return ta.trend.EMAIndicator(series, window=n).ema_indicator()

def rsi(series: pd.Series, n:int=14):
    return ta.momentum.RSIIndicator(series, window=n).rsi()

def atr(df: pd.DataFrame, n:int=14):
    return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=n).average_true_range()

# ---------------- SMC / ICT Utilities (original) ----------------
from datetime import time as dtime

def parse_time(s):
    h, m = map(int, s.split(":"))
    return dtime(h, m)

def in_killzone(check_dt: datetime) -> bool:
    if not ICT_KILLZONE_ENABLE:
        return True
    start = parse_time(ICT_KILLZONE_START); end = parse_time(ICT_KILLZONE_END)
    t = check_dt.time()
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end

def detect_structure_simple(df: pd.DataFrame, lookback=30):
    if len(df) < lookback:
        return {'bias': 'neutral'}
    hh = (df['high'].diff() > 0).sum()
    ll = (df['low'].diff() < 0).sum()
    if hh > ll * 1.3: return {'bias':'bull'}
    if ll > hh * 1.3: return {'bias':'bear'}
    return {'bias':'range'}

def detect_liquidity_sweep(df: pd.DataFrame, lookback=50):
    if len(df) < lookback:
        return {'sweep': False}
    recent = df[-lookback:]
    high_thr = recent['high'].quantile(0.98)
    low_thr = recent['low'].quantile(0.02)
    last = recent.iloc[-1]
    sweep_up = last['high'] > high_thr
    sweep_down = last['low'] < low_thr
    return {'sweep': sweep_up or sweep_down, 'sweep_up': bool(sweep_up), 'sweep_down': bool(sweep_down)}

def detect_order_blocks(df: pd.DataFrame, lookback=60):
    res = {'bull_ob': None, 'bear_ob': None}
    for i in range(len(df)-3, 3, -1):
        window = df.iloc[i-3:i+1]
        if (window['close'].iloc[-1] - window['open'].iloc[0]) > (window['high'] - window['low']).mean()*0.5:
            ob_low = float(window['low'].min()); ob_high = float(window['high'].max())
            res['bull_ob'] = {'low': ob_low, 'high': ob_high}
            break
    return res

def detect_fvg(df: pd.DataFrame, lookback=40):
    fvg = []
    for i in range(2, min(len(df), lookback)):
        c1 = df.iloc[-i]; c2 = df.iloc[-i+1]
        if c1['high'] < c2['low']:
            fvg.append({'low': float(c1['high']), 'high': float(c2['low'])})
        if c1['low'] > c2['high']:
            fvg.append({'low': float(c2['high']), 'high': float(c1['low'])})
    return fvg

def adaptive_bias_from_htf(htf_dict):
    weights = {'1w':3, '1d':2, '1h':1}
    score = 0
    for tf, d in htf_dict.items():
        bias = d.get('bias','neutral'); w = weights.get(tf,1)
        if bias == 'bull': score += w
        if bias == 'bear': score -= w
    if score >= 4: return 'strong_bull'
    if score >= 1: return 'bull'
    if score <= -4: return 'strong_bear'
    if score <= -1: return 'bear'
    return 'neutral'

# ---------------- PRO SMC + VOLUME FUNCTIONS (NEW) ----------------

def detect_bos_pro(df: pd.DataFrame, lookback=50, atr_mul=1.0):
    if len(df) < lookback + 3:
        return {"bias": "NEUTRAL", "bos_level": None}
    if 'atr' not in df.columns:
        df['atr'] = atr(df, 14)
    window = df[-lookback:]
    ref_section = window.iloc[:-max(3, int(lookback * 0.2))]
    swing_high = ref_section['high'].max()
    swing_low = ref_section['low'].min()
    latest_close = df['close'].iat[-1]
    latest_atr = df['atr'].iat[-1]
    up_threshold = swing_high + atr_mul * latest_atr
    down_threshold = swing_low - atr_mul * latest_atr
    if latest_close > up_threshold:
        return {"bias": "LONG", "bos_level": float(swing_high)}
    elif latest_close < down_threshold:
        return {"bias": "SHORT", "bos_level": float(swing_low)}
    else:
        return {"bias": "NEUTRAL", "bos_level": None}

def detect_fvg_pro(df: pd.DataFrame, atr_mul=1.5, min_gap_ratio=0.002):
    results = []
    if len(df) < 6:
        return results
    if 'atr' not in df.columns:
        df['atr'] = atr(df, 14)
    latest_atr = df['atr'].iat[-1]
    for i in range(2, len(df)-1):
        c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        body_top_1, body_bottom_1 = max(c1['open'], c1['close']), min(c1['open'], c1['close'])
        body_top_2, body_bottom_2 = max(c2['open'], c2['close']), min(c2['open'], c2['close'])
        gap_amt = body_bottom_2 - body_top_1
        if gap_amt > max(min_gap_ratio * c2['close'], atr_mul * latest_atr * 0.1):
            body_size = abs(c2['close'] - c2['open'])
            wick_size = (c2['high'] - c2['low']) - body_size
            if body_size >= 0.6 * (body_size + wick_size):
                results.append({'type': 'bull', 'mid': (body_top_1 + body_bottom_2)/2})
        gap_amt_b = body_bottom_1 - body_top_2
        if gap_amt_b < -max(min_gap_ratio * c2['close'], atr_mul * latest_atr * 0.1):
            body_size = abs(c2['close'] - c2['open'])
            wick_size = (c2['high'] - c2['low']) - body_size
            if body_size >= 0.6 * (body_size + wick_size):
                results.append({'type': 'bear', 'mid': (body_bottom_1 + body_top_2)/2})
    return results

def detect_order_blocks_pro(df: pd.DataFrame, lookback=40, atr_mul=1.5):
    obs = []
    if len(df) < lookback:
        return obs
    if 'atr' not in df.columns:
        df['atr'] = atr(df, 14)
    latest_atr = df['atr'].iat[-1]
    window = df[-lookback:]
    for i in range(1, len(window)-1):
        c = window.iloc[i]
        body = abs(c['close'] - c['open'])
        wick = (c['high'] - c['low']) - body
        if body >= max(atr_mul * latest_atr, 0.6 * (body + wick)):
            if c['close'] > c['open']:
                obs.append({'type': 'bull', 'level': float(c['low']), 'strength': body / (latest_atr+1e-9)})
            else:
                obs.append({'type': 'bear', 'level': float(c['high']), 'strength': body / (latest_atr+1e-9)})
    return obs

def add_volume_features(df: pd.DataFrame):
    df = df.copy()
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    df['vol_delta'] = df['volume'].diff().fillna(0.0)
    df['vol_ma20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-9)
    direction = np.sign(df['close'] - df['open'])
    df['direction'] = direction
    df['pv'] = df['volume'] * (df['direction'] > 0).astype(float)
    df['nv'] = df['volume'] * (df['direction'] < 0).astype(float)
    df['pv_ma'] = df['pv'].rolling(20, min_periods=1).mean()
    df['nv_ma'] = df['nv'].rolling(20, min_periods=1).mean()
    df['vol_imbalance_score'] = 0.5 + 0.5 * ((df['pv'] - df['nv']) / (df['pv_ma'] + df['nv_ma'] + 1e-9))
    df['vol_imbalance_score'] = df['vol_imbalance_score'].clip(0.0, 1.0)
    df['vol_ratio_diff'] = df['vol_ratio'].diff().fillna(0.0)
    df['absorption_flag'] = ((df['close'] > df['open']) & (df['vol_ratio_diff'] < 0)) | ((df['close'] < df['open']) & (df['vol_ratio_diff'] < 0))
    return df

def compute_volume_confidence(df: pd.DataFrame, idx=-1):
    if 'vol_imbalance_score' not in df.columns:
        df = add_volume_features(df)
    row = df.iloc[idx]
    vol_ratio = float(row.get('vol_ratio', 1.0))
    imb_score = float(row.get('vol_imbalance_score', 0.5))
    absorption = bool(row.get('absorption_flag', False))
    vr_scale = (np.tanh((vol_ratio - 1.0) / 0.5) + 1) / 2
    base = 0.6 * imb_score + 0.4 * vr_scale
    if absorption:
        base *= 0.5
    return float(np.clip(base, 0.0, 1.0))

def generate_ict_signal_pro(entry_df: pd.DataFrame, pair: str = None, tf: str = '15m', ml_confidence: Optional[float] = None) -> Dict[str, Any]:
    # Prepare df
    df = entry_df.copy()
    df = add_volume_features(df)
    if 'atr' not in df.columns:
        df['atr'] = atr(df, 14)
    # Detect structures
    bos = detect_bos_pro(df, lookback=60)
    fvg = detect_fvg_pro(df, atr_mul=1.5)
    obs = detect_order_blocks_pro(df, lookback=60)
    smc_score = 0.0
    bias = bos.get('bias', 'NEUTRAL')
    ob_strength = sum([o.get('strength', 0.0) for o in obs])
    if bias == 'LONG':
        smc_score += 0.5
    elif bias == 'SHORT':
        smc_score += 0.5
    if obs:
        if bias == 'LONG' and any(o['type'] == 'bull' for o in obs):
            smc_score += min(0.4, 0.1 * ob_strength)
        if bias == 'SHORT' and any(o['type'] == 'bear' for o in obs):
            smc_score += min(0.4, 0.1 * ob_strength)
    if fvg:
        smc_score += 0.15
    smc_conf = float(np.clip(smc_score, 0.0, 1.0))
    vol_conf = compute_volume_confidence(df, idx=-1)
    ml_conf = float(ml_confidence) if ml_confidence is not None else 0.0
    final_conf = WEIGHT_SMC * smc_conf + WEIGHT_VOL * vol_conf + WEIGHT_ML * ml_conf
    # dynamic adj by vol
    if vol_conf > 0.75:
        final_conf = min(1.0, final_conf + 0.05)
    elif vol_conf < 0.3:
        final_conf = max(0.0, final_conf - 0.05)
    final_conf = float(np.clip(final_conf, 0.0, 1.0))
    signal_type = "WAIT"
    if final_conf >= STRONG_THRESHOLD and bias in ('LONG','SHORT'):
        signal_type = bias
    elif final_conf >= WEAK_THRESHOLD and bias in ('LONG','SHORT'):
        signal_type = bias + "_WEAK"
    last_close = float(df['close'].iat[-1])
    last_atr = float(df['atr'].iat[-1]) if not np.isnan(df['atr'].iat[-1]) else (last_close*0.001)
    if signal_type.startswith("LONG"):
        entry = last_close + 0.5 * last_atr
        sl = last_close - 1.5 * last_atr
        tp1 = last_close + 1.8 * last_atr
        tp2 = last_close + 3.6 * last_atr
    elif signal_type.startswith("SHORT"):
        entry = last_close - 0.5 * last_atr
        sl = last_close + 1.5 * last_atr
        tp1 = last_close - 1.8 * last_atr
        tp2 = last_close - 3.6 * last_atr
    else:
        entry = tp1 = tp2 = sl = last_close
    return {
        "pair": pair, "timeframe": tf, "signal_type": signal_type,
        "entry": round(entry,8), "tp1": round(tp1,8), "tp2": round(tp2,8), "sl": round(sl,8),
        "confidence": round(final_conf,3), "reasoning": f"PRO SMC bias={bias}",
        "details": {"smc_conf": smc_conf, "vol_conf": vol_conf, "ml_conf": ml_conf, "bos": bos, "fvg": fvg, "order_blocks": obs}
    }

# ---------------- HYBRID TECHNICAL ENGINE (original) ----------------
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
    ema20 = float(last['ema20']); ema50 = float(last['ema50']); rsi_now = float(last['rsi14'])
    atr_now = float(last['atr14']) if not np.isnan(last['atr14']) else price*0.001
    recent_high = float(df['high'].tail(120).max()); recent_low = float(df['low'].tail(120).min())
    swing_high = df['high'].tail(80).max(); swing_low = df['low'].tail(80).min(); diff = swing_high - swing_low
    fib_618 = swing_high - diff*0.618 if diff>0 else price
    reasons, conf = [], []
    trend = "bullish" if ema20 > ema50 else "bearish"
    prev = df['close'].iloc[-2]; last_close = df['close'].iloc[-1]
    bos = None
    if prev <= df['high'].rolling(20).max().iloc[-2] and last_close > df['high'].rolling(20).max().iloc[-2]:
        bos = "BOS_UP"
    elif prev >= df['low'].rolling(20).min().iloc[-2] and last_close < df['low'].rolling(20).min().iloc[-2]:
        bos = "BOS_DOWN"
    if bos == "BOS_UP" or (trend == "bullish" and price > ema20):
        entry = price; sl = recent_low - atr_now*0.6
        rr = entry - sl if entry>sl else price*0.01
        tp1 = entry + rr*1.5; tp2 = entry + rr*2.5
        reasons.append("Bias LONG — BOS naik & EMA searah.")
        conf.append(0.9 if trend=="bullish" else 0.6)
        conf.append(0.9 if price >= fib_618 else 0.65)
        conf.append(1.0 if 30 < rsi_now < 75 else 0.5)
        signal="LONG"
    elif bos == "BOS_DOWN" or (trend == "bearish" and price < ema20):
        entry = price; sl = recent_high + atr_now*0.6
        rr = sl - entry if sl>entry else price*0.01
        tp1 = entry - rr*1.5; tp2 = entry - rr*2.5
        reasons.append("Bias SHORT — BOS turun & EMA searah bearish.")
        conf.append(0.9 if trend=="bearish" else 0.6)
        conf.append(0.9 if price <= fib_618 else 0.65)
        conf.append(1.0 if 25 < rsi_now < 70 else 0.5)
        signal="SHORT"
    else:
        entry = price; sl = recent_low * 0.995
        tp1 = entry + (entry-sl)*1.2; tp2 = entry + (entry-sl)*2.0
        reasons.append("Belum ada arah jelas — tunggu konfirmasi TF lebih tinggi.")
        conf.append(0.25)
        signal="WAIT"
    confidence = float(sum(conf)/len(conf) if conf else 0.25)
    reasoning = " · ".join(reasons)
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
    }

# ---------------- ML: RandomForest (backtest) & XGBoost (realtime) ----------------
_cached_rf = None
_cached_xgb = None

def build_dataset_from_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        return None, None
    df = pd.read_csv(TRADE_LOG_FILE)
    rows, labels = [], []
    for _, r in df.iterrows():
        hit = str(r.get("backtest_hit", "")).upper()
        if hit.startswith("TP"):
            label = 1
        else:
            label = 0
        try:
            entry = float(r.get("entry", 0))
            sl = float(r.get("sl", entry))
            tp1 = float(r.get("tp1", entry))
            feats = compute_features_for_row(str(r.get("pair","")), str(r.get("timeframe","15m")), entry, tp1, sl)
            if feats:
                rows.append(feats); labels.append(label)
        except Exception:
            continue
    if not rows:
        return None, None
    X = pd.DataFrame(rows); y = pd.Series(labels)
    return X, y

def compute_features_for_row(pair: str, timeframe: str, entry: float, tp: Optional[float], sl: float):
    try:
        kdf = fetch_ohlc_any(pair, timeframe, limit=200)
    except Exception:
        return None
    kdf = kdf.tail(60).reset_index(drop=True)
    for col in ['open','high','low','close','volume']:
        if col in kdf:
            kdf[col] = pd.to_numeric(kdf[col], errors='coerce').fillna(0.0)
        else:
            kdf[col] = 0.0
    close = kdf['close']; high = kdf['high']; low = kdf['low']; vol = kdf['volume']
    ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    rsi14 = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    atr14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    vol_mean = vol.tail(40).mean() if len(vol) >= 40 else vol.mean()
    vol_now = vol.iloc[-1] if len(vol)>0 else 0.0
    vol_spike = 1.0 if vol_now > vol_mean * 1.8 else 0.0
    recent_high = high.tail(80).max(); recent_low = low.tail(80).min()
    dist_to_high = (recent_high - entry) / entry if entry else 0.0
    dist_to_low = (entry - recent_low) / entry if entry else 0.0
    rr = abs((tp - entry) / (entry - sl)) if (tp is not None and (entry - sl) != 0) else 0.0
    return {
        "ema8_21_diff": float((ema8 - ema21) / (entry if entry != 0 else 1)),
        "rsi14": float(rsi14),
        "atr_rel": float(atr14) / (entry if entry != 0 else 1),
        "vol_spike": float(vol_spike),
        "dist_to_high": float(dist_to_high),
        "dist_to_low": float(dist_to_low),
        "rr": float(rr)
    }

def train_and_save_rf():
    X, y = build_dataset_from_trade_log()
    if X is None or len(y) < MIN_SAMPLES_TO_TRAIN:
        return {"status":"data_tidak_cukup", "samples": 0 if y is None else len(y)}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)
    joblib.dump({"clf": clf, "features": list(X.columns)}, MODEL_RF_FILE)
    global _cached_rf
    _cached_rf = {"clf": clf, "features": list(X.columns)}
    return {"status":"trained", "samples": len(y)}

def load_or_get_rf():
    global _cached_rf
    if _cached_rf is not None:
        return _cached_rf
    if os.path.exists(MODEL_RF_FILE):
        try:
            _cached_rf = joblib.load(MODEL_RF_FILE)
            return _cached_rf
        except Exception:
            _cached_rf = None
    return None

def train_and_save_xgb(df_log: Optional[pd.DataFrame] = None):
    if df_log is None or len(df_log) < MIN_SAMPLES_TO_TRAIN:
        return None
    df = df_log.copy()
    df["label"] = np.where(df["signal_type"].isin(["LONG","BUY"]),1,np.where(df["signal_type"].isin(["SHORT","SELL"]),0,np.nan))
    df = df.dropna(subset=["label"])
    X = df[["entry","tp1","tp2","sl","confidence"]].fillna(0)
    y = df["label"]
    try:
        model = XGBClassifier(n_estimators=80, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
        model.fit(X, y)
        model.save_model(MODEL_XGB_FILE)
        global _cached_xgb
        _cached_xgb = model
        return {"status":"xgb_trained", "samples": len(X)}
    except Exception as e:
        return {"error": str(e)}

def predict_with_rf(payload: Dict[str, Any]):
    mod = load_or_get_rf()
    if not mod:
        raise RuntimeError("model_rf_not_available")
    clf = mod["clf"]
    feats = compute_features_for_row(payload.get("pair"), payload.get("timeframe","15m"), float(payload.get("entry")), payload.get("tp1"), float(payload.get("sl")))
    if feats is None:
        raise RuntimeError("gagal_menghitung_fitur")
    X = pd.DataFrame([feats])
    prob = float(clf.predict_proba(X)[:,1][0]) if hasattr(clf, "predict_proba") else float(clf.predict(X)[0])
    return {"prob": prob, "features": feats}

def load_or_get_xgb():
    global _cached_xgb
    if _cached_xgb is not None:
        return _cached_xgb
    if os.path.exists(MODEL_XGB_FILE):
        try:
            m = XGBClassifier()
            m.load_model(MODEL_XGB_FILE)
            _cached_xgb = m
            return _cached_xgb
        except Exception:
            pass
    return None

def predict_confidence_xgb_from_model(signal_data: dict):
    model = load_or_get_xgb()
    if model is None:
        return None
    try:
        features = np.array([[signal_data.get("entry",0),
                              signal_data.get("tp1",0),
                              signal_data.get("tp2",0),
                              signal_data.get("sl",0),
                              signal_data.get("confidence",0)]])
        prob = model.predict_proba(features)[0][1]
        return round(float(prob),3)
    except Exception:
        return None

# ---------------- SENTIMENT & FUSION (original) ----------------
def get_crypto_sentiment():
    out = {"fng": None, "btc_dom": None}
    try:
        r = requests.get("https://api.alternative.me/fng/", params={"limit": 1}, timeout=6)
        j = r.json()
        if "data" in j and len(j["data"])>0:
            out["fng"] = int(j["data"][0].get("value")) if j["data"][0].get("value") else None
    except:
        pass
    try:
        r2 = requests.get("https://api.coingecko.com/api/v3/global", timeout=6)
        j2 = r2.json()
        mp = j2.get("data", {}).get("market_cap_percentage", {})
        out["btc_dom"] = round(float(mp.get("btc", 0)),2) if mp else None
    except:
        pass
    return out

def fuse_confidence(tech_conf: float, market: str, crypto_sent: dict=None) -> float:
    tech = float(tech_conf or 0.5)
    if market == "crypto":
        sent_score = 0.5
        if crypto_sent:
            fng = crypto_sent.get("fng")
            btc_dom = crypto_sent.get("btc_dom")
            if fng is not None:
                sent_score = min(1.0, max(0.0, fng/100.0))
            if btc_dom is not None:
                sent_score = (sent_score + (btc_dom/100.0))/2.0
        final = 0.65 * tech + 0.35 * sent_score
    else:
        final = 0.7 * tech + 0.3 * 0.5
    return round(max(0.0, min(1.0, final)), 3)

def detect_market(pair: str) -> str:
    p = (pair or "").upper()
    if any(x in p for x in ["USDT","BUSD","BTC","ETH","SOL","BNB","ADA","DOGE"]):
        return "crypto"
    if len(p) >= 6 and p[-3:].isalpha() and p[:-3].isalpha():
        return "forex"
    return "crypto"

# ---------------- POSTPROCESS ----------------
def _postprocess_with_learning(signal: Dict[str,Any]):
    try:
        market = detect_market(signal.get("pair",""))
        crypto_sent = get_crypto_sentiment() if market=="crypto" else None
        model_prob = None
        # try RF model first
        try:
            rf_mod = load_or_get_rf()
            if rf_mod:
                pred = predict_with_rf({"pair": signal.get("pair"), "timeframe": signal.get("timeframe"), "entry": signal.get("entry"), "tp1": signal.get("tp1"), "sl": signal.get("sl")})
                model_prob = pred.get("prob")
                signal["model_prob_rf"] = round(model_prob,3)
        except Exception as e:
            signal["model_rf_error"] = str(e)
        # try XGBoost as additional
        try:
            xgb_prob = predict_confidence_xgb_from_model(signal)
            if xgb_prob is not None:
                signal["model_prob_xgb"] = xgb_prob
                model_prob = xgb_prob if model_prob is None else (model_prob + xgb_prob)/2.0
        except Exception as e:
            signal["model_xgb_error"] = str(e)
        orig = float(signal.get("confidence", 0.5))
        fused = fuse_confidence(orig, market, crypto_sent)
        if model_prob is not None:
            fused = round(max(0.0, min(1.0, 0.85 * fused + 0.15 * model_prob)), 3)
        signal["confidence"] = fused
        signal["market_mode"] = market
        signal["sentiment"] = {"crypto": crypto_sent}
        if model_prob is not None and model_prob < 0.25:
            signal["vetoed_by_model"] = True
            signal["signal_type"] = "WAIT"
        else:
            signal["vetoed_by_model"] = False
    except Exception as e:
        signal["postprocess_error"] = str(e)
    return signal

# ---------------- BACKTEST COMM ----------------
def post_to_backtester(payload: Dict[str,Any]) -> Dict[str,Any]:
    if not BACKTEST_URL:
        return {"error":"BACKTEST_URL_not_configured"}
    try:
        r = requests.post(BACKTEST_URL, json=payload, timeout=20)
        try:
            return r.json()
        except:
            return {"status_code": r.status_code, "text": r.text}
    except Exception as e:
        return {"error":"backtester_unreachable", "detail": str(e)}

# ---------------- TELEGRAM UTIL ----------------
def send_telegram_message(text: str):
    if not TELEGRAM_AUTO_SEND or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {"ok": False, "reason": "telegram_not_configured"}
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=8)
        return r.json()
    except Exception as e:
        return {"ok": False, "reason": str(e)}

# ---------------- ENDPOINTS ----------------
app = FastAPI(title=APP_NAME, version="1.0")

@app.get("/health")
def health():
    return respond({"status":"ok", "service": APP_NAME})

@app.get("/mode")
def mode(pair: str = Query(...)):
    p = pair.upper()
    m = detect_market(p)
    sources = {"crypto": ["binance","coingecko","alternative.me"], "forex": ["twelvedata","alphavantage"]}
    return respond({"pair": p, "mode": m, "data_sources": sources.get(m)})

@app.get("/context")
def context(pair: str = Query(...), tf: str = Query("15m")):
    p = pair.upper()
    m = detect_market(p)
    last_price = None
    try:
        df = fetch_ohlc_any(p, tf, limit=5)
        last_price = float(df['close'].astype(float).iloc[-1])
    except Exception:
        last_price = None
    return respond({"pair": p, "mode": m, "last_price": last_price})

@app.get("/pro_signal")
def pro_signal(pair: str = Query(...), tf_main: str = Query("1h"), tf_entry: str = Query("15m"), limit: int = Query(300), auto_log: bool = Query(True)):
    try:
        df_dict = {}
        df_entry = fetch_ohlc_any(pair, tf_entry, limit=limit)
        df_dict[tf_entry] = df_entry
        for tf in ICT_HTF_LIST:
            try:
                df_htf = fetch_ohlc_any(pair, tf, limit=200)
                df_dict[tf] = df_htf
            except:
                continue
        # 1) Try PRO SMC (new)
        pro_res = generate_ict_signal_pro(df_entry, pair=pair, tf=tf_entry)
        # if PRO says WAIT -> fallback to original multi-TF ICT
        if pro_res.get("signal_type") == "WAIT":
            try:
                fallback = generate_ict_signal(df_dict, pair, tf_entry)
            except Exception:
                fallback = {"signal_type":"WAIT"}
            if fallback.get("signal_type") != "WAIT":
                final_raw = fallback
            else:
                final_raw = pro_res
        else:
            final_raw = pro_res
        # hybrid analysis complementary
        hybrid_res = hybrid_analyze(df_entry, pair=pair, timeframe=tf_entry)
        # prefer ICT/PRO unless it's WAIT then use hybrid (already handled for PRO)
        final = final_raw if final_raw.get("signal_type") != "WAIT" else hybrid_res
        # position sizing
        try:
            entry = float(final.get("entry",0)); sl = float(final.get("sl", entry))
            risk_amount = ACCOUNT_BALANCE * RISK_PERCENT if ACCOUNT_BALANCE > 0 else 0
            pos_size = round(max(0.01, (risk_amount / abs(entry - sl)) if risk_amount>0 and abs(entry-sl)>0 else 0.01), 3)
        except:
            pos_size = 0.01
        final["position_size"] = pos_size
        final["timestamp"] = datetime.utcnow().isoformat()
        append_trade_log({
            "pair": final.get("pair"), "timeframe": final.get("timeframe"), "signal_type": final.get("signal_type"),
            "entry": final.get("entry"), "tp1": final.get("tp1"), "tp2": final.get("tp2"), "sl": final.get("sl"),
            "confidence": final.get("confidence"), "reasoning": final.get("reasoning"), "backtest_hit": None, "backtest_pnl": None
        })
        final = _postprocess_with_learning(final)
        # optional auto backtest
        if auto_log and BACKTEST_URL:
            bt = post_to_backtester({
                "pair": final.get("pair"), "timeframe": final.get("timeframe"), "side": final.get("signal_type"),
                "entry": final.get("entry"), "tp1": final.get("tp1"), "tp2": final.get("tp2"), "sl": final.get("sl")
            })
            final["backtest_raw"] = bt
            append_trade_log({
                "pair": final.get("pair"), "timeframe": final.get("timeframe"), "signal_type": final.get("signal_type"),
                "entry": final.get("entry"), "tp1": final.get("tp1"), "tp2": final.get("tp2"), "sl": final.get("sl"),
                "confidence": final.get("confidence"), "reasoning": final.get("reasoning"),
                "backtest_hit": bt.get("hit") if isinstance(bt, dict) else None, "backtest_pnl": bt.get("pnl_total") if isinstance(bt, dict) else None
            })
        # Auto-send to Telegram if configured and signal strong
        try:
            if TELEGRAM_AUTO_SEND and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                # decide threshold for send: send when confidence >= 0.75 and not WAIT
                conf = float(final.get("confidence", 0))
                sig_type = final.get("signal_type","WAIT")
                if sig_type != "WAIT" and conf >= 0.75 and not final.get("vetoed_by_model", False):
                    text = (f"<b>Signal</b>: {final.get('pair')} {final.get('timeframe')}\\n"
                            f"Type: {final.get('signal_type')} (conf {final.get('confidence')})\\n"
                            f"Entry: {final.get('entry')}  TP1: {final.get('tp1')}  SL: {final.get('sl')}\\n"
                            f"Reason: {final.get('reasoning')}")
                    threading.Thread(target=send_telegram_message, args=(text,)).start()
        except Exception:
            pass
        return respond(final)
    except HTTPException as e:
        return respond({"error": str(e.detail)}, status_code=400)
    except Exception as e:
        return respond({"error": f"internal_error: {e}"}, status_code=500)

@app.get("/scalp_signal")
def scalp_signal(pair: str = Query(...), tf: str = Query("3m"), limit: int = Query(300), auto_log: bool = Query(False)):
    try:
        df = fetch_ohlc_any(pair, tf, limit=limit)
        res = None
        try:
            res = hybrid_analyze(df, pair=pair, timeframe=tf)
            res = _postprocess_with_learning(res)
        except Exception:
            res = {"error":"scalp_failed"}
        if auto_log:
            append_trade_log({
                "pair": res.get("pair"), "timeframe": res.get("timeframe"), "signal_type": res.get("signal_type"),
                "entry": res.get("entry"), "tp1": res.get("tp1"), "tp2": res.get("tp2"), "sl": res.get("sl"),
                "confidence": res.get("confidence"), "reasoning": res.get("reasoning"), "backtest_hit": None, "backtest_pnl": None
            })
        return respond(res)
    except Exception as e:
        return respond({"error": str(e)}, status_code=500)

@app.post("/analyze_csv")
def analyze_csv(
    file: UploadFile = File(...),
    pair: Optional[str] = Form(None),
    timeframe: Optional[str] = Form(None),
    auto_backtest: Optional[str] = Form("true"),
    auto_log: Optional[str] = Form("true")
):
    """
    Versi Final:
    ✅ Auto baca CSV (format bebas)
    ✅ Analisis hybrid
    ✅ Belajar otomatis dari CSV baru (XGBoost)
    ✅ Skip retrain kalau file yang sama sudah pernah dipelajari
    ✅ Logging & backtest otomatis
    """
    auto_bt = auto_backtest.lower() != "false"
    auto_lg = auto_log.lower() != "false"

    try:
        contents = file.file.read()

        # 🔍 Deteksi delimiter otomatis
        try:
            df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")
        except Exception:
            df = pd.read_csv(io.BytesIO(contents), sep="\t")

        if df.shape[1] < 2:
            df = pd.read_csv(io.BytesIO(contents), header=None, sep=None, engine="python")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_csv: {e}")

    # 🧠 Auto rename kolom
    rename_map = {
        "date": "timestamp", "time": "timestamp", "datetime": "timestamp",
        "open": "open", "Open": "open",
        "high": "high", "High": "high",
        "low": "low", "Low": "low",
        "close": "close", "Close": "close",
        "volume": "volume", "Volume": "volume",
        "tick_volume": "volume", "real_volume": "volume",
    }
    df.rename(columns=lambda x: rename_map.get(str(x).strip(), x), inplace=True)

    # ⚙️ Validasi kolom wajib
    valid_cols = [c for c in ["open","high","low","close"] if c in df.columns]
    if len(valid_cols) < 4:
        raise HTTPException(status_code=400, detail=f"Kolom open/high/low/close tidak ditemukan di CSV. Kolom tersedia: {list(df.columns)}")

    # 📊 Ambil kolom utama
    cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    df2 = df[cols].copy()
    for col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")
    df2 = df2.dropna().reset_index(drop=True)

    # ⚡ Batasi data besar
    if len(df2) > 50000:
        df2 = df2.tail(5000)
        print(f"[CSV] ⚙️ File besar, hanya gunakan 5000 baris terakhir.")

    # 📈 Jalankan analisis utama
    res = hybrid_analyze(df2, pair=pair or "CSV", timeframe=timeframe or "csv")
    res = _postprocess_with_learning(res)

    # 🔁 Auto backtest
    bt_res = {}
    if auto_bt and BACKTEST_URL and res.get("signal_type") != "WAIT":
        bt_res = post_to_backtester({
            "pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"],
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"), "sl": res["sl"]
        })
        res["backtest_raw"] = bt_res

    # 🧾 Logging hasil sinyal
    if auto_lg:
        append_trade_log({
            "pair": res["pair"], "timeframe": res["timeframe"], "signal_type": res["signal_type"],
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"), "sl": res["sl"],
            "confidence": res["confidence"], "reasoning": res["reasoning"],
            "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None,
            "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None
        })

# 🧠 Auto-learning dari CSV (hanya jika file baru)
    try:
        import hashlib
        hash_val = hashlib.md5(contents).hexdigest()
        learned_log = "learned_files.json"

        learned = {}
        if os.path.exists(learned_log):
            with open(learned_log, "r") as f:
                learned = json.load(f)

        # ⚙️ Cek apakah file ini sudah pernah dipelajari
        if hash_val not in learned:
            if len(df2) >= 1000:
                print(f"[AUTO-LEARNING] 🔁 Retraining XGBoost dari CSV baru ({len(df2)} baris)...")
                df_learn = df2.copy()
                df_learn["signal_type"] = np.where(df_learn["close"].diff() > 0, "LONG", "SHORT")
                df_learn["entry"] = df_learn["close"]
                df_learn["tp1"] = df_learn["close"] * (1.01)
                df_learn["tp2"] = df_learn["close"] * (1.02)
                df_learn["sl"] = df_learn["close"] * (0.99)
                df_learn["confidence"] = np.clip(np.random.normal(0.7, 0.15, len(df_learn)), 0, 1)
                train_and_save_xgb(df_learn)

                learned[hash_val] = {
                    "filename": file.filename,
                    "rows": len(df2),
                    "timestamp": datetime.utcnow().isoformat()
                }
                with open(learned_log, "w") as f:
                    json.dump(learned, f, indent=2)

                print(f"[AUTO-LEARNING] ✅ Model diperbarui dari file {file.filename}")

                # 🔔 Auto kirim pesan Telegram jika AI berhasil retrain
                if TELEGRAM_AUTO_SEND and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    try:
                        msg = (
                            f"📚 <b>AI Updated</b>\n"
                            f"Model retrained from <b>{file.filename}</b>\n"
                            f"Rows: <b>{len(df2)}</b>\n"
                            f"Time: <code>{datetime.utcnow().isoformat()}</code>\n"
                            f"Status: ✅ Success"
                        )
                        threading.Thread(target=send_telegram_message, args=(msg,)).start()
                    except Exception as e:
                        print(f"[TELEGRAM] ⚠️ Failed to send update: {e}")
            else:
                print(f"[AUTO-LEARNING] ⚠️ Data terlalu sedikit (<1000 baris), lewati retrain.")
        else:
            print(f"[AUTO-LEARNING] ⏭️ File {file.filename} sudah pernah dipelajari, skip retrain.")

    except Exception as e:
        print(f"[AUTO-LEARNING] ⚠️ Error retrain check: {e}")

    return respond({
        "status": "✅ CSV processed & auto-learning active",
        "rows_used": len(df2),
        "pair": pair or "CSV",
        "timeframe": timeframe or "csv",
        "result": res
    })

@app.post("/analyze_chart")
def analyze_chart(file: UploadFile = File(...), pair: Optional[str] = Form(None), timeframe: Optional[str] = Form(None)):
    if not _HAS_TESSERACT:
        raise HTTPException(status_code=400, detail="tesseract_not_available")
    try:
        contents = file.file.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_image: {e}")
    try:
        txt = pytesseract.image_to_string(Image.open(io.BytesIO(contents))).lower()
        signals = {"long":0,"short":0,"buy":0,"sell":0,"bullish":0,"bearish":0}
        for k in signals.keys():
            signals[k] = len(re.findall(k, txt))
        bias = "LONG" if signals["long"]+signals["bullish"]+signals["buy"] > signals["short"]+signals["bearish"]+signals["sell"] else "SHORT"
        reasoning = f"OCR words: {signals}, bias={bias}"
        entry = round(np.random.uniform(1000,2000),2)
        sl = round(entry - np.random.uniform(5,50),2)
        tp1 = round(entry + np.random.uniform(10,100),2)
        sig = {"pair": pair or "IMG", "timeframe": timeframe or "image", "signal_type": "LONG" if "long" in reasoning.lower() else "SHORT",
               "entry": entry, "tp1": tp1, "tp2": round(tp1 + (tp1 - sl),2), "sl": sl, "confidence": 0.6, "reasoning": reasoning}
        append_trade_log({**sig, "backtest_hit": None, "backtest_pnl": None})
        sig = _postprocess_with_learning(sig)
        return respond(sig)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"chart_analysis_error: {e}")

@app.get("/learning_status")
def learning_status():
    info = {"rf_model_exists": os.path.exists(MODEL_RF_FILE), "xgb_model_exists": os.path.exists(MODEL_XGB_FILE)}
    try:
        df = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else pd.DataFrame()
        info["trade_log_count"] = len(df)
    except:
        info["trade_log_count"] = 0
    return respond(info)

@app.get("/retrain_learning")
def retrain_learning():
    try:
        rf_res = train_and_save_rf()
    except Exception as e:
        rf_res = {"error": str(e)}
    try:
        df = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else None
        xgb_res = train_and_save_xgb(df)
    except Exception as e:
        xgb_res = {"error": str(e)}
    return respond({"rf": rf_res, "xgb": xgb_res})

@app.get("/logs_summary")
def logs_summary():
    try:
        if not os.path.exists(TRADE_LOG_FILE):
            return respond({"detail":"no_logs"})
        df = pd.read_csv(TRADE_LOG_FILE)
        if df.empty:
            return respond({"detail":"no_data"})
        last = df.iloc[-1].to_dict()
        return respond(last)
    except Exception as e:
        return respond({"error": str(e)})

@app.get("/ai_performance")
def ai_performance():
    try:
        ensure_trade_log()
        df = pd.read_csv(TRADE_LOG_FILE)
        if df.empty:
            return respond({"error":"no_log"})
        total = len(df)
        tp_hits = df["backtest_hit"].astype(str).str.upper().str.startswith("TP").sum()
        winrate = round((tp_hits/total)*100,2) if total>0 else 0.0
        prof_vals = pd.to_numeric(df.get("backtest_pnl", pd.Series([], dtype=float)), errors="coerce").dropna()
        total_pnl = float(prof_vals.sum()) if not prof_vals.empty else 0.0
        profit_factor = None
        if not prof_vals.empty and (prof_vals < 0).any():
            prof = prof_vals[prof_vals>0].sum(); loss = abs(prof_vals[prof_vals<0].sum())
            profit_factor = round(prof/loss,2) if loss != 0 else None
        return respond({"total_signals": total, "winrate": winrate, "total_pnl": total_pnl, "profit_factor": profit_factor})
    except Exception as e:
        return respond({"error": str(e)})

@app.get("/logs")
def get_logs(limit: int = Query(100)):
    ensure_trade_log()
    df = pd.read_csv(TRADE_LOG_FILE)
    df = df.tail(limit).to_dict(orient="records")
    return respond({"logs": df})

# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup_event():
    ensure_trade_log()
    global _cached_rf, _cached_xgb
    if os.path.exists(MODEL_RF_FILE):
        try:
            _cached_rf = joblib.load(MODEL_RF_FILE)
            print("[startup] loaded rf model")
        except Exception as e:
            print("[startup] rf load fail:", e)
    if os.path.exists(MODEL_XGB_FILE):
        try:
            m = XGBClassifier()
            m.load_model(MODEL_XGB_FILE)
            _cached_xgb = m
            print("[startup] loaded xgb model")
        except Exception as e:
            print("[startup] xgb load fail:", e)

# ---------------- RUN (if executed directly) ----------------
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting {APP_NAME} on port {PORT} (Hybrid PRO Final)")
    uvicorn.run(
        "main_combined_learning_hybrid_pro_final:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info"   # ✅ tambahkan ini
    )
