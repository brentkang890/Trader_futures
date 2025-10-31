# main_combined_learning.py
"""
Pro Trader AI - Combined + Learning (Bahasa Indonesia)
SMC Pro + Mode Hybrid + Auto Position Sizing (Risk 2% default)
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
    title="Pro Trader AI - Combined + Learning (ID) (SMC Pro)",
    description="Analisis Crypto (Binance) & Forex (TwelveData) + SMC Pro + Auto Sizing",
    version="1.1"
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

# --- RISK & MODE CONFIG ---
RISK_PERCENT = float(os.environ.get("RISK_PERCENT", 0.02))  # default 2%
ACCOUNT_BALANCE = float(os.environ.get("ACCOUNT_BALANCE", "0"))  # optional, for sizing
CURRENT_MODE = os.environ.get("TRADING_MODE", "auto").lower()  # auto|agresif|moderate|konservatif

# SMC profile file
SMC_PROFILE_FILE = os.environ.get("SMC_PROFILE_FILE", "smc_profiles.pkl")

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
        print("respond() fallback:", e)
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

# ---------------- MODE & POSITION SIZING HELPERS ----------------
def set_mode(mode: str):
    global CURRENT_MODE
    m = (mode or "auto").lower()
    if m not in ("auto","agresif","aggressive","moderate","moderat","konservatif"):
        raise ValueError("mode_tidak_valid")
    # normalize synonyms
    if m == "aggressive":
        m = "agresif"
    if m == "moderat":
        m = "moderate"
    CURRENT_MODE = m
    return CURRENT_MODE

def get_mode():
    return CURRENT_MODE

def position_size(entry: float, sl: float) -> dict:
    """Hitung ukuran posisi berdasar risiko RISK_PERCENT dari ACCOUNT_BALANCE (opsional)."""
    if entry <= 0 or sl <= 0 or entry == sl:
        return {"size": None, "risk_amount": None}
    if ACCOUNT_BALANCE <= 0:
        return {"size": None, "risk_amount": None}
    risk_amount = ACCOUNT_BALANCE * RISK_PERCENT
    per_unit_risk = abs(entry - sl)
    if per_unit_risk == 0:
        return {"size": None, "risk_amount": round(risk_amount, 2)}
    size = risk_amount / per_unit_risk
    return {"size": round(size, 6), "risk_amount": round(risk_amount, 2)}

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

# ---------------- SMC PROFILE & AUTO-TUNE (LOAD DEFAULTS) ----------------
import joblib as _joblib  # alias for smc profile save/load

_default_smc_profiles = {
    "crypto_major": {
        "pattern": ["BTC", "ETH", "BNB", "ADA", "SOL"],
        "bos_window": 18,
        "ob_lookback": 90,
        "fvg_lookback": 70,
        "tol_equal": 0.001,
        "rr_target": 2.2,
        "fib_zone": 0.5
    },
    "altcoin": {
        "pattern": ["USDT", "PEPE", "BONK", "SHIB", "SOL"],
        "bos_window": 14,
        "ob_lookback": 60,
        "fvg_lookback": 50,
        "tol_equal": 0.0015,
        "rr_target": 1.8,
        "fib_zone": 0.618
    },
    "forex": {
        "pattern": ["EUR", "USD", "JPY", "XAU", "GBP", "AUD"],
        "bos_window": 25,
        "ob_lookback": 200,
        "fvg_lookback": 100,
        "tol_equal": 0.0003,
        "rr_target": 3.0,
        "fib_zone": 0.382
    }
}

try:
    if os.path.exists(SMC_PROFILE_FILE):
        smc_profiles = _joblib.load(SMC_PROFILE_FILE)
    else:
        smc_profiles = _default_smc_profiles.copy()
        _joblib.dump(smc_profiles, SMC_PROFILE_FILE)
except Exception as e:
    print("SMC profile load error:", e)
    smc_profiles = _default_smc_profiles.copy()

def save_smc_profiles():
    try:
        _joblib.dump(smc_profiles, SMC_PROFILE_FILE)
    except Exception as e:
        print("Failed save smc profiles:", e)

def get_smc_profile(pair: str) -> dict:
    """Return SMC profile dict for given pair (best-effort)."""
    if not pair:
        return smc_profiles.get("crypto_major")
    p = pair.upper()
    # simple heuristics: check major symbols first
    for name, prof in smc_profiles.items():
        for pat in prof.get("pattern", []):
            if pat.upper() in p:
                return prof
    if any(x in p for x in ["USDT", "BUSD", "BTC", "ETH"]):
        return smc_profiles.get("crypto_major")
    if len(p) >= 6 and p[-3:].isalpha() and p[:-3].isalpha():
        return smc_profiles.get("forex")
    return smc_profiles.get("crypto_major")

# ---------------- SMC DETECTIONS (param-driven wrappers) ----------------
def detect_equal_levels(series: pd.Series, tol: float = 0.0005) -> list:
    """Cari equal highs/lows sederhana (liquidity)."""
    lvls = []
    s = series.tail(120).reset_index(drop=True)
    for i in range(2, len(s)):
        if s[i-1] == 0 or s[i] == 0:
            continue
        if abs(s[i] - s[i-1]) / s[i] < tol:
            lvls.append(float((s[i] + s[i-1]) / 2))
    return sorted(set(lvls))

def detect_fvg(df: pd.DataFrame, lookback: int = 60) -> list:
    """Cari FVG bullish/bearish (gap body-to-body sederhana)."""
    fvg = []
    sub = df.tail(lookback).reset_index(drop=True)
    for i in range(2, len(sub)):
        # Bullish FVG: low(i) > high(i-2)
        if float(sub['low'].iloc[i]) > float(sub['high'].iloc[i-2]):
            fvg.append({"type":"bull","upper": float(sub['low'].iloc[i]), "lower": float(sub['high'].iloc[i-2])})
        # Bearish FVG: high(i) < low(i-2)
        if float(sub['high'].iloc[i]) < float(sub['low'].iloc[i-2]):
            fvg.append({"type":"bear","upper": float(sub['low'].iloc[i-2]), "lower": float(sub['high'].iloc[i])})
    return fvg

def detect_order_blocks(df: pd.DataFrame, lookback: int = 120) -> list:
    """OB sederhana: candle berlawanan sebelum BOS; pakai body-range sebagai zona."""
    obs = []
    sub = df.tail(lookback).reset_index(drop=True)
    for i in range(3, len(sub)):
        prev_bull = float(sub['close'].iloc[i-1]) > float(sub['open'].iloc[i-1])
        prev_bear = float(sub['close'].iloc[i-1]) < float(sub['open'].iloc[i-1])
        # Bull OB: candle merah terakhir sebelum kenaikan kuat
        if prev_bear and float(sub['close'].iloc[i]) > float(sub['high'].iloc[i-1]):
            ob_low  = float(min(sub['open'].iloc[i-1], sub['close'].iloc[i-1]))
            ob_high = float(max(sub['open'].iloc[i-1], sub['close'].iloc[i-1]))
            obs.append({"type":"bull","low":ob_low,"high":ob_high})
        # Bear OB: candle hijau terakhir sebelum penurunan kuat
        if prev_bull and float(sub['close'].iloc[i]) < float(sub['low'].iloc[i-1]):
            ob_low  = float(min(sub['open'].iloc[i-1], sub['close'].iloc[i-1]))
            ob_high = float(max(sub['open'].iloc[i-1], sub['close'].iloc[i-1]))
            obs.append({"type":"bear","low":ob_low,"high":ob_high})
    # dedup & kompres
    dedup = []
    for z in obs:
        if not any(abs(z['low']-x['low'])<1e-9 and abs(z['high']-x['high'])<1e-9 and z['type']==x['type'] for x in dedup):
            dedup.append(z)
    return dedup

def smc_context(df: pd.DataFrame, profile: Optional[dict] = None) -> dict:
    """Return SMC context using profile parameters (bos_window, ob_lookback, fvg_lookback, tol_equal)."""
    if profile is None:
        profile = get_smc_profile(None)
    bos = breakout_of_structure(df, window=int(profile.get("bos_window", 20)))
    swing_high = float(df['high'].tail(80).max())
    swing_low  = float(df['low'].tail(80).min())
    mid = swing_low + (swing_high - swing_low) * (profile.get("fib_zone", 0.5))
    ob  = detect_order_blocks(df, lookback=int(profile.get("ob_lookback", 120)))
    fvg = detect_fvg(df, lookback=int(profile.get("fvg_lookback", 60)))
    eql_highs = detect_equal_levels(df['high'], tol=float(profile.get("tol_equal", 0.0005)))
    eql_lows  = detect_equal_levels(df['low'], tol=float(profile.get("tol_equal", 0.0005)))
    return {
        "bos": bos,
        "premium_discount_50": mid,
        "order_blocks": ob,
        "fvgs": fvg,
        "eq_highs": eql_highs,
        "eq_lows": eql_lows,
        "profile_used": profile
    }

# ---------------- PLACEHOLDER: hybrid_analyze (full implementation continues in Part 2) ----------------
def hybrid_analyze(df: pd.DataFrame, pair:Optional[str]=None, timeframe:Optional[str]=None) -> dict:
    """
    STRATEGI HYBRID (SMC Pro) - implementasi lengkap dilanjutkan di Part 2.
    Fungsi ini akan:
     - Hitung EMA/RSi/ATR
     - Ambil SMC context via get_smc_profile(pair)
     - Tentukan mode (auto/manual) dan sesuaikan TP/SL/Entry
     - Integrasikan ML model & sentiment
     - Kembalikan dict sinyal termasuk position sizing dan smc metadata
    """
    # Implementation continued in Part 2
    raise NotImplementedError("hybrid_analyze belum diimplementasikan di bagian ini. Lanjutkan ke Part 2.")
# ---------------- STRATEGI HYBRID (implementasi penuh) ----------------
def hybrid_analyze(df: pd.DataFrame, pair:Optional[str]=None, timeframe:Optional[str]=None) -> dict:
    df = df.copy().dropna().reset_index(drop=True)
    if df.shape[0] < 12:
        return {"error":"data_tidak_cukup", "message":"Perlu minimal 12 candle untuk analisis."}

    # pilih profile berdasarkan pair
    profile = get_smc_profile(pair or "")
    # compute basic indicators
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
    bos = breakout_of_structure(df, window=int(profile.get("bos_window", 20)))
    swing_high = float(df['high'].tail(80).max())
    swing_low  = float(df['low'].tail(80).min())
    diff = swing_high - swing_low if swing_high and swing_low else price*0.01
    fib_zone = float(profile.get("fib_zone", 0.5))
    fib_618 = swing_high - diff * 0.618 if diff>0 else price

    # SMC context with profile
    smc = smc_context(df, profile=profile)

    # Mode otomatis berdasar kondisi tren/volatilitas
    mode = get_mode()
    if mode == "auto":
        trend_strength = abs((ema20 - ema50) / price) if price else 0.0
        choppy = float(atr_now/price) < 0.003 if price else True
        if trend_strength > 0.01 and not choppy:
            mode = "agresif"
        elif trend_strength > 0.004:
            mode = "moderate"
        else:
            mode = "konservatif"

    reasons, conf = [], []
    trend = "bullish" if ema20 > ema50 else "bearish"

    # default safe values
    entry = price
    signal = "WAIT"
    sl = price * 0.995
    tp1 = price * 1.002
    tp2 = price * 1.004

    # LONG conditions
    if bos == "BOS_UP" or (trend == "bullish" and price > ema20):
        entry = price
        sl = recent_low - atr_now*0.6 if recent_low and atr_now else price - atr_now
        rr = entry - sl if entry>sl else max( (entry*0.01), 1e-8 )
        tp1 = entry + rr * 1.5
        tp2 = entry + rr * 2.5
        reasons.append("Bias LONG — BOS naik & EMA searah.")
        conf.append(0.9 if trend=="bullish" else 0.6)
        conf.append(0.9 if price >= fib_618 else 0.65)
        conf.append(1.0 if 30 < rsi_now < 75 else 0.5)
        signal="LONG"

        # Premium/Discount filter (beli lebih bagus di bawah mid)
        if price > smc['premium_discount_50'] and mode in ("moderate","konservatif"):
            conf.append(0.2)
            reasons.append("Harga di zona premium (PD) — hati-hati.")
        # Validasi OB/FVG/Liquidity (cek beberapa last zones)
        bull_ob_hit = any(z['type']=="bull" and z['low']<=price<=z['high'] for z in smc.get('order_blocks', [])[-6:])
        bull_fvg_near = any(g['type']=="bull" and g['lower']<=price<=g['upper'] for g in smc.get('fvgs', [])[-6:])
        liq_above = any(lvl>price for lvl in smc.get('eq_highs', []))
        if bull_ob_hit: conf.append(0.12); reasons.append("Mitigasi Bull OB.")
        if bull_fvg_near: conf.append(0.08); reasons.append("Di dalam/near Bull FVG.")
        if liq_above: conf.append(0.06); reasons.append("Ada equal highs di atas (target likuiditas).")
        # Mode tuning
        if mode == "agresif":
            tp1 = entry + rr*1.2; tp2 = entry + rr*2.0
        elif mode == "konservatif":
            tp1 = entry + rr*1.8; tp2 = entry + rr*3.0

    # SHORT conditions
    elif bos == "BOS_DOWN" or (trend == "bearish" and price < ema20):
        entry = price
        sl = recent_high + atr_now*0.6 if recent_high and atr_now else price + atr_now
        rr = sl - entry if sl>entry else max((price*0.01), 1e-8)
        tp1 = entry - rr*1.5
        tp2 = entry - rr*2.5
        reasons.append("Bias SHORT — BOS turun & EMA searah bearish.")
        conf.append(0.9 if trend=="bearish" else 0.6)
        conf.append(0.9 if price <= fib_618 else 0.65)
        conf.append(1.0 if 25 < rsi_now < 70 else 0.5)
        signal="SHORT"

        # Premium/Discount filter for sells
        if price < smc['premium_discount_50'] and mode in ("moderate","konservatif"):
            conf.append(0.2)
            reasons.append("Harga di zona discount untuk sell — hati-hati.")
        bear_ob_hit = any(z['type']=="bear" and z['low']<=price<=z['high'] for z in smc.get('order_blocks', [])[-6:])
        bear_fvg_near = any(g['type']=="bear" and g['lower']<=price<=g['upper'] for g in smc.get('fvgs', [])[-6:])
        liq_below = any(lvl<price for lvl in smc.get('eq_lows', []))
        if bear_ob_hit: conf.append(0.12); reasons.append("Mitigasi Bear OB.")
        if bear_fvg_near: conf.append(0.08); reasons.append("Di dalam/near Bear FVG.")
        if liq_below: conf.append(0.06); reasons.append("Ada equal lows di bawah (target likuiditas).")
        if mode == "agresif":
            tp1 = entry - rr*1.2; tp2 = entry - rr*2.0
        elif mode == "konservatif":
            tp1 = entry - rr*1.8; tp2 = entry - rr*3.0

    else:
        # WAIT / no clear bias
        entry = price
        sl = recent_low * 0.995 if recent_low else price * 0.997
        tp1 = entry + (entry-sl)*1.2
        tp2 = entry + (entry-sl)*2.0
        reasons.append("Belum ada arah jelas — tunggu konfirmasi TF lebih tinggi.")
        conf.append(0.25)
        signal="WAIT"

    # compute tech confidence
    try:
        confidence = float(sum(conf)/len(conf))
    except Exception:
        confidence = 0.5
    reasoning = " · ".join(reasons)

    # position sizing
    sizing = position_size(entry, sl)

    result = {
        "pair": pair or "",
        "timeframe": timeframe or "",
        "signal_type": signal,
        "entry": round(entry,8),
        "tp1": round(tp1,8),
        "tp2": round(tp2,8),
        "sl": round(sl,8),
        "confidence": round(confidence,3),
        "reasoning": reasoning,
        "risk_percent": RISK_PERCENT,
        "position_size": sizing.get("size"),
        "risk_amount": sizing.get("risk_amount"),
        "mode_used": mode,
        "smc": {
            "bos": smc.get("bos"),
            "pd": round(float(smc.get("premium_discount_50", price)), 8),
            "eq_highs": smc.get("eq_highs", [])[-3:],
            "eq_lows": smc.get("eq_lows", [])[-3:],
            "profile_used": smc.get("profile_used", {})
        }
    }

    return result

# ---------------- SCALPING ENGINE (diperbarui untuk sizing & smc profile) ----------------
def scalp_engine(df: pd.DataFrame, pair:Optional[str]=None, tf:Optional[str]=None) -> dict:
    if df.shape[0] < 30:
        return {"error": "data_tidak_cukup"}

    # pick profile
    profile = get_smc_profile(pair or "")
    df['ema8'] = ema(df['close'], 8)
    df['ema21'] = ema(df['close'], 21)
    df['rsi14'] = rsi(df['close'], 14)
    df['atr14'] = atr(df, 14)

    last = df.iloc[-1]
    price = float(last['close'])
    atr_now = float(last['atr14']) if not np.isnan(last['atr14']) else price * 0.001
    vol_mean = df['volume'].tail(40).mean() if df.shape[0] >= 40 else df['volume'].mean()
    vol_spike = float(last['volume']) > (vol_mean * 1.8 if vol_mean > 0 else False)

    smc = smc_context(df, profile=profile)
    mode = get_mode()
    if mode == "auto":
        trend_strength = abs((df['ema8'].iloc[-1] - df['ema21'].iloc[-1]) / price) if price else 0.0
        if trend_strength > 0.008:
            mode = "agresif"
        elif trend_strength > 0.003:
            mode = "moderate"
        else:
            mode = "konservatif"

    # conditions for scalp entries
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

    sizing = position_size(entry, sl)

    return {
        "pair": pair or "",
        "timeframe": tf or "",
        "signal_type": signal,
        "entry": round(entry, 8),
        "tp1": round(tp1, 8),
        "tp2": round(tp2, 8),
        "sl": round(sl, 8),
        "confidence": round(conf, 3),
        "reasoning": reason,
        "risk_percent": RISK_PERCENT,
        "position_size": sizing.get("size"),
        "risk_amount": sizing.get("risk_amount"),
        "mode_used": mode
    }

# ---------------- IMAGE OCR HELPERS (detect harga & candles dari gambar chart) ----------------
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

    # spawn autotune in background (non-blocking)
    try:
        t = threading.Thread(target=lambda: auto_tune_smc_parameters(min_signals_required=50), daemon=True)
        t.start()
    except Exception as e:
        print("autotune spawn failed:", e)

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

# Part 2 complete — lanjut ke Part 3 (endpoints, startup, backtester comms, smc autotune API).
# ---------------- FASTAPI ENDPOINTS ----------------

@app.get("/")
def root():
    return respond({
        "service": "Pro Trader AI (SMC Pro + AutoTune)",
        "status": "✅ aktif",
        "mode": get_mode(),
        "risk_percent": RISK_PERCENT,
        "smc_profiles_loaded": list(smc_profiles.keys())
    })

@app.get("/set_mode")
def set_trading_mode(mode: str = Query(...)):
    try:
        m = set_mode(mode)
        return respond({"ok": True, "mode": m})
    except Exception as e:
        return respond({"error": str(e)})

@app.get("/get_mode")
def get_trading_mode():
    return respond({"mode": get_mode()})

@app.get("/pro_signal")
def pro_signal(pair: str = Query(...), tf_main: str = Query("1h"), tf_entry: str = Query("15m"),
               auto_log: bool = Query(False)):
    try:
        df = fetch_ohlc_binance(pair, tf_entry, limit=500)
        result = hybrid_analyze(df, pair=pair, timeframe=tf_entry)
        # optional backtest/probability integration
        try:
            pred = predict_with_model(result)
            result["ml_prob"] = round(pred.get("prob", 0.0), 3)
            result["ml_features"] = pred.get("features", {})
        except Exception as e:
            result["ml_prob"] = None
            result["ml_error"] = str(e)

        if auto_log:
            append_trade_log(result)
            check_and_trigger_retrain_if_needed()
        return respond(result)
    except Exception as e:
        return respond({"error": str(e)})

@app.get("/scalp_signal")
def scalp_signal(pair: str = Query(...), tf: str = Query("3m"), auto_log: bool = Query(False)):
    try:
        df = fetch_ohlc_binance(pair, tf, limit=300)
        result = scalp_engine(df, pair=pair, tf=tf)
        if auto_log:
            append_trade_log(result)
            check_and_trigger_retrain_if_needed()
        return respond(result)
    except Exception as e:
        return respond({"error": str(e)})

@app.post("/analyze_chart")
def analyze_chart(file: UploadFile = File(...)):
    try:
        data = file.file.read()
        img_cv = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None:
            raise ValueError("gambar_tidak_valid")
        ymap = ocr_y_axis_prices(img_cv)
        df = detect_candles_from_plot(img_cv, ymap)
        if df.empty:
            raise ValueError("gagal_deteksi_candlestick")
        result = hybrid_analyze(df, pair="UNKNOWN", timeframe="chart")
        return respond(result)
    except Exception as e:
        return respond({"error": str(e)})

@app.post("/analyze_csv")
def analyze_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if not {"open","high","low","close"}.issubset(df.columns):
            raise ValueError("csv_tidak_valid: perlu kolom open,high,low,close")
        pair = file.filename.replace(".csv","")
        result = hybrid_analyze(df, pair=pair, timeframe="csv")
        return respond(result)
    except Exception as e:
        return respond({"error": str(e)})

@app.post("/retrain_learning")
def retrain_learning():
    try:
        res = train_and_save_model()
        return respond(res)
    except Exception as e:
        return respond({"error": str(e)})

@app.get("/learning_status")
def learning_status():
    try:
        model_exists = os.path.exists(MODEL_FILE)
        features = []
        if model_exists:
            mod = joblib.load(MODEL_FILE)
            features = mod.get("features", [])
        df = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else pd.DataFrame()
        return respond({
            "model_exists": model_exists,
            "trade_log_count": len(df),
            "features": features
        })
    except Exception as e:
        return respond({"error": str(e)})

@app.get("/ai_performance")
def ai_performance():
    try:
        df = pd.read_csv(TRADE_LOG_FILE)
        total = len(df)
        if total == 0:
            return respond({"error": "belum_ada_data"})
        hit = df['backtest_hit'].astype(str).str.upper()
        tp_hits = hit.str.startswith("TP").sum()
        sl_hits = hit.str.startswith("SL").sum()
        winrate = round(tp_hits/total*100,2) if total>0 else 0.0
        pnl = pd.to_numeric(df.get("backtest_pnl", pd.Series([0]*total)), errors='coerce').fillna(0)
        profit_factor = round(abs(pnl[pnl>0].sum() / abs(pnl[pnl<0].sum()+1e-9)),3) if len(pnl)>0 else 0
        return respond({
            "total_signals": total,
            "winrate": winrate,
            "profit_factor": profit_factor,
            "model_status": "trained" if os.path.exists(MODEL_FILE) else "not_trained"
        })
    except Exception as e:
        return respond({"error": str(e)})

@app.get("/logs_summary")
def logs_summary():
    try:
        if not os.path.exists(TRADE_LOG_FILE):
            return respond({"error":"tidak_ada_log"})
        df = pd.read_csv(TRADE_LOG_FILE)
        if df.empty:
            return respond({"error":"log_kosong"})
        last = df.iloc[-1].to_dict()
        return respond(last)
    except Exception as e:
        return respond({"error": str(e)})

@app.get("/download_logs")
def download_logs():
    if not os.path.exists(TRADE_LOG_FILE):
        raise HTTPException(status_code=404, detail="tidak_ada_log")
    return FileResponse(TRADE_LOG_FILE, filename="trade_log.csv", media_type="text/csv")

@app.get("/smc_profiles")
def smc_profiles_api():
    return respond(smc_profiles)

@app.get("/force_autotune")
def force_autotune(min_signals: int = Query(50)):
    try:
        res = auto_tune_smc_parameters(min_signals_required=min_signals, force=True)
        return respond(res)
    except Exception as e:
        return respond({"error": str(e)})

# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup_event():
    ensure_trade_log()
    # load model cache
    global _cached_model
    if os.path.exists(MODEL_FILE):
        try:
            _cached_model = joblib.load(MODEL_FILE)
            print("[Startup] model loaded:", MODEL_FILE)
        except Exception as e:
            print("[Startup] gagal load model:", e)
    else:
        print("[Startup] model belum ada, nanti dilatih otomatis.")

    print("[Startup] mode awal:", get_mode(), "risk:", RISK_PERCENT)
    print("[Startup] smc profiles:", list(smc_profiles.keys()))

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8000)))
