"""
main_combined_learning_hybrid_final.py
Combined Hybrid: SMC/ICT PRO + Hybrid Technical Engine + XGBoost + RandomForest + Data Fallback
Compatibility: Designed to work with telegram_bot (2).py as-is (endpoints /pro_signal, /scalp_signal, /analyze_chart, /analyze_csv, /learning_status, /retrain_learning, /ai_performance, /logs_summary, /mode, /context, /health)
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
APP_NAME = "Pro Trader AI - Hybrid Final"
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
        # silence, fallback later
        pass
    raise RuntimeError(f"Binance fetch fail for {symbol}")

def _format_twelvedata_symbol(s: str) -> str:
    s2 = s.upper().replace(" ", "").replace("_","")
    # if crypto pair endswith USDT -> map to USD symbol for TwelveData e.g. BTCUSDT -> BTC/USD
    if s2.endswith("USDT"):
        return f"{s2[:-4]}/USD"
    if len(s2) == 6 and s2.endswith("USD"):
        # e.g. XAUUSD -> XAU/USD
        return f"{s2[:3]}/{s2[3:]}"
    if "/" in s2:
        return s2
    # default try s as-is
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
    # ensure numeric
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    df["timestamp"] = pd.to_datetime(df.get("datetime", pd.Series(np.arange(len(df)))), errors="coerce")
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
    # find timeseries key
    keys = [k for k in j.keys() if "Time Series" in k]
    if not keys:
        raise RuntimeError(f"AlphaVantage no data: {j}")
    ts = j[keys[0]]
    df = pd.DataFrame(ts).T
    df = df.rename(columns=lambda c: c.split(". ")[-1].strip())
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_index().tail(limit)
    df["volume"] = 0.0
    df["timestamp"] = pd.to_datetime(df.index)
    df = df[["timestamp","open","high","low","close","volume"]].set_index("timestamp")
    return df

def fetch_ohlc_any(symbol: str, interval: str="15m", limit:int=500) -> pd.DataFrame:
    """Try Binance, then TwelveData, then AlphaVantage (with detailed logging)"""
    symbol = symbol.upper().replace(" ", "").replace("/", "")
    print(f"[FETCH] 🔎 Requesting OHLC for {symbol} ({interval}) with limit={limit}")

    # 1️⃣ Binance
    try:
        print(f"[FETCH] 🟢 Trying Binance for {symbol}")
        df = fetch_ohlc_binance(symbol, interval, limit)
        print(f"[FETCH] ✅ Binance OK — got {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        print(f"[FETCH] ⚠️ Binance failed for {symbol}: {e}")

    # 2️⃣ TwelveData
    try:
        print(f"[FETCH] 🟡 Trying TwelveData for {symbol}")
        df = fetch_ohlc_twelvedata(symbol, interval, limit)
        print(f"[FETCH] ✅ TwelveData OK — got {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        print(f"[FETCH] ⚠️ TwelveData failed for {symbol}: {e}")

    # 3️⃣ AlphaVantage
    try:
        print(f"[FETCH] 🔵 Trying AlphaVantage for {symbol}")
        df = fetch_ohlc_alpha_forex(symbol, interval, limit)
        print(f"[FETCH] ✅ AlphaVantage OK — got {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        print(f"[FETCH] ❌ AlphaVantage failed for {symbol}: {e}")

    raise RuntimeError(f"All data sources failed for {symbol}")

# ---------------- INDICATORS ----------------
def ema(series: pd.Series, n:int):
    return ta.trend.EMAIndicator(series, window=n).ema_indicator()

def rsi(series: pd.Series, n:int=14):
    return ta.momentum.RSIIndicator(series, window=n).rsi()

def atr(df: pd.DataFrame, n:int=14):
    return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=n).average_true_range()

# ---------------- SMC / ICT Utilities ----------------
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

def generate_ict_signal(df_dict: Dict[str, pd.DataFrame], pair: str, entry_tf: str):
    htf_analysis = {}
    for tf in ICT_HTF_LIST:
        if tf in df_dict:
            htf_analysis[tf] = detect_structure_simple(df_dict[tf], lookback=40)
    bias = adaptive_bias_from_htf(htf_analysis)
    entry_df = df_dict.get(entry_tf)
    if entry_df is None:
        return {'error': 'entry_tf_missing'}
    sweep = detect_liquidity_sweep(entry_df, lookback=80)
    ob = detect_order_blocks(entry_df, lookback=80)
    fvg = detect_fvg(entry_df, lookback=80)
    is_kz = in_killzone(datetime.utcnow())
    score = 0; reasons = []
    if bias in ('bull','strong_bull'): score += 1; reasons.append(f"HTF-bias:{bias}")
    if bias in ('bear','strong_bear'): score -= 1; reasons.append(f"HTF-bias:{bias}")
    if sweep.get('sweep_down'): score += 0.8; reasons.append("Liquidity sweep down")
    if sweep.get('sweep_up'): score -= 0.8; reasons.append("Liquidity sweep up")
    if ob.get('bull_ob'): score += 0.3; reasons.append("Bullish OB present")
    if len(fvg)>0: score += 0.2; reasons.append("FVG detected")
    if is_kz: score *= 1.1; reasons.append("In killzone")
    conf_raw = max(min(score/3, 1), -1)
    confidence = abs(round(conf_raw,3))
    signal_type = "LONG" if conf_raw >= ICT_MIN_CONFIRM else ("SHORT" if conf_raw <= -ICT_MIN_CONFIRM else "WAIT")
    last = entry_df.iloc[-1]
    atr_v = (entry_df['high'] - entry_df['low']).rolling(14).mean().iloc[-1] if len(entry_df)>14 else (entry_df['high']-entry_df['low']).mean()
    entry = float(last['close'])
    if signal_type == "LONG":
        sl = entry - 1.5*atr_v; tp1 = entry + 1.8*atr_v; tp2 = entry + 3.6*atr_v
    elif signal_type == "SHORT":
        sl = entry + 1.5*atr_v; tp1 = entry - 1.8*atr_v; tp2 = entry - 3.6*atr_v
    else:
        sl, tp1, tp2 = entry, entry, entry
    reasoning = "; ".join(reasons) or "No strong ICT/SMC cues"
    return {
        "pair": pair, "timeframe": entry_tf, "signal_type": signal_type,
        "entry": round(entry,8), "tp1": round(tp1,8), "tp2": round(tp2,8), "sl": round(sl,8),
        "confidence": confidence, "reasoning": reasoning
    }

# ---------------- HYBRID TECHNICAL ENGINE (from combined v1) ----------------
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
    # Break of Structure detection
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
    # Build X from trade_log similar to earlier XGBoost approach if df_log provided
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

# ---------------- SENTIMENT & FUSION (lightweight) ----------------
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
        # collect multi timeframe data: entry tf + HTF list if possible
        df_dict = {}
        # entry timeframe
        df_entry = fetch_ohlc_any(pair, tf_entry, limit=limit)
        df_dict[tf_entry] = df_entry
        # gather HTF samples (try to fetch if available)
        for tf in ICT_HTF_LIST:
            try:
                df_htf = fetch_ohlc_any(pair, tf, limit=200)
                df_dict[tf] = df_htf
            except:
                continue
        # generate ICT signal
        ict_res = generate_ict_signal(df_dict, pair, tf_entry)
        if "error" in ict_res:
            raise HTTPException(status_code=400, detail=ict_res.get("error"))
        # Also run hybrid technical engine on entry TF for complementary signal
        hybrid_res = hybrid_analyze(df_entry, pair=pair, timeframe=tf_entry)
        # prefer ICT signal_type unless it's WAIT then use hybrid
        final = ict_res if ict_res.get("signal_type") != "WAIT" else hybrid_res
        # ensure pos size safe
        try:
            entry = float(final.get("entry",0)); sl = float(final.get("sl", entry))
            risk_amount = ACCOUNT_BALANCE * RISK_PERCENT if ACCOUNT_BALANCE > 0 else 0
            pos_size = round(max(0.01, (risk_amount / abs(entry - sl)) if risk_amount>0 and abs(entry-sl)>0 else 0.01), 3)
        except:
            pos_size = 0.01
        final["position_size"] = pos_size
        final["timestamp"] = datetime.utcnow().isoformat()
        # append raw log before postprocess
        append_trade_log({
            "pair": final.get("pair"), "timeframe": final.get("timeframe"), "signal_type": final.get("signal_type"),
            "entry": final.get("entry"), "tp1": final.get("tp1"), "tp2": final.get("tp2"), "sl": final.get("sl"),
            "confidence": final.get("confidence"), "reasoning": final.get("reasoning"), "backtest_hit": None, "backtest_pnl": None
        })
        # try ML models to refine confidence
        final = _postprocess_with_learning(final)
        # optional auto backtest
        if auto_log and BACKTEST_URL:
            bt = post_to_backtester({
                "pair": final.get("pair"), "timeframe": final.get("timeframe"), "side": final.get("signal_type"),
                "entry": final.get("entry"), "tp1": final.get("tp1"), "tp2": final.get("tp2"), "sl": final.get("sl")
            })
            final["backtest_raw"] = bt
            # update last log with backtest results (append new record)
            append_trade_log({
                "pair": final.get("pair"), "timeframe": final.get("timeframe"), "signal_type": final.get("signal_type"),
                "entry": final.get("entry"), "tp1": final.get("tp1"), "tp2": final.get("tp2"), "sl": final.get("sl"),
                "confidence": final.get("confidence"), "reasoning": final.get("reasoning"),
                "backtest_hit": bt.get("hit") if isinstance(bt, dict) else None, "backtest_pnl": bt.get("pnl_total") if isinstance(bt, dict) else None
            })
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
def analyze_csv(file: UploadFile = File(...), pair: Optional[str] = Form(None), timeframe: Optional[str] = Form(None), auto_backtest: Optional[str] = Form("true"), auto_log: Optional[str] = Form("true")):
    auto_bt = auto_backtest.lower() != "false"
    auto_lg = auto_log.lower() != "false"
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_csv: {e}")
    df.columns = [c.strip().lower() for c in df.columns]
    def find_col(k): return next((c for c in df.columns if k in c), None)
    o,h,l,c = find_col('open'), find_col('high'), find_col('low'), find_col('close')
    if not all([o,h,l,c]):
        raise HTTPException(status_code=400, detail="kolom_tidak_lengkap (butuh open, high, low, close)")
    df2 = df[[o,h,l,c]].rename(columns={o:'open', h:'high', l:'low', c:'close'})
    for col in ['open','high','low','close']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2 = df2.dropna().reset_index(drop=True)
    res = hybrid_analyze(df2, pair=pair or "CSV", timeframe=timeframe or "csv")
    res = _postprocess_with_learning(res)
    bt_res = {}
    if auto_bt and BACKTEST_URL and res.get("signal_type") != "WAIT":
        bt_res = post_to_backtester({
            "pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"],
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"), "sl": res["sl"]
        })
        res["backtest_raw"] = bt_res
    if auto_lg:
        append_trade_log({
            "pair": res["pair"], "timeframe": res["timeframe"], "signal_type": res["signal_type"],
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"), "sl": res["sl"],
            "confidence": res["confidence"], "reasoning": res["reasoning"],
            "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None, "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None
        })
    return respond(res)

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
    # Best-effort OCR + candle detection (reuse earlier helpers if available)
    try:
        # Simple OCR for words (long/buy/sell)
        txt = pytesseract.image_to_string(Image.open(io.BytesIO(contents))).lower()
        signals = {"long":0,"short":0,"buy":0,"sell":0,"bullish":0,"bearish":0}
        for k in signals.keys():
            signals[k] = len(re.findall(k, txt))
        bias = "LONG" if signals["long"]+signals["bullish"]+signals["buy"] > signals["short"]+signals["bearish"]+signals["sell"] else "SHORT"
        reasoning = f"OCR words: {signals}, bias={bias}"
        # fallback fake numeric guess
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
    # retrain xgb from CSV log if exists
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
    print(f"Starting {APP_NAME} on port {PORT}")
    uvicorn.run("main_combined_learning_hybrid_final:app", host="0.0.0.0", port=PORT, reload=False)
