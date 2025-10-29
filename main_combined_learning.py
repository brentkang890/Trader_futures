# main_combined_learning.py
"""
Pro Trader AI - FULL SMC (ICT/Alchemist) + Learning
- Market Structure (BOS/CHOCH), Liquidity Sweep, Order Block (OB), Fair Value Gap (FVG/Imbalance)
- POI selection (ranked), Apex zone, Left Shoulder (LS) pattern heuristic
- Multi-TP (TP1..TP3..TP5), SL at invalidation, RR targeting, killzone timing weight
- Hybrid confirmation with classic TA (EMA/RSI/ATR) untuk robustness
- Learning: RandomForest dari trade_log.csv (probabilitas sukses)
- Endpoints:
  /pro_signal, /scalp_signal, /analyze_chart, /analyze_csv,
  /learning_status, /retrain_learning, /ai_performance,
  /logs, /logs_summary, /download_logs, /model_debug,
  /sentiment, /mode, /context
"""

import os
import io
import re
import csv
import math
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import requests
import pandas as pd
import numpy as np

from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
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
    title="Pro Trader AI - FULL SMC + Learning (ID)",
    description="ICT/Alchemist SMC + Integrated Learning + Backtest + CSV/Chart",
    version="2.0"
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

ALPHA_API_KEY = os.environ.get("ALPHA_API_KEY", "")
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
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "pair", "timeframe", "signal_type",
                "entry", "tp1", "tp2", "tp3", "tp4", "tp5",
                "sl", "confidence", "reasoning",
                "backtest_hit", "backtest_pnl"
            ])

def append_trade_log(logrec: Dict[str, Any]):
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
            logrec.get("tp3"),
            logrec.get("tp4"),
            logrec.get("tp5"),
            logrec.get("sl"),
            logrec.get("confidence"),
            logrec.get("reasoning"),
            logrec.get("backtest_hit"),
            logrec.get("backtest_pnl")
        ])

def detect_market(pair: str) -> str:
    p = (pair or "").upper()
    if any(x in p for x in ["USDT", "BUSD", "BTC", "ETH", "SOL", "BNB", "ADA", "DOGE"]):
        return "crypto"
    if len(p) >= 6 and p[-3:].isalpha() and p[:-3].isalpha():
        return "forex"
    return "crypto"

# ---------------- FETCH OHLC ----------------
def fetch_ohlc_twelvedata(symbol: str, interval: str="15m", limit: int=500) -> pd.DataFrame:
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

    try:
        return fetch_ohlc_twelvedata(symbol, interval, limit)
    except Exception as e:
        try:
            return fetch_ohlc_alpha_forex(symbol, interval, limit)
        except Exception:
            raise RuntimeError(f"fetch_ohlc_failed_for_{symbol}: {e}")

def fetch_ohlc_alpha_forex(symbol: str, interval: str="15m", limit: int=500) -> pd.DataFrame:
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

# ---------------- BASE INDICATORS ----------------
def ema(series: pd.Series, n: int):
    return ta.trend.EMAIndicator(series, window=n).ema_indicator()

def rsi(series: pd.Series, n: int=14):
    return ta.momentum.RSIIndicator(series, window=n).rsi()

def atr(df: pd.DataFrame, n: int=14):
    return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=n).average_true_range()

# ---------------- SMC CORE ----------------
def swing_points(df: pd.DataFrame, lookback: int = 3) -> Tuple[List[int], List[int]]:
    """Find swing highs and swing lows indexes using simple fractal rule."""
    highs, lows = [], []
    h = df["high"].values
    l = df["low"].values
    for i in range(lookback, len(df)-lookback):
        if h[i] == max(h[i-lookback:i+lookback+1]):
            highs.append(i)
        if l[i] == min(l[i-lookback:i+lookback+1]):
            lows.append(i)
    return highs, lows

def label_structure(df: pd.DataFrame, highs: List[int], lows: List[int]) -> Dict[str, Any]:
    """Label BOS/CHOCH + trend via last swings."""
    labels = {"events": []}
    if len(highs) < 2 or len(lows) < 2:
        labels["trend"] = "sideways"
        return labels
    # Recent pivots
    last_h = highs[-2:]
    last_l = lows[-2:]
    last_close = float(df["close"].iloc[-1])

    hh_prev = float(df["high"].iloc[last_h[-2]])
    hh_now  = float(df["high"].iloc[last_h[-1]])
    ll_prev = float(df["low"].iloc[last_l[-2]])
    ll_now  = float(df["low"].iloc[last_l[-1]])

    trend = "bullish" if hh_now > hh_prev and ll_now > ll_prev else "bearish" if hh_now < hh_prev and ll_now < ll_prev else "sideways"
    labels["trend"] = trend

    # BOS if price closes beyond swing extreme
    bos = None
    if last_close > hh_now:
        bos = "BOS_UP"
    elif last_close < ll_now:
        bos = "BOS_DOWN"

    # CHoCH if break direction changes vs previous impulse
    choch = None
    if bos == "BOS_UP" and trend != "bullish":
        choch = "CHOCH_UP"
    if bos == "BOS_DOWN" and trend != "bearish":
        choch = "CHOCH_DOWN"

    if bos: labels["events"].append(bos)
    if choch: labels["events"].append(choch)
    labels["ref_high"] = hh_now
    labels["ref_low"]  = ll_now
    return labels

def detect_liquidity_sweep(df: pd.DataFrame, window: int = 30, tolerance: float = 0.0005) -> Optional[str]:
    """Detect sweep: wick takes previous high/low but close rejects back inside."""
    if len(df) < window + 2:
        return None
    sub = df.tail(window+2).reset_index(drop=True)
    prev_high = sub["high"].iloc[:-2].max()
    prev_low  = sub["low"].iloc[:-2].min()
    # last candle
    h = float(sub["high"].iloc[-2])
    l = float(sub["low"].iloc[-2])
    c = float(sub["close"].iloc[-2])
    o = float(sub["open"].iloc[-2])

    # sweep up if high takes prev_high but close back below it
    if h > prev_high and c < prev_high * (1 + tolerance):
        return "SWEEP_UP"
    # sweep down if low takes prev_low but close back above it
    if l < prev_low and c > prev_low * (1 - tolerance):
        return "SWEEP_DOWN"
    return None

def find_fvg(df: pd.DataFrame, max_lookback: int = 60, min_gap_ratio: float = 0.0008) -> List[Dict[str, float]]:
    """Find fair value gaps (3-candle)."""
    res = []
    n = len(df)
    start = max(2, n - max_lookback)
    for i in range(start, n-1):
        c1_h = float(df["high"].iloc[i-2]) if i-2 >= 0 else None
        c1_l = float(df["low"].iloc[i-2]) if i-2 >= 0 else None
        c2_h = float(df["high"].iloc[i-1])
        c2_l = float(df["low"].iloc[i-1])
        c3_h = float(df["high"].iloc[i])
        c3_l = float(df["low"].iloc[i])

        if c1_h is None: 
            continue

        # bullish FVG if c1_high < c3_low
        if c1_h < c3_l and (c3_l - c1_h) / c2_h > min_gap_ratio:
            res.append({"type": "BULL", "low": c1_h, "high": c3_l, "mid": (c1_h + c3_l) / 2.0})
        # bearish FVG if c1_low > c3_high
        if c1_l > c3_h and (c1_l - c3_h) / c2_l > min_gap_ratio:
            res.append({"type": "BEAR", "low": c3_h, "high": c1_l, "mid": (c3_h + c1_l) / 2.0})
    return res[-8:]  # keep last few

def find_order_blocks(df: pd.DataFrame, highs: List[int], lows: List[int], lookback: int = 80) -> List[Dict[str, Any]]:
    """
    Simple OB: last opposite candle before impulse that broke structure.
    Long OB: last bearish candle before up-impulse (close > range high)
    Short OB: last bullish candle before down-impulse (close < range low)
    """
    res = []
    rng = range(max(5, len(df)-lookback), len(df)-1)
    for i in rng:
        # bullish impulse
        if df["close"].iloc[i] > df["high"].rolling(5).max().iloc[i-1]:
            # find last bearish body prior to impulse
            j = i-1
            while j >= max(0, i-10):
                o = float(df["open"].iloc[j]); c = float(df["close"].iloc[j])
                if c < o:  # bearish
                    ob_low = float(min(df["open"].iloc[j], df["close"].iloc[j], df["low"].iloc[j]))
                    ob_high = float(max(df["open"].iloc[j], df["close"].iloc[j], df["high"].iloc[j]))
                    res.append({"type":"BULL","low":ob_low,"high":ob_high,"index":j})
                    break
                j -= 1
        # bearish impulse
        if df["close"].iloc[i] < df["low"].rolling(5).min().iloc[i-1]:
            j = i-1
            while j >= max(0, i-10):
                o = float(df["open"].iloc[j]); c = float(df["close"].iloc[j])
                if c > o:  # bullish
                    ob_low = float(min(df["open"].iloc[j], df["close"].iloc[j], df["low"].iloc[j]))
                    ob_high = float(max(df["open"].iloc[j], df["close"].iloc[j], df["high"].iloc[j]))
                    res.append({"type":"BEAR","low":ob_low,"high":ob_high,"index":j})
                    break
                j -= 1
    # dedupe / keep latest
    res = sorted(res, key=lambda x: x["index"])[-6:]
    return res

def detect_apex_zone(df: pd.DataFrame, window: int = 30) -> Optional[Dict[str, float]]:
    """Apex heuristik: area konfluensi range squeeze (ATR menurun) + MA crossover rapat."""
    if len(df) < window + 20:
        return None
    sub = df.tail(window+20).reset_index(drop=True)
    atr14 = ta.volatility.AverageTrueRange(sub["high"], sub["low"], sub["close"], window=14).average_true_range().values
    ema20 = sub["close"].ewm(span=20, adjust=False).mean().values
    ema50 = sub["close"].ewm(span=50, adjust=False).mean().values
    tight = np.mean(atr14[-10:]) < np.mean(atr14[-20:-10]) * 0.8
    cross = abs(ema20[-1] - ema50[-1]) / sub["close"].iloc[-1] < 0.003
    if tight and cross:
        cen = float(sub["close"].iloc[-1])
        return {"center": cen, "low": cen * 0.996, "high": cen * 1.004}
    return None

def detect_left_shoulder(df: pd.DataFrame, lookback: int = 80) -> Optional[Dict[str, Any]]:
    """
    LS (left shoulder) heuristik: pola tiga puncak/lembah kasar.
    Untuk bearish HnS: left shoulder < head peak > right shoulder, neckline relatif datar.
    Kita return level neckline saja sebagai referensi invalidate.
    """
    if len(df) < lookback:
        return None
    sub = df.tail(lookback)
    highs_idx = sub["high"].rolling(3, center=True).apply(lambda x: x[1] == max(x), raw=True)
    peaks = [i for i,v in enumerate(highs_idx) if v==1.0]
    if len(peaks) < 3:
        return None
    # ambil 3 puncak terakhir
    p = peaks[-3:]
    h = sub["high"].values
    a,b,c = h[p[0]], h[p[1]], h[p[2]]
    if b > a*1.01 and b > c*1.01:
        # neckline kira-kira dari lows di antara puncak
        nl1 = float(sub["low"].iloc[p[0]:p[1]].min())
        nl2 = float(sub["low"].iloc[p[1]:p[2]].min())
        neckline = (nl1+nl2)/2.0
        return {"type":"HNS_BEAR","neckline":neckline,"head":float(b)}
    return None

def killzone_weight(ts_ms: int, market: str) -> float:
    """
    Weight untuk jam aktif: London/NY.
    Input open_time ms dari Binance; untuk TwelveData/AlphaVantage kita pakai indeks.
    Kita pakai UTC jam karena sumber bervariasi; heuristik saja.
    """
    try:
        dt = datetime.fromtimestamp(ts_ms/1000.0, tz=timezone.utc)
        hour = dt.hour
    except Exception:
        # fallback: bobot netral
        return 1.0
    # London 07-10 UTC, NY 12-16 UTC
    if (7 <= hour <= 10) or (12 <= hour <= 16):
        return 1.08
    return 0.98 if market == "forex" else 1.0

def rank_pois(df: pd.DataFrame, ob_list: List[Dict[str,Any]], fvg_list: List[Dict[str,Any]], sweep: Optional[str], labels: Dict[str,Any], market:str) -> List[Dict[str,Any]]:
    price = float(df["close"].iloc[-1])
    ts = int(df["open_time"].iloc[-1]) if "open_time" in df.columns else int(time.time()*1000)
    kz = killzone_weight(ts, market)

    ranked = []
    trend = labels.get("trend","sideways")
    bos = "BOS_UP" in labels.get("events",[]) or "CHOCH_UP" in labels.get("events",[])
    bos_down = "BOS_DOWN" in labels.get("events",[]) or "CHOCH_DOWN" in labels.get("events",[])

    # OBs
    for ob in ob_list:
        width = max(1e-9, ob["high"] - ob["low"])
        dist = abs(price - (ob["low"]+ob["high"])/2.0) / price
        score = 1.0 / (1.0 + dist*200) * (1.0 / (1.0 + width/price*500))
        if trend == "bullish" and ob["type"] == "BULL": score *= 1.2
        if trend == "bearish" and ob["type"] == "BEAR": score *= 1.2
        if bos and ob["type"]=="BULL": score *= 1.15
        if bos_down and ob["type"]=="BEAR": score *= 1.15
        if sweep == "SWEEP_UP" and ob["type"]=="BEAR": score *= 1.1
        if sweep == "SWEEP_DOWN" and ob["type"]=="BULL": score *= 1.1
        score *= kz
        ranked.append({"poi":"OB","side":"LONG" if ob["type"]=="BULL" else "SHORT","low":ob["low"],"high":ob["high"],"score":score})

    # FVGs
    for g in fvg_list:
        width = max(1e-9, g["high"] - g["low"])
        dist = abs(price - g["mid"]) / price
        score = 0.8 / (1.0 + dist*180) * (1.0 / (1.0 + width/price*400))
        if trend == "bullish" and g["type"]=="BULL": score *= 1.15
        if trend == "bearish" and g["type"]=="BEAR": score *= 1.15
        if sweep == "SWEEP_UP" and g["type"]=="BEAR": score *= 1.08
        if sweep == "SWEEP_DOWN" and g["type"]=="BULL": score *= 1.08
        score *= kz
        ranked.append({"poi":"FVG","side":"LONG" if g["type"]=="BULL" else "SHORT","low":g["low"],"high":g["high"],"score":score})

    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    return ranked[:6]

def refine_entry_from_poi(side: str, price: float, low: float, high: float) -> Tuple[float, float]:
    """Entry di mid zone, SL di ujung invalidation."""
    zone_mid = (low + high) / 2.0
    if side == "LONG":
        entry = max(zone_mid, low + (high - low) * 0.35)
        sl = low - (high - low) * 0.2
    else:
        entry = min(zone_mid, high - (high - low) * 0.35)
        sl = high + (high - low) * 0.2
    return float(entry), float(sl)

def build_multi_tp(side: str, entry: float, sl: float, rr_steps: List[float]) -> List[float]:
    """TPs dengan RR target bertingkat."""
    rr_unit = abs(entry - sl)
    if rr_unit <= 0:
        rr_unit = entry * 0.005
    tps = []
    for rr in rr_steps:
        if side == "LONG":
            tps.append(entry + rr_unit * rr)
        else:
            tps.append(entry - rr_unit * rr)
    return [float(round(x, 8)) for x in tps]

def smc_signal(df: pd.DataFrame, pair: Optional[str], timeframe: Optional[str]) -> Dict[str, Any]:
    """
    Pipeline FULL SMC:
    - swing points, BOS/CHOCH, sweep
    - OB, FVG, POI ranking
    - apex + LS heuristics (info tambahan)
    - entry refinement + multi-TP
    - fusion dengan TA dasar untuk confidence
    """
    if df.shape[0] < 50:
        return {"error": "data_tidak_cukup", "message": "Perlu >= 50 candle"}

    # TA baseline
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)
    price = float(df["close"].iloc[-1])

    highs, lows = swing_points(df, lookback=3)
    labels = label_structure(df, highs, lows)
    sweep = detect_liquidity_sweep(df, window=40)
    fvg = find_fvg(df, max_lookback=80)
    obs = find_order_blocks(df, highs, lows, lookback=120)
    apex = detect_apex_zone(df, window=40)
    ls = detect_left_shoulder(df, lookback=120)

    market = detect_market(pair or "")
    pois = rank_pois(df, obs, fvg, sweep, labels, market)

    # default WAIT jika tidak ada POI
    if not pois:
        return {
            "pair": pair or "",
            "timeframe": timeframe or "",
            "signal_type": "WAIT",
            "entry": price,
            "tp1": None, "tp2": None, "tp3": None, "tp4": None, "tp5": None,
            "sl": None,
            "confidence": 0.3,
            "reasoning": "Tidak ada POI valid. Tunggu setup bersih.",
            "smc": {"trend": labels.get("trend"), "events": labels.get("events"), "sweep": sweep, "apex": apex, "ls": ls, "pois": pois}
        }

    top = pois[0]  # pilih POI terbaik
    side = "LONG" if top["side"] == "LONG" else "SHORT"
    entry, sl = refine_entry_from_poi(side, price, low=top["low"], high=top["high"])

    # RR dan multi TP bertingkat
    rr_steps = [1.0, 2.0, 3.0, 4.0, 5.0]
    tps = build_multi_tp(side, entry, sl, rr_steps)
    tp1, tp2, tp3, tp4, tp5 = tps[:5]

    # baseline confidence
    confs = []
    # trend-konfluensi
    trend = labels.get("trend")
    if side == "LONG" and trend == "bullish":
        confs.append(0.9)
    elif side == "SHORT" and trend == "bearish":
        confs.append(0.9)
    else:
        confs.append(0.55)
    # EMA konfluensi
    ema_ok = df["ema20"].iloc[-1] > df["ema50"].iloc[-1]
    if side == "LONG" and ema_ok:
        confs.append(0.85)
    elif side == "SHORT" and not ema_ok:
        confs.append(0.85)
    else:
        confs.append(0.6)
    # RSI equilibrium
    rsi_now = float(df["rsi14"].iloc[-1])
    if 35 < rsi_now < 70:
        confs.append(0.8)
    else:
        confs.append(0.6)
    # sweep boost
    if (sweep == "SWEEP_DOWN" and side == "LONG") or (sweep == "SWEEP_UP" and side == "SHORT"):
        confs.append(0.85)
    # apex presence -> kemungkinan impuls breakout
    if apex is not None:
        confs.append(0.75)
    # LS sebagai heads-up pembalikan (turunkan sedikit jika kontra)
    if ls and ls.get("type") == "HNS_BEAR" and side == "LONG":
        confs.append(0.6)

    confidence = round(float(sum(confs) / len(confs)), 3)

    # reasoning ringkas
    reasons = []
    reasons.append(f"POI {top['poi']} {side}")
    if "events" in labels and labels["events"]:
        reasons.append("Structure: " + ",".join(labels["events"]))
    if sweep:
        reasons.append(f"Sweep: {sweep}")
    reasons.append(f"Trend: {trend}")
    reasons.append("EMA20/50 confirm" if (side=="LONG" and ema_ok) or (side=="SHORT" and not ema_ok) else "EMA diverge")
    if apex: reasons.append("Apex squeeze")
    if ls: reasons.append("LS pattern caution")
    reasoning = " | ".join(reasons)

    return {
        "pair": pair or "",
        "timeframe": timeframe or "",
        "signal_type": side,
        "entry": round(float(entry), 8),
        "tp1": tp1, "tp2": tp2, "tp3": tp3, "tp4": tp4, "tp5": tp5,
        "sl": round(float(sl), 8),
        "confidence": confidence,
        "reasoning": reasoning,
        "smc": {
            "trend": trend,
            "events": labels.get("events"),
            "sweep": sweep,
            "apex": apex,
            "ls": ls,
            "pois": pois
        }
    }

# ---------------- LEARNING ----------------
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
        "ema8_21_diff": float((ema8 - ema21) / (entry if entry != 0 else 1)),
        "rsi14": float(rsi14),
        "atr_rel": float(atr14) / (entry if entry != 0 else 1),
        "vol_spike": float(vol_spike),
        "dist_to_high": float(dist_to_high),
        "dist_to_low": float(dist_to_low),
        "rr": float(rr)
    }

def build_dataset_from_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        return None, None
    df = pd.read_csv(TRADE_LOG_FILE)
    rows, labels = [], []
    for _, r in df.iterrows():
        hit = str(r.get("backtest_hit", "")).upper()
        label = 1 if hit.startswith("TP") else 0
        try:
            feats = compute_features_for_row(
                str(r.get('pair',"")),
                str(r.get('timeframe',"15m")),
                float(r.get('entry',0)),
                r.get('tp1', None),
                float(r.get('sl',0))
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
    global _last_retrain_count, _cached_model
    X, y = build_dataset_from_trade_log()
    if X is None or len(y) < MIN_SAMPLES_TO_TRAIN:
        return {"status": "data_tidak_cukup", "samples": 0 if y is None else len(y)}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=1, max_depth=None)
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
    return {"status": "trained", "samples": int(len(y)), "auc": auc, "report": report}

def predict_with_model(payload: Dict[str, Any]):
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
    def worker():
        try:
            res = train_and_save_model()
            print("Retrain result:", res)
        except Exception as e:
            print("Retrain error:", e)
    t = threading.Thread(target=worker, daemon=True)
    t.start()

def check_and_trigger_retrain_if_needed():
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

# ---------------- SENTIMENT ----------------
def get_crypto_sentiment():
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
    out = {"dxy": None, "vix": None, "snp_change": None, "source": []}
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
        final = 0.7 * tech + 0.3 * 0.5
    return round(max(0.0, min(1.0, final)), 3)

def _postprocess_with_learning(signal: Dict[str, Any]) -> Dict[str, Any]:
    try:
        market = detect_market(signal.get("pair", ""))
        crypto_sent = get_crypto_sentiment() if market == "crypto" else None
        macro_sent = get_macro_sentiment() if market == "forex" else None

        model_prob = None
        if os.path.exists(MODEL_FILE) and signal.get("sl") is not None:
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
            fused = round(max(0.0, min(1.0, 0.82 * fused + 0.18 * model_prob)), 3)
        signal["confidence"] = fused
        signal["market_mode"] = market
        signal["sentiment"] = {"crypto": crypto_sent, "macro": macro_sent}
        if model_prob is not None and model_prob < 0.22:
            signal["vetoed_by_model"] = True
            signal["signal_type"] = "WAIT"
        else:
            signal["vetoed_by_model"] = False
    except Exception as e:
        signal["postprocess_error"] = str(e)
    return signal

# ---------------- IMAGE OCR HELPERS (best effort) ----------------
def ocr_y_axis_prices(img_cv):
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
        # heuristik open/close
        bullish = ch < (plot.shape[0] * 0.15)  # dummy; warna sulit konsisten
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
    lastc = df["close"].iloc[-1] if not df["close"].isna().all() else 1.0
    if lastc != 0:
        df = df / lastc
    return df.tail(max_bars).reset_index(drop=True)

# ---------------- ENDPOINTS ----------------
@app.get("/health")
def health():
    return respond({"status": "ok", "service": "Pro Trader AI - FULL SMC + Learning"})

def fetch_generic(pair: str, tf: str, limit: int=300) -> pd.DataFrame:
    df = fetch_ohlc_binance(pair, tf, limit=limit)
    # ensure dtype
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)

def _maybe_backtest_and_log(res: Dict[str,Any], auto_log: bool):
    if not auto_log:
        return res
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
        "tp3": res.get("tp3"), "tp4": res.get("tp4"), "tp5": res.get("tp5"),
        "sl": res["sl"], "confidence": res["confidence"], "reasoning": res["reasoning"],
        "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None,
        "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None
    }
    append_trade_log(logrec)
    check_and_trigger_retrain_if_needed()
    return res

@app.get("/pro_signal")
def pro_signal(
    pair: str = Query(...),
    tf_main: str = Query("1h"),
    tf_entry: str = Query("15m"),
    limit: int = Query(300),
    auto_log: bool = Query(False)
):
    try:
        # gunakan TF entry untuk eksekusi SMC
        df_entry = fetch_generic(pair, tf_entry, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch_failed: {e}")

    res = smc_signal(df_entry, pair=pair, timeframe=tf_entry)

    # Tambahkan konteks TF utama (arah HTF)
    try:
        df_main = fetch_generic(pair, tf_main, 200)
        ema20_main = float(ema(df_main['close'], 20).iloc[-1])
        ema50_main = float(ema(df_main['close'], 50).iloc[-1])
        res['context_main_trend'] = "bullish" if ema20_main > ema50_main else "bearish"
        if res["signal_type"] == "LONG" and res['context_main_trend']=="bullish":
            res["confidence"] = round(min(1.0, res["confidence"] + 0.05), 3)
        if res["signal_type"] == "SHORT" and res['context_main_trend']=="bearish":
            res["confidence"] = round(min(1.0, res["confidence"] + 0.05), 3)
    except:
        pass

    res = _postprocess_with_learning(res)
    res = _maybe_backtest_and_log(res, auto_log)
    return respond(res)

@app.get("/scalp_signal")
def scalp_signal(pair: str = Query(...), tf: str = Query("3m"), limit: int = Query(300), auto_log: bool = Query(False)):
    df = fetch_generic(pair, tf, limit)
    # gunakan SMC juga untuk scalp (lebih agresif)
    res = smc_signal(df, pair=pair, timeframe=tf)
    res = _postprocess_with_learning(res)
    res = _maybe_backtest_and_log(res, auto_log)
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

    df_ohlc = detect_candles_from_plot(img_cv, y_map, max_bars=240)
    if df_ohlc.empty:
        raise HTTPException(status_code=400, detail="gagal_membaca_chart")

    for col in ['open', 'high', 'low', 'close']:
        df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors='coerce')
    df_ohlc = df_ohlc.dropna().reset_index(drop=True)

    res = smc_signal(df_ohlc, pair=pair or "IMG", timeframe=timeframe or "img")
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
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"),
            "tp3": res.get("tp3"), "tp4": res.get("tp4"), "tp5": res.get("tp5"),
            "sl": res["sl"], "confidence": res["confidence"], "reasoning": res["reasoning"],
            "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None,
            "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None
        })
        check_and_trigger_retrain_if_needed()

    res['bars_used'] = int(df_ohlc.shape[0])
    return respond(res)

@app.post("/analyze_csv")
def analyze_csv(
    file: UploadFile = File(...),
    pair: Optional[str] = Form(None),
    timeframe: Optional[str] = Form(None),
    auto_backtest: Optional[str] = Form("true"),
    auto_log: Optional[str] = Form("true")
):
    auto_bt = auto_backtest.lower() != "false"
    auto_lg = auto_log.lower() != "false"

    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_csv: {e}")

    df.columns = [c.strip().lower() for c in df.columns]
    def find_col(k): return next((c for c in df.columns if k in c), None)
    o, h, l, ccol = find_col('open'), find_col('high'), find_col('low'), find_col('close')
    if not all([o, h, l, ccol]):
        raise HTTPException(status_code=400, detail="kolom_tidak_lengkap (butuh open, high, low, close)")

    df2 = df[[o, h, l, ccol]].rename(columns={o: 'open', h: 'high', l: 'low', ccol: 'close'})
    for col in ['open', 'high', 'low', 'close']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2 = df2.dropna().reset_index(drop=True)

    res = smc_signal(df2, pair=pair or "CSV", timeframe=timeframe or "csv")
    res = _postprocess_with_learning(res)

    bt_res = {}
    if auto_bt and res.get("signal_type") != "WAIT":
        bt_payload = {
            "pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"],
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"),
            "sl": res["sl"], "confidence": res["confidence"], "reason": res["reasoning"]
        }
        bt_res = post_to_backtester(bt_payload)
        res["backtest_raw"] = bt_res

    if auto_lg:
        backtest_hit = bt_res.get("hit") or bt_res.get("result") or bt_res.get("outcome")
        backtest_pnl = bt_res.get("pnl_total") or bt_res.get("pnl") or bt_res.get("profit")
        append_trade_log({
            "pair": res["pair"],
            "timeframe": res["timeframe"],
            "signal_type": res["signal_type"],
            "entry": res["entry"],
            "tp1": res.get("tp1"), "tp2": res.get("tp2"),
            "tp3": res.get("tp3"), "tp4": res.get("tp4"), "tp5": res.get("tp5"),
            "sl": res["sl"],
            "confidence": res["confidence"],
            "reasoning": res["reasoning"],
            "backtest_hit": backtest_hit,
            "backtest_pnl": backtest_pnl
        })
        check_and_trigger_retrain_if_needed()

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
            "tp3": last.get("tp3", ""),
            "tp4": last.get("tp4", ""),
            "tp5": last.get("tp5", ""),
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
            "total_signals": int(total),
            "tp_hits": int(tp_hits),
            "sl_hits": int(sl_hits),
            "winrate": winrate,
            "avg_confidence": avg_conf,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "pair_stats": pair_stats,
            "tf_stats": tf_stats,
            "model_status": "Sudah Dilatih" if model_exists else "Belum Ada Model"
        }
        return respond(data)
    except Exception as e:
        return respond({"error": str(e)})

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

# Jalankan:
# uvicorn main_combined_learning:app --host 0.0.0.0 --port $PORT
