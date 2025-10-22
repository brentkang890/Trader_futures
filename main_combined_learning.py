# main_protrader.py
"""
Pro Trader AI — single-file FastAPI service
- Hybrid data source: Binance (crypto) with AlphaVantage fallback (forex)
- Hybrid strategy (hybrid_analyze + scalp_engine)
- Chart OCR (OpenCV + pytesseract) best-effort
- Logging to CSV, RandomForest learning & cache, retrain trigger
- Debug endpoints for model & logs
"""

import os
import io
import re
import time
import math
import json
import joblib
import logging
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

# technical libs
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# image libs
from PIL import Image
import cv2
try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    pytesseract = None
    _HAS_TESSERACT = False

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("protrader")

app = FastAPI(title="ProTraderAI", version="1.0")

# ---------------- CONFIG ----------------
TRADE_LOG_FILE = os.environ.get("TRADE_LOG_FILE", "trade_log.csv")
MODEL_FILE = os.environ.get("MODEL_FILE", "rf_model.pkl")
MIN_SAMPLES_TO_TRAIN = int(os.environ.get("MIN_SAMPLES_TO_TRAIN", 50))
N_SIGNALS_TO_RETRAIN = int(os.environ.get("N_SIGNALS_TO_RETRAIN", 50))
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

ALPHA_API_KEY = os.environ.get("ALPHA_API_KEY", "")
ALPHA_URL = "https://www.alphavantage.co/query"
BACKTEST_URL = os.environ.get("BACKTEST_URL", "")

# cache / threading
_lock = threading.Lock()
_last_retrain_count = 0
_cached_model = None

# ---------------- UTIL & DATA FETCH ----------------
def fetch_ohlc_alpha_forex(symbol: str, interval: str="15m", limit:int=500) -> pd.DataFrame:
    if not ALPHA_API_KEY:
        raise RuntimeError("ALPHA_API_KEY_not_set")
    symbol = symbol.upper()
    # expect symbol like EURUSD or XAUUSD where first 3 = from, last 3 = to (works for many)
    if len(symbol) < 6:
        raise RuntimeError("symbol_format_invalid_for_alpha")
    from_sym = symbol[:3]
    to_sym = symbol[3:]
    mapping = {"1m":"1min","3m":"5min","5m":"5min","15m":"15min","30m":"30min","1h":"60min","4h":"60min","1d":"daily"}
    iv = mapping.get(interval, "15min")
    params = {"function": "FX_INTRADAY", "from_symbol": from_sym, "to_symbol": to_sym,
              "interval": iv, "apikey": ALPHA_API_KEY, "outputsize":"compact"}
    r = requests.get(ALPHA_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    # find time series key
    ts_key = next((k for k in data.keys() if "Time Series" in k), None)
    if not ts_key:
        # fallback: maybe daily/time series for FX not available
        raise RuntimeError(f"Alpha no data or limit reached: {data}")
    ts = data[ts_key]
    df = pd.DataFrame(ts).T
    # columns format: '1. open' etc -> normalize
    df.columns = [c.split('. ')[-1] for c in df.columns]
    df = df.rename(columns=lambda c: c.strip())
    for c in ['open','high','low','close']:
        if c not in df.columns:
            raise RuntimeError("alpha_missing_ohlc_columns")
    df = df[['open','high','low','close']].astype(float)
    df = df.sort_index().tail(limit).reset_index(drop=True)
    df['volume'] = 0.0
    df.insert(0, 'open_time', pd.RangeIndex(start=0, stop=len(df)))
    return df[['open_time','open','high','low','close','volume']]

def fetch_ohlc_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    symbol = symbol.upper()
    # try Binance (crypto)
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
            return df[["open_time","open","high","low","close","volume"]].reset_index(drop=True)
    except Exception as e:
        logger.debug("Binance fetch failed: %s", e)
    # fallback to Alpha for forex
    try:
        return fetch_ohlc_alpha_forex(symbol, interval, limit)
    except Exception as e:
        raise RuntimeError(f"fetch_failed_for_{symbol}: {e}")

# ---------------- INDICATORS & STRATEGIES ----------------
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
        conf.extend([0.9 if trend=="bullish" else 0.6,
                     0.9 if price >= fib_618 else 0.65,
                     1.0 if 30 < rsi_now < 75 else 0.5])
        signal="LONG"
    elif bos == "BOS_DOWN" or (trend == "bearish" and price < ema20):
        entry = price
        sl = recent_high + atr_now*0.6
        rr = sl - entry if sl>entry else price*0.01
        tp1 = entry - rr*1.5
        tp2 = entry - rr*2.5
        reasons.append("Bias SHORT — BOS turun & EMA searah bearish.")
        conf.extend([0.9 if trend=="bearish" else 0.6,
                     0.9 if price <= fib_618 else 0.65,
                     1.0 if 25 < rsi_now < 70 else 0.5])
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
    vol_now = float(last['volume']) if 'volume' in last else 0.0
    vol_spike = vol_now > (vol_mean * 1.8 if vol_mean > 0 else False)

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

# ---------------- LOGGING & LEARNING ----------------
def ensure_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        df = pd.DataFrame(columns=[
            "id","timestamp","pair","timeframe","signal_type",
            "entry","tp1","tp2","sl","confidence","reasoning",
            "backtest_hit","backtest_pnl"
        ])
        df.to_csv(TRADE_LOG_FILE, index=False)

def append_trade_log(record: Dict[str,Any]) -> int:
    ensure_trade_log()
    df = pd.read_csv(TRADE_LOG_FILE)
    next_id = int(df['id'].max()) + 1 if not df.empty else 1
    record_row = {
        "id": next_id,
        "timestamp": datetime.utcnow().isoformat(),
        "pair": record.get("pair"),
        "timeframe": record.get("timeframe"),
        "signal_type": record.get("signal_type"),
        "entry": record.get("entry"),
        "tp1": record.get("tp1"),
        "tp2": record.get("tp2"),
        "sl": record.get("sl"),
        "confidence": record.get("confidence"),
        "reasoning": record.get("reasoning"),
        "backtest_hit": record.get("backtest_hit"),
        "backtest_pnl": record.get("backtest_pnl")
    }
    df = pd.concat([df, pd.DataFrame([record_row])], ignore_index=True)
    df.to_csv(TRADE_LOG_FILE, index=False)
    return next_id

def post_to_backtester(payload: Dict[str,Any]) -> Dict[str,Any]:
    if not BACKTEST_URL:
        return {"error":"BACKTEST_URL_not_configured"}
    try:
        r = requests.post(BACKTEST_URL, json=payload, timeout=15)
        try:
            return r.json()
        except Exception:
            return {"status_code": r.status_code, "text": r.text}
    except Exception as e:
        return {"error":"backtester_unreachable", "detail": str(e)}

# ---------------- OCR helpers ----------------
def ocr_y_axis_prices(img_cv):
    if not _HAS_TESSERACT:
        return {}
    h, w = img_cv.shape[:2]
    # crop right side region where y-axis numbers usually are
    crop = img_cv[int(h*0.02):int(h*0.98), int(w*0.72):w].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 3)
    config = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.,"
    txt = pytesseract.image_to_string(th, config=config)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    vals = []
    for ln in lines:
        s = ln.replace(",", ".").replace(" ", "")
        m = re.findall(r"[\d\.]+", s)
        if m:
            try:
                vals.append(float(m[0]))
            except:
                pass
    if not vals:
        return {}
    ys = np.linspace(int(h*0.02), int(h*0.98), num=len(vals)).astype(int).tolist()
    prices = vals
    return {int(y): float(p) for y,p in zip(ys, prices)}

def detect_candles_from_plot(img_cv, y_map, max_bars=200):
    h,w,_ = img_cv.shape
    plot = img_cv[int(h*0.06):int(h*0.94), int(w*0.06):int(w*0.94)].copy()
    hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 40]); upper_red1 = np.array([12,255,255])
    lower_red2 = np.array([160,40,40]); upper_red2 = np.array([180,255,255])
    lower_green = np.array([35,40,40]); upper_green = np.array([95,255,255])

    mask_r = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                            cv2.inRange(hsv, lower_red2, upper_red2))
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(mask_r, mask_g)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candles = []
    ys = sorted(y_map.keys()) if y_map else None
    ps = [y_map[y] for y in ys] if ys else None

    for c in contours:
        x,y,cw,ch = cv2.boundingRect(c)
        if cw < 4 or ch < 6:
            continue
        global_y_top = y + int(h*0.06)
        global_y_bot = y+ch + int(h*0.06)
        if ys and len(ys)>1:
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
    if not ys or len(ys)<=1:
        lastc = df["close"].iloc[-1] if not df["close"].isna().all() else 1.0
        df = df / lastc
    return df.tail(max_bars).reset_index(drop=True)

# ---------------- FEATURE ENGINEERING & MODEL ----------------
def compute_features_for_row(pair: str, timeframe: str, entry: float, tp: Optional[float], sl: float) -> Optional[Dict[str,float]]:
    try:
        kdf = fetch_ohlc_binance(pair, timeframe, limit=200)
    except Exception as e:
        logger.debug("compute_features: fetch failed: %s", e)
        return None
    kdf = kdf.tail(60).reset_index(drop=True)
    close = kdf['close'].astype(float)
    high = kdf['high'].astype(float)
    low = kdf['low'].astype(float)
    vol = kdf['volume'].astype(float) if 'volume' in kdf else pd.Series([0.0]*len(kdf))

    ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    rsi14 = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    atr14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    vol_mean = vol.tail(40).mean() if len(vol) >= 40 else vol.mean()
    vol_now = vol.iloc[-1] if len(vol)>0 else 0.0
    vol_spike = 1.0 if vol_now > vol_mean * 1.8 else 0.0

    recent_high = high.tail(80).max()
    recent_low = low.tail(80).min()
    dist_to_high = (recent_high - entry) / (entry if entry!=0 else 1)
    dist_to_low = (entry - recent_low) / (entry if entry!=0 else 1)
    rr = abs((tp - entry) / (entry - sl)) if (tp is not None and (entry - sl) != 0) else 0.0

    return {
        "ema8_21_diff": (ema8 - ema21) / (entry if entry!=0 else 1),
        "rsi14": float(rsi14),
        "atr_rel": float(atr14) / (entry if entry!=0 else 1),
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
        hit = str(r.get("backtest_hit","")).upper()
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
                str(r['timeframe'] or "15m"),
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
    global _last_retrain_count, _cached_model
    X, y = build_dataset_from_trade_log()
    if X is None or len(y) < MIN_SAMPLES_TO_TRAIN:
        return {"status":"data_tidak_cukup", "samples": 0 if y is None else len(y)}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    yprob = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else None
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, yprob) if yprob is not None else None
    joblib.dump({"clf": clf, "features": list(X.columns)}, MODEL_FILE)
    _cached_model = {"clf": clf, "features": list(X.columns)}
    try:
        df = pd.read_csv(TRADE_LOG_FILE)
        _last_retrain_count = len(df)
    except:
        _last_retrain_count = 0
    return {"status":"trained", "samples": len(y), "auc": auc, "report": report}

def predict_with_model(payload: Dict[str,Any]):
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
    prob = float(clf.predict_proba(X)[:,1][0]) if hasattr(clf, "predict_proba") else float(clf.predict(X)[0])
    return {"prob": prob, "features": feats}

def maybe_trigger_retrain_background():
    def worker():
        try:
            res = train_and_save_model()
            logger.info("Retrain result: %s", res)
        except Exception as e:
            logger.exception("Retrain error: %s", e)
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
            logger.exception("check_retrain error: %s", e)

# ---------------- POSTPROCESS / MODEL INTEGRATION ----------------
def _postprocess_with_learning(signal: Dict[str,Any]) -> Dict[str,Any]:
    try:
        if os.path.exists(MODEL_FILE):
            pred = predict_with_model({
                "pair": signal.get("pair"),
                "timeframe": signal.get("timeframe"),
                "entry": signal.get("entry"),
                "tp": signal.get("tp1"),
                "sl": signal.get("sl")
            })
            prob = pred.get("prob", 0.0)
            orig = float(signal.get("confidence", 0.5))
            new_conf = round(max(min(1.0, 0.9 * orig + 0.1 * prob), 0.0), 3)
            signal["confidence"] = new_conf
            signal["model_prob"] = round(prob, 3)
            if prob < 0.35:
                signal["vetoed_by_model"] = True
                signal["signal_type"] = "WAIT"
            else:
                signal["vetoed_by_model"] = False
        else:
            signal["model_prob"] = None
            signal["vetoed_by_model"] = False
    except Exception as e:
        signal["model_error"] = str(e)
    return signal

# ---------------- FASTAPI endpoints ----------------
@app.get("/health")
def health():
    return {"status":"ok","service":"ProTraderAI"}

@app.get("/pro_signal")
def pro_signal(pair: str = Query(...), tf_main: str = Query("1h"), tf_entry: str = Query("15m"),
               limit: int = Query(300), auto_log: bool = Query(False)):
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
    except Exception:
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

    return JSONResponse(res)

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

    return JSONResponse(res)

@app.post("/analyze_chart")
def analyze_chart(file: UploadFile = File(...), pair: Optional[str] = Form(None),
                  timeframe: Optional[str] = Form(None), auto_backtest: Optional[str] = Form("true")):
    auto_flag = auto_backtest.lower() != "false"
    try:
        contents = file.file.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.exception("analyze_chart: invalid image: %s", e)
        raise HTTPException(status_code=400, detail=f"gambar_tidak_valid: {e}")

    if not _HAS_TESSERACT:
        logger.warning("Tesseract not available - OCR disabled")
        # we continue without OCR; detect_candles_from_plot will fallback
    y_map = {}
    try:
        if _HAS_TESSERACT:
            y_map = ocr_y_axis_prices(img_cv)
    except Exception as e:
        logger.warning("ocr_y_axis_prices failed: %s", e)
        y_map = {}

    try:
        df_ohlc = detect_candles_from_plot(img_cv, y_map, max_bars=200)
    except Exception as e:
        logger.exception("detect_candles_from_plot failed: %s", e)
        raise HTTPException(status_code=500, detail=f"gagal_mendeteksi_candles: {e}")

    if df_ohlc.empty:
        raise HTTPException(status_code=400, detail="gagal_membaca_chart")

    for col in ['open','high','low','close']:
        df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors='coerce')
    df_ohlc = df_ohlc.dropna().reset_index(drop=True)
    if df_ohlc.shape[0] < 12:
        raise HTTPException(status_code=400, detail="data_tidak_cukup_dari_gambar")

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
    return JSONResponse(res)

@app.post("/analyze_csv")
def analyze_csv(file: UploadFile = File(...), pair: Optional[str] = Form(None), timeframe: Optional[str] = Form(None)):
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_csv: {e}")
    df.columns = [c.strip().lower() for c in df.columns]
    def find_col(k): return next((c for c in df.columns if k in c), None)
    o,h,l,ccol = find_col('open'), find_col('high'), find_col('low'), find_col('close')
    if not all([o,h,l,ccol]):
        raise HTTPException(status_code=400, detail="kolom_tidak_lengkap (butuh open,high,low,close)")
    df2 = df[[o,h,l,ccol]].rename(columns={o:'open',h:'high',l:'low',ccol:'close'})
    for col in ['open','high','low','close']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2 = df2.dropna().reset_index(drop=True)
    if df2.shape[0] < 12:
        raise HTTPException(status_code=400, detail="data_csv_tidak_cukup")
    res = hybrid_analyze(df2, pair=pair or "CSV", timeframe=timeframe or "csv")
    res = _postprocess_with_learning(res)
    return JSONResponse(res)

@app.get("/learning_status")
def learning_status():
    info = {"model_exists": os.path.exists(MODEL_FILE)}
    if info["model_exists"]:
        try:
            mod = joblib.load(MODEL_FILE)
            info["features"] = mod.get("features")
        except Exception as e:
            info["features"] = None
            info["model_load_error"] = str(e)
    try:
        df = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else pd.DataFrame()
        info["trade_log_count"] = len(df)
    except:
        info["trade_log_count"] = 0
    return JSONResponse(info)

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
    return JSONResponse(info)

@app.get("/retrain_learning")
def retrain_learning():
    res = train_and_save_model()
    return JSONResponse(res)

@app.get("/logs")
def get_logs(limit: int = Query(100)):
    ensure_trade_log()
    df = pd.read_csv(TRADE_LOG_FILE)
    df = df.tail(limit).to_dict(orient="records")
    return JSONResponse({"logs": df})

@app.get("/logs_summary")
def logs_summary():
    try:
        if not os.path.exists(TRADE_LOG_FILE):
            return JSONResponse({"detail":"Belum ada log sinyal tersimpan."})
        df = pd.read_csv(TRADE_LOG_FILE)
        if df.empty:
            return JSONResponse({"detail":"Belum ada data sinyal terbaru."})
        last = df.iloc[-1]
        data = {
            "pair": last.get("pair",""),
            "timeframe": last.get("timeframe",""),
            "signal_type": last.get("signal_type",""),
            "entry": last.get("entry",""),
            "tp1": last.get("tp1",""),
            "tp2": last.get("tp2",""),
            "sl": last.get("sl",""),
            "confidence": last.get("confidence",""),
            "reasoning": last.get("reasoning","")
        }
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.get("/download_logs")
def download_logs():
    ensure_trade_log()
    return FileResponse(TRADE_LOG_FILE, media_type="text/csv", filename="trade_log.csv")

@app.get("/ai_performance")
def ai_performance():
    try:
        ensure_trade_log()
        df = pd.read_csv(TRADE_LOG_FILE)
        if df.empty:
            return JSONResponse({"error":"Belum ada data sinyal untuk dianalisis."})
        total = len(df)
        tp_hits = df["backtest_hit"].astype(str).str.upper().str.startswith("TP").sum()
        sl_hits = df["backtest_hit"].astype(str).str.upper().str.startswith("SL").sum()
        winrate = round((tp_hits / total) * 100, 2) if total > 0 else 0
        avg_conf = round(pd.to_numeric(df.get("confidence", pd.Series([])), errors="coerce").mean(), 3)
        pnl_values = pd.to_numeric(df.get("backtest_pnl", pd.Series([])), errors="coerce").dropna()
        total_pnl = float(pnl_values.sum()) if not pnl_values.empty else 0.0
        profit_factor = None
        if not pnl_values.empty and (pnl_values < 0).any():
            prof = pnl_values[pnl_values>0].sum()
            loss = abs(pnl_values[pnl_values<0].sum())
            profit_factor = round(prof / loss, 2) if loss!=0 else None
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
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)})

# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup_event():
    ensure_trade_log()
    global _cached_model
    if os.path.exists(MODEL_FILE):
        try:
            _cached_model = joblib.load(MODEL_FILE)
            logger.info("Loaded cached model on startup.")
        except Exception as e:
            logger.exception("Failed load cached model on startup: %s", e)

# Note: run with uvicorn main_protrader:app --host 0.0.0.0 --port <port>
