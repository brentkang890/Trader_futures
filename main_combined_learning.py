# main_combined_learning_fix_safe_v1.py
"""
Pro Trader AI - Combined + Learning (Bahasa Indonesia)
Versi: fix_safe_v1 (stabil dan tahan error)
- Analisis Crypto (Binance) & Forex (AlphaVantage)
- Auto-logging sinyal ke trade_log.csv
- Pembelajaran AI (RandomForest) terintegrasi + retrain otomatis
- OCR Chart dari Gambar (dengan fallback aman)
- 100% sinkron dengan telegram_bot.py
"""

import os
import io
import math
import json
import threading
import time
import re
from datetime import datetime
from typing import Optional, Dict, Any

import requests
import pandas as pd
import numpy as np

from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

# Technical Analysis & ML
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Image Processing
from PIL import Image
import cv2
try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False
    pytesseract = None

app = FastAPI(
    title="Pro Trader AI - Combined + Learning (ID)",
    description="Analisis Crypto (Binance) & Forex (AlphaVantage) + Pembelajaran Terintegrasi",
    version="1.0"
)

# ---------------- KONFIG ----------------
BACKTEST_URL = os.environ.get("BACKTEST_URL")
TRADE_LOG_FILE = "trade_log.csv"
MODEL_FILE = "rf_model.pkl"
MIN_SAMPLES_TO_TRAIN = int(os.environ.get("MIN_SAMPLES_TO_TRAIN", 50))
N_SIGNALS_TO_RETRAIN = int(os.environ.get("N_SIGNALS_TO_RETRAIN", 50))
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

ALPHA_API_KEY = os.environ.get("ALPHA_API_KEY", "")
ALPHA_URL = "https://www.alphavantage.co/query"

_lock = threading.Lock()
_last_retrain_count = 0
_cached_model = None  # Cache model yang sudah dimuat dari disk

# ---------------- UTILITAS & INDIKATOR ----------------
def fetch_ohlc_alpha_forex(symbol: str, interval: str="15m", limit: int=500) -> pd.DataFrame:
    """Ambil data Forex dari AlphaVantage"""
    if not ALPHA_API_KEY:
        raise RuntimeError("ALPHA_API_KEY_not_set")
    symbol = symbol.upper()
    from_sym = symbol[:3]
    to_sym = symbol[3:]
    mapping = {
        "1m": "1min", "3m": "5min", "5m": "5min",
        "15m": "15min", "30m": "30min",
        "1h": "60min", "4h": "60min", "1d": "daily"
    }
    iv = mapping.get(interval, "15min")
    params = {
        "function": "FX_INTRADAY", "from_symbol": from_sym, "to_symbol": to_sym,
        "interval": iv, "apikey": ALPHA_API_KEY, "outputsize": "compact"
    }
    r = requests.get(ALPHA_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    keys = [k for k in data.keys() if "Time Series" in k]
    if not keys:
        raise ValueError(f"AlphaVantage no data: {data}")
    key = keys[0]
    ts = data[key]
    df = pd.DataFrame(ts).T
    df.columns = [c.split('. ')[-1] for c in df.columns]
    df = df.rename(columns=lambda c: c.strip())
    df = df[["open", "high", "low", "close"]].astype(float)
    df = df.sort_index().tail(limit).reset_index(drop=True)
    df["volume"] = 0.0
    df.insert(0, "open_time", pd.RangeIndex(start=0, stop=len(df)))
    return df

def fetch_ohlc_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Ambil data dari Binance, fallback ke AlphaVantage jika gagal"""
    symbol = symbol.upper()
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(BINANCE_KLINES, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "num_trades", "tb_base", "tb_quote", "ignore"
            ])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[["open_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
            return df
    except Exception:
        pass

    # fallback AlphaVantage
    return fetch_ohlc_alpha_forex(symbol, interval, limit)

# ---------------- INDIKATOR ----------------
def ema(series: pd.Series, n: int):
    return ta.trend.EMAIndicator(series, window=n).ema_indicator()

def rsi(series: pd.Series, n: int=14):
    return ta.momentum.RSIIndicator(series, window=n).rsi()

def atr(df: pd.DataFrame, n: int=14):
    return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=n).average_true_range()

def detect_sr(df: pd.DataFrame, lookback:int=120):
    """Deteksi support dan resistance terakhir"""
    return float(df['high'].tail(lookback).max()), float(df['low'].tail(lookback).min())

def breakout_of_structure(df: pd.DataFrame, window:int=20):
    """Deteksi breakout BOS naik/turun"""
    if df.shape[0] < window + 2:
        return None
    high_sw = df['high'].rolling(window).max().iloc[-2]
    low_sw = df['low'].rolling(window).min().iloc[-2]
    last, prev = df['close'].iloc[-1], df['close'].iloc[-2]
    if prev <= high_sw and last > high_sw:
        return "BOS_UP"
    if prev >= low_sw and last < low_sw:
        return "BOS_DOWN"
    return None
    
    # ---------------- STRATEGI HYBRID ----------------
def hybrid_analyze(df: pd.DataFrame, pair: Optional[str] = None, timeframe: Optional[str] = None) -> dict:
    """Analisis utama: gabungan trend EMA, RSI, ATR, BOS, dan Fibo"""
    df = df.copy().dropna().reset_index(drop=True)
    if df.shape[0] < 12:
        return {"error": "data_tidak_cukup", "message": "Perlu minimal 12 candle untuk analisis."}

    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['rsi14'] = rsi(df['close'], 14)
    df['atr14'] = atr(df, 14)

    last = df.iloc[-1]
    price, ema20, ema50 = float(last['close']), float(last['ema20']), float(last['ema50'])
    rsi_now = float(last['rsi14'])
    atr_now = float(last['atr14']) if not np.isnan(last['atr14']) else price * 0.001

    recent_high, recent_low = detect_sr(df, 120)
    bos = breakout_of_structure(df, 20)

    swing_high = df['high'].tail(80).max()
    swing_low = df['low'].tail(80).min()
    diff = swing_high - swing_low
    fib_618 = swing_high - diff * 0.618 if diff > 0 else price

    trend = "bullish" if ema20 > ema50 else "bearish"
    reasons, conf = [], []

    if bos == "BOS_UP" or (trend == "bullish" and price > ema20):
        entry, sl = price, recent_low - atr_now * 0.6
        rr = max(entry - sl, price * 0.01)
        tp1, tp2 = entry + rr * 1.5, entry + rr * 2.5
        reasons.append("Bias LONG — BOS naik & EMA searah.")
        conf += [0.9 if trend == "bullish" else 0.6,
                 0.9 if price >= fib_618 else 0.65,
                 1.0 if 30 < rsi_now < 75 else 0.5]
        signal = "LONG"

    elif bos == "BOS_DOWN" or (trend == "bearish" and price < ema20):
        entry, sl = price, recent_high + atr_now * 0.6
        rr = max(sl - entry, price * 0.01)
        tp1, tp2 = entry - rr * 1.5, entry - rr * 2.5
        reasons.append("Bias SHORT — BOS turun & EMA searah bearish.")
        conf += [0.9 if trend == "bearish" else 0.6,
                 0.9 if price <= fib_618 else 0.65,
                 1.0 if 25 < rsi_now < 70 else 0.5]
        signal = "SHORT"

    else:
        entry, sl = price, recent_low * 0.995
        tp1, tp2 = entry + (entry - sl) * 1.2, entry + (entry - sl) * 2.0
        reasons.append("Belum ada arah jelas — tunggu konfirmasi TF lebih tinggi.")
        conf = [0.25]
        signal = "WAIT"

    confidence = float(sum(conf) / len(conf))
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

# ---------------- SCALPING ENGINE ----------------
def scalp_engine(df: pd.DataFrame, pair: Optional[str] = None, tf: Optional[str] = None) -> dict:
    """Strategi cepat untuk scalping timeframe kecil"""
    if df.shape[0] < 30:
        return {"error": "data_tidak_cukup"}

    df['ema8'] = ema(df['close'], 8)
    df['ema21'] = ema(df['close'], 21)
    df['rsi14'] = rsi(df['close'], 14)
    df['atr14'] = atr(df, 14)

    last = df.iloc[-1]
    price, atr_now = float(last['close']), float(last['atr14'])
    vol_mean = df['volume'].tail(40).mean()
    vol_spike = float(last['volume']) > (vol_mean * 1.8 if vol_mean > 0 else False)

    if last['ema8'] > last['ema21'] and vol_spike and 35 < last['rsi14'] < 75:
        entry, sl = price, price - atr_now * 0.6
        tp1, tp2 = price + atr_now * 0.8, price + atr_now * 1.4
        reason, conf, signal = "Scalp LONG — EMA8 > EMA21, volume tinggi.", 0.9, "LONG"
    elif last['ema8'] < last['ema21'] and vol_spike and 25 < last['rsi14'] < 65:
        entry, sl = price, price + atr_now * 0.6
        tp1, tp2 = price - atr_now * 0.8, price - atr_now * 1.4
        reason, conf, signal = "Scalp SHORT — EMA8 < EMA21, volume tinggi.", 0.9, "SHORT"
    else:
        entry, sl = price, price * 0.998
        tp1, tp2 = price * 1.002, price * 1.004
        reason, conf, signal = "Tidak ada peluang bersih.", 0.3, "WAIT"

    return {
        "pair": pair or "", "timeframe": tf or "", "signal_type": signal,
        "entry": round(entry, 8), "tp1": round(tp1, 8), "tp2": round(tp2, 8),
        "sl": round(sl, 8), "confidence": conf, "reasoning": reason
    }

# ---------------- LOGGING ----------------
def ensure_trade_log():
    """Pastikan file trade_log.csv ada"""
    if not os.path.exists(TRADE_LOG_FILE):
        df = pd.DataFrame(columns=[
            "id", "timestamp", "pair", "timeframe", "signal_type",
            "entry", "tp1", "tp2", "sl", "confidence", "reasoning",
            "backtest_hit", "backtest_pnl"
        ])
        df.to_csv(TRADE_LOG_FILE, index=False)

def append_trade_log(record: Dict[str, Any]) -> int:
    """Tambahkan sinyal baru ke trade_log.csv"""
    ensure_trade_log()
    df = pd.read_csv(TRADE_LOG_FILE)
    next_id = int(df['id'].max()) + 1 if not df.empty else 1
    record_row = {
        "id": next_id, "timestamp": datetime.utcnow().isoformat(),
        "pair": record.get("pair"), "timeframe": record.get("timeframe"),
        "signal_type": record.get("signal_type"), "entry": record.get("entry"),
        "tp1": record.get("tp1"), "tp2": record.get("tp2"), "sl": record.get("sl"),
        "confidence": record.get("confidence"), "reasoning": record.get("reasoning"),
        "backtest_hit": record.get("backtest_hit"), "backtest_pnl": record.get("backtest_pnl")
    }
    df = pd.concat([df, pd.DataFrame([record_row])], ignore_index=True)
    df.to_csv(TRADE_LOG_FILE, index=False)
    return next_id
    
    # ---------------- MACHINE LEARNING SYSTEM ----------------

def compute_features_for_row(pair: str, timeframe: str, entry: float, tp: Optional[float], sl: float) -> Optional[Dict[str, float]]:
    """Hitung fitur untuk satu trade (EMA diff, RSI, ATR, dll.)"""
    try:
        kdf = fetch_ohlc_binance(pair, timeframe, limit=200)
    except Exception:
        return None

    kdf = kdf.tail(60).reset_index(drop=True)
    close, high, low, vol = kdf['close'], kdf['high'], kdf['low'], kdf['volume']

    ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    rsi14 = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    atr14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    vol_mean = vol.tail(40).mean() if len(vol) >= 40 else vol.mean()
    vol_spike = 1.0 if vol.iloc[-1] > vol_mean * 1.8 else 0.0

    recent_high, recent_low = high.tail(80).max(), low.tail(80).min()
    dist_to_high = (recent_high - entry) / entry if entry else 0.0
    dist_to_low = (entry - recent_low) / entry if entry else 0.0
    rr = abs((tp - entry) / (entry - sl)) if tp and (entry - sl) != 0 else 0.0

    return {
        "ema8_21_diff": (ema8 - ema21) / entry,
        "rsi14": float(rsi14),
        "atr_rel": float(atr14) / entry,
        "vol_spike": float(vol_spike),
        "dist_to_high": float(dist_to_high),
        "dist_to_low": float(dist_to_low),
        "rr": float(rr)
    }

def build_dataset_from_trade_log():
    """Bangun dataset X, y dari trade_log.csv"""
    if not os.path.exists(TRADE_LOG_FILE):
        return None, None

    df = pd.read_csv(TRADE_LOG_FILE)
    rows, labels = [], []

    for _, r in df.iterrows():
        hit = str(r.get("backtest_hit", "")).upper()
        label = 1 if hit.startswith("TP") else 0
        feats = compute_features_for_row(
            str(r['pair']), str(r.get('timeframe', "15m")),
            float(r['entry']), r.get('tp1', None), float(r['sl'])
        )
        if feats:
            rows.append(feats)
            labels.append(label)

    if not rows:
        return None, None
    return pd.DataFrame(rows), pd.Series(labels)

def train_and_save_model():
    """Latih model baru & simpan ke disk"""
    global _cached_model, _last_retrain_count
    X, y = build_dataset_from_trade_log()
    if X is None or len(y) < MIN_SAMPLES_TO_TRAIN:
        return {"status": "data_tidak_cukup", "samples": len(y) if y is not None else 0}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    report = classification_report(y_test, clf.predict(X_test), output_dict=True)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    joblib.dump({"clf": clf, "features": list(X.columns)}, MODEL_FILE)
    _cached_model = {"clf": clf, "features": list(X.columns)}
    _last_retrain_count = len(pd.read_csv(TRADE_LOG_FILE))

    return {"status": "trained", "samples": len(y), "auc": auc, "report": report}

def predict_with_model(payload: Dict[str, Any]):
    """Prediksi probabilitas sinyal berhasil (pakai model cache jika ada)"""
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(MODEL_FILE):
            raise RuntimeError("model_belum_dilatih")
        _cached_model = joblib.load(MODEL_FILE)

    clf = _cached_model["clf"]
    feats = compute_features_for_row(
        payload.get("pair"), payload.get("timeframe", "15m"),
        float(payload.get("entry")), payload.get("tp1"), float(payload.get("sl"))
    )
    if feats is None:
        raise RuntimeError("gagal_menghitung_fitur")

    X = pd.DataFrame([feats])
    prob = float(clf.predict_proba(X)[:, 1][0])
    return {"prob": prob, "features": feats}

def maybe_trigger_retrain_background():
    """Jalankan retrain model di thread terpisah"""
    def worker():
        try:
            print("🔄 Auto retrain model berjalan...")
            res = train_and_save_model()
            print("✅ Retrain selesai:", res)
        except Exception as e:
            print("⚠️ Retrain gagal:", e)
    threading.Thread(target=worker, daemon=True).start()

def check_and_trigger_retrain_if_needed():
    """Cek jumlah sinyal & retrain otomatis bila cukup"""
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
            print("check_retrain error:", e)
            
# ---------------- ENDPOINT FASTAPI ----------------

@app.get("/health")
def health():
    """Cek status server"""
    return {"status": "ok", "service": "Pro Trader AI - Learning (ID)"}

def _postprocess_with_learning(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Gabungkan hasil teknikal + prediksi AI"""
    try:
        if os.path.exists(MODEL_FILE):
            pred = predict_with_model({
                "pair": signal.get("pair"),
                "timeframe": signal.get("timeframe"),
                "entry": signal.get("entry"),
                "tp1": signal.get("tp1"),
                "sl": signal.get("sl")
            })
            prob = pred.get("prob", 0.0)
            base_conf = float(signal.get("confidence", 0.5))
            new_conf = round(min(1.0, (0.9 * base_conf + 0.1 * prob)), 3)
            signal["confidence"] = new_conf
            signal["model_prob"] = round(prob, 3)
            signal["vetoed_by_model"] = prob < 0.35
            if signal["vetoed_by_model"]:
                signal["signal_type"] = "WAIT"
        else:
            signal["model_prob"] = None
    except Exception as e:
        signal["model_error"] = str(e)
    return signal

@app.get("/pro_signal")
def pro_signal(pair: str = Query(...), tf_main: str = Query("1h"),
               tf_entry: str = Query("15m"), limit: int = Query(300),
               auto_log: bool = Query(False)):
    """Analisis sinyal utama (multi-timeframe)"""
    try:
        df_entry = fetch_ohlc_binance(pair, tf_entry, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch_failed: {e}")

    res = hybrid_analyze(df_entry, pair=pair, timeframe=tf_entry)
    try:
        df_main = fetch_ohlc_binance(pair, tf_main, limit=200)
        ema20, ema50 = float(ema(df_main['close'], 20).iloc[-1]), float(ema(df_main['close'], 50).iloc[-1])
        res['context_main_trend'] = "bullish" if ema20 > ema50 else "bearish"
    except Exception:
        res['context_main_trend'] = "unknown"

    res = _postprocess_with_learning(res)

    if auto_log:
        payload_bt = {
            "pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"],
            "entry": res["entry"], "tp1": res["tp1"], "tp2": res["tp2"],
            "sl": res["sl"], "confidence": res["confidence"], "reason": res["reasoning"]
        }
        bt_res = post_to_backtester(payload_bt)
        res["backtest_raw"] = bt_res
        append_trade_log({
            "pair": res["pair"], "timeframe": res["timeframe"], "signal_type": res["signal_type"],
            "entry": res["entry"], "tp1": res["tp1"], "tp2": res["tp2"], "sl": res["sl"],
            "confidence": res["confidence"], "reasoning": res["reasoning"],
            "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None,
            "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None
        })
        check_and_trigger_retrain_if_needed()

    return JSONResponse(res)

@app.get("/scalp_signal")
def scalp_signal(pair: str = Query(...), tf: str = Query("3m"),
                 limit: int = Query(300), auto_log: bool = Query(False)):
    """Analisis scalping cepat"""
    try:
        df = fetch_ohlc_binance(pair, tf, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch_failed: {e}")

    res = scalp_engine(df, pair=pair, tf=tf)
    res = _postprocess_with_learning(res)

    if auto_log:
        payload_bt = {
            "pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"],
            "entry": res["entry"], "tp1": res["tp1"], "tp2": res["tp2"],
            "sl": res["sl"], "confidence": res["confidence"], "reason": res["reasoning"]
        }
        bt_res = post_to_backtester(payload_bt)
        res["backtest_raw"] = bt_res
        append_trade_log({
            "pair": res["pair"], "timeframe": res["timeframe"], "signal_type": res["signal_type"],
            "entry": res["entry"], "tp1": res["tp1"], "tp2": res["tp2"], "sl": res["sl"],
            "confidence": res["confidence"], "reasoning": res["reasoning"],
            "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None,
            "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None
        })
        check_and_trigger_retrain_if_needed()

    return JSONResponse(res)

@app.post("/analyze_chart")
def analyze_chart(file: UploadFile = File(...),
                  pair: Optional[str] = Form(None),
                  timeframe: Optional[str] = Form(None),
                  auto_backtest: Optional[str] = Form("true")):
    """Analisis chart dari gambar (TradingView Screenshot)"""
    auto_flag = auto_backtest.lower() != "false"
    try:
        contents = file.file.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"gambar_tidak_valid: {e}")

    if not _HAS_TESSERACT:
        raise HTTPException(status_code=400, detail="tesseract_tidak_tersedia")

    y_map = {}
    try:
        y_map = ocr_y_axis_prices(img_cv)
    except Exception:
        y_map = {}

    df_ohlc = detect_candles_from_plot(img_cv, y_map, max_bars=200)
    if df_ohlc.empty:
        raise HTTPException(status_code=400, detail="gagal_membaca_chart")

    for c in ['open', 'high', 'low', 'close']:
        df_ohlc[c] = pd.to_numeric(df_ohlc[c], errors='coerce')

    res = hybrid_analyze(df_ohlc, pair=pair or "IMG", timeframe=timeframe or "img")
    res = _postprocess_with_learning(res)

    if auto_flag:
        bt_res = post_to_backtester({
            "pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"],
            "entry": res["entry"], "tp1": res["tp1"], "tp2": res["tp2"],
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
    
# ---------------- MONITORING & ANALISIS TAMBAHAN ----------------

@app.get("/logs")
def get_logs(limit: int = Query(100)):
    """Lihat log sinyal terakhir"""
    ensure_trade_log()
    df = pd.read_csv(TRADE_LOG_FILE)
    df = df.tail(limit).to_dict(orient="records")
    return JSONResponse({"logs": df})

@app.get("/logs_summary")
def logs_summary():
    """Ringkasan sinyal terakhir"""
    ensure_trade_log()
    df = pd.read_csv(TRADE_LOG_FILE)
    if df.empty:
        return JSONResponse({"detail": "Belum ada data sinyal tersimpan."})
    last = df.iloc[-1]
    return JSONResponse({
        "pair": last.get("pair", ""),
        "timeframe": last.get("timeframe", ""),
        "signal_type": last.get("signal_type", ""),
        "entry": last.get("entry", ""),
        "tp1": last.get("tp1", ""),
        "tp2": last.get("tp2", ""),
        "sl": last.get("sl", ""),
        "confidence": last.get("confidence", ""),
        "reasoning": last.get("reasoning", "")
    })

@app.get("/ai_performance")
def ai_performance():
    """Analisis performa AI berdasarkan trade log"""
    ensure_trade_log()
    df = pd.read_csv(TRADE_LOG_FILE)
    if df.empty:
        return JSONResponse({"error": "Belum ada sinyal yang tercatat."})

    total = len(df)
    tp_hits = df["backtest_hit"].astype(str).str.upper().str.startswith("TP").sum()
    sl_hits = df["backtest_hit"].astype(str).str.upper().str.startswith("SL").sum()
    winrate = round((tp_hits / total) * 100, 2)
    avg_conf = round(pd.to_numeric(df["confidence"], errors="coerce").mean(), 3)

    pnl_values = pd.to_numeric(df["backtest_pnl"], errors="coerce").dropna()
    total_pnl = pnl_values.sum() if not pnl_values.empty else 0.0

    profit_factor, max_drawdown = None, 0.0
    if not pnl_values.empty:
        gains = pnl_values[pnl_values > 0].sum()
        losses = abs(pnl_values[pnl_values < 0].sum())
        profit_factor = round(gains / losses, 2) if losses != 0 else None
        max_drawdown = round(pnl_values.min(), 3) if not pnl_values.empty else 0.0

    model_exists = os.path.exists(MODEL_FILE)
    return JSONResponse({
        "total_signals": total,
        "tp_hits": int(tp_hits),
        "sl_hits": int(sl_hits),
        "winrate": winrate,
        "avg_confidence": avg_conf,
        "total_pnl": float(total_pnl),
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "model_status": "✅ Sudah Dilatih" if model_exists else "❌ Belum Ada Model"
    })

@app.get("/model_debug")
def model_debug():
    """Info debug model: fitur, waktu retrain, importance"""
    info = {"model_exists": os.path.exists(MODEL_FILE)}
    if info["model_exists"]:
        try:
            mod = joblib.load(MODEL_FILE)
            clf = mod.get("clf")
            features = mod.get("features")
            info["features"] = features
            info["last_trained"] = datetime.fromtimestamp(os.path.getmtime(MODEL_FILE)).isoformat()
            if hasattr(clf, "feature_importances_"):
                info["feature_importance"] = {
                    f: round(float(v), 4) for f, v in zip(features, clf.feature_importances_)
                }
        except Exception as e:
            info["error_loading_model"] = str(e)
    else:
        info["features"] = []
    return JSONResponse(info)

@app.get("/retrain_learning")
def retrain_learning():
    """Paksa retrain model manual"""
    res = train_and_save_model()
    return JSONResponse(res)

@app.get("/learning_status")
def learning_status():
    """Cek status pembelajaran AI"""
    info = {"model_exists": os.path.exists(MODEL_FILE)}
    if info["model_exists"]:
        mod = joblib.load(MODEL_FILE)
        info["features"] = mod.get("features")
    try:
        df = pd.read_csv(TRADE_LOG_FILE)
        info["trade_log_count"] = len(df)
    except:
        info["trade_log_count"] = 0
    return JSONResponse(info)

@app.get("/download_logs")
def download_logs():
    """Unduh semua log sinyal"""
    ensure_trade_log()
    return FileResponse(TRADE_LOG_FILE, media_type="text/csv", filename="trade_log.csv")

# ---------------- STARTUP EVENT ----------------
@app.on_event("startup")
def startup_event():
    """Inisialisasi sistem saat container pertama kali aktif"""
    ensure_trade_log()
    global _cached_model
    if os.path.exists(MODEL_FILE):
        try:
            _cached_model = joblib.load(MODEL_FILE)
            print("✅ Model berhasil dimuat dari cache.")
        except Exception as e:
            print("⚠️ Gagal memuat model:", e)
    else:
        print("ℹ️ Belum ada model yang dilatih. Akan dibuat otomatis nanti.")

# Jalankan server (Railway otomatis memanggil)
# uvicorn main_combined_learning:app --host 0.0.0.0 --port $PORT
