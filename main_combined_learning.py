"""
Pro Trader AI - Combined + Learning (Stable Release)
- Analisis Crypto (Binance) & Forex (CSV)
- Auto-log ke trade_log.csv
- Learning terintegrasi (RandomForest)
- Full bahasa Indonesia
"""

import os, io, json, threading
from datetime import datetime
from typing import Optional, Dict, Any
import requests, pandas as pd, numpy as np
from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from PIL import Image
import cv2

try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

# ============ KONFIGURASI ============
app = FastAPI(title="Pro Trader AI - Learning (Stable)", version="1.1")
BACKTEST_URL = os.environ.get("BACKTEST_URL")
TRADE_LOG_FILE = "trade_log.csv"
MODEL_FILE = "rf_model.pkl"
MIN_SAMPLES_TO_TRAIN = int(os.environ.get("MIN_SAMPLES_TO_TRAIN", 50))
N_SIGNALS_TO_RETRAIN = int(os.environ.get("N_SIGNALS_TO_RETRAIN", 50))
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

_lock = threading.Lock()
_last_retrain_count = 0


# ============ UTILITAS TEKNIKAL ============
def fetch_ohlc_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(BINANCE_KLINES, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "qav","num_trades","tb_base","tb_quote","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time","open","high","low","close","volume"]]

def ema(series, n): return ta.trend.EMAIndicator(series, window=n).ema_indicator()
def rsi(series, n=14): return ta.momentum.RSIIndicator(series, window=n).rsi()
def atr(df, n=14): return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=n).average_true_range()


# ============ STRATEGI HYBRID ============
def hybrid_analyze(df: pd.DataFrame, pair=None, timeframe=None) -> dict:
    df = df.copy().dropna()
    if len(df) < 12:
        return {"error":"data_tidak_cukup","message":"Perlu minimal 12 candle."}

    df['ema20'], df['ema50'] = ema(df['close'],20), ema(df['close'],50)
    df['rsi14'], df['atr14'] = rsi(df['close'],14), atr(df,14)

    last = df.iloc[-1]
    price = float(last['close'])
    ema20, ema50 = float(last['ema20']), float(last['ema50'])
    rsi_now = float(last['rsi14'])
    atr_now = float(last['atr14']) if not np.isnan(last['atr14']) else price*0.001
    trend = "bullish" if ema20 > ema50 else "bearish"

    high = df['high'].tail(100).max()
    low = df['low'].tail(100).min()
    signal, reasons, conf = "WAIT", [], []

    if trend == "bullish" and price > ema20:
        signal = "LONG"
        reasons.append("Bias LONG — harga di atas EMA & tren naik.")
        conf.append(0.9)
    elif trend == "bearish" and price < ema20:
        signal = "SHORT"
        reasons.append("Bias SHORT — harga di bawah EMA & tren turun.")
        conf.append(0.9)
    else:
        reasons.append("Belum ada arah jelas.")
        conf.append(0.3)

    entry = price
    rr = price * 0.01
    sl = entry - rr if signal == "LONG" else entry + rr
    tp1 = entry + rr*1.5 if signal == "LONG" else entry - rr*1.5
    tp2 = entry + rr*2.5 if signal == "LONG" else entry - rr*2.5
    confidence = round(sum(conf)/len(conf),3)

    return {
        "pair": pair or "", "timeframe": timeframe or "",
        "signal_type": signal, "entry": round(entry,6),
        "tp1": round(tp1,6), "tp2": round(tp2,6),
        "sl": round(sl,6), "confidence": confidence,
        "reasoning": " ".join(reasons)
    }


# ============ LOGGING ============
def ensure_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        cols = ["id","timestamp","pair","timeframe","signal_type","entry","tp1","tp2","sl","confidence","reasoning","backtest_hit","backtest_pnl"]
        pd.DataFrame(columns=cols).to_csv(TRADE_LOG_FILE, index=False)

def append_trade_log(record: Dict[str, Any]):
    ensure_trade_log()
    df = pd.read_csv(TRADE_LOG_FILE)
    next_id = int(df["id"].max()) + 1 if not df.empty else 1
    record["id"] = next_id
    record["timestamp"] = datetime.utcnow().isoformat()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(TRADE_LOG_FILE, index=False)
    return next_id


# ============ MODEL LEARNING ============
def build_dataset_from_trade_log():
    if not os.path.exists(TRADE_LOG_FILE): return None, None
    df = pd.read_csv(TRADE_LOG_FILE)
    if df.empty: return None, None
    rows, labels = [], []
    for _, r in df.iterrows():
        hit = str(r.get("backtest_hit", "")).upper()
        label = 1 if hit.startswith("TP") else 0
        feats = {
            "confidence": float(r.get("confidence",0.5))
        }
        rows.append(feats); labels.append(label)
    if not rows: return None, None
    return pd.DataFrame(rows), pd.Series(labels)

def train_and_save_model():
    global _last_retrain_count
    X,y = build_dataset_from_trade_log()
    if X is None or len(y) < MIN_SAMPLES_TO_TRAIN:
        return {"status":"data_tidak_cukup"}
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train,y_train)
    joblib.dump({"clf":clf,"features":list(X.columns)},MODEL_FILE)
    _last_retrain_count = len(y)
    return {"status":"trained","samples":len(y)}

def predict_with_model(signal: dict):
    if not os.path.exists(MODEL_FILE):
        raise RuntimeError("model_belum_dilatih")
    mod = joblib.load(MODEL_FILE)
    clf = mod["clf"]
    X = pd.DataFrame([{"confidence": signal.get("confidence",0.5)}])
    prob = float(clf.predict_proba(X)[:,1][0])
    return prob


# ============ RETRAIN SYSTEM ============
def check_and_trigger_retrain_if_needed():
    global _last_retrain_count
    with _lock:
        df = pd.read_csv(TRADE_LOG_FILE)
        total = len(df)
        if _last_retrain_count == 0: _last_retrain_count = total
        if total - _last_retrain_count >= N_SIGNALS_TO_RETRAIN:
            threading.Thread(target=train_and_save_model, daemon=True).start()
            _last_retrain_count = total


# ============ ENDPOINT ============
@app.get("/health")
def health():
    return {"status":"ok","service":"Pro Trader AI - Learning"}

@app.get("/pro_signal")
def pro_signal(pair: Optional[str] = Query(None), tf_main: str = "1h", tf_entry: str = "15m", auto_log: bool = False):
    if not pair:
        raise HTTPException(status_code=400, detail="parameter 'pair' wajib diisi.")
    try:
        df = fetch_ohlc_binance(pair, tf_entry, limit=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch_failed: {e}")

    res = hybrid_analyze(df, pair, tf_entry)
    try:
        if os.path.exists(MODEL_FILE):
            prob = predict_with_model(res)
            res["model_prob"] = round(prob,3)
            res["confidence"] = round((res["confidence"]*0.7 + prob*0.3),3)
    except Exception as e:
        res["model_error"] = str(e)

    if auto_log:
        append_trade_log(res)
        check_and_trigger_retrain_if_needed()

    return JSONResponse(res)


@app.get("/logs_summary")
def logs_summary():
    ensure_trade_log()
    df = pd.read_csv(TRADE_LOG_FILE)
    if df.empty:
        return {"detail": "Belum ada data sinyal terbaru."}
    last = df.iloc[-1]
    return last.to_dict()


@app.get("/learning_status_summary")
def learning_status_summary():
    model_exists = os.path.exists(MODEL_FILE)
    df = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else pd.DataFrame()
    logs_count = len(df)
    return {
        "model_status": "✅ Sudah Dilatih" if model_exists else "❌ Belum Ada Model",
        "log_count": logs_count,
        "learning_ready": logs_count >= MIN_SAMPLES_TO_TRAIN,
        "description": "AI siap retrain otomatis saat data cukup."
    }


@app.get("/ai_performance")
def ai_performance():
    ensure_trade_log()
    df = pd.read_csv(TRADE_LOG_FILE)
    if df.empty:
        return {"error": "Belum ada data sinyal untuk dianalisis."}

    tp_hits = df["backtest_hit"].astype(str).str.upper().str.startswith("TP").sum()
    total = len(df)
    winrate = round((tp_hits / total) * 100, 2)
    pnl_values = pd.to_numeric(df.get("backtest_pnl", []), errors="coerce").dropna()

    if (pnl_values < 0).any() and (pnl_values > 0).any():
        profit_factor = round(abs(pnl_values[pnl_values > 0].sum() / abs(pnl_values[pnl_values < 0].sum())), 2)
    else:
        profit_factor = None

    return {
        "total_signals": total,
        "winrate": winrate,
        "profit_factor": profit_factor,
        "model_status": "✅" if os.path.exists(MODEL_FILE) else "❌"
    }


# ============ PLACEHOLDER UNTUK CHART OCR ============
def ocr_y_axis_prices(img): return {}
def detect_candles_from_plot(img, y_map, max_bars=200): return pd.DataFrame(columns=["open","high","low","close"])

# Jalankan dengan: uvicorn main_combined_learning:app --host 0.0.0.0 --port $PORT
ensure_trade_log()
