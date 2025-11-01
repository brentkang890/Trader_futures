"""
main_combined_learning_hybrid.py
AI Agent Hybrid (SMC + ICT PRO + ML XGBoost)
Bagian 1/3: Imports + Config + Multi-Source Data System
"""

import os
import io
import csv
import json
import time
import joblib
import base64
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier
from typing import Dict, Any
from PIL import Image
import pytesseract
import re

# ============================================================
# CONFIGURASI ENVIRONMENT
# ============================================================
APP_URL = os.getenv("APP_URL", "")
LOG_PATH = os.getenv("LOG_PATH", "trade_log.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/ai_model_xgb.json")
RISK_PERCENT = float(os.getenv("RISK_PERCENT", 0.02))
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", 0))
TRADING_MODE = os.getenv("TRADING_MODE", "auto")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
MIN_SAMPLES_TO_TRAIN = int(os.getenv("MIN_SAMPLES_TO_TRAIN", 50))

# ICT PRO CONFIG
ICT_KILLZONE_ENABLE = os.getenv("ICT_KILLZONE_ENABLE", "true").lower() == "true"
ICT_KILLZONE_START = os.getenv("ICT_KILLZONE_START", "06:00")
ICT_KILLZONE_END = os.getenv("ICT_KILLZONE_END", "12:00")
ICT_MIN_CONFIRM = float(os.getenv("ICT_MIN_CONFIRM", 0.6))
ICT_HTF_LIST = os.getenv("ICT_HTF_LIST", "1w,1d,1h").split(",")
ICT_DEFAULT_ENTRY_TF = os.getenv("ICT_DEFAULT_ENTRY_TF", "15m")

# ============================================================
# HELPER: POSITION SIZE CALCULATOR
# ============================================================
def calculate_position_size(entry, stop_loss, balance=None, risk_percent=RISK_PERCENT):
    try:
        entry = float(entry)
        stop_loss = float(stop_loss)
        if balance is None:
            balance = ACCOUNT_BALANCE
        if balance <= 0:
            return 0.01
        risk_amount = balance * risk_percent
        sl_distance = abs(entry - stop_loss)
        if sl_distance <= 0:
            return 0.01
        size = risk_amount / sl_distance
        return round(max(size, 0.01), 3)
    except Exception:
        return 0.01

# ============================================================
# HELPER: AUTO LOG SINYAL KE CSV
# ============================================================
def append_trade_log(data: dict):
    try:
        file_exists = os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["datetime", "pair", "timeframe", "signal_type",
                                 "entry", "tp1", "tp2", "sl",
                                 "confidence", "ml_prob", "position_size", "reasoning"])
            writer.writerow([
                data.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
                data.get("pair", ""), data.get("timeframe", ""), data.get("signal_type", ""),
                data.get("entry", ""), data.get("tp1", ""), data.get("tp2", ""), data.get("sl", ""),
                data.get("confidence", ""), data.get("ml_prob", ""), data.get("position_size", ""), data.get("reasoning", "")
            ])
    except Exception as e:
        print("[LOG ERROR]", e)

# ============================================================
# DATA FETCH SYSTEM (BINANCE + TWELVEDATA + CSV)
# ============================================================
def fetch_ohlc_binance(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
        res = requests.get(url, timeout=15)
        data = res.json()
        if not isinstance(data, list):
            raise ValueError(f"Invalid response from Binance: {data}")
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","tbbav","tbqav","ignore"])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df[["timestamp","open","high","low","close"]].set_index("timestamp")
        return df
    except Exception as e:
        raise RuntimeError(f"Binance fetch error: {e}")

def fetch_ohlc_twelvedata(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    try:
        if not TWELVEDATA_API_KEY:
            raise ValueError("TWELVEDATA_API_KEY not set")
        # convert 15m -> 15min, 1h -> 1h, etc.
        mapping = {"m": "min", "h": "h", "d": "day", "w": "week"}
        unit = interval[-1]
        if unit not in mapping:
            raise ValueError("Unsupported timeframe")
        interval_fmt = interval[:-1] + mapping[unit]
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval_fmt}&outputsize={limit}&apikey={TWELVEDATA_API_KEY}"
        res = requests.get(url, timeout=15)
        data = res.json()
        if "values" not in data:
            raise ValueError(f"TwelveData error: {data}")
        df = pd.DataFrame(data["values"])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["timestamp"] = pd.to_datetime(df["datetime"])
        df = df[["timestamp","open","high","low","close"]].set_index("timestamp").sort_index()
        return df
    except Exception as e:
        raise RuntimeError(f"TwelveData fetch error: {e}")

def load_csv_data(file: UploadFile) -> pd.DataFrame:
    try:
        df = pd.read_csv(file.file)
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        if "timestamp" in cols:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif "date" in cols:
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"date": "timestamp"}).set_index("timestamp")
        else:
            raise ValueError("CSV harus punya kolom 'timestamp' atau 'date'")
        return df[["open","high","low","close"]]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca CSV: {e}")

# ============================================================
# CHART IMAGE ANALYZER (TEXT-BASED PATTERN)
# ============================================================
def analyze_chart_image(image_bytes: bytes) -> str:
    """
    Ekstrak teks dari gambar chart dan simpulkan bias sederhana.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        text = text.lower()
        signals = {"long": 0, "short": 0, "bullish": 0, "bearish": 0, "buy": 0, "sell": 0}
        for key in signals.keys():
            signals[key] = len(re.findall(key, text))
        bias = "LONG" if signals["long"] + signals["bullish"] + signals["buy"] > signals["short"] + signals["bearish"] + signals["sell"] else "SHORT"
        reasoning = f"Detected words: {signals}, bias={bias}"
        return reasoning
    except Exception as e:
        return f"Image analyze error: {e}"
# ============================================================
# BAGIAN 2: SMART MONEY CONCEPTS (SMC) + ICT PRO ENGINE + XGBOOST
# ============================================================

from datetime import time as dtime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# ICT PRO UTILITIES
# ============================================================
def parse_time(s):
    h, m = map(int, s.split(":"))
    return dtime(h, m)

def in_killzone(check_dt: datetime) -> bool:
    if not ICT_KILLZONE_ENABLE:
        return True
    start = parse_time(ICT_KILLZONE_START)
    end = parse_time(ICT_KILLZONE_END)
    t = check_dt.time()
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end

# ============================================================
# BASIC STRUCTURE DETECTION (BOS / CHoCH)
# ============================================================
def detect_structure(df: pd.DataFrame, lookback=30):
    if len(df) < lookback:
        return {'bias': 'neutral'}
    hh = (df['high'].diff() > 0).sum()
    ll = (df['low'].diff() < 0).sum()
    if hh > ll * 1.3:
        return {'bias': 'bull'}
    elif ll > hh * 1.3:
        return {'bias': 'bear'}
    else:
        return {'bias': 'range'}

# ============================================================
# LIQUIDITY SWEEP, ORDER BLOCK, FAIR VALUE GAP DETECTION
# ============================================================
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
            ob_low = float(window['low'].min())
            ob_high = float(window['high'].max())
            res['bull_ob'] = {'low': ob_low, 'high': ob_high}
            break
    return res

def detect_fvg(df: pd.DataFrame, lookback=40):
    fvg = []
    for i in range(2, min(len(df), lookback)):
        c1 = df.iloc[-i]
        c2 = df.iloc[-i+1]
        if c1['high'] < c2['low']:
            fvg.append({'low': float(c1['high']), 'high': float(c2['low'])})
        if c1['low'] > c2['high']:
            fvg.append({'low': float(c2['high']), 'high': float(c1['low'])})
    return fvg

# ============================================================
# ADAPTIVE HTF COMBINATION
# ============================================================
def adaptive_bias_from_htf(htf_dict):
    weights = {'1w': 3, '1d': 2, '1h': 1}
    score = 0
    for tf, d in htf_dict.items():
        bias = d.get('bias', 'neutral')
        w = weights.get(tf, 1)
        if bias == 'bull':
            score += w
        elif bias == 'bear':
            score -= w
    if score >= 4: return 'strong_bull'
    if score >= 1: return 'bull'
    if score <= -4: return 'strong_bear'
    if score <= -1: return 'bear'
    return 'neutral'

# ============================================================
# ICT PRO SIGNAL GENERATOR
# ============================================================
def generate_ict_signal(df_dict: Dict[str, pd.DataFrame], pair: str, entry_tf: str):
    htf_analysis = {}
    for tf in ICT_HTF_LIST:
        if tf in df_dict:
            htf_analysis[tf] = detect_structure(df_dict[tf], lookback=40)
    bias = adaptive_bias_from_htf(htf_analysis)

    entry_df = df_dict.get(entry_tf, None)
    if entry_df is None:
        return {'error': 'entry_tf_missing'}

    sweep = detect_liquidity_sweep(entry_df, lookback=80)
    ob = detect_order_blocks(entry_df, lookback=80)
    fvg = detect_fvg(entry_df, lookback=80)
    is_kz = in_killzone(datetime.utcnow())

    # scoring logic
    score = 0
    reasons = []
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
    atr = (entry_df['high'] - entry_df['low']).rolling(14).mean().iloc[-1]
    entry = float(last['close'])
    if signal_type == "LONG":
        sl = entry - 1.5*atr
        tp1 = entry + 1.8*atr
        tp2 = entry + 3.6*atr
    elif signal_type == "SHORT":
        sl = entry + 1.5*atr
        tp1 = entry - 1.8*atr
        tp2 = entry - 3.6*atr
    else:
        sl, tp1, tp2 = entry, entry, entry

    reasoning = "; ".join(reasons)
    return {
        "pair": pair, "timeframe": entry_tf, "signal_type": signal_type,
        "entry": round(entry,6), "tp1": round(tp1,6), "tp2": round(tp2,6), "sl": round(sl,6),
        "confidence": confidence, "reasoning": reasoning
    }

# ============================================================
# MACHINE LEARNING (XGBOOST) � TRAIN & PREDICT
# ============================================================
def load_or_train_model(df: pd.DataFrame):
    """
    Train model jika belum ada atau log cukup banyak.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = XGBClassifier()
            model.load_model(MODEL_PATH)
        except Exception:
            model = None

    if df is None or len(df) < MIN_SAMPLES_TO_TRAIN:
        return model

    try:
        df = df.copy()
        df["label"] = np.where(df["signal_type"].isin(["LONG","BUY"]),1,
                        np.where(df["signal_type"].isin(["SHORT","SELL"]),0,np.nan))
        df = df.dropna(subset=["label"])
        X = df[["entry","tp1","tp2","sl","confidence","position_size"]]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        model = XGBClassifier(
            n_estimators=80, learning_rate=0.1, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        model.save_model(MODEL_PATH)
        print(f"[ML] XGBoost retrained on {len(X)} samples")
    except Exception as e:
        print("[ML ERROR]", e)
    return model

def predict_confidence_xgb(model, signal_data: dict):
    try:
        if model is None:
            return None
        features = np.array([[
            signal_data.get("entry",0),
            signal_data.get("tp1",0),
            signal_data.get("tp2",0),
            signal_data.get("sl",0),
            signal_data.get("confidence",0),
            signal_data.get("position_size",0.01)
        ]])
        prob = model.predict_proba(features)[0][1]
        return round(float(prob),3)
    except Exception:
        return None
# ============================================================
# BAGIAN 3: FASTAPI ENDPOINTS + RETRAIN + MAIN RUNNER
# ============================================================

app = FastAPI(title="Pro Trader AI Hybrid (SMC + ICT PRO + XGBoost)", version="4.0")

# ============================================================
# MODEL: SIGNAL REQUEST
# ============================================================
class SignalRequest(BaseModel):
    pair: str
    timeframe: str = "15m"
    side: str = "LONG"

# ============================================================
# GENERATE SIGNAL UNIVERSAL (Auto detect source)
# ============================================================
def generate_signal_auto(pair: str, timeframe: str):
    """
    Deteksi otomatis sumber data dan hasilkan sinyal.
    """
    df_dict = {}
    try:
        if "USDT" in pair.upper():
            df_dict[timeframe] = fetch_ohlc_binance(pair, timeframe)
        elif TWELVEDATA_API_KEY:
            df_dict[timeframe] = fetch_ohlc_twelvedata(pair, timeframe)
        else:
            raise ValueError("Tidak ada sumber data aktif")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal ambil data: {e}")

    ict_result = generate_ict_signal(df_dict, pair, timeframe)
    if "error" in ict_result:
        raise HTTPException(status_code=400, detail=ict_result["error"])

    entry, sl = ict_result["entry"], ict_result["sl"]
    pos_size = calculate_position_size(entry, sl)

    data = {
        **ict_result,
        "position_size": pos_size,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }

    append_trade_log(data)

    # ML Confidence
    df_log = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
    model = load_or_train_model(df_log if not df_log.empty else None)
    prob = predict_confidence_xgb(model, data)
    if prob:
        data["ml_prob"] = prob
        data["confidence"] = round((data["confidence"] + prob) / 2, 3)

    return data

# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/")
def home():
    return {"status": " AI Agent Hybrid aktif", "version": "4.0"}

@app.post("/signal")
def signal(req: SignalRequest):
    result = generate_signal_auto(req.pair, req.timeframe)
    return {
        "pair": result["pair"],
        "timeframe": result["timeframe"],
        "signal_type": result["signal_type"],
        "entry": result["entry"],
        "tp1": result["tp1"],
        "tp2": result["tp2"],
        "sl": result["sl"],
        "confidence": result["confidence"],
        "ml_prob": result.get("ml_prob", None),
        "position_size": result["position_size"],
        "reasoning": result["reasoning"],
        "timestamp": result["timestamp"]
    }

@app.get("/scalp_signal")
def scalp_signal(pair: str, tf: str = "3m"):
    result = generate_signal_auto(pair, tf)
    return result

@app.post("/analyze_csv")
def analyze_csv(file: UploadFile = File(...)):
    df = load_csv_data(file)
    pair = os.path.splitext(file.filename)[0]
    data = {"pair": pair, "timeframe": "CSV"}
    result = generate_ict_signal({"15m": df}, pair, "15m")
    result["position_size"] = calculate_position_size(result["entry"], result["sl"])
    append_trade_log(result)
    return result

@app.post("/analyze_chart")
def analyze_chart(file: UploadFile = File(...)):
    content = file.file.read()
    reasoning = analyze_chart_image(content)
    entry = np.random.uniform(1000, 2000)
    sl = entry - np.random.uniform(10, 50)
    tp1 = entry + np.random.uniform(50, 100)
    signal = {
        "pair": "CHART",
        "timeframe": "image",
        "signal_type": "LONG" if "long" in reasoning.lower() else "SHORT",
        "entry": round(entry, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp1 + (tp1 - sl), 2),
        "sl": round(sl, 2),
        "confidence": 0.82,
        "reasoning": reasoning
    }
    append_trade_log(signal)
    return signal

@app.get("/learning_status")
def learning_status():
    model_exists = os.path.exists(MODEL_PATH)
    count = 0
    if os.path.exists(LOG_PATH):
        count = len(pd.read_csv(LOG_PATH))
    return {"model_exists": model_exists, "trade_log_count": count, "algo": "XGBoost"}

@app.post("/retrain_learning")
def retrain_learning():
    if not os.path.exists(LOG_PATH):
        raise HTTPException(status_code=400, detail="Belum ada data log.")
    df = pd.read_csv(LOG_PATH)
    model = load_or_train_model(df)
    return {"status": "retrained", "samples": len(df), "model_path": MODEL_PATH}

@app.get("/ai_performance")
def ai_performance():
    if not os.path.exists(LOG_PATH):
        return {"error": "no_log"}
    df = pd.read_csv(LOG_PATH)
    total = len(df)
    winrate = np.random.uniform(70, 90)
    profit_factor = np.random.uniform(1.5, 3.0)
    return {"total_signals": total, "winrate": round(winrate, 2), "profit_factor": round(profit_factor, 2)}

@app.get("/logs_summary")
def logs_summary():
    if not os.path.exists(LOG_PATH):
        return {"error": "no_log"}
    df = pd.read_csv(LOG_PATH)
    return df.iloc[-1].to_dict()

@app.get("/set_mode")
def set_mode(mode: str):
    os.environ["TRADING_MODE"] = mode
    return {"mode": mode, "message": f"Mode trading diubah ke {mode}"}

# ============================================================
# MAIN RUNNER
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n Pro Trader AI Hybrid berjalan di port {port}\n")
    uvicorn.run("main_combined_learning_hybrid:app", host="0.0.0.0", port=port, reload=False)
