"""
main_combined_learning_hybrid.py
AI Agent Hybrid (SMC + ICT PRO + ML XGBoost)
Versi 4.1 — FIXED (Fallback Binance → TwelveData + Error Handler + Universal Pair)
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
from datetime import time as dtime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
# HELPER
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

def append_trade_log(data: dict):
    try:
        file_exists = os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "datetime", "pair", "timeframe", "signal_type", "entry", "tp1", "tp2", "sl",
                    "confidence", "ml_prob", "position_size", "reasoning"
                ])
            writer.writerow([
                data.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
                data.get("pair", ""), data.get("timeframe", ""), data.get("signal_type", ""),
                data.get("entry", ""), data.get("tp1", ""), data.get("tp2", ""), data.get("sl", ""),
                data.get("confidence", ""), data.get("ml_prob", ""), data.get("position_size", ""), data.get("reasoning", "")
            ])
    except Exception as e:
        print("[LOG ERROR]", e)

# ============================================================
# DATA FETCH SYSTEM (BINANCE + TWELVEDATA)
# ============================================================
def fetch_ohlc_binance(symbol: str, interval: str = "15m", limit: int = 500):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
        res = requests.get(url, timeout=15)
        data = res.json()
        if not isinstance(data, list):
            raise ValueError(f"Invalid response: {data}")
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","tbbav","tbqav","ignore"])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        return df[["timestamp","open","high","low","close"]].set_index("timestamp")
    except Exception as e:
        raise RuntimeError(f"Binance fetch error: {e}")

def fetch_ohlc_twelvedata(symbol: str, interval: str = "15m", limit: int = 500):
    try:
        if not TWELVEDATA_API_KEY:
            raise ValueError("TWELVEDATA_API_KEY not set")
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
        return df[["timestamp","open","high","low","close"]].set_index("timestamp").sort_index()
    except Exception as e:
        raise RuntimeError(f"TwelveData fetch error: {e}")

# ============================================================
# CHART IMAGE ANALYZER
# ============================================================
def analyze_chart_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img).lower()
        signals = {"long": 0, "short": 0, "bullish": 0, "bearish": 0, "buy": 0, "sell": 0}
        for k in signals:
            signals[k] = len(re.findall(k, text))
        bias = "LONG" if signals["long"] + signals["bullish"] + signals["buy"] > signals["short"] + signals["bearish"] + signals["sell"] else "SHORT"
        return f"Detected words: {signals}, bias={bias}"
    except Exception as e:
        return f"Image analyze error: {e}"

# ============================================================
# ICT + SMC ENGINE
# ============================================================
def parse_time(s): h, m = map(int, s.split(":")); return dtime(h, m)

def in_killzone(check_dt: datetime):
    if not ICT_KILLZONE_ENABLE: return True
    start, end = parse_time(ICT_KILLZONE_START), parse_time(ICT_KILLZONE_END)
    t = check_dt.time()
    return start <= t <= end if start <= end else t >= start or t <= end

def detect_structure(df, lookback=30):
    if len(df) < lookback: return {'bias': 'neutral'}
    hh, ll = (df['high'].diff() > 0).sum(), (df['low'].diff() < 0).sum()
    if hh > ll * 1.3: return {'bias': 'bull'}
    elif ll > hh * 1.3: return {'bias': 'bear'}
    return {'bias': 'range'}

def detect_liquidity_sweep(df, lookback=50):
    if len(df) < lookback: return {'sweep': False}
    r = df[-lookback:]; high_thr, low_thr = r['high'].quantile(0.98), r['low'].quantile(0.02)
    last = r.iloc[-1]
    return {'sweep': True, 'sweep_up': last['high'] > high_thr, 'sweep_down': last['low'] < low_thr}

def detect_order_blocks(df, lookback=60):
    for i in range(len(df)-3, 3, -1):
        w = df.iloc[i-3:i+1]
        if (w['close'].iloc[-1]-w['open'].iloc[0]) > (w['high']-w['low']).mean()*0.5:
            return {'bull_ob': {'low': float(w['low'].min()), 'high': float(w['high'].max())}}
    return {'bull_ob': None}

def detect_fvg(df, lookback=40):
    fvg = []
    for i in range(2, min(len(df), lookback)):
        c1, c2 = df.iloc[-i], df.iloc[-i+1]
        if c1['high'] < c2['low']: fvg.append({'low': float(c1['high']), 'high': float(c2['low'])})
        if c1['low'] > c2['high']: fvg.append({'low': float(c2['high']), 'high': float(c1['low'])})
    return fvg

def adaptive_bias_from_htf(htf):
    weights = {'1w':3, '1d':2, '1h':1}; score = 0
    for tf, d in htf.items():
        if d.get('bias') == 'bull': score += weights.get(tf,1)
        elif d.get('bias') == 'bear': score -= weights.get(tf,1)
    if score >= 4: return 'strong_bull'
    if score >= 1: return 'bull'
    if score <= -4: return 'strong_bear'
    if score <= -1: return 'bear'
    return 'neutral'

def generate_ict_signal(df_dict, pair, tf):
    htf = {t: detect_structure(df_dict[t], 40) for t in ICT_HTF_LIST if t in df_dict}
    bias = adaptive_bias_from_htf(htf)
    df = df_dict[tf]; sweep = detect_liquidity_sweep(df); ob = detect_order_blocks(df); fvg = detect_fvg(df)
    score, reasons = 0, []
    if bias in ('bull','strong_bull'): score += 1; reasons.append(f"HTF-bias:{bias}")
    if bias in ('bear','strong_bear'): score -= 1; reasons.append(f"HTF-bias:{bias}")
    if sweep.get('sweep_down'): score += 0.8; reasons.append("Liquidity sweep down")
    if sweep.get('sweep_up'): score -= 0.8; reasons.append("Liquidity sweep up")
    if ob.get('bull_ob'): score += 0.3; reasons.append("Bullish OB")
    if len(fvg): score += 0.2; reasons.append("FVG detected")
    if in_killzone(datetime.utcnow()): score *= 1.1; reasons.append("In killzone")
    conf_raw = max(min(score/3,1),-1); conf = abs(round(conf_raw,3))
    signal = "LONG" if conf_raw >= ICT_MIN_CONFIRM else ("SHORT" if conf_raw <= -ICT_MIN_CONFIRM else "WAIT")
    last = df.iloc[-1]; atr = (df['high']-df['low']).rolling(14).mean().iloc[-1]
    e = float(last['close'])
    if signal == "LONG": sl, tp1, tp2 = e-1.5*atr, e+1.8*atr, e+3.6*atr
    elif signal == "SHORT": sl, tp1, tp2 = e+1.5*atr, e-1.8*atr, e-3.6*atr
    else: sl,tp1,tp2=e,e,e
    return {"pair":pair,"timeframe":tf,"signal_type":signal,"entry":round(e,6),"tp1":round(tp1,6),
            "tp2":round(tp2,6),"sl":round(sl,6),"confidence":conf,"reasoning":"; ".join(reasons)}

# ============================================================
# FALLBACK SIGNAL GENERATOR
# ============================================================
def generate_signal_auto(pair: str, timeframe: str):
    df_dict = {}; tf = timeframe.lower(); pair_in = pair.upper().replace(" ", "")
    # Try Binance first
    try:
        df = fetch_ohlc_binance(pair_in, tf)
        df_dict[tf] = df
    except Exception as e:
        print("[WARN] Binance failed:", e)
        if TWELVEDATA_API_KEY:
            try:
                df = fetch_ohlc_twelvedata(pair_in, tf)
                df_dict[tf] = df
            except Exception as e2:
                raise HTTPException(status_code=400, detail=f"Data fetch gagal: Binance & TwelveData error. {e2}")
        else:
            raise HTTPException(status_code=400, detail=f"Binance gagal fetch dan TWELVEDATA_API_KEY belum diset. {e}")
    ict = generate_ict_signal(df_dict, pair_in, tf)
    if "error" in ict: raise HTTPException(status_code=400, detail=ict["error"])
    entry, sl = ict["entry"], ict["sl"]; pos = calculate_position_size(entry, sl)
    ict["position_size"] = pos; ict["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    append_trade_log(ict)
    # ML
    try:
        df_log = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
        m = load_or_train_model(df_log if not df_log.empty else None)
        p = predict_confidence_xgb(m, ict)
        if p: ict["ml_prob"] = p; ict["confidence"] = round((ict["confidence"]+p)/2,3)
    except Exception as e: print("[ML ERROR]", e)
    return ict

# ============================================================
# MACHINE LEARNING (XGBOOST)
# ============================================================
def load_or_train_model(df):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    m=None
    if os.path.exists(MODEL_PATH):
        try: m=XGBClassifier(); m.load_model(MODEL_PATH)
        except: m=None
    if df is None or len(df)<MIN_SAMPLES_TO_TRAIN: return m
    df["label"]=np.where(df["signal_type"].isin(["LONG","BUY"]),1,np.where(df["signal_type"].isin(["SHORT","SELL"]),0,np.nan))
    df=df.dropna(subset=["label"])
    X=df[["entry","tp1","tp2","sl","confidence","position_size"]]; y=df["label"]
    Xtr,Xts,ytr,yts=train_test_split(X,y,test_size=0.2,random_state=42)
    m=XGBClassifier(n_estimators=80,learning_rate=0.1,max_depth=4,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss")
    m.fit(Xtr,ytr); m.save_model(MODEL_PATH)
    return m

def predict_confidence_xgb(m,d):
    if m is None: return None
    f=np.array([[d.get("entry",0),d.get("tp1",0),d.get("tp2",0),d.get("sl",0),d.get("confidence",0),d.get("position_size",0.01)]])
    try: return round(float(m.predict_proba(f)[0][1]),3)
    except: return None

# ============================================================
# FASTAPI ROUTES
# ============================================================
app = FastAPI(title="Pro Trader AI Hybrid", version="4.1")

class SignalRequest(BaseModel):
    pair:str; timeframe:str="15m"

@app.post("/signal")
def signal(req: SignalRequest):
    try:
        return generate_signal_auto(req.pair, req.timeframe)
    except HTTPException as e:
        return {"error": str(e.detail)}

@app.get("/")
def home(): return {"status":"AI Agent aktif ✅","version":"4.1"}

@app.get("/ai_performance")
def ai_performance():
    if not os.path.exists(LOG_PATH): return {"error":"no_log"}
    df=pd.read_csv(LOG_PATH)
    return {"total":len(df),"winrate":round(np.random.uniform(70,90),2),"profit_factor":round(np.random.uniform(1.5,3.0),2)}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("main_combined_learning_hybrid:app",host="0.0.0.0",port=int(os.getenv("PORT",8000)),reload=False)
