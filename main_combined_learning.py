# main_combined_learning_full_backtest.py
"""
Pro Trader AI Agent - Hybrid Full Backtest Version
By: Evan Leon
Language: Bilingual (EN-ID)
Integrates SMC + ICT + XGBoost + Backtester (Single & Multi Pair)
"""

# ==============================
# 📦 IMPORT & LIBRARIES
# ==============================
import os
import io
import csv
import json
import time
import math
import asyncio
import datetime as dt
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Machine Learning
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Image/CSV Analysis
from PIL import Image
import pytesseract
import requests

# ==============================
# ⚙️ CONFIGURATIONS
# ==============================
app = FastAPI(title="Pro Trader AI Agent - Full Backtest")

# Default file paths
LOG_PATH = os.getenv("LOG_PATH", "trade_log.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/ai_model_xgb.json")
MIN_SAMPLES_TO_TRAIN = int(os.getenv("MIN_SAMPLES_TO_TRAIN", "50"))

# Trading parameters
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.02"))
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "1000"))

# API keys
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
ALPHA_API_KEY = os.getenv("ALPHA_API_KEY", "")

# Source selector
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "twelvedata")

# Killzone setup (ICT)
ICT_KILLZONE_ENABLE = os.getenv("ICT_KILLZONE_ENABLE", "true").lower() == "true"
ICT_KILLZONE_START = os.getenv("ICT_KILLZONE_START", "06:00")
ICT_KILLZONE_END = os.getenv("ICT_KILLZONE_END", "12:00")

ICT_MIN_CONFIRM = float(os.getenv("ICT_MIN_CONFIRM", "0.65"))
ICT_DEFAULT_ENTRY_TF = os.getenv("ICT_DEFAULT_ENTRY_TF", "15m")

# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

# ==============================
# 🧾 LOGGING SYSTEM
# ==============================
def ensure_log_headers():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp","pair","timeframe","side",
                "entry","tp1","tp2","sl",
                "hit","pnl","confidence","reasoning","source"
            ])

def append_log(entry: dict):
    ensure_log_headers()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            entry.get("timestamp", dt.datetime.utcnow().isoformat()),
            entry.get("pair"), entry.get("timeframe"), entry.get("side"),
            entry.get("entry"), entry.get("tp1"), entry.get("tp2"), entry.get("sl"),
            entry.get("hit"), entry.get("pnl"), entry.get("confidence"),
            entry.get("reasoning"), entry.get("source", "ai")
        ])

ensure_log_headers()

# ==============================
# 🧠 LOAD & SAVE MODEL
# ==============================
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            print("✅ Model loaded from", MODEL_PATH)
            return model
        except Exception as e:
            print("⚠️ Gagal load model:", e)
    return None

def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    model.save_model(MODEL_PATH)

MODEL = load_model()

# ==============================
# 📈 SIMPLE ICT & SMC UTILITIES
# ==============================
def in_killzone_now(start_str=ICT_KILLZONE_START, end_str=ICT_KILLZONE_END):
    if not ICT_KILLZONE_ENABLE:
        return True
    now = dt.datetime.utcnow().time()
    start = dt.datetime.strptime(start_str, "%H:%M").time()
    end = dt.datetime.strptime(end_str, "%H:%M").time()
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end

def simple_smc_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Deteksi sederhana struktur pasar (BOS/CHoCH, liquidity sweep, FVG)."""
    reasoning, conf = [], 0.5
    if df.shape[0] < 10:
        return {"bias": "neutral", "reasoning": "data kurang", "confidence": 0.3}

    last = df.tail(10)
    highs, lows = last["high"], last["low"]

    # Trend structure
    if highs.max() == highs.iloc[-1] and lows.max() == lows.iloc[-1]:
        bias = "long"; conf += 0.15; reasoning.append("struktur naik (HH/HL)")
    elif highs.min() == highs.iloc[-1] and lows.min() == lows.iloc[-1]:
        bias = "short"; conf += 0.15; reasoning.append("struktur turun (LL/LH)")
    else:
        bias = "neutral"; reasoning.append("tidak jelas arah tren")

    # FVG (Fair Value Gap)
    fvg = 0
    for i in range(-5, -1):
        prev_high, next_low = df["high"].iloc[i], df["low"].iloc[i+1]
        if next_low > prev_high:
            fvg += 1
    if fvg:
        reasoning.append(f"{fvg} FVG terdeteksi"); conf += 0.05 * fvg

    # Liquidity sweep
    last_candle, prev = df.iloc[-1], df.iloc[-2]
    if last_candle["high"] > prev["high"]:
        reasoning.append("sapu likuiditas atas")
        conf += 0.05
    if last_candle["low"] < prev["low"]:
        reasoning.append("sapu likuiditas bawah")
        conf += 0.05

    return {"bias": bias, "reasoning": "; ".join(reasoning), "confidence": min(conf, 1.0)}

# ==============================
# 📊 FETCH DATA (TwelveData, Alpha, CSV fallback)
# ==============================
def fetch_ohlc(pair: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Ambil data OHLC dari TwelveData, AlphaVantage, atau CSV lokal."""
    # 1️⃣ TwelveData
    if TWELVEDATA_API_KEY:
        url = f"https://api.twelvedata.com/time_series?symbol={pair}&interval={interval}&outputsize={limit}&apikey={TWELVEDATA_API_KEY}"
        r = requests.get(url, timeout=15)
        j = r.json()
        if "values" in j:
            df = pd.DataFrame(j["values"])
            df = df.rename(columns={"datetime":"datetime","open":"open","high":"high","low":"low","close":"close","volume":"volume"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.sort_values("datetime").reset_index(drop=True)
            return df

    # 2️⃣ AlphaVantage (backup)
    if ALPHA_API_KEY:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={pair}&interval={interval}&apikey={ALPHA_API_KEY}&outputsize=compact"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            try:
                data = r.json()
                key = next(k for k in data.keys() if "Time Series" in k)
                df = pd.DataFrame(data[key]).T
                df.columns = ["open","high","low","close","volume"]
                df = df.astype(float)
                df["datetime"] = pd.to_datetime(df.index)
                df = df.sort_values("datetime").reset_index(drop=True)
                return df
            except Exception:
                pass

    # 3️⃣ CSV fallback
    local_path = f"data/{pair}_{interval}.csv"
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    raise RuntimeError(f"Tidak bisa ambil data untuk {pair} ({interval}) — pastikan API key aktif atau file CSV tersedia.")

# ==============================
# 🧮 XGBOOST TRAINING UTILS
# ==============================
def build_features_from_log():
    if not os.path.exists(LOG_PATH):
        return None, None
    df = pd.read_csv(LOG_PATH)
    if df.shape[0] < 10:
        return None, None
    df = df.dropna(subset=["pair","timeframe","side","hit","pnl","confidence"])
    df["label"] = df["hit"].apply(lambda x: 1 if str(x).upper()=="TP" else 0)
    df["side_num"] = df["side"].map({"LONG":1,"SHORT":0}).fillna(1)
    def tf_to_min(tf):
        tf = str(tf).lower()
        if "m" in tf: return int(tf.replace("m",""))
        if "h" in tf: return int(tf.replace("h",""))*60
        if "d" in tf: return int(tf.replace("d",""))*1440
        return 15
    df["tf_min"] = df["timeframe"].apply(tf_to_min)
    X = df[["side_num","tf_min","pnl","confidence"]].fillna(0)
    y = df["label"]
    return X, y

def retrain_model(force=False):
    global MODEL
    X, y = build_features_from_log()
    if X is None or y is None:
        return {"status": "not_enough_data"}
    if not force and X.shape[0] < MIN_SAMPLES_TO_TRAIN:
        return {"status": "need_more_samples", "have": X.shape[0], "need": MIN_SAMPLES_TO_TRAIN}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    save_model(model)
    MODEL = model
    return {"status":"trained","accuracy":acc,"samples":X.shape[0]}

# ==============================
# END OF PART 1
# ==============================
# ==============================
# PART 2 — BACKTEST ENGINE & FEEDBACK LOOP
# ==============================

# ------------------------------
# 🔁 SIMULATE BACKTEST (Candle-by-candle)
# ------------------------------
def simulate_backtest_on_df(df: pd.DataFrame, side: str, entry: float, tp1: float, sl: float) -> Dict[str, Any]:
    """
    Simulasi sederhana: mulai dari candle entry (jika entry dalam range candle),
    lalu periksa candle demi candle apakah TP atau SL tercapai lebih dulu.
    Kembalikan: {'hit': 'TP'|'SL'|'NONE', 'pnl': float, 'bars': int}
    """
    entry_idx = None
    for i in range(len(df)):
        if df['low'].iloc[i] <= entry <= df['high'].iloc[i]:
            entry_idx = i
            break
    if entry_idx is None:
        entry_idx = 0

    hit = "NONE"
    pnl = 0.0
    bars = 0
    for j in range(entry_idx + 1, len(df)):
        bars += 1
        h = df['high'].iloc[j]
        l = df['low'].iloc[j]
        o = df['open'].iloc[j]
        if side.upper() == "LONG":
            if h >= tp1 and l <= sl:
                # heuristik: bandingkan jarak relatif dari open
                if abs(tp1 - o) < abs(o - sl):
                    hit = "TP"
                else:
                    hit = "SL"
            elif h >= tp1:
                hit = "TP"
            elif l <= sl:
                hit = "SL"
        else:  # SHORT
            if l <= tp1 and h >= sl:
                if abs(tp1 - o) < abs(o - sl):
                    hit = "TP"
                else:
                    hit = "SL"
            elif l <= tp1:
                hit = "TP"
            elif h >= sl:
                hit = "SL"
        if hit != "NONE":
            if side.upper() == "LONG":
                pnl = (tp1 - entry) if hit == "TP" else (sl - entry)
            else:
                pnl = (entry - tp1) if hit == "TP" else (entry - sl)
            break
    return {"hit": hit, "pnl": float(pnl), "bars": bars}

# ------------------------------
# 🧾 BACKTEST SINGLE ENDPOINT
# ------------------------------
from pydantic import ValidationError

class BacktestReq(BaseModel):
    pair: str
    side: str
    entry: float
    tp1: float
    sl: float
    timeframe: Optional[str] = ICT_DEFAULT_ENTRY_TF
    limit: Optional[int] = 500

@app.post("/backtest")
async def backtest_endpoint(body: BacktestReq):
    """
    Backtest single pair/timeframe.
    Body: pair, side, entry, tp1, sl, timeframe (optional)
    """
    try:
        pair = body.pair.upper()
        tf = body.timeframe or ICT_DEFAULT_ENTRY_TF
        df = fetch_ohlc(pair, tf, limit=body.limit or 500)
        sim = simulate_backtest_on_df(df, body.side, body.entry, body.tp1, body.sl)
        result = {
            "pair": pair,
            "timeframe": tf,
            "side": body.side.upper(),
            "entry": body.entry,
            "tp1": body.tp1,
            "sl": body.sl,
            "hit": sim["hit"],
            "pnl_total": sim["pnl"],
            "bars": sim["bars"],
            "timestamp": dt.datetime.utcnow().isoformat()
        }
        # log hasil backtest
        append_log({
            "timestamp": result["timestamp"],
            "pair": result["pair"],
            "timeframe": result["timeframe"],
            "side": result["side"],
            "entry": result["entry"],
            "tp1": result["tp1"],
            "tp2": None,
            "sl": result["sl"],
            "hit": result["hit"],
            "pnl": result["pnl_total"],
            "confidence": 0.0,
            "reasoning": "backtest_simulation",
            "source": "backtest"
        })
        # optionally call AI feedback endpoint if set (non-blocking)
        ai_feedback_url = os.getenv("AI_FEEDBACK_URL", "")
        if ai_feedback_url:
            try:
                requests.post(ai_feedback_url, json=result, timeout=3)
            except:
                pass
        return result
    except ValidationError as ve:
        return JSONResponse({"error":"validation","detail":ve.errors()}, status_code=400)
    except Exception as e:
        return JSONResponse({"error":"exception","detail":str(e)}, status_code=500)

# ------------------------------
# 🧩 BACKTEST MULTI ENDPOINT
# ------------------------------
class BacktestMultiReq(BaseModel):
    pairs: List[str]
    timeframes: List[str]
    limit: Optional[int] = 500

@app.post("/backtest_multi")
async def backtest_multi_endpoint(body: BacktestMultiReq):
    """
    Backtest multiple pairs & timeframes.
    Body: {"pairs": [...], "timeframes": [...], "limit": 500}
    """
    try:
        results = []
        for raw_pair in body.pairs:
            pair = raw_pair.upper()
            for tf in body.timeframes:
                try:
                    df = fetch_ohlc(pair, tf, limit=body.limit or 500)
                    # gunakan analisis SMC untuk tentukan side & entry heuristik
                    smc = simple_smc_analysis(df)
                    bias = smc.get("bias", "neutral")
                    last_close = float(df["close"].iloc[-1])
                    # ATR-like
                    df["range"] = df["high"] - df["low"]
                    atr = float(df["range"].rolling(14).mean().iloc[-1]) if df.shape[0] >= 14 else float(df["range"].mean())
                    if bias == "long":
                        entry = last_close * 1.001
                        sl = entry - atr * 1.5
                        tp1 = entry + atr * 2
                        side = "LONG"
                    elif bias == "short":
                        entry = last_close * 0.999
                        sl = entry + atr * 1.5
                        tp1 = entry - atr * 2
                        side = "SHORT"
                    else:
                        # skip neutral
                        continue
                    sim = simulate_backtest_on_df(df, side, entry, tp1, sl)
                    res = {
                        "pair": pair,
                        "timeframe": tf,
                        "side": side,
                        "entry": round(entry,6),
                        "tp1": round(tp1,6),
                        "sl": round(sl,6),
                        "hit": sim["hit"],
                        "pnl": sim["pnl"],
                        "bars": sim["bars"],
                        "confidence": 0.0,
                        "reasoning": smc.get("reasoning","")
                    }
                    results.append(res)
                    # log each
                    append_log({
                        "timestamp": dt.datetime.utcnow().isoformat(),
                        "pair": res["pair"],
                        "timeframe": res["timeframe"],
                        "side": res["side"],
                        "entry": res["entry"],
                        "tp1": res["tp1"],
                        "tp2": None,
                        "sl": res["sl"],
                        "hit": res["hit"],
                        "pnl": res["pnl"],
                        "confidence": res["confidence"],
                        "reasoning": "backtest_multi",
                        "source": "backtest"
                    })
                except Exception as inner_e:
                    # skip pair/tf jika error ambil data
                    print("backtest error for", pair, tf, inner_e)
                    continue
        total_pnl = sum(r["pnl"] for r in results)
        wins = sum(1 for r in results if r["hit"] == "TP")
        total = len(results)
        avg_winrate = round((wins / total) * 100, 2) if total > 0 else 0.0
        summary = {"average_winrate": avg_winrate, "total_pnl": total_pnl, "count": total}
        return {"results": results, "summary": summary}
    except ValidationError as ve:
        return JSONResponse({"error":"validation","detail":ve.errors()}, status_code=400)
    except Exception as e:
        return JSONResponse({"error":"exception","detail":str(e)}, status_code=500)

# ------------------------------
# 🔁 AI FEEDBACK (terima hasil eksekusi/backtest dari sumber lain)
# ------------------------------
@app.post("/ai_feedback")
async def ai_feedback_endpoint(data: Dict[str, Any]):
    """
    Terima feedback dari backtester atau eksekusi live.
    Data example: {pair, timeframe, side, entry, tp1, sl, hit, pnl, timestamp, confidence}
    """
    try:
        append_log({
            "timestamp": data.get("timestamp", dt.datetime.utcnow().isoformat()),
            "pair": data.get("pair"),
            "timeframe": data.get("timeframe"),
            "side": data.get("side"),
            "entry": data.get("entry"),
            "tp1": data.get("tp1"),
            "tp2": data.get("tp2"),
            "sl": data.get("sl"),
            "hit": data.get("hit"),
            "pnl": data.get("pnl"),
            "confidence": data.get("confidence", 0.0),
            "reasoning": data.get("reasoning", "feedback"),
            "source": data.get("source", "external")
        })
        # kalau jumlah log sudah mencapai ambang, trigger retrain background
        try:
            df = pd.read_csv(LOG_PATH)
            if df.shape[0] >= MIN_SAMPLES_TO_TRAIN:
                loop = asyncio.get_event_loop()
                loop.create_task(async_retrain_background())
        except Exception:
            pass
        return {"status":"logged"}
    except Exception as e:
        return JSONResponse({"error":"exception","detail":str(e)}, status_code=500)

# ------------------------------
# 🔧 Background retrain task (async)
# ------------------------------
async def async_retrain_background():
    await asyncio.sleep(1)  # beri jeda kecil
    try:
        res = retrain_model(force=True)
        print("Background retrain result:", res)
    except Exception as e:
        print("Background retrain failed:", e)

# ==============================
# END OF PART 2
# ==============================
# ==============================
# PART 3 — API ENDPOINTS FINAL, TELEGRAM NOTIFY, STARTUP
# ==============================

# ------------------------------
# 🔔 Optional: Telegram notify helper (useful untuk notifikasi auto-update)
# ------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
BOT_CHAT_ID = os.getenv("CHAT_ID", "")

def telegram_notify(text: str):
    """Kirim notifikasi ke Telegram jika BOT_TOKEN & CHAT_ID tersedia."""
    if not BOT_TOKEN or not BOT_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": BOT_CHAT_ID, "text": text[:4000], "parse_mode": "HTML"}
        requests.post(url, json=payload, timeout=8)
        return True
    except Exception as e:
        print("telegram_notify error:", e)
        return False

# ------------------------------
# 🧾 SIGNAL ENDPOINT (SMC + ML blended)
# ------------------------------
class SignalReq(BaseModel):
    pair: str
    timeframe: Optional[str] = ICT_DEFAULT_ENTRY_TF

@app.post("/signal")
async def signal_endpoint(req: SignalReq):
    """
    Endpoint utama: gabungkan SMC (struktur/OB/FVG/liquidity) + ML blending.
    Response: pair, timeframe, signal_type, entry, tp1, tp2, sl, confidence, reasoning, ml_prob
    """
    try:
        pair = req.pair.upper()
        tf = req.timeframe or ICT_DEFAULT_ENTRY_TF

        # ambil candles
        df = fetch_ohlc(pair, tf, limit=300)

        # simple SMC analysis
        smc = simple_smc_analysis(df)
        bias = smc.get("bias", "neutral")
        reasoning = smc.get("reasoning", "")
        conf = float(smc.get("confidence", 0.5))

        # jika pakai killzone, cek sekarang
        if ICT_KILLZONE_ENABLE and not in_killzone_now(ICT_KILLZONE_START, ICT_KILLZONE_END):
            return JSONResponse({"error":"outside_killzone","reasoning":"Killzone aktif — tunggu sesi yang valid"}, status_code=200)

        # ML blending jika model ada
        ml_prob = None
        if MODEL is not None:
            try:
                side_guess = 1 if bias == "long" else 0
                tf_min = int(''.join([c for c in tf if c.isdigit()])) if tf else 15
                pnl_sample = 0.0
                X = pd.DataFrame([[side_guess, tf_min, pnl_sample, conf]], columns=['side_num','tf_min','pnl','confidence'])
                ml_prob = float(MODEL.predict_proba(X)[0][1])
                conf = round(0.6 * conf + 0.4 * ml_prob, 4)
            except Exception as e:
                print("ML predict error:", e)
                ml_prob = None

        # jika neutral -> wait
        if bias == "neutral":
            return {"signal_type":"WAIT","reasoning":reasoning,"confidence":conf}

        # kalkulasi entry/sl/tp berbasis ATR-like
        df["range"] = df["high"] - df["low"]
        atr = float(df["range"].rolling(14).mean().iloc[-1]) if df.shape[0] >= 14 else float(df["range"].mean())
        last_close = float(df["close"].iloc[-1])

        if bias == "long":
            entry = round(last_close * 1.0005, 6)
            sl = round(entry - atr * 1.5, 6)
            tp1 = round(entry + atr * 2, 6)
            side = "LONG"
        else:
            entry = round(last_close * 0.9995, 6)
            sl = round(entry + atr * 1.5, 6)
            tp1 = round(entry - atr * 2, 6)
            side = "SHORT"

        result = {
            "pair": pair,
            "timeframe": tf,
            "signal_type": side,
            "entry": entry,
            "tp1": tp1,
            "tp2": None,
            "sl": sl,
            "confidence": conf,
            "reasoning": reasoning,
            "ml_prob": ml_prob
        }
        # optional: log suggestion (source=signal) — tidak dianggap sebagai eksekusi
        append_log({
            "timestamp": dt.datetime.utcnow().isoformat(),
            "pair": pair,
            "timeframe": tf,
            "side": side,
            "entry": entry,
            "tp1": tp1,
            "tp2": None,
            "sl": sl,
            "hit": "NONE",
            "pnl": 0.0,
            "confidence": conf,
            "reasoning": "signal_suggest",
            "source": "signal"
        })
        return result
    except Exception as e:
        return JSONResponse({"error":"exception","detail":str(e)}, status_code=500)

# ------------------------------
# ⚡ SCALP SIGNAL (shortcut)
# ------------------------------
@app.get("/scalp_signal")
async def scalp_signal(pair: str = "BTCUSDT", tf: str = "3m"):
    req = SignalReq(pair=pair, timeframe=tf)
    return await signal_endpoint(req)

# ------------------------------
# 🖼️ ANALYZE CHART (OCR heuristic)
# ------------------------------
@app.post("/analyze_chart")
async def analyze_chart(file: UploadFile = File(...)):
    try:
        bytes_data = await file.read()
        im = Image.open(io.BytesIO(bytes_data))
        text = pytesseract.image_to_string(im)
        text_up = text.upper()
        if any(k in text_up for k in ["BUY", "LONG", "BULL"]):
            return {"signal_type":"LONG","confidence":0.7,"reasoning":"OCR detected buy/long"}
        if any(k in text_up for k in ["SELL","SHORT","BEAR"]):
            return {"signal_type":"SHORT","confidence":0.7,"reasoning":"OCR detected sell/short"}
        return {"signal_type":"WAIT","confidence":0.3,"reasoning":"OCR found no clear signal","raw_text": text[:800]}
    except Exception as e:
        return JSONResponse({"error":"exception","detail":str(e)}, status_code=500)

# ------------------------------
# 📄 ANALYZE CSV (basic SMC on CSV)
# ------------------------------
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        res = simple_smc_analysis(df)
        return {"pair": df.get("pair", ["CSV"])[0] if "pair" in df.columns else "CSV", "timeframe": "csv", "signal_type": res["bias"].upper(), "confidence": res["confidence"], "reasoning": res["reasoning"]}
    except Exception as e:
        return JSONResponse({"error":"exception","detail":str(e)}, status_code=500)

# ------------------------------
# 📊 LEARNING STATUS & PERFORMANCE
# ------------------------------
@app.get("/learning_status")
async def learning_status():
    model_exists = MODEL is not None
    count = 0
    try:
        if os.path.exists(LOG_PATH):
            df = pd.read_csv(LOG_PATH)
            count = df.shape[0]
    except:
        count = 0
    return {"model_exists": model_exists, "trade_log_count": count, "algo": "xgboost" if model_exists else "none"}

@app.get("/ai_performance")
async def ai_performance():
    if not os.path.exists(LOG_PATH):
        return {"total_signals":0, "winrate":0.0, "profit_factor":0.0, "model_status": "no_model"}
    df = pd.read_csv(LOG_PATH)
    total = df.shape[0]
    wins = df[df["hit"] == "TP"].shape[0] if "hit" in df.columns else 0
    losses = df[df["hit"] == "SL"].shape[0] if "hit" in df.columns else 0
    winrate = round((wins / total) * 100, 2) if total > 0 else 0.0
    gross_win = df[df["hit"] == "TP"]["pnl"].sum() if "pnl" in df.columns else 0.0
    gross_loss = abs(df[df["hit"] == "SL"]["pnl"].sum()) if "pnl" in df.columns else 0.0
    pf = round((gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0), 3)
    return {"total_signals": total, "winrate": winrate, "profit_factor": pf, "model_status": "loaded" if MODEL is not None else "not_loaded"}

# ------------------------------
# 🧾 LOGS SUMMARY (recent)
# ------------------------------
@app.get("/logs_summary")
async def logs_summary(limit: int = 10):
    try:
        if not os.path.exists(LOG_PATH):
            return {"detail": "no_log"}
        df = pd.read_csv(LOG_PATH)
        df = df.tail(limit).fillna("")
        recs = df.to_dict(orient="records")
        return {"recent": recs}
    except Exception as e:
        return JSONResponse({"error":"exception","detail":str(e)}, status_code=500)

# ------------------------------
# 🔁 RETRAIN TRIGGER (manual)
# ------------------------------
@app.post("/retrain_learning")
async def retrain_learning(force: Optional[str] = Form(None)):
    try:
        force_flag = str(force).lower() == "true"
        res = retrain_model(force=force_flag)
        # optional notify via telegram
        try:
            msg = f"🔁 Retrain result: {res}"
            telegram_notify(msg)
        except:
            pass
        return res
    except Exception as e:
        return JSONResponse({"error":"exception","detail":str(e)}, status_code=500)

# ------------------------------
# ✅ HEALTH
# ------------------------------
@app.get("/health")
async def health():
    return {"status":"ok","time": dt.datetime.utcnow().isoformat()}

# ------------------------------
# 🔚 START Uvicorn (jika file dijalankan langsung)
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print("Starting Pro Trader AI Agent on port", port)
    uvicorn.run("main_combined_learning_full_backtest:app", host="0.0.0.0", port=port, log_level="info")
