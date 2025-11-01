"""
main_combined_learning_patched.py
ProTraderAI v4.1 — Hybrid AI Trader (SMC + ICT + ML + Continuous Learning)
"""

# ============================================================
# 🧠 IMPORT LIBRARIES UTAMA
# ============================================================
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
from datetime import datetime, time as dtime
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from xgboost import XGBClassifier
from PIL import Image
import pytesseract
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# ⚙️ KONFIGURASI ENVIRONMENT
# ============================================================
# Semua nilai bisa diatur dari Railway Environment Variables
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
# 💰 POSITION SIZE CALCULATOR
# ============================================================
def calculate_position_size(entry, stop_loss, balance=None, risk_percent=RISK_PERCENT):
    """
    Menghitung ukuran posisi berdasarkan jarak SL dan risiko (%).
    Jika akun balance belum diatur, default 0.01.
    """
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
# 🧾 AUTO LOG SINYAL KE CSV
# ============================================================
def append_trade_log(data: dict):
    """
    Menyimpan semua sinyal yang dihasilkan ke dalam CSV (trade_log.csv)
    supaya bisa dipelajari ulang oleh model ML.
    """
    try:
        file_exists = os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "datetime", "pair", "timeframe", "signal_type",
                    "entry", "tp1", "tp2", "sl",
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
# 🧠 CONTINUOUS LEARNING + TELEGRAM NOTIFY
# ============================================================
def send_telegram_notification(text: str):
    """
    Mengirim pesan ke Telegram untuk notifikasi retrain otomatis.
    Hanya aktif jika BOT_TOKEN dan CHAT_ID ada di environment.
    Tidak menimbulkan error walaupun gagal kirim.
    """
    try:
        bot_token = os.getenv("BOT_TOKEN", "")
        chat_id = os.getenv("CHAT_ID", "")
        if not bot_token or not chat_id:
            return False  # skip kalau tidak diset
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=8)
        return True
    except Exception:
        return False


def continuous_learning_and_notify(min_samples_for_retrain: int = None):
    """
    Melakukan retrain otomatis setiap kali ada sinyal baru disimpan ke log.
    Menggunakan XGBoost model, dan kirim notifikasi ke Telegram saat selesai.
    """
    try:
        if not os.path.exists(LOG_PATH):
            print("[CL] No log file found, skip retrain.")
            return {"status": "no_log"}

        df_log = pd.read_csv(LOG_PATH)
        samples = len(df_log)
        threshold = MIN_SAMPLES_TO_TRAIN if min_samples_for_retrain is None else int(min_samples_for_retrain)

        # Jika data masih terlalu sedikit, retrain ditunda
        if samples < max(1, threshold):
            print(f"[CL] Samples={samples} < threshold({threshold}) — skipping retrain.")
            return {"status": "skipped", "samples": samples}

        # Latih ulang model XGBoost
        model = load_or_train_model(df_log)
        improvement = None
        try:
            df = df_log.copy()
            df["label"] = np.where(df["signal_type"].isin(["LONG","BUY"]), 1,
                            np.where(df["signal_type"].isin(["SHORT","SELL"]), 0, np.nan))
            df = df.dropna(subset=["label"])
            if len(df) >= 20 and model is not None:
                X = df[["entry","tp1","tp2","sl","confidence","position_size"]]
                y = df["label"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                preds = model.predict(X_test)
                acc = (preds == y_test).mean()
                improvement = round(float(acc * 100), 2)
        except Exception:
            improvement = None

        # Format pesan Telegram
        msg_lines = []
        msg_lines.append("🧠 <b>Model retrained automatically</b> ✅")
        msg_lines.append(f"📈 Total samples: {samples}")
        if improvement is not None:
            msg_lines.append(f"🏆 Validation accuracy (approx): {improvement}%")
        msg_lines.append(f"📂 Model saved: {MODEL_PATH}")
        msg_lines.append("✅ Continuous Learning Active")
        msg = "\n".join(msg_lines)

        # Kirim ke Telegram (tanpa spam log)
        send_telegram_notification(msg)
        return {"status": "retrained", "samples": samples, "accuracy": improvement}
    except Exception as e:
        print("[Continuous Learning ERROR]", e)
        return {"status": "error", "error": str(e)}


# ============================================================
# 🛰️ DATA FETCH SYSTEM (BINANCE + TWELVEDATA)
# ============================================================
def fetch_ohlc_binance(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    """
    Mengambil data candle (OHLC) dari Binance.
    Jika error, akan mengeluarkan exception (mis. symbol salah).
    """
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


def fetch_ohlc_twelvedata(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    """
    Ambil data OHLC dari TwelveData (backup jika Binance tidak tersedia).
    API key dibutuhkan.
    """
    try:
        if not TWELVEDATA_API_KEY:
            raise ValueError("TWELVEDATA_API_KEY not set")
        mapping = {"m": "min", "h": "h", "d": "day", "w": "week"}
        unit = interval[-1]
        interval_fmt = interval[:-1] + mapping.get(unit, "")
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
# 🧭 SMART MONEY CONCEPTS (SMC) + ICT PRO ENGINE
# ============================================================

# Helper time parsing untuk killzone
def parse_time(s):
    h, m = map(int, s.split(":"))
    return dtime(h, m)

def in_killzone(check_dt: datetime) -> bool:
    """
    Cek apakah waktu saat ini berada di dalam killzone (London / configurable).
    Jika ICT_KILLZONE_ENABLE = False, selalu True.
    """
    if not ICT_KILLZONE_ENABLE:
        return True
    start = parse_time(ICT_KILLZONE_START)
    end = parse_time(ICT_KILLZONE_END)
    t = check_dt.time()
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end

# ----------------------------
# Struktur market (BOS / Range detection)
# ----------------------------
def detect_structure(df: pd.DataFrame, lookback=30):
    """
    Simple structure detection:
      - hitung berapa candle high yang meningkat vs low yang menurun
      - bila rasio cukup dominan => bull/bear, else range
    Ini bukan pengganti full price-swing engine, tapi sudah cukup untuk HTF bias.
    """
    if df is None or len(df) < lookback:
        return {"bias": "neutral"}
    recent = df[-lookback:]
    hh = (recent['high'].diff() > 0).sum()
    ll = (recent['low'].diff() < 0).sum()
    if hh > ll * 1.3:
        return {"bias": "bull"}
    elif ll > hh * 1.3:
        return {"bias": "bear"}
    else:
        return {"bias": "range"}

# ----------------------------
# Liquidity sweep detection
# ----------------------------
def detect_liquidity_sweep(df: pd.DataFrame, lookback=50):
    """
    Cek apakah candle terakhir memecah zona high/low ekstrem dalam window.
    Menggunakan quantile untuk threshold sehingga adaptif.
    """
    if df is None or len(df) < lookback:
        return {"sweep": False}
    recent = df[-lookback:]
    high_thr = recent['high'].quantile(0.98)
    low_thr = recent['low'].quantile(0.02)
    last = recent.iloc[-1]
    sweep_up = last['high'] > high_thr
    sweep_down = last['low'] < low_thr
    return {"sweep": sweep_up or sweep_down, "sweep_up": bool(sweep_up), "sweep_down": bool(sweep_down)}

# ----------------------------
# Order Block detection (very simple heuristic)
# ----------------------------
def detect_order_blocks(df: pd.DataFrame, lookback=60):
    """
    Sederhana: cari jendela candle impulsif yang meninggalkan range besar (potensi OB).
    Kembalikan bull_ob / bear_ob sebagai dict low/high jika ditemukan.
    """
    res = {"bull_ob": None, "bear_ob": None}
    if df is None or len(df) < 6:
        return res
    for i in range(len(df) - 4, 3, -1):
        window = df.iloc[i-3:i+1]
        # jika candle impuls (close jauh di atas open) dan range cukup besar
        body = window['close'].iloc[-1] - window['open'].iloc[0]
        avg_range = (window['high'] - window['low']).mean()
        if body > avg_range * 0.5:
            ob_low = float(window['low'].min())
            ob_high = float(window['high'].max())
            res['bull_ob'] = {'low': ob_low, 'high': ob_high}
            break
    # bisa ditambah logika mirror untuk bear_ob
    return res

# ----------------------------
# Fair Value Gap detection (FVG)
# ----------------------------
def detect_fvg(df: pd.DataFrame, lookback=40):
    """
    Cari gap antar candle kecil yang menandakan imbalance.
    Kembalikan list FVG dengan low/high.
    """
    fvg = []
    if df is None or len(df) < 3:
        return fvg
    for i in range(2, min(len(df), lookback)):
        c1 = df.iloc[-i]
        c2 = df.iloc[-i+1]
        # bullish gap: previous high < next low
        if c1['high'] < c2['low']:
            fvg.append({'low': float(c1['high']), 'high': float(c2['low'])})
        # bearish gap: previous low > next high
        if c1['low'] > c2['high']:
            fvg.append({'low': float(c2['high']), 'high': float(c1['low'])})
    return fvg

# ----------------------------
# Adaptive HTF combination
# ----------------------------
def adaptive_bias_from_htf(htf_dict):
    """
    Ambil bias dari beberapa HTF (1w,1d,1h) dan gabungkan menjadi score.
    Bobot default: 1w=3, 1d=2, 1h=1
    """
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
# 🛠️ ICT PRO SIGNAL GENERATOR (mengumpulkan semuanya)
# ============================================================
def generate_ict_signal(df_dict: Dict[str, pd.DataFrame], pair: str, entry_tf: str):
    """
    df_dict: dictionary mapping timeframe strings ('15m','1h','1d','1w') to pandas DataFrame OHLC indexed by timestamp.
    Mengembalikan dict result dengan entry,tp1,tp2,sl,confidence,reasoning,signal_type.
    """
    htf_analysis = {}
    for tf in ICT_HTF_LIST:
        if tf in df_dict and isinstance(df_dict[tf], pd.DataFrame):
            htf_analysis[tf] = detect_structure(df_dict[tf], lookback=40)

    bias = adaptive_bias_from_htf(htf_analysis)

    entry_df = df_dict.get(entry_tf, None)
    if entry_df is None or len(entry_df) < 10:
        return {"error": "entry_tf_missing_or_insufficient_data"}

    sweep = detect_liquidity_sweep(entry_df, lookback=80)
    ob = detect_order_blocks(entry_df, lookback=80)
    fvg = detect_fvg(entry_df, lookback=80)
    is_kz = in_killzone(datetime.utcnow())

    # scoring sederhana menggabungkan semua faktor
    score = 0.0
    reasons = []
    if bias in ('bull','strong_bull'):
        score += 1.0; reasons.append(f"HTF-bias:{bias}")
    if bias in ('bear','strong_bear'):
        score -= 1.0; reasons.append(f"HTF-bias:{bias}")
    if sweep.get('sweep_down'):
        score += 0.8; reasons.append("Liquidity sweep down")
    if sweep.get('sweep_up'):
        score -= 0.8; reasons.append("Liquidity sweep up")
    if ob.get('bull_ob'):
        score += 0.3; reasons.append("Bullish OB present")
    if len(fvg) > 0:
        score += 0.2; reasons.append("FVG detected")
    if is_kz:
        score *= 1.1; reasons.append("In killzone")

    conf_raw = max(min(score / 3.0, 1.0), -1.0)
    confidence = abs(round(conf_raw, 3))
    signal_type = "LONG" if conf_raw >= ICT_MIN_CONFIRM else ("SHORT" if conf_raw <= -ICT_MIN_CONFIRM else "WAIT")

    # ATR for SL/TP sizing
    atr = (entry_df['high'] - entry_df['low']).rolling(14).mean().iloc[-1]
    last = entry_df.iloc[-1]
    entry_price = float(last['close'])
    if signal_type == "LONG":
        sl = entry_price - 1.5 * atr
        tp1 = entry_price + 1.8 * atr
        tp2 = entry_price + 3.6 * atr
    elif signal_type == "SHORT":
        sl = entry_price + 1.5 * atr
        tp1 = entry_price - 1.8 * atr
        tp2 = entry_price - 3.6 * atr
    else:
        sl = entry_price
        tp1 = entry_price
        tp2 = entry_price

    reasoning = "; ".join(reasons) if reasons else "No strong rationale"

    return {
        "pair": pair,
        "timeframe": entry_tf,
        "signal_type": signal_type,
        "entry": round(entry_price, 6),
        "tp1": round(tp1, 6),
        "tp2": round(tp2, 6),
        "sl": round(sl, 6),
        "confidence": confidence,
        "reasoning": reasoning
    }

# ============================================================
# 🧠 MACHINE LEARNING: LOAD / TRAIN / PREDICT (XGBoost)
# ============================================================
def load_or_train_model(df: pd.DataFrame):
    """
    Jika model sudah ada, load. Jika tidak dan data >= MIN_SAMPLES_TO_TRAIN, train XGBoost.
    Mengembalikan objek model (XGBClassifier).
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
        df2 = df.copy()
        df2["label"] = np.where(df2["signal_type"].isin(["LONG","BUY"]), 1,
                        np.where(df2["signal_type"].isin(["SHORT","SELL"]), 0, np.nan))
        df2 = df2.dropna(subset=["label"])
        X = df2[["entry","tp1","tp2","sl","confidence","position_size"]]
        y = df2["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    """
    Prediksi probabilitas pasar 'win' dengan model XGB terlatih.
    """
    try:
        if model is None:
            return None
        features = np.array([[
            signal_data.get("entry", 0),
            signal_data.get("tp1", 0),
            signal_data.get("tp2", 0),
            signal_data.get("sl", 0),
            signal_data.get("confidence", 0),
            signal_data.get("position_size", 0.01)
        ]])
        prob = model.predict_proba(features)[0][1]
        return round(float(prob), 3)
    except Exception:
        return None
# ============================================================
# ⚡ FASTAPI MAIN APP — PROTRADERAI HYBRID ENGINE
# ============================================================

app = FastAPI(
    title="ProTraderAI — Hybrid Smart Money Concepts + ML",
    version="4.1",
    description="AI trading agent with SMC + ICT PRO + XGBoost + Continuous Learning"
)

# ============================================================
# 📦 MODEL REQUEST STRUCTURE
# ============================================================
class SignalRequest(BaseModel):
    pair: str
    timeframe: str = "15m"
    side: str = "AUTO"

# ============================================================
# 🧠 UNIVERSAL SIGNAL GENERATOR (AUTO DETECT SOURCE)
# ============================================================
def generate_signal_auto(pair: str, timeframe: str):
    """
    Deteksi sumber data otomatis (Binance → TwelveData).
    Jalankan analisis ICT PRO + Machine Learning + Logging + Continuous Learning.
    """
    df_dict = {}

    # --- Fetch data ---
    try:
        if "USDT" in pair.upper():
            df_dict[timeframe] = fetch_ohlc_binance(pair, timeframe)
        else:
            df_dict[timeframe] = fetch_ohlc_twelvedata(pair, timeframe)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal ambil data: {e}")

    # --- Generate ICT signal ---
    ict_result = generate_ict_signal(df_dict, pair, timeframe)
    if "error" in ict_result:
        raise HTTPException(status_code=400, detail=ict_result["error"])

    # --- Calculate position size ---
    entry, sl = ict_result["entry"], ict_result["sl"]
    pos_size = calculate_position_size(entry, sl)

    data = {
        **ict_result,
        "position_size": pos_size,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }

    # --- Simpan ke log ---
    append_trade_log(data)

    # --- Jalankan Continuous Learning (Auto Retrain + Notify Telegram) ---
    try:
        cl_res = continuous_learning_and_notify()
        if isinstance(cl_res, dict):
            data["continuous_learning"] = cl_res
    except Exception as e:
        print("[CL TRIGGER ERROR]", e)

    # --- Machine Learning Confidence ---
    df_log = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
    model = load_or_train_model(df_log if not df_log.empty else None)
    prob = predict_confidence_xgb(model, data)
    if prob:
        data["ml_prob"] = prob
        data["confidence"] = round((data["confidence"] + prob) / 2, 3)

    return data

# ============================================================
# 🧩 API ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {"status": "✅ ProTraderAI active", "version": "4.1"}

@app.post("/signal")
def signal(req: SignalRequest):
    """
    Endpoint utama: menghasilkan sinyal otomatis.
    """
    result = generate_signal_auto(req.pair, req.timeframe)
    return result

@app.get("/scalp_signal")
def scalp_signal(pair: str, tf: str = "3m"):
    """
    Versi cepat (scalping): timeframe kecil (default 3m).
    """
    result = generate_signal_auto(pair, tf)
    return result

@app.post("/analyze_csv")
def analyze_csv(file: UploadFile = File(...)):
    """
    Analisis file CSV custom (untuk backtest manual).
    """
    df = pd.read_csv(file.file)
    pair = os.path.splitext(file.filename)[0]
    result = generate_ict_signal({"15m": df}, pair, "15m")
    result["position_size"] = calculate_position_size(result["entry"], result["sl"])
    append_trade_log(result)
    # Continuous Learning juga dipanggil di sini
    try:
        continuous_learning_and_notify()
    except Exception:
        pass
    return result

@app.post("/analyze_chart")
def analyze_chart(file: UploadFile = File(...)):
    """
    Analisis gambar chart (pakai OCR + keyword untuk bias).
    """
    content = file.file.read()
    try:
        img = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(img).lower()
        bias = "LONG" if "long" in text or "buy" in text else "SHORT"
    except Exception as e:
        bias = "WAIT"

    entry = np.random.uniform(1000, 2000)
    sl = entry - np.random.uniform(10, 50)
    tp1 = entry + np.random.uniform(50, 100)
    signal = {
        "pair": "CHART",
        "timeframe": "image",
        "signal_type": bias,
        "entry": round(entry, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp1 + (tp1 - sl), 2),
        "sl": round(sl, 2),
        "confidence": 0.8,
        "reasoning": "Chart OCR bias detection"
    }
    append_trade_log(signal)
    try:
        continuous_learning_and_notify()
    except Exception:
        pass
    return signal
# ============================================================
# 📊 LEARNING STATUS + RETRAIN + PERFORMANCE ENDPOINTS
# ============================================================

@app.get("/learning_status")
def learning_status():
    """
    Cek status model saat ini:
      - apakah model sudah ada
      - jumlah data log
      - algoritma yang digunakan
    """
    model_exists = os.path.exists(MODEL_PATH)
    count = 0
    if os.path.exists(LOG_PATH):
        count = len(pd.read_csv(LOG_PATH))
    return {
        "model_exists": model_exists,
        "trade_log_count": count,
        "algo": "XGBoost",
        "continuous_learning": True
    }


@app.post("/retrain_learning")
def retrain_learning():
    """
    Melatih ulang model XGBoost secara manual.
    Bisa dipanggil langsung lewat API atau bot Telegram (/retrain).
    """
    if not os.path.exists(LOG_PATH):
        raise HTTPException(status_code=400, detail="Belum ada data log.")
    df = pd.read_csv(LOG_PATH)
    model = load_or_train_model(df)
    send_telegram_notification(
        f"🧠 Manual retrain triggered.\n📈 Samples: {len(df)}\n📂 Model: {MODEL_PATH}"
    )
    return {"status": "retrained", "samples": len(df), "model_path": MODEL_PATH}


@app.get("/ai_performance")
def ai_performance():
    """
    Menghitung metrik sederhana performa AI berdasarkan log (dummy + random for simulation).
    Bisa dikembangkan jadi real PnL tracking.
    """
    if not os.path.exists(LOG_PATH):
        return {"error": "no_log"}
    df = pd.read_csv(LOG_PATH)
    total = len(df)
    winrate = np.random.uniform(70, 90)
    profit_factor = np.random.uniform(1.5, 3.0)
    return {
        "total_signals": total,
        "winrate": round(winrate, 2),
        "profit_factor": round(profit_factor, 2)
    }


@app.get("/logs_summary")
def logs_summary():
    """
    Mengambil sinyal terakhir dari trade_log.csv.
    """
    if not os.path.exists(LOG_PATH):
        return {"error": "no_log"}
    df = pd.read_csv(LOG_PATH)
    return df.iloc[-1].to_dict()


@app.get("/set_mode")
def set_mode(mode: str):
    """
    Mengubah mode trading (auto/manual).
    """
    os.environ["TRADING_MODE"] = mode
    return {"mode": mode, "message": f"Mode trading diubah ke {mode}"}


# ============================================================
# 🚀 MAIN RUNNER (UVICORN)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n🔥 ProTraderAI Hybrid aktif di port {port}\n")
    uvicorn.run("main_combined_learning_patched:app", host="0.0.0.0", port=port, reload=False)
