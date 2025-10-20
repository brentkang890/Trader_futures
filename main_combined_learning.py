main_combined_learning_v3.py

""" Pro Trader AI v3 - Combined + Learning + Vision + Robust Startup

FastAPI service for signals (crypto + csv + image)

RandomForest auto-learning from logged signals

Telegram integration (text & image) non-blocking

Keep-alive + auto-redeploy hooks (safe/no auto-trigger)


Notes:

This file is production-ready for a small Railway-like container.

Configure environment variables listed below before deploy.


ENV VARS REQUIRED / OPTIONAL:

APP_URL (optional for keep-alive; recommended)

TELEGRAM_TOKEN (optional: to enable bot)

CHAT_ID (optional: default chat for notifications)

BACKTEST_URL (optional: endpoint to send signals for evaluation)

RAILWAY_TOKEN / PROJECT_ID (optional: if you add auto-redeploy using Railway API)


Run with: uvicorn main_combined_learning_v3:app --host 0.0.0.0 --port $PORT """

import os import io import time import threading import requests import logging from datetime import datetime from typing import Optional, Dict, Any

import numpy as np import pandas as pd from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException from fastapi.responses import JSONResponse

optional libs that may be heavy on some platforms

try: import ta except Exception: ta = None

try: from sklearn.ensemble import RandomForestClassifier from sklearn.model_selection import train_test_split from sklearn.metrics import classification_report, roc_auc_score import joblib except Exception: RandomForestClassifier = None train_test_split = None classification_report = None roc_auc_score = None joblib = None

image/ocr libs

from PIL import Image import cv2 try: import pytesseract _HAS_TESSERACT = True except Exception: pytesseract = None _HAS_TESSERACT = False

telegram (optional)

try: import telebot except Exception: telebot = None

configure logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s") logger = logging.getLogger("pro_trader_ai")

app = FastAPI(title="Pro Trader AI v3", version="3.0")

---------------- CONFIG ----------------

BACKTEST_URL = os.getenv("BACKTEST_URL") TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trade_log.csv") MODEL_FILE = os.getenv("MODEL_FILE", "rf_model.pkl") MIN_SAMPLES_TO_TRAIN = int(os.getenv("MIN_SAMPLES_TO_TRAIN", "50")) N_SIGNALS_TO_RETRAIN = int(os.getenv("N_SIGNALS_TO_RETRAIN", "50")) BINANCE_KLINES = os.getenv("BINANCE_KLINES", "https://api.binance.com/api/v3/klines") APP_URL = os.getenv("APP_URL") TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") CHAT_ID = os.getenv("CHAT_ID")

thread-safety / state

_lock = threading.Lock() _last_retrain_count = 0

---------------- UTILITIES ----------------

def fetch_ohlc_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame: """Fetch klines from Binance and return standardized DataFrame.""" if not symbol: raise ValueError("symbol_required") r = requests.get(BINANCE_KLINES, params={"symbol": symbol.upper(), "interval": interval, "limit": limit}, timeout=10) r.raise_for_status() data = r.json() df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","close_time","qav","num_trades","tb_base","tb_quote","ignore"])  # Binance full columns for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c], errors="coerce") return df[["open_time","open","high","low","close","volume"]].rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})

defensive wrappers for ta

def _ema(series, n): if ta is None: raise RuntimeError("ta_library_missing") return ta.trend.EMAIndicator(series, window=n).ema_indicator()

def _rsi(series, n=14): if ta is None: raise RuntimeError("ta_library_missing") return ta.momentum.RSIIndicator(series, window=n).rsi()

def _atr(df, n=14): if ta is None: raise RuntimeError("ta_library_missing") return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=n).average_true_range()

small helpers

def detect_sr(df: pd.DataFrame, lookback:int=120): recent_h = df['high'].tail(lookback).max() recent_l = df['low'].tail(lookback).min() return float(recent_h), float(recent_l)

def breakout_of_structure(df: pd.DataFrame, window:int=20): if df.shape[0] < window+2: return None high_sw = df['high'].rolling(window).max().iloc[-2] low_sw = df['low'].rolling(window).min().iloc[-2] last = df['close'].iloc[-1] prev = df['close'].iloc[-2] if prev <= high_sw and last > high_sw: return "BOS_UP" if prev >= low_sw and last < low_sw: return "BOS_DOWN" return None

---------------- STRATEGY ----------------

def hybrid_analyze(df: pd.DataFrame, pair:Optional[str]=None, timeframe:Optional[str]=None) -> dict: df = df.copy().dropna().reset_index(drop=True) if df.shape[0] < 12: return {"error":"data_tidak_cukup", "message":"Perlu minimal 12 candle untuk analisis."}

df['ema20'] = _ema(df['close'], 20)
df['ema50'] = _ema(df['close'], 50)
df['rsi14'] = _rsi(df['close'], 14)
df['atr14'] = _atr(df, 14)

last = df.iloc[-1]
price = float(last['close'])
ema20 = float(last['ema20'])
ema50 = float(last['ema50'])
rsi_now = float(last['rsi14'])
atr_now = float(last['atr14']) if not np.isnan(last['atr14']) else max(price*0.001, 1e-8)

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
    conf.append(0.9 if trend=="bullish" else 0.6)
    conf.append(0.9 if price >= fib_618 else 0.65)
    conf.append(1.0 if 30 < rsi_now < 75 else 0.5)
    signal="LONG"
elif bos == "BOS_DOWN" or (trend == "bearish" and price < ema20):
    entry = price
    sl = recent_high + atr_now*0.6
    rr = sl - entry if sl>entry else price*0.01
    tp1 = entry - rr*1.5
    tp2 = entry - rr*2.5
    reasons.append("Bias SHORT — BOS turun & EMA searah bearish.")
    conf.append(0.9 if trend=="bearish" else 0.6)
    conf.append(0.9 if price <= fib_618 else 0.65)
    conf.append(1.0 if 25 < rsi_now < 70 else 0.5)
    signal="SHORT"
else:
    entry = price
    sl = recent_low * 0.995
    tp1 = entry + (entry-sl)*1.2
    tp2 = entry + (entry-sl)*2.0
    reasons.append("Belum ada arah jelas — tunggu konfirmasi TF lebih tinggi.")
    conf.append(0.25)
    signal="WAIT"

confidence = float(sum(conf)/len(conf)) if conf else 0.0
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

---------------- SCALP ENGINE ----------------

def scalp_engine(df: pd.DataFrame, pair:Optional[str]=None, tf:Optional[str]=None) -> dict: if df.shape[0] < 30: return {"error": "data_tidak_cukup"}

df['ema8'] = _ema(df['close'], 8)
df['ema21'] = _ema(df['close'], 21)
df['rsi14'] = _rsi(df['close'], 14)
df['atr14'] = _atr(df, 14)

last = df.iloc[-1]
price = float(last['close'])
atr_now = float(last['atr14']) if not np.isnan(last['atr14']) else price * 0.001
vol_mean = df['volume'].tail(40).mean() if df.shape[0] >= 40 else df['volume'].mean()
vol_spike = float(last['volume']) > (vol_mean * 1.8 if vol_mean > 0 else False)

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

---------------- LOGGING ----------------

def ensure_trade_log(): if not os.path.exists(TRADE_LOG_FILE): df = pd.DataFrame(columns=[ "id", "timestamp", "pair", "timeframe", "signal_type", "entry", "tp1", "tp2", "sl", "confidence", "reasoning", "backtest_hit", "backtest_pnl" ]) df.to_csv(TRADE_LOG_FILE, index=False)

def append_trade_log(record: Dict[str, Any]) -> int: ensure_trade_log() df = pd.read_csv(TRADE_LOG_FILE) next_id = int(df['id'].max()) + 1 if not df.empty else 1 record_row = { "id": next_id, "timestamp": datetime.utcnow().isoformat(), "pair": record.get("pair"), "timeframe": record.get("timeframe"), "signal_type": record.get("signal_type"), "entry": record.get("entry"), "tp1": record.get("tp1"), "tp2": record.get("tp2"), "sl": record.get("sl"), "confidence": record.get("confidence"), "reasoning": record.get("reasoning"), "backtest_hit": record.get("backtest_hit"), "backtest_pnl": record.get("backtest_pnl") } df = pd.concat([df, pd.DataFrame([record_row])], ignore_index=True) df.to_csv(TRADE_LOG_FILE, index=False) return next_id

---------------- BACKTEST COMM ----------------

def post_to_backtester(payload: Dict[str, Any]) -> Dict[str, Any]: if not BACKTEST_URL: return {"error": "BACKTEST_URL_not_configured"} try: r = requests.post(BACKTEST_URL, json=payload, timeout=15) try: return r.json() except Exception: return {"status_code": r.status_code, "text": r.text} except Exception as e: return {"error": "backtester_unreachable", "detail": str(e)}

---------------- FEATURE ENGINEERING ----------------

def compute_features_for_row(pair: str, timeframe: str, entry: float, tp: Optional[float], sl: float) -> Optional[Dict[str, float]]: try: kdf = fetch_ohlc_binance(pair, timeframe, limit=200) except Exception: return None

kdf = kdf.tail(60).reset_index(drop=True)
close = kdf['close'].astype(float)
high = kdf['high'].astype(float)
low = kdf['low'].astype(float)
vol = kdf['volume'].astype(float)

ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
rsi14 = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1] if ta else 0.0
atr14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1] if ta else 0.0
vol_mean = vol.tail(40).mean() if len(vol) >= 40 else vol.mean()
vol_now = vol.iloc[-1]
vol_spike = 1.0 if vol_now > vol_mean * 1.8 else 0.0

recent_high = high.tail(80).max()
recent_low = low.tail(80).min()
dist_to_high = (recent_high - entry) / entry if entry else 0.0
dist_to_low = (entry - recent_low) / entry if entry else 0.0
rr = abs((tp - entry) / (entry - sl)) if (tp is not None and (entry - sl) != 0) else 0.0

return {
    "ema8_21_diff": (ema8 - ema21) / entry if entry else 0.0,
    "rsi14": float(rsi14),
    "atr_rel": float(atr14) / entry if entry else 0.0,
    "vol_spike": float(vol_spike),
    "dist_to_high": float(dist_to_high),
    "dist_to_low": float(dist_to_low),
    "rr": float(rr)
}

---------------- LEARNING SYSTEM ----------------

def build_dataset_from_trade_log(): if not os.path.exists(TRADE_LOG_FILE): return None, None df = pd.read_csv(TRADE_LOG_FILE) rows, labels = [], [] for _, r in df.iterrows(): hit = str(r.get("backtest_hit", "")).upper() if hit == "" or hit == "NO_HIT": label = 0 elif hit.startswith("TP"): label = 1 elif hit.startswith("SL"): label = 0 else: label = 0

feats = compute_features_for_row(
        str(r['pair']),
        str(r['timeframe'] or "15m"),
        float(r['entry']),
        r.get('tp1', None),
        float(r['sl'])
    )
    if feats is None:
        continue
    rows.append(feats)
    labels.append(label)

if not rows:
    return None, None
X = pd.DataFrame(rows)
y = pd.Series(labels)
return X, y

def train_and_save_model(): global _last_retrain_count if RandomForestClassifier is None or joblib is None: return {"status": "sklearn_missing"}

X, y = build_dataset_from_trade_log()
if X is None or len(y) < MIN_SAMPLES_TO_TRAIN:
    return {"status": "data_tidak_cukup", "samples": 0 if y is None else len(y)}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
yprob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

report = classification_report(y_test, y_pred, output_dict=True) if classification_report else None
auc = roc_auc_score(y_test, yprob) if (roc_auc_score and yprob is not None) else None

joblib.dump({"clf": clf, "features": list(X.columns)}, MODEL_FILE)

try:
    df = pd.read_csv(TRADE_LOG_FILE)
    _last_retrain_count = len(df)
except Exception:
    _last_retrain_count = 0

return {"status": "trained", "samples": len(y), "auc": auc, "report": report}

def predict_with_model(payload: Dict[str, Any]): if joblib is None: raise RuntimeError("joblib_missing") if not os.path.exists(MODEL_FILE): raise RuntimeError("model_belum_dilatih") mod = joblib.load(MODEL_FILE) clf = mod["clf"] pair = payload.get("pair") timeframe = payload.get("timeframe") or "15m" entry = float(payload.get("entry")) tp = payload.get("tp") or payload.get("tp1") sl = float(payload.get("sl"))

feats = compute_features_for_row(pair, timeframe, entry, tp, sl)
if feats is None:
    raise RuntimeError("gagal_menghitung_fitur")
X = pd.DataFrame([feats])
prob = float(clf.predict_proba(X)[:, 1][0]) if hasattr(clf, "predict_proba") else float(clf.predict(X)[0])
return {"prob": prob, "features": feats}

---------------- BACKGROUND RETRAIN ----------------

def maybe_trigger_retrain_background(): def worker(): try: res = train_and_save_model() logger.info(f"Retrain result: {res}") except Exception as e: logger.exception("Retrain error") t = threading.Thread(target=worker, daemon=True) t.start()

def check_and_trigger_retrain_if_needed(): global _last_retrain_count with _lock: try: df = pd.read_csv(TRADE_LOG_FILE) total = len(df) if _last_retrain_count == 0: _last_retrain_count = total if total - _last_retrain_count >= N_SIGNALS_TO_RETRAIN: _last_retrain_count = total maybe_trigger_retrain_background() except Exception as e: logger.exception("check_retrain error")

---------------- POSTPROCESS WITH LEARNING ----------------

def _postprocess_with_learning(signal: Dict[str, Any]) -> Dict[str, Any]: try: if os.path.exists(MODEL_FILE): pred = predict_with_model({ "pair": signal.get("pair"), "timeframe": signal.get("timeframe"), "entry": signal.get("entry"), "tp": signal.get("tp1"), "sl": signal.get("sl") }) prob = pred.get("prob", 0.0) orig = float(signal.get("confidence", 0.5)) new_conf = round(max(min(1.0, 0.9 * orig + 0.1 * prob), 0.0), 3) signal["confidence"] = new_conf signal["model_prob"] = round(prob, 3) if prob < 0.35: signal["vetoed_by_model"] = True signal["signal_type"] = "WAIT" else: signal["vetoed_by_model"] = False else: signal["model_prob"] = None signal["vetoed_by_model"] = False except Exception as e: signal["model_error"] = str(e) return signal

---------------- API ENDPOINTS ----------------

@app.get("/") def root(): return {"status": "ok", "service": "Pro Trader AI v3", "time": datetime.utcnow().isoformat()}

@app.get("/health") def health(): return {"status": "ok", "service": "Pro Trader AI v3"}

@app.get("/pro_signal") def pro_signal(pair: str = Query(...), tf_main: str = Query("1h"), tf_entry: str = Query("15m"), limit: int = Query(300), auto_log: bool = Query(False)): try: df_entry = fetch_ohlc_binance(pair, tf_entry, limit=limit) except Exception as e: raise HTTPException(status_code=500, detail=f"fetch_failed: {e}")

res = hybrid_analyze(df_entry, pair=pair, timeframe=tf_entry)
try:
    df_main = fetch_ohlc_binance(pair, tf_main, limit=200)
    ema20_main = _ema(df_main['close'], 20).iloc[-1]
    ema50_main = _ema(df_main['close'], 50).iloc[-1]
    res['context_main_trend'] = "bullish" if float(ema20_main) > float(ema50_main) else "bearish"
except Exception:
    pass

res = _postprocess_with_learning(res)

if auto_log:
    payload_bt = {"pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"], "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"), "sl": res["sl"], "confidence": res["confidence"], "reason": res["reasoning"]}
    bt_res = post_to_backtester(payload_bt)
    res["backtest_raw"] = bt_res
    logrec = {"pair": res["pair"], "timeframe": res["timeframe"], "signal_type": res["signal_type"], "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"), "sl": res["sl"], "confidence": res["confidence"], "reasoning": res["reasoning"], "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None, "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None}
    append_trade_log(logrec)
    check_and_trigger_retrain_if_needed()

return JSONResponse(res)

@app.get("/scalp_signal") def scalp_signal(pair: str = Query(...), tf: str = Query("3m"), limit: int = Query(300), auto_log: bool = Query(False)): try: df = fetch_ohlc_binance(pair, tf, limit=limit) except Exception as e: raise HTTPException(status_code=500, detail=f"fetch_failed: {e}")

res = scalp_engine(df, pair=pair, tf=tf)
res = _postprocess_with_learning(res)

if auto_log:
    payload_bt = {"pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"], "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"), "sl": res["sl"], "confidence": res["confidence"], "reason": res["reasoning"]}
    bt_res = post_to_backtester(payload_bt)
    res["backtest_raw"] = bt_res
    logrec = {"pair": res["pair"], "timeframe": res["timeframe"], "signal_type": res["signal_type"], "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"), "sl": res["sl"], "confidence": res["confidence"], "reasoning": res["reasoning"], "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None, "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None}
    append_trade_log(logrec)
    check_and_trigger_retrain_if_needed()

return JSONResponse(res)

@app.post("/analyze_chart") def analyze_chart(file: UploadFile = File(...), pair: Optional[str] = Form(None), timeframe: Optional[str] = Form(None), auto_backtest: Optional[str] = Form("true")): auto_flag = str(auto_backtest).lower() != "false" try: contents = file.file.read() pil = Image.open(io.BytesIO(contents)).convert("RGB") img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR) except Exception as e: raise HTTPException(status_code=400, detail=f"gambar_tidak_valid: {e}")

if not _HAS_TESSERACT:
    raise HTTPException(status_code=400, detail="tesseract_ocr_tidak_tersedia")

try:
    y_map = ocr_y_axis_prices(img_cv)
except Exception:
    y_map = {}

df_ohlc = detect_candles_from_plot(img_cv, y_map, max_bars=200)
if df_ohlc.empty:
    raise HTTPException(status_code=400, detail="gagal_membaca_chart")

for col in ['open','high','low','close']:
    df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors='coerce')

res = hybrid_analyze(df_ohlc, pair=pair or "IMG", timeframe=timeframe or "img")
res = _postprocess_with_learning(res)

if auto_flag:
    bt_res = post_to_backtester({"pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"], "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"), "sl": res["sl"], "confidence": res["confidence"], "reason": res["reasoning"]})
    res["backtest_raw"] = bt_res
    append_trade_log({"pair": res["pair"], "timeframe": res["timeframe"], "signal_type": res["signal_type"], "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"), "sl": res["sl"], "confidence": res["confidence"], "reasoning": res["reasoning"]})
    check_and_trigger_retrain_if_needed()

res['bars_used'] = int(df_ohlc.shape[0])
return JSONResponse(res)

@app.post("/analyze_csv") def analyze_csv(file: UploadFile = File(...), pair: Optional[str] = Form(None), timeframe: Optional[str] = Form(None)): try: contents = file.file.read() df = pd.read_csv(io.BytesIO(contents)) except Exception as e: raise HTTPException(status_code=400, detail=f"invalid_csv: {e}")

df.columns = [c.strip().lower() for c in df.columns]
def find_col(k):
    return next((c for c in df.columns if k in c), None)

o, h, l, ccol = find_col('open'), find_col('high'), find_col('low'), find_col('close')
if not all([o, h, l, ccol]):
    raise HTTPException(status_code=400, detail="kolom_tidak_lengkap (butuh open, high, low, close)")

df2 = df[[o, h, l, ccol]].rename(columns={o: 'open', h: 'high', l: 'low', ccol: 'close'})
for col in ['open','high','low','close']:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')
df2 = df2.dropna().reset_index(drop=True)

res = hybrid_analyze(df2, pair=pair or "CSV", timeframe=timeframe or "csv")
res = _postprocess_with_learning(res)
return JSONResponse(res)

@app.get("/learning_status") def learning_status(): info = {"model_exists": os.path.exists(MODEL_FILE)} if info["model_exists"] and joblib is not None: mod = joblib.load(MODEL_FILE) info["features"] = mod.get("features") try: df = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else pd.DataFrame() info["trade_log_count"] = len(df) except Exception: info["trade_log_count"] = 0 return JSONResponse(info)

@app.get("/retrain_learning") def retrain_learning(): res = train_and_save_model() return JSONResponse(res)

@app.get("/logs") def get_logs(limit: int = Query(100)): ensure_trade_log() df = pd.read_csv(TRADE_LOG_FILE) df = df.tail(limit).to_dict(orient="records") return JSONResponse({"logs": df})

---------------- OCR / VISION HELPERS ----------------

def ocr_y_axis_prices(img): if not _HAS_TESSERACT: return {} gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT) y_map = {} for i, text in enumerate(data.get("text", [])): t = str(text).strip().replace(',', '') try: # accept numbers like 1234.56 if t and (t.replace('.', '', 1).isdigit() or (t.replace('.', '', 1).replace('-', '', 1).isdigit())): y = data["top"][i] val = float(t) y_map[y] = val except Exception: continue return y_map

def detect_candles_from_plot(img, y_map, max_bars=200): gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) , th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) candles = [] if len(y_map) < 2: return pd.DataFrame(columns=["open","high","low","close"]) ys = sorted(y_map.keys()); y0, y1 = ys[0], ys[-1] v0, v1 = y_map[y0], y_map[y1] def px_to_price(y): return v0 + (v1 - v0) * ((y - y0) / (y1 - y0)) for c in contours: x,y,w,h = cv2.boundingRect(c) if 5 < w < 200 and 5 < h < 800: high = px_to_price(y) low = px_to_price(y+h) open = (high + low) / 2 close_ = open_ candles.append([open_, high, low, close_]) df = pd.DataFrame(candles[-max_bars:], columns=["open","high","low","close"]) return df

---------------- TELEGRAM BOT (NON-BLOCKING) ----------------

if telebot and TELEGRAM_TOKEN: try: bot = telebot.TeleBot(TELEGRAM_TOKEN)

@bot.message_handler(commands=['start'])
    def _welcome(msg):
        bot.reply_to(msg, "👋 Halo! Kirim pair dan timeframe (contoh: BTCUSDT 1h) atau kirim gambar chart.")

    @bot.message_handler(func=lambda m: True, content_types=['text'])
    def _handle_text(msg):
        try:
            parts = msg.text.strip().upper().split()
            if len(parts) < 1:
                bot.reply_to(msg, "Format salah. Contoh: BTCUSDT 1H")
                return
            pair = parts[0]
            tf = parts[1] if len(parts) > 1 else '15m'
            # call pro_signal endpoint
            url = f"{APP_URL}/pro_signal?pair={pair}&tf_main={tf}&tf_entry={tf}"
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                d = r.json()
                bot.send_message(msg.chat.id, f"📈 {pair} ({tf})\nEntry: {d.get('entry')}\nTP1: {d.get('tp1')} | TP2: {d.get('tp2')}\nSL: {d.get('sl')}\nConf: {d.get('confidence')}\n{d.get('reasoning')}", parse_mode='HTML')
            else:
                bot.send_message(msg.chat.id, f"Gagal: {r.text}")
        except Exception as e:
            bot.send_message(msg.chat.id, f"Error: {e}")

    @bot.message_handler(content_types=['photo'])
    def _handle_photo(msg):
        try:
            file_id = msg.photo[-1].file_id
            file_info = bot.get_file(file_id)
            file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_info.file_path}"
            img_data = requests.get(file_url).content
            files = {'file': ('chart.png', img_data, 'image/png')}
            data = {'pair': 'IMG', 'timeframe': 'chart'}
            r = requests.post(f"{APP_URL}/analyze_chart", files=files, data=data, timeout=40)
            if r.status_code == 200:
                d = r.json()
                bot.send_message(msg.chat.id, f"📊 Analisis Chart\nEntry: {d.get('entry')}\nTP1: {d.get('tp1')} | TP2: {d.get('tp2')}\nSL: {d.get('sl')}\nSignal: {d.get('signal_type')}\nConf: {d.get('confidence')}\n{d.get('reasoning')}", parse_mode='HTML')
            else:
                bot.send_message(msg.chat.id, f"Gagal analisa gambar: {r.text}")
        except Exception as e:
            bot.send_message(msg.chat.id, f"Error: {e}")

    threading.Thread(target=lambda: bot.polling(non_stop=True), daemon=True).start()
    logger.info("Telegram bot started (non-blocking)")
except Exception as e:
    logger.exception("Failed to start telegram bot")

else: logger.info("Telegram not configured or telebot lib missing")

---------------- KEEP-ALIVE (non-blocking) ----------------

def keep_alive_loop(): while True: try: if APP_URL: try: r = requests.get(APP_URL, timeout=10) logger.info(f"[KEEP-ALIVE] {r.status_code} {datetime.utcnow().isoformat()}") except Exception as e: logger.warning(f"[KEEP-ALIVE] ping failed: {e}") else: logger.debug("APP_URL not set; skipping keep-alive ping") except Exception as e: logger.exception("Keep alive loop error") time.sleep(300)

threading.Thread(target=keep_alive_loop, daemon=True).start()

logger.info("Pro Trader AI v3 ready")

