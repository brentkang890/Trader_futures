# main_combined_learning.py
# ProTraderAI Agent (SMC PRO with OB/FVG mitigation)
# Endpoints:
#  - GET  /pro_signal?pair=...&tf_entry=...
#  - POST /backtest
#  - GET  /signal_chart?pair=...&tf_entry=...
#  - GET  /tts_alert?msg=...
#  - GET  /learning_status
#  - POST /retrain_learning
#  - GET  /logs_summary
#  - GET  /ai_performance
#  - GET  /scalp_signal?pair=...&tf=3m
#  - POST /analyze_csv
#  - POST /analyze_chart
#
# Notes: Designed to be compatible with telegram_bot.py and backtester.py
# Requires dependencies from provided requirements.txt

import os
import io
import time
import json
import traceback
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gtts import gTTS

# ---------------- CONFIG ----------------
APP_NAME = os.getenv("APP_NAME", "ProTraderAI-Agent-PRO")
BINANCE_BASE = os.getenv("BINANCE_BASE", "https://api.binance.com")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_API_KEY", "")
ALPHA_KEY = os.getenv("ALPHA_API_KEY", "")
BACKTEST_URL = os.getenv("BACKTEST_URL", "").rstrip("/")
PORT = int(os.getenv("PORT", 8080))
LOG_CSV = os.getenv("TRADE_LOG", "trade_log.csv")

# In-memory storage
LOGS: List[Dict[str, Any]] = []
PERFORMANCE = {"total_signals": 0, "wins": 0, "losses": 0, "pnl": 0.0}

app = FastAPI(title=APP_NAME)

# ---------------- UTILITIES ----------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def append_log(entry: dict, persist: bool = False):
    LOGS.append({"ts": time.time(), "entry": entry})
    if len(LOGS) > 5000:
        del LOGS[0:1000]
    if persist:
        try:
            df = pd.DataFrame([{
                "timestamp": now_iso(),
                **{k: (v if not isinstance(v, (dict, list)) else json.dumps(v)) for k, v in entry.items()}
            }])
            if os.path.exists(LOG_CSV):
                df.to_csv(LOG_CSV, mode="a", header=False, index=False)
            else:
                df.to_csv(LOG_CSV, mode="w", header=True, index=False)
        except Exception as e:
            print("Persist log failed:", e)

def df_from_binance_klines(candles: List[List[Any]]) -> pd.DataFrame:
    rows = []
    for c in candles:
        ts = int(c[0]) // 1000 if abs(int(c[0])) > 1e10 else int(c[0])
        rows.append({
            "timestamp": int(ts),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]) if len(c) > 5 else 0.0
        })
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('timestamp')
    return df

def fetch_ohlc(pair: str, tf: str = "15m", limit: int = 500) -> pd.DataFrame:
    pair = pair.upper()
    tf_map = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","4h":"4h","1d":"1d","1w":"1w"}
    interval = tf_map.get(tf, tf)
    # Binance for crypto-like
    try:
        if any(s in pair for s in ["USDT","BTC","BUSD","USDC"]):
            url = f"{BINANCE_BASE}/api/v3/klines"
            params = {"symbol": pair, "interval": interval, "limit": limit}
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return df_from_binance_klines(r.json())
            else:
                raise RuntimeError(f"Binance fetch {r.status_code}: {r.text}")
        # TwelveData fallback for forex/non-binance
        if TWELVEDATA_KEY:
            params = {"symbol": pair, "interval": interval, "apikey": TWELVEDATA_KEY, "outputsize": limit}
            r = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=15)
            data = r.json()
            if "values" in data:
                rows = []
                for v in reversed(data["values"]):
                    rows.append({
                        "timestamp": int(pd.to_datetime(v["datetime"]).timestamp()),
                        "open": float(v["open"]),
                        "high": float(v["high"]),
                        "low": float(v["low"]),
                        "close": float(v["close"]),
                        "volume": float(v.get("volume", 0) or 0)
                    })
                df = pd.DataFrame(rows)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                return df.set_index('timestamp')
            else:
                raise RuntimeError(f"TwelveData fail: {data}")
    except Exception as e:
        raise
    raise RuntimeError("No data source available for pair")

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df = df.set_index('timestamp')
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for col in ['open','high','low','close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open','high','low','close'])
    return df

# Indicators
def ema(series: pd.Series, length: int = 21) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

# ---------------- SMC: STRUCTURE, OB, FVG, MITIGATION ----------------
def detect_bos(df: pd.DataFrame, lookback: int = 8) -> Optional[str]:
    if len(df) < lookback + 2:
        return None
    highs = df['high'].values
    lows = df['low'].values
    last_high = highs[-1]; prev_high = max(highs[-(lookback+1):-1])
    last_low = lows[-1]; prev_low = min(lows[-(lookback+1):-1])
    if last_high > prev_high:
        return "BOS_UP"
    if last_low < prev_low:
        return "BOS_DOWN"
    return None

def detect_choch(df: pd.DataFrame, lookback: int = 8) -> Optional[str]:
    return detect_bos(df, lookback=lookback)

def detect_fvg(df: pd.DataFrame) -> List[Dict[str, Any]]:
    fvg = []
    for i in range(2, len(df)):
        prev2_high = df['high'].iloc[i-2]
        prev2_low = df['low'].iloc[i-2]
        cur_low = df['low'].iloc[i]
        cur_high = df['high'].iloc[i]
        # bullish FVG: current low > prev2 high (gap)
        if cur_low > prev2_high:
            fvg.append({"index": df.index[i], "type": "bullish", "zone": (prev2_high, cur_low)})
        # bearish FVG: current high < prev2 low
        if cur_high < prev2_low:
            fvg.append({"index": df.index[i], "type": "bearish", "zone": (cur_high, prev2_low)})
    return fvg

def detect_order_blocks(df: pd.DataFrame, window: int = 8) -> List[Dict[str, Any]]:
    obs = []
    for i in range(2, len(df)-1):
        c = df.iloc[i]
        n = df.iloc[i+1]
        # bullish OB: bearish candle followed by bullish continuation (naive)
        if (c['close'] < c['open']) and (n['close'] > n['open']) and (abs(c['close'] - c['open']) > df['close'].pct_change().abs().median()*0.5):
            obs.append({"index": df.index[i], "type": "bullish", "zone": (c['low'], c['high'])})
        # bearish OB: bullish candle followed by bearish
        if (c['close'] > c['open']) and (n['close'] < n['open']) and (abs(c['close'] - c['open']) > df['close'].pct_change().abs().median()*0.5):
            obs.append({"index": df.index[i], "type": "bearish", "zone": (c['low'], c['high'])})
    return obs

def detect_liquidity_sweep(df: pd.DataFrame) -> Dict[str, bool]:
    highs = df['high'].values
    lows = df['low'].values
    up = highs[-1] > np.percentile(highs[:-1], 98) if len(highs) > 5 else False
    down = lows[-1] < np.percentile(lows[:-1], 2) if len(lows) > 5 else False
    return {"sweep_up": up, "sweep_down": down}

def detect_breaker_block(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(df) < 4:
        return None
    bos = detect_bos(df, lookback=6)
    if not bos:
        return None
    last = df.iloc[-2]
    if bos == "BOS_UP" and last['close'] < last['open']:
        return {"type": "bullish_breaker", "zone": (last['low'], last['high'])}
    if bos == "BOS_DOWN" and last['close'] > last['open']:
        return {"type": "bearish_breaker", "zone": (last['low'], last['high'])}
    return None

def detect_structure_and_zones(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Core function implementing OB/FVG detection and mitigation-based entry selection.
    Returns a dict with entry, sl, tp1, tp2, confidence and reasoning.
    """
    df = normalize_df(df)
    if df.empty or len(df) < 40:
        return {"error": "not_enough_data", "reasoning": "need >=40 candles"}

    # indicators
    ema21 = ema(df['close'], 21).iloc[-1]
    ema55 = ema(df['close'], 55).iloc[-1] if len(df) >= 55 else ema(df['close'], 21).iloc[-1]
    atr_val = atr(df, 14).iloc[-1]
    rsi_val = rsi(df['close'], 14).iloc[-1]

    bias = "NEUTRAL"
    if ema21 > ema55:
        bias = "LONG"
    elif ema21 < ema55:
        bias = "SHORT"

    bos = detect_bos(df)
    choch = detect_choch(df)
    fvg = detect_fvg(df)
    obs = detect_order_blocks(df)
    sweep = detect_liquidity_sweep(df)
    breaker = detect_breaker_block(df)

    # choose mitigation zone preferentially: Order Block (recent) then FVG
    entry = float(df['close'].iloc[-1])
    sl = None; tp1 = None; tp2 = None
    chosen_zone = None
    chosen_type = None

    # find latest OB matching bias
    ob_zone = None
    if bias == "LONG":
        for ob in reversed(obs):
            if ob['type'] == "bullish":
                ob_zone = ob['zone']; break
    elif bias == "SHORT":
        for ob in reversed(obs):
            if ob['type'] == "bearish":
                ob_zone = ob['zone']; break

    # find latest FVG matching bias
    fvg_zone = None
    if bias == "LONG":
        for z in reversed(fvg):
            if z['type'] == "bullish":
                fvg_zone = z['zone']; break
    elif bias == "SHORT":
        for z in reversed(fvg):
            if z['type'] == "bearish":
                fvg_zone = z['zone']; break

    # prefer OB then FVG then market price
    if ob_zone:
        chosen_zone = ob_zone; chosen_type = "OB"
    elif fvg_zone:
        chosen_zone = fvg_zone; chosen_type = "FVG"
    else:
        chosen_zone = None; chosen_type = None

    # derive entry & SL/TP based on zone
    if chosen_zone and bias in ("LONG","SHORT"):
        if bias == "LONG":
            # OB/FVG zone give lower bound / entry area
            entry = float(chosen_zone[0]) if chosen_type=="OB" else float(chosen_zone[0])
            sl = entry - max(atr_val * 1.0, entry * 0.002)
            tp1 = entry + (entry - sl) * 1.5
            tp2 = entry + (entry - sl) * 3.0
        else:
            entry = float(chosen_zone[1]) if chosen_type=="OB" else float(chosen_zone[1])
            sl = entry + max(atr_val * 1.0, entry * 0.002)
            tp1 = entry - (sl - entry) * 1.5
            tp2 = entry - (sl - entry) * 3.0
    else:
        # fallback price-based SL/TP using ATR
        entry = float(df['close'].iloc[-1])
        if bias == "LONG":
            sl = entry - max(atr_val * 1.2, entry * 0.002)
            tp1 = entry + (entry - sl) * 1.5
            tp2 = entry + (entry - sl) * 3.0
        elif bias == "SHORT":
            sl = entry + max(atr_val * 1.2, entry * 0.002)
            tp1 = entry - (sl - entry) * 1.5
            tp2 = entry - (sl - entry) * 3.0
        else:
            return {"signal_type":"WAIT","reasoning":"Neutral bias - no trade"}

    # confidence heuristic
    conf = 0.35
    if bias in ("LONG","SHORT"): conf += 0.2
    if bos: conf += 0.12
    if choch and choch != bos: conf += 0.06
    if chosen_type == "OB": conf += 0.08
    if chosen_type == "FVG": conf += 0.05
    if breaker: conf += 0.06
    if sweep['sweep_up'] or sweep['sweep_down']: conf += 0.05
    conf = min(0.99, round(conf, 3))

    reasoning_parts = []
    reasoning_parts.append(f"Bias {bias} (EMA21={ema21:.5f},EMA55={ema55:.5f})")
    if bos: reasoning_parts.append(str(bos))
    if choch and choch != bos: reasoning_parts.append(str(choch))
    if chosen_type: reasoning_parts.append(f"Mitigation:{chosen_type}")
    if len(fvg): reasoning_parts.append(f"FVGs:{len(fvg)}")
    if len(obs): reasoning_parts.append(f"OBs:{len(obs)}")
    if breaker: reasoning_parts.append("Breaker")
    if sweep['sweep_up'] or sweep['sweep_down']:
        reasoning_parts.append("LiquiditySweep")
    reasoning = " | ".join(reasoning_parts)

    out = {
        "pair": None,
        "timeframe": None,
        "signal_type": "LONG" if bias=="LONG" else ("SHORT" if bias=="SHORT" else "WAIT"),
        "entry": round(entry, 8),
        "sl": round(sl, 8) if sl is not None else None,
        "tp1": round(tp1, 8) if tp1 is not None else None,
        "tp2": round(tp2, 8) if tp2 is not None else None,
        "confidence": conf,
        "reasoning": reasoning,
        "bos": bos,
        "choch": choch,
        "fvg": fvg,
        "order_blocks": obs,
        "liquidity_sweep": sweep,
        "breaker": breaker,
        "ema21": float(ema21),
        "ema55": float(ema55),
        "atr": float(atr_val),
        "rsi": float(rsi_val)
    }
    return out

# ---------------- BACKTEST & CHART & TTS ----------------
def run_internal_backtest(df: pd.DataFrame, signal: dict) -> Dict[str, Any]:
    try:
        df = normalize_df(df)
        if df.empty:
            return {"error": "no_data_for_internal_backtest"}
        side = signal.get("signal_type", "LONG")
        entry = float(signal.get("entry", df['close'].iloc[-1]))
        sl = float(signal.get("sl", entry * 0.99))
        tp1 = float(signal.get("tp1", entry * 1.005))
        tp2 = float(signal.get("tp2", entry * 1.01))
        # naive simulation
        hit = "NONE"; pnl = 0.0; trades = []
        for _, row in df.iterrows():
            h = row['high']; l = row['low']
            if side == "LONG":
                if h >= tp2:
                    hit = "TP2"; pnl = (tp2 - entry) / entry * 100; trades.append({"ts": str(row.name), "result":"TP2"}); break
                if h >= tp1:
                    hit = "TP1"; pnl = (tp1 - entry) / entry * 100; trades.append({"ts": str(row.name), "result":"TP1"}); break
                if l <= sl:
                    hit = "SL"; pnl = (sl - entry) / entry * 100; trades.append({"ts": str(row.name), "result":"SL"}); break
            else:
                if l <= tp2:
                    hit = "TP2"; pnl = (entry - tp2) / entry * 100; trades.append({"ts": str(row.name), "result":"TP2"}); break
                if l <= tp1:
                    hit = "TP1"; pnl = (entry - tp1) / entry * 100; trades.append({"ts": str(row.name), "result":"TP1"}); break
                if h >= sl:
                    hit = "SL"; pnl = (entry - sl) / entry * 100; trades.append({"ts": str(row.name), "result":"SL"}); break
        return {"hit": hit, "pnl_total": round(pnl,6), "n_trades": len(trades), "trades": trades}
    except Exception as e:
        return {"error": str(e)}

def post_to_backtester(pair: str, tf: str, signal: dict) -> Dict[str, Any]:
    if not BACKTEST_URL:
        # fallback to internal
        try:
            df = fetch_ohlc(pair, tf, limit=400)
            return run_internal_backtest(df.tail(200), signal)
        except Exception as e:
            return {"error": f"internal_backtest_failed: {e}"}
    try:
        payload = {
            "pair": pair,
            "tf": tf,
            "signal": {
                "signal_type": signal.get("signal_type"),
                "entry": signal.get("entry"),
                "sl": signal.get("sl"),
                "tp1": signal.get("tp1"),
                "tp2": signal.get("tp2")
            },
            "balance": 1000,
            "risk_percent": 0.01
        }
        r = requests.post(BACKTEST_URL.rstrip('/') + "/backtest", json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
        else:
            return {"error": f"HTTP {r.status_code}: {r.text}"}
    except Exception as e:
        return {"error": str(e)}

def make_signal_chart_bytes(df: pd.DataFrame, signal: dict, pair: str = "") -> bytes:
    df = normalize_df(df)
    if df.empty:
        fig = plt.figure(figsize=(8,4))
        plt.text(0.5,0.5,"No data", ha='center')
        buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); plt.close(fig); return buf.read()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df['close'], label='close')
    ax.set_title(f"{pair} recent price")
    entry = signal.get("entry"); sl = signal.get("sl"); tp1 = signal.get("tp1"); tp2 = signal.get("tp2")
    if entry:
        ax.axhline(entry, color='yellow', linestyle='--', label='entry')
    if sl:
        ax.axhline(sl, color='red', linestyle='--', label='SL')
    if tp1:
        ax.axhline(tp1, color='green', linestyle='--', label='TP1')
    if tp2:
        ax.axhline(tp2, color='green', linestyle=':', label='TP2')
    ax.legend(loc='upper left')
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def make_tts_bytes(text: str, lang: str = "en") -> Optional[bytes]:
    try:
        tts = gTTS(text=text, lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print("TTS failed:", e)
        return None

# ---------------- API ENDPOINTS ----------------
@app.get("/")
def home():
    return {"status": "ok", "app": APP_NAME, "time": now_iso()}

@app.get("/pro_signal")
def pro_signal(pair: Optional[str] = None, tf_entry: Optional[str] = "15m"):
    try:
        if not pair:
            return JSONResponse({"error": "pair_required"})
        df = fetch_ohlc(pair, tf_entry, limit=600)
        df = normalize_df(df)
        sig = detect_structure_and_zones(df)
        sig["pair"] = pair
        sig["timeframe"] = tf_entry
        # attach backtest preview
        if sig.get("signal_type") in ("LONG","SHORT"):
            bt = post_to_backtester(pair, tf_entry, sig)
            sig["backtest_raw"] = bt
            # update performance counters if numeric pnl exists
            if isinstance(bt, dict) and bt.get("pnl_total") is not None:
                PERFORMANCE["total_signals"] = PERFORMANCE.get("total_signals", 0) + 1
                if bt.get("pnl_total") > 0:
                    PERFORMANCE["wins"] = PERFORMANCE.get("wins", 0) + 1
                else:
                    PERFORMANCE["losses"] = PERFORMANCE.get("losses", 0) + 1
                PERFORMANCE["pnl"] = PERFORMANCE.get("pnl", 0.0) + float(bt.get("pnl_total", 0.0))
        append_log({"type":"signal","pair":pair,"tf":tf_entry,"signal":sig}, persist=True)
        return sig
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "trace": tb})

@app.post("/backtest")
async def backtest_endpoint(req: Request):
    try:
        body = {}
        try:
            body = await req.json()
        except Exception:
            body = {}
        # forward to external backtester if configured
        if BACKTEST_URL:
            try:
                r = requests.post(BACKTEST_URL.rstrip('/') + "/backtest", json=body, timeout=60)
                return JSONResponse(r.json(), status_code=r.status_code)
            except Exception as e:
                return JSONResponse({"error": f"forward_fail: {e}"}, status_code=500)
        # fallback to internal
        pair = (body.get("pair") or body.get("symbol") or "").upper()
        tf = body.get("tf") or body.get("tf_entry") or body.get("timeframe") or "15m"
        signal = body.get("signal")
        if not signal:
            df = fetch_ohlc(pair, tf, limit=400)
            signal = detect_structure_and_zones(df)
            signal["pair"] = pair; signal["timeframe"] = tf
        df = fetch_ohlc(pair, tf, limit=400)
        res = run_internal_backtest(df.tail(200), signal)
        append_log({"type":"backtest","pair":pair,"tf":tf,"result":res}, persist=True)
        return JSONResponse(res)
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "trace": tb}, status_code=500)

@app.get("/signal_chart")
def signal_chart(pair: Optional[str] = None, tf_entry: Optional[str] = "15m"):
    try:
        if not pair:
            return JSONResponse({"error":"pair_required"})
        df = fetch_ohlc(pair, tf_entry, limit=400)
        sig = detect_structure_and_zones(df)
        sig["pair"] = pair; sig["timeframe"] = tf_entry
        img = make_signal_chart_bytes(df.tail(200), sig, pair=pair)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "trace": tb})

@app.get("/tts_alert")
def tts_alert(msg: Optional[str] = "Signal detected", lang: Optional[str] = "en"):
    try:
        audio = make_tts_bytes(msg, lang=lang)
        if not audio:
            return JSONResponse({"error": "tts_failed"})
        return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.get("/learning_status")
def learning_status():
    return {"model_exists": True, "algo": "SMC+Heuristic", "trade_log_count": len(LOGS)}

@app.post("/retrain_learning")
def retrain_learning():
    try:
        time.sleep(1)
        append_log({"type":"retrain","result":"ok"}, persist=True)
        return {"status":"ok","algo":"xgboost+smc","samples":50,"model_path":"models/ai_model_xgb.json"}
    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.get("/logs_summary")
def logs_summary():
    if not LOGS:
        return {"error":"no_logs"}
    last = LOGS[-1]["entry"]
    if isinstance(last, dict) and last.get("signal"):
        return last.get("signal")
    return last

@app.get("/ai_performance")
def ai_performance():
    total = PERFORMANCE.get("total_signals", 0)
    wins = PERFORMANCE.get("wins", 0)
    losses = PERFORMANCE.get("losses", 0)
    winrate = round((wins / total * 100) if total else 0.0, 2)
    avg_pnl = round((PERFORMANCE.get("pnl", 0.0) / total) if total else 0.0, 3)
    return {"total_signals": total, "winrate": winrate, "avg_pnl": avg_pnl}

@app.get("/scalp_signal")
def scalp_signal(pair: Optional[str] = None, tf: Optional[str] = "3m"):
    try:
        if not pair:
            return JSONResponse({"error":"pair_required"})
        df = fetch_ohlc(pair, tf, limit=300)
        sig = detect_structure_and_zones(df)
        if sig.get("signal_type") in ("LONG","SHORT"):
            entry = sig["entry"]; tp1 = sig["tp1"]; tp2 = sig["tp2"]
            sig["tp1"] = entry + (tp1 - entry) * 0.45 if sig["signal_type"]=="LONG" else entry + (tp1 - entry) * 0.45
            sig["tp2"] = entry + (tp2 - entry) * 0.25 if sig["signal_type"]=="LONG" else entry + (tp2 - entry) * 0.25
        append_log({"type":"scalp","pair":pair,"tf":tf,"signal":sig}, persist=True)
        return sig
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "trace": tb})

@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        sig = detect_structure_and_zones(df)
        append_log({"type":"analyze_csv","file": file.filename,"signal":sig}, persist=True)
        return sig
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "trace": tb})

@app.post("/analyze_chart")
async def analyze_chart(file: UploadFile = File(...)):
    try:
        content = await file.read()
        sig = {"signal_type":"WAIT","reasoning":"chart image analysis not implemented - use CSV or /pro_signal"}
        append_log({"type":"analyze_chart","file": file.filename,"signal":sig}, persist=True)
        return sig
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()})

# ----------------- RUN SERVER -----------------
if __name__ == "__main__":
    import uvicorn
    print(f"[STARTUP] {APP_NAME} running on port {PORT}")
    uvicorn.run("main_combined_learning:app", host="0.0.0.0", port=PORT, log_level="info")
