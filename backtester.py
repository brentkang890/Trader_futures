# backtester.py
# ProTraderAI Backtester (PRO)
# Endpoints:
#  - POST /backtest         (main entry - accepts JSON payload)
#  - GET  /ping             (health quick)
#  - GET  /health           (full health)
#  - GET  /logs             (recent backtest logs)
#  - GET  /backtest_chart   (returns PNG chart for last/backtest id)
#
# Usage: uvicorn backtester:app --host 0.0.0.0 --port $PORT
#
# Env:
#  - BINANCE_BASE (optional, default https://api.binance.com)
#  - TWELVEDATA_API_KEY (optional, fallback for forex)
#  - BACKTEST_LOG (optional, default backtest_log.csv)
#  - PORT (optional)
#
import os
import io
import csv
import time
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
APP = "ProTraderAI-Backtester-PRO"
BINANCE_BASE = os.getenv("BINANCE_BASE", "https://api.binance.com")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_API_KEY", "")
BACKTEST_LOG = os.getenv("BACKTEST_LOG", "backtest_log.csv")
PORT = int(os.getenv("PORT", 8001))

# In-memory index of logs (small cache)
LOGS: List[Dict[str, Any]] = []

app = FastAPI(title=APP, version="1.0")

# ---------------- UTILITIES ----------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def ensure_log_csv():
    if not os.path.exists(BACKTEST_LOG):
        with open(BACKTEST_LOG, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "id", "timestamp", "pair", "timeframe", "side", "entry", "sl", "tp1", "tp2",
                "hit", "pnl_total", "n_trades", "notes"
            ])

def append_log_csv(record: Dict[str, Any]):
    ensure_log_csv()
    with open(BACKTEST_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            record.get("id"),
            record.get("timestamp"),
            record.get("pair"),
            record.get("timeframe"),
            record.get("side"),
            record.get("entry"),
            record.get("sl"),
            record.get("tp1"),
            record.get("tp2"),
            record.get("hit"),
            record.get("pnl_total"),
            record.get("n_trades"),
            record.get("notes", "")
        ])

# ---------------- DATA FETCH ----------------
def df_from_binance_klines(candles: List[List[Any]]) -> pd.DataFrame:
    rows = []
    for c in candles:
        ts = int(c[0]) // 1000 if abs(int(c[0])) > 1e10 else int(c[0])
        rows.append({
            "timestamp": pd.to_datetime(int(ts), unit="s"),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]) if len(c) > 5 else 0.0
        })
    df = pd.DataFrame(rows).set_index("timestamp")
    return df

def fetch_ohlc(pair: str, tf: str = "15m", limit: int = 1000) -> pd.DataFrame:
    pair = pair.upper()
    # map timeframe to binance interval (common)
    tf_map = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","4h":"4h","1d":"1d","1w":"1w"}
    interval = tf_map.get(tf, tf)
    # try Binance for crypto-like pairs
    try:
        if any(s in pair for s in ["USDT","BTC","BUSD","USDC"]):
            url = f"{BINANCE_BASE}/api/v3/klines"
            r = requests.get(url, params={"symbol": pair, "interval": interval, "limit": limit}, timeout=15)
            if r.status_code == 200:
                return df_from_binance_klines(r.json())
            else:
                raise RuntimeError(f"Binance fetch {r.status_code}: {r.text}")
        # fallback: TwelveData for forex / other
        if TWELVEDATA_KEY:
            # TwelveData interval mapping: minimal transform
            td_interval = interval
            r = requests.get("https://api.twelvedata.com/time_series",
                             params={"symbol": pair, "interval": td_interval, "outputsize": limit, "apikey": TWELVEDATA_KEY},
                             timeout=15)
            data = r.json()
            if "values" in data:
                rows = []
                for v in reversed(data["values"]):
                    rows.append({
                        "timestamp": pd.to_datetime(v["datetime"]),
                        "open": float(v["open"]),
                        "high": float(v["high"]),
                        "low": float(v["low"]),
                        "close": float(v["close"]),
                        "volume": float(v.get("volume", 0) or 0)
                    })
                df = pd.DataFrame(rows).set_index("timestamp")
                return df
            else:
                raise RuntimeError(f"TwelveData fail: {data}")
    except Exception as e:
        raise
    raise RuntimeError("No data source available for pair")

# ---------------- BACKTEST SIMULATOR ----------------
def simulate_backtest(df: pd.DataFrame, side: str, entry: float, sl: float, tp1: float, tp2: float) -> Dict[str, Any]:
    """
    Simulate forward: iterate candles (df chronological), return first hit result.
    Returns detailed trades list, hit, pnl_total (percentage), number of trades scanned.
    """
    result = {"trades": [], "hit": "NONE", "pnl_total": 0.0, "n_trades": 0}
    if df is None or df.empty:
        result["hit"] = "NO_DATA"
        return result

    # iterate each candle after the 'current' (we assume df contains future candles relative to entry)
    for _, row in df.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        # LONG logic
        if side.upper() == "LONG":
            # TP2 priority, then TP1, then SL
            if high >= tp2:
                pnl = (tp2 - entry) / entry * 100
                result.update({"hit":"TP2","pnl_total": round(pnl,6)})
                result["trades"].append({"timestamp": str(row.name), "result":"TP2"})
                break
            if high >= tp1:
                pnl = (tp1 - entry) / entry * 100
                result.update({"hit":"TP1","pnl_total": round(pnl,6)})
                result["trades"].append({"timestamp": str(row.name), "result":"TP1"})
                break
            if low <= sl:
                pnl = (sl - entry) / entry * 100
                result.update({"hit":"SL","pnl_total": round(pnl,6)})
                result["trades"].append({"timestamp": str(row.name), "result":"SL"})
                break
        else:
            # SHORT
            if low <= tp2:
                pnl = (entry - tp2) / entry * 100
                result.update({"hit":"TP2","pnl_total": round(pnl,6)})
                result["trades"].append({"timestamp": str(row.name), "result":"TP2"})
                break
            if low <= tp1:
                pnl = (entry - tp1) / entry * 100
                result.update({"hit":"TP1","pnl_total": round(pnl,6)})
                result["trades"].append({"timestamp": str(row.name), "result":"TP1"})
                break
            if high >= sl:
                pnl = (entry - sl) / entry * 100
                result.update({"hit":"SL","pnl_total": round(pnl,6)})
                result["trades"].append({"timestamp": str(row.name), "result":"SL"})
                break

    result["n_trades"] = len(result["trades"]) or (0 if result["hit"]=="NONE" else 1)
    return result

# ---------------- CHARTS ----------------
def make_backtest_chart(df: pd.DataFrame, entry: float, sl: float, tp1: float, tp2: float, pair: str = "") -> bytes:
    df = df.copy().tail(200)
    if df.empty:
        fig = plt.figure(figsize=(8,4))
        plt.text(0.5,0.5,"No data", ha='center')
        buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); plt.close(fig); return buf.read()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df['close'], label='close')
    ax.set_title(f"{pair} - recent close")
    if entry is not None:
        ax.axhline(entry, color='yellow', linestyle='--', label='entry')
    if sl is not None:
        ax.axhline(sl, color='red', linestyle='--', label='SL')
    if tp1 is not None:
        ax.axhline(tp1, color='green', linestyle='--', label='TP1')
    if tp2 is not None:
        ax.axhline(tp2, color='green', linestyle=':', label='TP2')
    ax.legend(loc='upper left')
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ---------------- ENDPOINTS ----------------
@app.get("/ping")
def ping():
    return {"pong": True, "time": now_iso()}

@app.get("/health")
def health():
    return {"status": "ok", "app": APP, "time": now_iso(), "logs_cached": len(LOGS)}

@app.get("/logs")
def get_logs(limit: int = Query(50, ge=1, le=1000)):
    # return tail logs (from CSV if exists)
    try:
        if os.path.exists(BACKTEST_LOG):
            df = pd.read_csv(BACKTEST_LOG)
            records = df.tail(limit).to_dict(orient="records")
            return {"logs": records}
        else:
            return {"logs": LOGS[-limit:]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/backtest")
async def backtest(req: Request):
    """
    Accept flexible payload:
    {
      "pair": "BTCUSDT",
      "tf": "15m",
      "signal": {"signal_type":"LONG","entry":68500,"sl":68100,"tp1":68800,"tp2":69200},
      "ohlc": [ {timestamp, open, high, low, close, volume}, ... ]   # optional
    }
    If ohlc provided, simulator uses it; otherwise attempts to fetch via exchange.
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    try:
        pair = (body.get("pair") or body.get("symbol") or "").upper()
        tf = body.get("tf") or body.get("tf_entry") or body.get("timeframe") or "15m"
        signal = body.get("signal") or {}
        # parse signal fields
        side = (signal.get("signal_type") or signal.get("side") or "LONG").upper()
        entry = signal.get("entry")
        sl = signal.get("sl")
        tp1 = signal.get("tp1")
        tp2 = signal.get("tp2")

        # if OHLC provided in body, convert to df, else fetch
        if "ohlc" in body and body["ohlc"]:
            df = pd.DataFrame(body["ohlc"])
            # try normalize timestamp if numeric (seconds)
            if "timestamp" in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                    df = df.set_index('timestamp')
                except Exception:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df = df.set_index('timestamp')
        else:
            df = fetch_ohlc(pair, tf, limit=800)

        df = df.sort_index()
        # if signal incomplete, attempt to build simple signal from last candle
        if entry is None:
            entry = float(df['close'].iloc[-1])
        if sl is None or tp1 is None or tp2 is None:
            # simple sl/tp estimate using ATR
            highs = df['high']; lows = df['low']; close = df['close']
            tr = pd.concat([highs - lows, (highs - close.shift(1)).abs(), (lows - close.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=1).mean().iloc[-1]
            sl_dist = max(atr * 1.2, entry * 0.0025)
            if side == "LONG":
                sl = entry - sl_dist
                tp1 = entry + sl_dist * 1.5
                tp2 = entry + sl_dist * 3.0
            else:
                sl = entry + sl_dist
                tp1 = entry - sl_dist * 1.5
                tp2 = entry - sl_dist * 3.0

        # choose forward candles after current for simulation. Use tail window
        future_df = df.tail(400)
        sim = simulate_backtest(future_df, side, float(entry), float(sl), float(tp1), float(tp2))

        # prepare response
        run_id = int(time.time() * 1000)
        record = {
            "id": run_id,
            "timestamp": now_iso(),
            "pair": pair,
            "timeframe": tf,
            "side": side,
            "entry": float(entry),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "hit": sim.get("hit"),
            "pnl_total": sim.get("pnl_total"),
            "n_trades": sim.get("n_trades"),
            "trades": sim.get("trades"),
            "notes": ""
        }

        # append in-memory and CSV
        LOGS.append(record)
        append_log_csv(record)

        # include small performance summary
        summary = {
            "hit": record["hit"],
            "pnl_total": record["pnl_total"],
            "n_trades": record["n_trades"]
        }
        resp = {"id": record["id"], "pair": pair, "timeframe": tf, "summary": summary, "detail": record}
        return JSONResponse(resp)
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "trace": tb}, status_code=500)

@app.get("/backtest_chart")
def backtest_chart(id: Optional[int] = None, pair: Optional[str] = None, tf: Optional[str] = "15m"):
    """
    Return PNG chart for a previous backtest (by id) or current pair/timeframe.
    If id provided and exists in LOGS or CSV, uses that run; otherwise fetch current OHLC for pair.
    """
    try:
        target = None
        if id:
            # search in in-memory logs first
            for r in reversed(LOGS):
                if r.get("id") == int(id):
                    target = r
                    break
            if not target and os.path.exists(BACKTEST_LOG):
                df = pd.read_csv(BACKTEST_LOG)
                row = df[df['id'] == int(id)]
                if not row.empty:
                    row0 = row.iloc[-1].to_dict()
                    target = row0
        if target:
            # load OHLC for pair/tf near that run
            pair = pair or target.get("pair")
            tf = tf or target.get("timeframe", "15m")
            df = fetch_ohlc(pair, tf, limit=400)
            img = make_backtest_chart(df, target.get("entry"), target.get("sl"), target.get("tp1"), target.get("tp2"), pair=pair)
            return StreamingResponse(io.BytesIO(img), media_type="image/png")
        else:
            # if no id, require pair to fetch current and generate chart
            if not pair:
                raise HTTPException(status_code=400, detail="id or pair required")
            df = fetch_ohlc(pair, tf, limit=400)
            # naive last-entry
            entry = float(df['close'].iloc[-1])
            # produce default sl/tp scale
            highs = df['high']; lows = df['low']; close = df['close']
            tr = pd.concat([highs - lows, (highs - close.shift(1)).abs(), (lows - close.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=1).mean().iloc[-1]
            sl_dist = max(atr * 1.2, entry * 0.0025)
            sl = entry - sl_dist
            tp1 = entry + sl_dist * 1.5
            tp2 = entry + sl_dist * 3.0
            img = make_backtest_chart(df.tail(200), entry, sl, tp1, tp2, pair=pair)
            return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "trace": tb}, status_code=500)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    ensure_log_csv()
    print(f"[STARTUP] {APP} starting on port {PORT}")
    import uvicorn
    uvicorn.run("backtester:app", host="0.0.0.0", port=int(os.getenv("PORT", PORT)), log_level="info")
