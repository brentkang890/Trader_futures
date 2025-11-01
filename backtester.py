# ai_backtester.py
# FastAPI microservice to receive signals from AI Agent and simulate short-term backtests

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# ============= CONFIG =============
BINANCE_URL = "https://api.binance.com/api/v3/klines"
DEFAULT_LOOKAHEAD = int(os.getenv("LOOKAHEAD_BARS", "100"))  # number of candles to look ahead
INTERVAL = os.getenv("DEFAULT_INTERVAL", "15m")
TIMEFRAME_MAP = {"m": "min", "h": "hour", "d": "day"}

app = FastAPI(title="AI Backtester", version="1.0")

# ============= MODELS =============
class SignalInput(BaseModel):
    pair: str
    timeframe: str
    side: str
    entry: float
    tp1: float
    tp2: float
    sl: float
    confidence: float
    timestamp: str

# ============= HELPERS =============
def fetch_ohlc_binance(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    try:
        url = f"{BINANCE_URL}?symbol={symbol.upper()}&interval={interval}&limit={limit}"
        r = requests.get(url, timeout=15)
        data = r.json()
        if not isinstance(data, list):
            raise ValueError(f"Invalid Binance response: {data}")
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","tbbav","tbqav","ignore"
        ])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        return df[["timestamp","open","high","low","close"]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch error: {e}")

def simulate_trade(df: pd.DataFrame, signal: SignalInput) -> dict:
    """
    Simulate what happens after the signal entry.
    We'll check if TP1, TP2, or SL hit first.
    """
    # only consider candles *after* the signal time
    start_idx = df[df["timestamp"] > pd.to_datetime(signal.timestamp)].index
    if len(start_idx) == 0:
        return {"error": "No future data to backtest."}
    start_idx = start_idx[0]
    future_df = df.iloc[start_idx:start_idx+DEFAULT_LOOKAHEAD].copy()
    hit, pnl, bars = "NO_HIT", 0.0, 0
    if signal.side.upper() == "LONG":
        for i, row in enumerate(future_df.itertuples(), 1):
            if row.low <= signal.sl:
                hit, pnl = "SL", (signal.sl - signal.entry) / signal.entry * 100
                bars = i
                break
            if row.high >= signal.tp1:
                hit, pnl = "TP1", (signal.tp1 - signal.entry) / signal.entry * 100
                bars = i
                if row.high >= signal.tp2:
                    hit, pnl = "TP2", (signal.tp2 - signal.entry) / signal.entry * 100
                break
    else:
        for i, row in enumerate(future_df.itertuples(), 1):
            if row.high >= signal.sl:
                hit, pnl = "SL", (signal.entry - signal.sl) / signal.entry * 100
                bars = i
                break
            if row.low <= signal.tp1:
                hit, pnl = "TP1", (signal.entry - signal.tp1) / signal.entry * 100
                bars = i
                if row.low <= signal.tp2:
                    hit, pnl = "TP2", (signal.entry - signal.tp2) / signal.entry * 100
                break
    if bars == 0:
        bars = len(future_df)
    return {
        "hit": hit,
        "pnl_total": round(pnl, 3),
        "bars_to_hit": bars,
        "details": {
            "duration_minutes": bars * int(signal.timeframe[:-1]) if signal.timeframe[-1] == "m" else bars,
            "timestamp_entry": signal.timestamp,
            "timestamp_end": str(future_df.iloc[-1]["timestamp"])
        }
    }

# ============= ROUTES =============
@app.get("/")
def home():
    return {"status": "AI Backtester running", "version": "1.0"}

@app.post("/backtest")
def backtest_signal(sig: SignalInput):
    df = fetch_ohlc_binance(sig.pair)
    result = simulate_trade(df, sig)
    return result

# optional healthcheck route
@app.get("/ping")
def ping():
    return {"pong": True, "time": datetime.utcnow().isoformat()}

# ============= MAIN =============
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    print(f" AI Backtester running on port {port}")
    uvicorn.run("ai_backtester:app", host="0.0.0.0", port=port, reload=False)
