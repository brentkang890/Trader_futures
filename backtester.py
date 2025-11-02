# backtester.py
# Compatible with ProTraderAI main_combined_learning_hybrid_final.py and telegram_bot.py

import os
import csv
import math
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import requests

app = FastAPI(title="ProTraderAI Backtester", version="1.0")

BACKTEST_LOG = os.getenv("BACKTEST_LOG", "backtest_log.csv")
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

# ========== UTILITIES ==========
def ensure_backtest_log():
    if not os.path.exists(BACKTEST_LOG):
        with open(BACKTEST_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","pair","timeframe","side","entry","tp1","tp2","sl","hit","pnl_total"])

def fetch_ohlc(symbol: str, interval: str="15m", limit: int=200):
    """Fetch recent OHLC from Binance."""
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(BINANCE_KLINES, params=params, timeout=10)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["timestamp","open","high","low","close"]].dropna().reset_index(drop=True)

def calc_backtest(pair, tf, side, entry, tp1, tp2, sl):
    """Simulate whether TP or SL was hit first after entry."""
    df = fetch_ohlc(pair, tf, limit=120)
    after_entry = df[df["close"] > 0].tail(100)

    hit = "NONE"
    pnl = 0.0
    if len(after_entry) < 5:
        return {"hit": "NO_DATA", "pnl_total": 0.0}

    for _, row in after_entry.iterrows():
        high, low = row["high"], row["low"]
        if side == "LONG":
            if high >= tp2:
                hit = "TP2"; pnl = (tp2 - entry) / entry * 100; break
            elif high >= tp1:
                hit = "TP1"; pnl = (tp1 - entry) / entry * 100; break
            elif low <= sl:
                hit = "SL"; pnl = (sl - entry) / entry * 100; break
        elif side == "SHORT":
            if low <= tp2:
                hit = "TP2"; pnl = (entry - tp2) / entry * 100; break
            elif low <= tp1:
                hit = "TP1"; pnl = (entry - tp1) / entry * 100; break
            elif high >= sl:
                hit = "SL"; pnl = (entry - sl) / entry * 100; break
    if hit == "NONE":
        # assume partial movement
        last_close = after_entry["close"].iloc[-1]
        pnl = ((last_close - entry) / entry * 100) if side == "LONG" else ((entry - last_close) / entry * 100)

    return {"hit": hit, "pnl_total": round(pnl, 3)}

# ========== ENDPOINT ==========
@app.post("/backtest")
async def backtest(request: Request):
    try:
        data = await request.json()
        pair = data.get("pair", "").upper()
        tf = data.get("timeframe", "15m")
        side = data.get("side", "LONG").upper()
        entry = float(data.get("entry", 0))
        tp1 = float(data.get("tp1", entry))
        tp2 = float(data.get("tp2", entry))
        sl = float(data.get("sl", entry))

        res = calc_backtest(pair, tf, side, entry, tp1, tp2, sl)
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "pair": pair,
            "timeframe": tf,
            "side": side,
            "entry": entry,
            "tp1": tp1,
            "tp2": tp2,
            "sl": sl,
            "hit": res["hit"],
            "pnl_total": res["pnl_total"]
        }

        ensure_backtest_log()
        with open(BACKTEST_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)

        return JSONResponse(content=record)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/logs")
def get_logs(limit: int = 50):
    ensure_backtest_log()
    df = pd.read_csv(BACKTEST_LOG)
    return JSONResponse(content={"logs": df.tail(limit).to_dict(orient="records")})

@app.get("/health")
def health():
    return {"status": "ok", "service": "ProTraderAI Backtester"}

# ========== RUN ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    print(f"🚀 Starting Backtester on port {port}")
    uvicorn.run("backtester:app", host="0.0.0.0", port=port)
