# ================================================================
# 📊 PRO TRADER AI - BACKTESTER SERVICE
# Evaluasi sinyal dari AI Agent (TP / SL hit detection)
# ================================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import pandas as pd
from datetime import datetime

app = FastAPI(
    title="Pro Trader AI - Backtester",
    description="Server evaluator untuk menguji sinyal AI (TP/SL) & hitung profit",
    version="1.0"
)

# Binance API endpoint
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

# ------------------- UTILITAS -------------------
def fetch_recent_data(symbol: str, interval="15m", limit=100) -> pd.DataFrame:
    """Ambil data OHLC dari Binance."""
    try:
        r = requests.get(
            BINANCE_KLINES,
            params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
            timeout=10
        )
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "num_trades", "tb_base", "tb_quote", "ignore"
            ])
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["open_time", "open", "high", "low", "close"]]
        else:
            print("Binance fetch failed:", r.text)
            return pd.DataFrame()
    except Exception as e:
        print("Error fetch_recent_data:", e)
        return pd.DataFrame()

# ------------------- BACKTEST LOGIC -------------------
@app.post("/backtest")
async def backtest(req: Request):
    """Terima sinyal dari AI dan evaluasi apakah kena TP atau SL."""
    try:
        payload = await req.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"})

    pair = str(payload.get("pair", "")).upper()
    side = payload.get("side", "WAIT")
    entry = float(payload.get("entry", 0))
    tp1 = float(payload.get("tp1", 0))
    sl = float(payload.get("sl", 0))
    tf = payload.get("timeframe", "15m")

    if not pair or entry <= 0 or sl <= 0:
        return JSONResponse({"error": "invalid_payload", "data": payload})

    df = fetch_recent_data(pair, interval=tf)
    if df.empty:
        return JSONResponse({"error": "no_data_fetched", "pair": pair})

    hit, pnl = "NO_HIT", 0.0
    for _, row in df.iterrows():
        high, low = float(row["high"]), float(row["low"])

        if side == "LONG":
            if low <= sl:
                hit, pnl = "SL", round((sl - entry) / entry * 100, 3)
                break
            if high >= tp1:
                hit, pnl = "TP1", round((tp1 - entry) / entry * 100, 3)
                break

        elif side == "SHORT":
            if high >= sl:
                hit, pnl = "SL", round((entry - sl) / entry * 100, 3)
                break
            if low <= tp1:
                hit, pnl = "TP1", round((entry - tp1) / entry * 100, 3)
                break

    result = {
        "pair": pair,
        "side": side,
        "entry": entry,
        "tp1": tp1,
        "sl": sl,
        "hit": hit,
        "pnl_total": pnl,
        "timestamp": datetime.utcnow().isoformat()
    }

    print(f"[BACKTEST] {pair} {side} → {hit} ({pnl}%)")
    return JSONResponse(result)

@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "msg": "Backtester aktif dan siap menerima sinyal AI."}
