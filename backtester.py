# ================================================================
# 📊 PRO TRADER AI - UNIVERSAL BACKTESTER (Crypto + Forex)
# ================================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import pandas as pd
from datetime import datetime
import os

app = FastAPI(
    title="Pro Trader AI - Backtester",
    description="Evaluator sinyal AI (Crypto + Forex) dengan TP/SL detection",
    version="2.0"
)

# ------------------- API CONFIG -------------------
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
ALPHA_API_KEY = os.environ.get("ALPHA_API_KEY", "")
ALPHA_URL = "https://www.alphavantage.co/query"

# ------------------- DETECT PASAR -------------------
def detect_market(pair: str):
    p = pair.upper()
    if any(x in p for x in ["USDT", "BTC", "ETH", "SOL", "BNB", "DOGE", "ADA"]):
        return "crypto"
    return "forex"

# ------------------- FETCH DATA -------------------
def fetch_binance_data(symbol: str, interval="15m", limit=100):
    try:
        r = requests.get(BINANCE_KLINES, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "num_trades", "tb_base", "tb_quote", "ignore"
            ])
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["open_time", "open", "high", "low", "close"]]
    except Exception as e:
        print("Binance error:", e)
    return pd.DataFrame()

def fetch_forex_data(symbol: str, interval="15m"):
    """Ambil data forex dari AlphaVantage"""
    if not ALPHA_API_KEY:
        return pd.DataFrame()
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "60min"}
    iv = mapping.get(interval, "15min")
    from_sym, to_sym = symbol[:3], symbol[3:]
    try:
        r = requests.get(ALPHA_URL, params={
            "function": "FX_INTRADAY",
            "from_symbol": from_sym,
            "to_symbol": to_sym,
            "interval": iv,
            "apikey": ALPHA_API_KEY
        }, timeout=15)
        data = r.json()
        key = [k for k in data.keys() if "Time Series" in k]
        if not key:
            return pd.DataFrame()
        df = pd.DataFrame(data[key[0]]).T
        df.columns = [c.split(". ")[1] for c in df.columns]
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_index().tail(100).reset_index(drop=True)
        return df[["open", "high", "low", "close"]]
    except Exception as e:
        print("AlphaVantage error:", e)
        return pd.DataFrame()

# ------------------- BACKTEST -------------------
@app.post("/backtest")
async def backtest(req: Request):
    payload = await req.json()
    pair = str(payload.get("pair", "")).upper()
    side = payload.get("side", "WAIT")
    entry = float(payload.get("entry", 0))
    tp1 = float(payload.get("tp1", 0))
    sl = float(payload.get("sl", 0))
    tf = payload.get("timeframe", "15m")

    market = detect_market(pair)
    df = fetch_binance_data(pair, tf) if market == "crypto" else fetch_forex_data(pair, tf)

    if df.empty:
        return JSONResponse({"error": f"no_data_for_{pair}", "market": market})

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
        "market": market,
        "side": side,
        "entry": entry,
        "tp1": tp1,
        "sl": sl,
        "hit": hit,
        "pnl_total": pnl,
        "timestamp": datetime.utcnow().isoformat()
    }
    print(f"[BACKTEST] {pair} ({market}) {side} → {hit} ({pnl}%)")
    return JSONResponse(result)

@app.get("/")
def root():
    return {"status": "ok", "msg": "Backtester aktif untuk Crypto & Forex"}
