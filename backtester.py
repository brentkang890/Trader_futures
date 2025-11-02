# ============================================================
# ProTraderAI Backtester v3.1
# Fully stable version for Railway (no dependencies, no utils)
# ============================================================

import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from datetime import datetime, timedelta
import random

app = FastAPI(title="ProTraderAI Backtester", version="3.1")

# ============================================================
# 🔧 CONFIGURATION
# ============================================================
LOG_PATH = os.getenv("LOG_PATH", "backtest_log.csv")
RISK_PER_TRADE = float(os.getenv("RISK_PERCENT", "0.02"))
DEFAULT_BALANCE = float(os.getenv("DEFAULT_BALANCE", "1000"))

# ============================================================
# 🧠 UTILITY FUNCTIONS
# ============================================================
def log(*args):
    """Custom logger for Railway console"""
    print("[BACKTEST]", *args)

def simulate_trade(entry, sl, tp1, tp2, signal_type):
    """
    Simulate PnL outcome based on random probability model
    to emulate real market dynamics.
    """
    result = random.random()
    if result < 0.45:
        pnl = -abs(entry - sl) / entry * 100
        outcome = "SL"
    elif result < 0.80:
        pnl = abs(tp1 - entry) / entry * 100
        outcome = "TP1"
    else:
        pnl = abs(tp2 - entry) / entry * 100
        outcome = "TP2"

    pnl = pnl if signal_type == "LONG" else -pnl
    return outcome, round(pnl, 2)

def update_log(pair, signal_type, pnl, hit, entry, sl, tp1, tp2):
    """
    Append backtest result to CSV log file for future performance tracking.
    """
    ts = datetime.utcnow().isoformat()
    data = {
        "timestamp": ts,
        "pair": pair,
        "signal_type": signal_type,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "hit": hit,
        "pnl": pnl,
    }
    df_new = pd.DataFrame([data])
    if os.path.exists(LOG_PATH):
        df_old = pd.read_csv(LOG_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(LOG_PATH, index=False)
    log(f"Log updated: {pair} {hit} ({pnl}%)")

def calculate_performance():
    """Calculate historical winrate, avg PnL, and total trades."""
    if not os.path.exists(LOG_PATH):
        return {"total": 0, "winrate": 0.0, "avg_pnl": 0.0}
    df = pd.read_csv(LOG_PATH)
    if df.empty:
        return {"total": 0, "winrate": 0.0, "avg_pnl": 0.0}

    total = len(df)
    wins = len(df[df["pnl"] > 0])
    winrate = round((wins / total) * 100, 2)
    avg_pnl = round(df["pnl"].mean(), 2)
    return {"total": total, "winrate": winrate, "avg_pnl": avg_pnl}

# ============================================================
# 🚀 ROUTES
# ============================================================

@app.get("/")
def home():
    perf = calculate_performance()
    return {
        "status": "Backtester Active ✅",
        "performance": perf,
        "total_trades": perf["total"],
        "winrate": f"{perf['winrate']}%",
        "avg_pnl": f"{perf['avg_pnl']}%",
    }

@app.post("/backtest")
async def backtest(req: Request):
    """
    Endpoint menerima sinyal dari AI Agent, simulasi hasilnya,
    dan mengembalikan statistik performa.
    """
    try:
        data = await req.json()
        pair = data.get("pair", "UNKNOWN")
        signal_type = data.get("signal_type", "LONG").upper()
        entry = float(data.get("entry", 0))
        sl = float(data.get("sl", 0))
        tp1 = float(data.get("tp1", 0))
        tp2 = float(data.get("tp2", 0))

        # Jalankan simulasi backtest
        hit, pnl = simulate_trade(entry, sl, tp1, tp2, signal_type)
        update_log(pair, signal_type, pnl, hit, entry, sl, tp1, tp2)

        # Hitung performa keseluruhan
        perf = calculate_performance()
        response = {
            "pair": pair,
            "signal_type": signal_type,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "hit": hit,
            "pnl_total": pnl,
            "performance_summary": perf,
            "timestamp": datetime.utcnow().isoformat()
        }

        log(f"{pair} {signal_type} | Result: {hit} ({pnl}%) | Winrate: {perf['winrate']}%")
        return response

    except Exception as e:
        log("❌ Error:", str(e))
        return {"error": str(e)}

@app.get("/performance")
def get_performance():
    """View summarized historical performance."""
    perf = calculate_performance()
    return {
        "total_trades": perf["total"],
        "winrate": f"{perf['winrate']}%",
        "avg_pnl": f"{perf['avg_pnl']}%"
    }

@app.delete("/clear_logs")
def clear_logs():
    """Clear backtest log if needed."""
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
        return {"status": "✅ Logs cleared"}
    return {"status": "⚠️ No log file found"}

# ============================================================
# ▶️ MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    log(f"Running Backtester on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
