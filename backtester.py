# backtester.py
"""
🤖 Pro Trader AI - Smart Backtester (Ultimate Version)
-------------------------------------------------------
✅ Single signal test (/backtest)
✅ Batch CSV test (/analyze_batch)
✅ Auto-feedback ke AI model (learning)
✅ Auto-save ke CSV log
✅ Monitoring endpoint (/logs, /health)
-------------------------------------------------------
Compatible with:
- main_combined_learning.py (AI agent)
- telegram_bot.py (Telegram interface)
"""

import os
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File
import pandas as pd
import numpy as np
import requests

# ---------------- CONFIG ----------------
app = FastAPI(title="ProTrader AI Backtester", description="Smart Backtester with AI Feedback")

SAVE_LOG = os.environ.get("SAVE_LOG", "true").lower() == "true"
LOG_FILE = os.environ.get("LOG_FILE", "backtest_log.csv")
AI_FEEDBACK_URL = os.environ.get("AI_FEEDBACK_URL", "")  # contoh: https://your-ai-service.up.railway.app/learning_feedback
FEEDBACK_ENABLED = bool(AI_FEEDBACK_URL)

# ---------------- CORE LOGIC ----------------
def simulate_backtest(payload: dict):
    """Simulasi 1 sinyal dengan probabilitas realistis"""
    pair = payload.get("pair", "Unknown")
    side = payload.get("side", "LONG").upper()
    entry = float(payload.get("entry", 0))
    tp1 = float(payload.get("tp1", entry * 1.02))
    tp2 = float(payload.get("tp2", entry * 1.03)) if payload.get("tp2") else None
    sl = float(payload.get("sl", entry * 0.98))
    confidence = float(payload.get("confidence", 0))
    reason = payload.get("reason", "")

    rnd = np.random.random()
    if rnd < confidence * 0.6:
        hit = "TP2" if tp2 and rnd > confidence * 0.4 else "TP1"
    else:
        hit = "SL"

    if side == "LONG":
        pnl = ((tp1 - entry) / entry * 100) if "TP" in hit else ((sl - entry) / entry * 100)
    else:
        pnl = ((entry - tp1) / entry * 100) if "TP" in hit else ((entry - sl) / entry * 100)

    result = {
        "pair": pair,
        "side": side,
        "entry": entry,
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "hit": hit,
        "pnl_total": round(pnl, 3),
        "confidence": confidence,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    # Simpan ke log
    if SAVE_LOG:
        try:
            df = pd.DataFrame([result])
            if not os.path.exists(LOG_FILE):
                df.to_csv(LOG_FILE, index=False)
            else:
                df.to_csv(LOG_FILE, mode="a", header=False, index=False)
        except Exception as e:
            print(f"⚠️ Gagal simpan log: {e}")

    # Kirim feedback ke AI
    if FEEDBACK_ENABLED:
        try:
            requests.post(f"{AI_FEEDBACK_URL}", json=result, timeout=10)
        except Exception as e:
            print(f"⚠️ Gagal kirim feedback ke AI: {e}")

    return result


# ---------------- ENDPOINTS ----------------
@app.post("/backtest")
async def backtest(request: Request):
    """Backtest satu sinyal"""
    try:
        payload = await request.json()
        result = simulate_backtest(payload)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/analyze_batch")
async def analyze_batch(file: UploadFile = File(...)):
    """
    Upload CSV untuk batch backtest:
    pair,side,entry,tp1,tp2,sl,confidence,reason
    """
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as e:
        return {"error": f"Gagal baca CSV: {e}"}

    required = ["pair", "side", "entry", "tp1", "sl"]
    if not all(col in df.columns for col in required):
        return {"error": f"CSV harus memiliki kolom: {', '.join(required)}"}

    results = []
    for _, row in df.iterrows():
        try:
            res = simulate_backtest(row.to_dict())
            results.append(res)
        except Exception as e:
            results.append({"pair": row.get("pair", "?"), "error": str(e)})

    total = len(results)
    wins = sum(1 for r in results if "TP" in str(r.get("hit", "")))
    losses = sum(1 for r in results if "SL" in str(r.get("hit", "")))
    avg_pnl = np.mean([r.get("pnl_total", 0) for r in results])
    winrate = round((wins / total) * 100, 2) if total > 0 else 0.0

    summary = {
        "total": total,
        "tp_hits": wins,
        "sl_hits": losses,
        "winrate": winrate,
        "avg_pnl": round(float(avg_pnl), 3),
        "feedback_to_ai": FEEDBACK_ENABLED
    }

    return {"summary": summary, "results": results[:30]}


@app.get("/logs")
def get_logs(limit: int = 50):
    """Melihat hasil backtest terakhir"""
    if not os.path.exists(LOG_FILE):
        return {"detail": "Belum ada hasil backtest"}
    try:
        df = pd.read_csv(LOG_FILE).tail(limit)
        return {"logs": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ProTrader Smart Backtester",
        "ai_feedback_enabled": FEEDBACK_ENABLED,
        "log_file": LOG_FILE
    }


@app.on_event("startup")
def startup():
    print("✅ Smart Backtester aktif dan siap menerima data AI & Telegram")


# ✅ Jalankan server di Railway:
# uvicorn backtester:app --host 0.0.0.0 --port $PORT
