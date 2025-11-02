# ======================================================
#  AI Agent - ProTrader Hybrid ICT + ML
#  Version: Final Integrated with Backtester & Hybrid Self-Learning
# ======================================================

import os, json, time, threading, requests, pandas as pd, numpy as np
from datetime import datetime
from fastapi import FastAPI, Query
from typing import Optional
from utils import (
    generate_signal,
    append_trade_log,
    send_telegram_notification,
    retrain_learning
)
import schedule

app = FastAPI(title="ProTraderAI", version="1.6")

# ======================================================
#   SIGNAL GENERATOR (MAIN)
# ======================================================
@app.get("/pro_signal")
def pro_signal(
    pair: str = Query(...),
    tf_main: str = Query("1h"),
    tf_entry: str = Query("15m"),
    auto_log: bool = Query(True)
):
    """
    Generate AI-driven trading signal (ICT + Hybrid + ML)
    Now automatically sends the result to Backtester.
    """
    print(f"[PRO_SIGNAL] Generating signal for {pair} ({tf_entry}) at {datetime.utcnow()}")

    try:
        # ======== 1️⃣ Jalankan Analisis ========
        signal_result = generate_signal(pair, tf_main, tf_entry)
        if not signal_result:
            return {"error": "no signal generated"}

        # Tambah metadata
        signal_result["pair"] = pair.upper()
        signal_result["tf_entry"] = tf_entry
        signal_result["timestamp"] = datetime.utcnow().isoformat()

        # ======== 2️⃣ Kirim ke Backtester ========
        BACKTEST_URL = os.getenv("BACKTEST_URL", "")
        if BACKTEST_URL:
            try:
                payload = signal_result.copy()
                payload["timeframe"] = payload.pop("tf_entry", "15m")
                payload["side"] = payload.pop("signal_type", "LONG")
                print(f"[PRO_SIGNAL] Sending signal to Backtester: {BACKTEST_URL}")
                resp = requests.post(BACKTEST_URL, json=payload, timeout=15)
                bt_result = resp.json()
                print(f"[PRO_SIGNAL] Backtest response: {bt_result}")
                signal_result["backtest_result"] = bt_result

                # Telegram Notif Backtest
                if "hit" in bt_result:
                    msg = (
                        f"📋 <b>Backtest Result</b>\n"
                        f"Pair: <code>{signal_result['pair']}</code>\n"
                        f"TF: {signal_result.get('timeframe','15m')}\n"
                        f"Result: <b>{bt_result.get('hit','-')}</b>\n"
                        f"PnL: {bt_result.get('pnl_total',0)}%\n"
                        f"Bars: {bt_result.get('bars_to_hit','-')}"
                    )
                    try:
                        send_telegram_notification(msg)
                    except Exception as e:
                        print(f"[AI→TG] ⚠️ Telegram notify failed: {e}")

            except Exception as e:
                print(f"[AI→BT] ⚠️ Backtest error: {e}")
        else:
            print("[AI→BT] ⚠️ BACKTEST_URL not set")

        # ======== 3️⃣ Simpan Log ========
        try:
            log_path = os.getenv("LOG_PATH", "trade_log.csv")
            append_trade_log(signal_result, log_path)
            print(f"[PRO_SIGNAL] ✅ Signal saved to {log_path}")
        except Exception as e:
            print(f"[PRO_SIGNAL] ⚠️ Log save error: {e}")

        return signal_result

    except Exception as e:
        print(f"[PRO_SIGNAL] ❌ Error generating signal: {e}")
        return {"error": str(e)}

# ======================================================
#   HYBRID SELF-LEARNING SYSTEM (Auto Retrain)
# ======================================================

def evaluate_ai_performance():
    """
    Evaluate last 100 backtested trades from trade_log.csv
    and trigger auto retrain if accuracy drops.
    """
    log_path = os.getenv("LOG_PATH", "trade_log.csv")
    if not os.path.exists(log_path):
        print("[AI] ⚠️ No trade_log.csv found for evaluation.")
        return

    try:
        df = pd.read_csv(log_path)
        if len(df) < 20:
            print("[AI] Not enough data for evaluation.")
            return

        df = df.tail(100)
        df["result"] = df["backtest_result"].apply(
            lambda x: "TP" if "TP" in str(x) else ("SL" if "SL" in str(x) else "NONE")
        )
        total = len(df)
        wins = len(df[df["result"].str.contains("TP")])
        winrate = round((wins / total) * 100, 2)

        def extract_pnl(val):
            try:
                if isinstance(val, str) and "pnl_total" in val:
                    import re, json
                    j = json.loads(val.replace("'", '"'))
                    return j.get("pnl_total", 0)
            except Exception:
                return 0
            return 0

        df["pnl"] = df["backtest_result"].apply(extract_pnl)
        avg_pnl = round(df["pnl"].mean(), 3)
        profit_factor = round(abs(
            df[df["pnl"] > 0]["pnl"].sum() / df[df["pnl"] < 0]["pnl"].sum()), 2
        ) if len(df[df["pnl"] < 0]) > 0 else np.inf

        report_msg = (
            f"📈 <b>Weekly AI Performance</b>\n"
            f"✅ Winrate: {winrate}%\n"
            f"💰 Avg PnL: {avg_pnl}%\n"
            f"⚙️ Profit Factor: {profit_factor}\n"
        )

        print(report_msg)
        send_telegram_notification(report_msg)

        # retrain jika performa turun
        if winrate < 60:
            retrain_msg = f"⚠️ Winrate dropped to {winrate}%. Retraining AI model..."
            print(retrain_msg)
            send_telegram_notification(retrain_msg)
            try:
                retrain_learning()
                send_telegram_notification("✅ Retraining completed successfully.")
            except Exception as e:
                print("[AI Retrain Error]", e)
                send_telegram_notification(f"❌ Retraining failed: {e}")

    except Exception as e:
        print("[AI Eval Error]", e)

# Jadwal evaluasi setiap 7 hari
def start_scheduler():
    schedule.every(7).days.do(evaluate_ai_performance)
    while True:
        schedule.run_pending()
        time.sleep(3600)  # cek setiap jam

# Thread background agar jalan terus
threading.Thread(target=start_scheduler, daemon=True).start()

# ======================================================
#   HEALTH CHECK
# ======================================================
@app.get("/ping")
def ping():
    return {"pong": True, "time": datetime.utcnow().isoformat()}

# ======================================================
#   MAIN APP RUNNER
# ======================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    print(f"\n🔥 ProTraderAI Final Active - Port {port}\n")
    uvicorn.run("main_combined_learning:app", host="0.0.0.0", port=port, reload=False)
