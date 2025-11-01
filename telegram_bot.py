# telegram_bot.py
"""
🤖 Pro Trader AI Telegram Bot — Evan Leon Edition (Bilingual)
Fitur:
- Analisis sinyal SMC + ICT via /signal endpoint
- Backtest 1 pair & multi pair (tabel emoji)
- Statistik performa AI (winrate, PF)
- Retrain model otomatis
- Analisis chart & CSV
- Output bilingual (ID + EN)
"""

import os
import time
import requests
import threading

# ==============================
# ⚙️ CONFIG
# ==============================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
APP_URL = os.getenv("APP_URL")
BACKTEST_URL = f"{APP_URL.rstrip('/')}/backtest"
BACKTEST_MULTI_URL = f"{APP_URL.rstrip('/')}/backtest_multi"

AUTO_MODE = os.getenv("AUTO_MODE", "HYBRID")  # SIGNAL / STATUS / HYBRID
AUTO_INTERVAL = int(os.getenv("AUTO_INTERVAL_MIN", "60"))

if not BOT_TOKEN or not CHAT_ID or not APP_URL:
    raise ValueError("❌ BOT_TOKEN, CHAT_ID, APP_URL wajib diatur di environment!")

# ==============================
# 🔗 Helper Functions
# ==============================
def send_message(text, parse_mode="HTML"):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text[:4096], "parse_mode": parse_mode}
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("[ERROR] send_message:", e)

def get_updates(offset=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params = {"timeout": 100, "offset": offset}
        return requests.get(url, params=params, timeout=120).json()
    except Exception as e:
        print("[ERROR] get_updates:", e)
        return {}

# ==============================
# 🧠 COMMAND HANDLER
# ==============================
def handle_command(text):
    if not text:
        return "⚠️ Pesan kosong."
    t = text.strip().lower()

    # START
    if t in ("/start", "start"):
        return (
            "🤖 <b>Pro Trader AI Bot Aktif!</b>\n"
            "By: <b>Evan Leon</b>\n\n"
            "📈 Contoh perintah:\n"
            "• BTCUSDT 15m → analisis sinyal\n"
            "• backtest BTCUSDT → uji strategi satu pair\n"
            "• backtest multi BTCUSDT, XAUUSD → multi backtest\n\n"
            "🧠 Perintah lain:\n"
            "/status | /stats | /retrain"
        )

    # ==============================
    # 🧪 BACKTEST SINGLE
    # ==============================
    if t.startswith("backtest "):
        try:
            parts = t.split()
            if len(parts) < 2:
                return "⚙️ Format: <code>backtest BTCUSDT</code>"
            pair = parts[1].upper()
            r = requests.post(BACKTEST_URL, json={
                "pair": pair,
                "side": "LONG",
                "entry": 30000,
                "tp1": 31000,
                "sl": 29500,
                "timeframe": "15m"
            }, timeout=30)
            d = r.json()
            if "error" in d:
                return f"⚠️ Error: {d['error']}"
            return (
                f"📊 <b>Backtest {d.get('pair')}</b>\n"
                f"⏱ Timeframe: {d.get('timeframe')}\n"
                f"💡 Side: {d.get('side')}\n"
                f"🎯 Hit: {d.get('hit')}\n"
                f"💰 PnL: {d.get('pnl_total')}\n"
                f"🧠 Bars checked: {d.get('bars')}"
            )
        except Exception as e:
            return f"⚠️ Gagal backtest: {e}"

    # ==============================
    # 📊 BACKTEST MULTI
    # ==============================
    if t.startswith("backtest multi"):
        try:
            pairs_text = t.split("multi", 1)[1].strip()
            pairs = [p.strip().upper() for p in pairs_text.split(",") if p.strip()]
            if not pairs:
                return "⚙️ Format: <code>backtest multi BTCUSDT, XAUUSD</code>"
            payload = {"pairs": pairs, "timeframes": ["15m", "1h"]}
            r = requests.post(BACKTEST_MULTI_URL, json=payload, timeout=90)
            d = r.json()
            if "error" in d:
                return f"⚠️ Error: {d['error']}"
            msg = "📊 <b>Backtest Multi Result</b>\n────────────────────────────\n"
            for res in d.get("results", []):
                symbol = "✅" if res["hit"] == "TP" else "❌"
                msg += f"{res['pair']} ({res['timeframe']}) → {symbol} {res['hit']} | {res['pnl']:.2f}\n"
            s = d.get("summary", {})
            msg += "────────────────────────────\n"
            msg += f"📈 Winrate: {s.get('average_winrate',0)}%\n💹 Total PnL: {s.get('total_pnl',0):.2f}"
            return msg
        except Exception as e:
            return f"⚠️ Gagal multi-backtest: {e}"

    # ==============================
    # 🧠 RETRAIN
    # ==============================
    if t == "/retrain":
        try:
            r = requests.post(f"{APP_URL.rstrip('/')}/retrain_learning", data={"force": "true"}, timeout=60)
            d = r.json()
            if "error" in d:
                return f"⚠️ Error retrain: {d['error']}"
            return (
                "🧠 <b>Model retrained!</b>\n"
                f"Accuracy: {d.get('accuracy',0):.2f}\n"
                f"Samples: {d.get('samples',0)}"
            )
        except Exception as e:
            return f"⚠️ Retrain gagal: {e}"

    # ==============================
    # 📈 STATS
    # ==============================
    if t == "/stats":
        try:
            r = requests.get(f"{APP_URL.rstrip('/')}/ai_performance", timeout=20)
            d = r.json()
            return (
                f"📊 <b>AI Performance</b>\n"
                f"Total Sinyal: {d.get('total_signals')}\n"
                f"✅ Winrate: {d.get('winrate')}%\n"
                f"💹 Profit Factor: {d.get('profit_factor')}\n"
                f"🤖 Model: {d.get('model_status')}"
            )
        except Exception as e:
            return f"⚠️ Error ambil performa: {e}"

    # ==============================
    # 📦 STATUS
    # ==============================
    if t == "/status":
        try:
            r = requests.get(f"{APP_URL.rstrip('/')}/learning_status", timeout=15)
            d = r.json()
            return (
                f"🤖 <b>Status Model</b>\n"
                f"Model: {'✅ Aktif' if d.get('model_exists') else '❌ Tidak ada'}\n"
                f"🧮 Data Log: {d.get('trade_log_count',0)} sinyal"
            )
        except Exception as e:
            return f"⚠️ Error ambil status: {e}"

    # ==============================
    # 🔍 SIGNAL (DEFAULT)
    # ==============================
    parts = t.split()
    if len(parts) == 0:
        return "❌ Format salah. Contoh: <code>BTCUSDT 15m</code>"
    pair = parts[0].upper()
    tf = parts[1] if len(parts) > 1 else "15m"
    try:
        r = requests.post(f"{APP_URL.rstrip('/')}/signal", json={"pair": pair, "timeframe": tf}, timeout=25)
        d = r.json()
        if "error" in d:
            return f"⚠️ {d['error']}"
        emoji = "🟢" if d.get("signal_type") == "LONG" else "🔴" if d.get("signal_type") == "SHORT" else "⚪"
        return (
            f"{emoji} <b>{pair} ({d.get('timeframe')})</b>\n"
            f"💡 Signal: {d.get('signal_type')}\n"
            f"🎯 Entry: {d.get('entry')}\n"
            f"🏁 TP: {d.get('tp1')} | 🛑 SL: {d.get('sl')}\n"
            f"📊 Confidence: {d.get('confidence')}\n"
            f"🧠 Reasoning: {d.get('reasoning')}"
        )
    except Exception as e:
        return f"⚠️ Gagal ambil sinyal: {e}"

# ==============================
# 🔁 AUTO MODE THREAD
# ==============================
def auto_update_loop():
    while True:
        try:
            if AUTO_MODE.lower() in ("signal", "hybrid"):
                try:
                    r = requests.post(f"{APP_URL.rstrip('/')}/signal", json={"pair":"BTCUSDT","timeframe":"15m"}, timeout=25)
                    d = r.json()
                    msg = (
                        f"📊 <b>Auto Update</b>\n"
                        f"Pair: {d.get('pair')} ({d.get('timeframe')})\n"
                        f"💡 Signal: {d.get('signal_type')}\n"
                        f"Confidence: {d.get('confidence')}\n"
                    )
                    send_message(msg)
                except:
                    pass
            if AUTO_MODE.lower() in ("status", "hybrid"):
                try:
                    r = requests.get(f"{APP_URL.rstrip('/')}/ai_performance", timeout=20)
                    d = r.json()
                    msg = (
                        f"📈 <b>Auto Performance</b>\n"
                        f"Winrate: {d.get('winrate')}%\n"
                        f"Profit Factor: {d.get('profit_factor')}\n"
                        f"Signals: {d.get('total_signals')}"
                    )
                    send_message(msg)
                except:
                    pass
            time.sleep(AUTO_INTERVAL * 60)
        except Exception as e:
            print("[AUTO LOOP ERROR]", e)
            time.sleep(60)

# ==============================
# 🚀 MAIN LOOP
# ==============================
def main():
    offset = None
    send_message("🤖 Pro Trader AI Bot aktif!\nKetik /start untuk bantuan.\nBy: <b>Evan Leon</b>")
    threading.Thread(target=auto_update_loop, daemon=True).start()

    while True:
        try:
            upd = get_updates(offset)
            if "result" in upd:
                for u in upd["result"]:
                    offset = u["update_id"] + 1
                    msg = u.get("message", {})
                    if "text" in msg:
                        reply = handle_command(msg["text"])
                        send_message(reply)
            time.sleep(1.5)
        except Exception as e:
            print("[MAIN LOOP ERROR]", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
