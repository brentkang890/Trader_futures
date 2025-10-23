# ======================================================
# 🤖 PRO TRADER AI - FINAL TELEGRAM BOT
# Analisis Otomatis Crypto & Forex + Chart Reader + Backtest
# ======================================================

import os
import time
import requests

# ---------------- KONFIG ----------------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL")  # URL AI Agent
BACKTEST_URL = os.environ.get("BACKTEST_URL")  # URL Backtester (Railway)

if not BOT_TOKEN or not CHAT_ID or not APP_URL:
    raise ValueError("❌ Variabel BOT_TOKEN, CHAT_ID, atau APP_URL belum diatur di Railway!")

# ---------------- UTILITAS ----------------
def send_message(text, parse_mode="HTML"):
    """Kirim pesan ke Telegram"""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text[:4096], "parse_mode": parse_mode}
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("[ERROR] Kirim pesan gagal:", e)


def get_updates(offset=None):
    """Ambil pesan terbaru dari Telegram"""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params = {"timeout": 100, "offset": offset}
        return requests.get(url, params=params, timeout=120).json()
    except Exception as e:
        print("[ERROR] Gagal ambil update:", e)
        return {}


# ---------------- COMMAND HANDLER ----------------
def handle_command(text):
    if not text:
        return "⚠️ Pesan kosong."
    text = text.strip().lower()

    # ============================
    # 🔹 BACKTEST COMMAND
    # ============================
    if text.startswith("backtest"):
        try:
            parts = text.split()
            if len(parts) < 2:
                return "⚙️ Format: <code>backtest BTCUSDT</code> atau <code>backtest XAUUSD</code>"

            pair = parts[1].upper()

            payload = {
                "pair": pair,
                "side": "LONG",      # bisa diubah jadi SHORT
                "entry": 30000,
                "tp1": 31000,
                "sl": 29500,
                "timeframe": "15m"
            }

            target_url = BACKTEST_URL or f"{APP_URL.replace('production','faithful-truth')}/backtest"
            r = requests.post(target_url, json=payload, timeout=30)
            d = r.json()

            if "error" in d:
                return f"⚠️ Backtest gagal: {d['error']}"

            msg = (
                f"🧮 <b>Backtest Result</b>\n"
                f"📊 Pair: {d.get('pair')}\n"
                f"💡 Side: {d.get('side')}\n"
                f"🎯 Hit: {d.get('hit')}\n"
                f"💰 PnL: {d.get('pnl_total')}%\n"
                f"📈 Market: {d.get('market')}"
            )
            return msg
        except Exception as e:
            return f"⚠️ Gagal jalankan backtest: {e}"

    # ============================
    # 🔹 STATUS MODEL
    # ============================
    if text == "status":
        try:
            r = requests.get(f"{APP_URL}/learning_status", timeout=20)
            d = r.json()
            msg = (
                "🤖 <b>Status AI Agent</b>\n"
                f"📦 Model: {'✅ Ada' if d.get('model_exists') else '❌ Tidak ada'}\n"
                f"📊 Log Data: {d.get('trade_log_count', 0)} sinyal"
            )
            return msg
        except Exception as e:
            return f"⚠️ Gagal ambil status: {e}"

    # ============================
    # 🔹 STATISTIK
    # ============================
    if text == "stats":
        try:
            r = requests.get(f"{APP_URL}/ai_performance", timeout=25)
            d = r.json()
            if "error" in d:
                return f"⚠️ {d['error']}"
            msg = (
                "📈 <b>Statistik Performa AI</b>\n"
                f"📊 Total sinyal: {d['total_signals']}\n"
                f"✅ Winrate: {d['winrate']}%\n"
                f"💰 Profit Factor: {d.get('profit_factor', 'N/A')}\n"
                f"📉 Max Drawdown: {d.get('max_drawdown', 'N/A')}\n"
            )
            return msg
        except Exception as e:
            return f"⚠️ Gagal ambil statistik: {e}"

    # ============================
    # 🔹 SCALP CEPAT
    # ============================
    if text.startswith("scalp "):
        pair = text.split()[1].upper()
        try:
            r = requests.get(f"{APP_URL}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=25)
            d = r.json()
            msg = (
                f"⚡️ <b>Scalp {d.get('pair')}</b> ({d.get('timeframe')})\n"
                f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
                f"🎯 Entry: {d.get('entry')}\n"
                f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
                f"🛡 SL: {d.get('sl')}\n"
                f"📈 Confidence: {d.get('confidence')}\n"
                f"🧠 {d.get('reasoning')}"
            )
            return msg
        except Exception as e:
            return f"⚠️ Gagal ambil scalp signal: {e}"

    # ============================
    # 🔹 SIGNAL NORMAL
    # ============================
    parts = text.split()
    if len(parts) == 0:
        return "⚠️ Format salah. Contoh: <code>BTCUSDT 15m</code>"
    elif len(parts) == 1:
        pair, tf = parts[0], "15m"
    else:
        pair, tf = parts[0], parts[1]

    try:
        url = f"{APP_URL}/pro_signal?pair={pair.upper()}&tf_main=1h&tf_entry={tf}&auto_log=true"
        r = requests.get(url, timeout=30)
        d = r.json()
        if "error" in d:
            return f"⚠️ {d['error']}"

        msg = (
            f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
            f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
            f"🎯 Entry: {d.get('entry')}\n"
            f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
            f"🛡 SL: {d.get('sl')}\n"
            f"📈 Confidence: {d.get('confidence')}\n"
            f"🧠 {d.get('reasoning')}"
        )
        return msg
    except Exception as e:
        return f"❌ Error fetch AI Agent: {e}"


# ---------------- PROGRAM UTAMA ----------------
def main():
    offset = None
    print(f"🤖 BOT AKTIF — Hubung ke: {APP_URL}")
    send_message(
        "🤖 <b>Pro Trader AI Bot Aktif!</b>\n"
        "Ketik contoh:\n"
        "- BTCUSDT 15m atau XAUUSD 1h\n"
        "Perintah lain:\n"
        "- backtest BTCUSDT\n"
        "- status\n- stats\n- scalp BTCUSDT"
    )

    while True:
        try:
            updates = get_updates(offset)
            if "result" in updates:
                for update in updates["result"]:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})

                    if "text" in msg:
                        reply = handle_command(msg["text"])
                        send_message(reply)

            time.sleep(3)
        except Exception as e:
            print("[ERROR LOOP]", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
