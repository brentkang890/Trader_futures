# ======================================================
# 💬 PRO TRADER AI TELEGRAM BOT (AUTO LOG + AUTO RETRAIN)
# ======================================================

import os
import requests
import time
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# ===================== CONFIG =====================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
APP_URL = os.environ.get("APP_URL", "").rstrip("/")

if not BOT_TOKEN or not APP_URL:
    print("⚠️ BOT_TOKEN atau APP_URL belum diatur.")
    exit(1)

# ===================== CORE FUNCTION =====================
async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text:
        return

    await update.message.reply_text("⏳ Sedang memproses sinyal...")

    try:
        if text.lower().startswith("scalp"):
            # Contoh: "scalp BTCUSDT 3m"
            parts = text.split()
            if len(parts) >= 2:
                pair = parts[1].upper()
                tf = parts[2] if len(parts) >= 3 else "3m"
            else:
                await update.message.reply_text("Format: scalp BTCUSDT [timeframe]")
                return

            url = f"{APP_URL}/scalp_signal?pair={pair}&tf={tf}&auto_log=true"
        else:
            # Contoh: "BTCUSDT 15m"
            parts = text.split()
            pair = parts[0].upper()
            tf = parts[1] if len(parts) >= 2 else "15m"
            url = f"{APP_URL}/pro_signal?pair={pair}&tf_main=1h&tf_entry={tf}&auto_log=true"

        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            await update.message.reply_text(f"❌ Gagal mengambil data dari AI Server: {r.text}")
            return

        d = r.json()
        if "error" in d:
            await update.message.reply_text(f"⚠️ Error: {d['error']}")
            return

        # Format pesan hasil sinyal
        msg = (
            f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
            f"💡 Sinyal: <b>{d.get('signal_type')}</b>\n"
            f"🎯 Entry: {d.get('entry')}\n"
            f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
            f"🛡 SL: {d.get('sl')}\n"
            f"📈 Confidence: {d.get('confidence')}\n"
            f"🧠 {d.get('reasoning')}"
        )

        # Coba retrain otomatis jika cukup data
        try:
            status = requests.get(f"{APP_URL}/learning_status", timeout=10).json()
            trade_count = status.get("trade_log_count", 0)
            if trade_count >= 50:
                requests.get(f"{APP_URL}/retrain_learning", timeout=60)
                msg += f"\n\n🔄 Model diretrain otomatis ({trade_count} sinyal)"
            else:
                msg += f"\n\n📦 Total sinyal tercatat: {trade_count}"
        except Exception as e:
            msg += f"\n\n⚠️ Gagal cek retrain otomatis: {e}"

        await update.message.reply_text(msg, parse_mode="HTML")

    except Exception as e:
        await update.message.reply_text(f"⚠️ Terjadi kesalahan: {str(e)}")

# ===================== COMMANDS =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Selamat datang di Pro Trader AI Bot!\n"
        "Kirimkan pair + timeframe, contoh:\n"
        "<code>BTCUSDT 15m</code>\n\n"
        "Untuk scalping:\n"
        "<code>scalp BTCUSDT 3m</code>",
        parse_mode="HTML"
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        res = requests.get(f"{APP_URL}/ai_performance", timeout=20).json()
        if "error" in res:
            await update.message.reply_text(f"⚠️ {res['error']}")
            return
        msg = (
            f"📈 <b>AI Performance</b>\n"
            f"Total: {res['total_signals']} sinyal\n"
            f"✅ TP: {res['tp_hits']} | ❌ SL: {res['sl_hits']}\n"
            f"🏆 Winrate: {res['winrate']}%\n"
            f"💰 Total PnL: {res['total_pnl']}%\n"
            f"🤖 Model: {res['model_status']}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Gagal mengambil performa AI: {e}")

async def retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Melatih ulang model...")
    try:
        r = requests.get(f"{APP_URL}/retrain_learning", timeout=120)
        await update.message.reply_text(f"✅ Retrain selesai: {r.text}")
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal retrain: {e}")

# ===================== MAIN =====================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("retrain", retrain))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_command))

    print("🤖 Telegram Bot aktif dan menunggu perintah...")
    app.run_polling()

if __name__ == "__main__":
    main()
