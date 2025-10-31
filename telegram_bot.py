import os
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Load environment variables
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL", "http://127.0.0.1:8000")

# Helper function
async def send_message(context, text, chat_id=None):
    try:
        await context.bot.send_message(chat_id=chat_id or CHAT_ID, text=text)
    except Exception as e:
        print("Send message error:", e)

# ---------------- COMMANDS ----------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 *Pro Trader AI Bot (SMC Pro)* Siap digunakan!\n\n"
        "Perintah yang tersedia:\n"
        "/scalp <pair> — Sinyal scalping cepat (contoh: /scalp BTCUSDT)\n"
        "/pro <pair> — Analisis penuh SMC Pro (contoh: /pro SOLUSDT)\n"
        "/mode <auto/agresif/moderate/konservatif> — Ubah mode strategi\n"
        "/performance — Cek performa AI\n"
        "/logs — Lihat hasil sinyal terakhir\n"
        "/retrain — Latih ulang model AI\n"
        "/autotune — Jalankan auto-tune SMC\n"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def scalp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ Gunakan format: /scalp BTCUSDT")
        return
    pair = context.args[0].upper()
    url = f"{APP_URL}/scalp_signal?pair={pair}&auto_log=true"
    try:
        res = requests.get(url, timeout=30).json()
        text = f"⚡ *Scalp Signal* — {pair}\n" \
               f"Signal: {res.get('signal_type')}\n" \
               f"Entry: {res.get('entry')}\n" \
               f"TP1: {res.get('tp1')}\nTP2: {res.get('tp2')}\nSL: {res.get('sl')}\n" \
               f"Confidence: {res.get('confidence')}\nMode: {res.get('mode_used')}"
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def pro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ Gunakan format: /pro BTCUSDT")
        return
    pair = context.args[0].upper()
    url = f"{APP_URL}/pro_signal?pair={pair}&auto_log=true"
    try:
        res = requests.get(url, timeout=60).json()
        text = f"📊 *Pro Signal (SMC)* — {pair}\n" \
               f"Signal: {res.get('signal_type')}\nEntry: {res.get('entry')}\n" \
               f"TP1: {res.get('tp1')} | TP2: {res.get('tp2')}\nSL: {res.get('sl')}\n" \
               f"Mode: {res.get('mode_used')}\nRisk: {res.get('risk_percent')*100:.1f}%\n" \
               f"Position Size: {res.get('position_size')}\n" \
               f"Confidence: {res.get('confidence')}\n" \
               f"Reasoning: {res.get('reasoning')[:400]}..."
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Gunakan format: /mode auto atau /mode agresif")
        return
    mode = context.args[0].lower()
    url = f"{APP_URL}/set_mode?mode={mode}"
    try:
        res = requests.get(url).json()
        msg = f"✅ Mode trading diubah ke *{res.get('mode')}*"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        res = requests.get(f"{APP_URL}/ai_performance").json()
        text = f"📈 *AI Performance*\n" \
               f"Total Sinyal: {res.get('total_signals')}\n" \
               f"Winrate: {res.get('winrate')}%\nProfit Factor: {res.get('profit_factor')}\n" \
               f"Model: {res.get('model_status')}"
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        res = requests.get(f"{APP_URL}/logs_summary").json()
        text = f"🧾 *Log Terakhir*\nPair: {res.get('pair')}\nSignal: {res.get('signal_type')}\n" \
               f"Entry: {res.get('entry')}\nSL: {res.get('sl')}\nTP1: {res.get('tp1')}\n" \
               f"Confidence: {res.get('confidence')}\nReason: {res.get('reasoning')[:300]}..."
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧠 Melatih ulang model AI... tunggu sebentar.")
    try:
        res = requests.post(f"{APP_URL}/retrain_learning").json()
        await update.message.reply_text(f"✅ Retrain selesai!\nStatus: {res.get('status')}\nSamples: {res.get('samples')}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error retrain: {e}")

async def autotune(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚙️ Menjalankan auto-tune SMC parameters...")
    try:
        res = requests.get(f"{APP_URL}/force_autotune").json()
        tuned = res.get("tuned", [])
        msg = f"✅ Auto-Tune selesai. {len(tuned)} profil disesuaikan."
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Error autotune: {e}")

# ---------------- MAIN ----------------
def main():
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN belum diatur di environment!")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scalp", scalp))
    app.add_handler(CommandHandler("pro", pro))
    app.add_handler(CommandHandler("mode", mode))
    app.add_handler(CommandHandler("performance", performance))
    app.add_handler(CommandHandler("logs", logs))
    app.add_handler(CommandHandler("retrain", retrain))
    app.add_handler(CommandHandler("autotune", autotune))

    print("🤖 Telegram bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
