import os
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Load environment variables
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL", "http://127.0.0.1:8000")

# Helper: kirim pesan aman ke Telegram
async def send_message(context, text, chat_id=None):
    try:
        await context.bot.send_message(chat_id=chat_id or CHAT_ID, text=text, parse_mode="HTML")
    except Exception as e:
        print("Send message error:", e)

# ---------------- COMMANDS ----------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 <b>Pro Trader AI Bot (SMC Pro)</b> siap digunakan!<br><br>"
        "<b>Perintah yang tersedia:</b><br>"
        "/scalp &lt;pair&gt; — Sinyal scalping cepat (contoh: /scalp BTCUSDT)<br>"
        "/pro &lt;pair&gt; — Analisis penuh SMC Pro (contoh: /pro SOLUSDT)<br>"
        "/mode &lt;auto/agresif/moderate/konservatif&gt; — Ubah mode strategi<br>"
        "/performance — Cek performa AI<br>"
        "/logs — Lihat sinyal terakhir<br>"
        "/retrain — Latih ulang model AI<br>"
        "/autotune — Jalankan auto-tune SMC<br>"
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def scalp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ Gunakan format: <b>/scalp BTCUSDT</b>", parse_mode="HTML")
        return
    pair = context.args[0].upper()
    url = f"{APP_URL}/scalp_signal?pair={pair}&auto_log=true"
    try:
        res = requests.get(url, timeout=30).json()
        text = (
            f"⚡ <b>Scalp Signal</b> — {pair}<br>"
            f"<b>Signal:</b> {res.get('signal_type')}<br>"
            f"<b>Entry:</b> {res.get('entry')}<br>"
            f"<b>TP1:</b> {res.get('tp1')}<br>"
            f"<b>TP2:</b> {res.get('tp2')}<br>"
            f"<b>SL:</b> {res.get('sl')}<br>"
            f"<b>Confidence:</b> {res.get('confidence')}<br>"
            f"<b>Mode:</b> {res.get('mode_used')}"
        )
        await update.message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}", parse_mode="HTML")

async def pro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ Gunakan format: <b>/pro BTCUSDT</b>", parse_mode="HTML")
        return
    pair = context.args[0].upper()
    url = f"{APP_URL}/pro_signal?pair={pair}&auto_log=true"
    try:
        res = requests.get(url, timeout=60).json()
        text = (
            f"📊 <b>Pro Signal (SMC)</b> — {pair}<br>"
            f"<b>Signal:</b> {res.get('signal_type')}<br>"
            f"<b>Entry:</b> {res.get('entry')}<br>"
            f"<b>TP1:</b> {res.get('tp1')} | <b>TP2:</b> {res.get('tp2')}<br>"
            f"<b>SL:</b> {res.get('sl')}<br>"
            f"<b>Mode:</b> {res.get('mode_used')}<br>"
            f"<b>Risk:</b> {res.get('risk_percent') * 100:.1f}%<br>"
            f"<b>Position Size:</b> {res.get('position_size')}<br>"
            f"<b>Confidence:</b> {res.get('confidence')}<br>"
            f"<b>Reasoning:</b> {res.get('reasoning')[:500]}..."
        )
        await update.message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}", parse_mode="HTML")

async def mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Gunakan format: <b>/mode auto</b> atau <b>/mode agresif</b>", parse_mode="HTML")
        return
    mode = context.args[0].lower()
    url = f"{APP_URL}/set_mode?mode={mode}"
    try:
        res = requests.get(url).json()
        msg = f"✅ Mode trading diubah ke <b>{res.get('mode')}</b>"
        await update.message.reply_text(msg, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}", parse_mode="HTML")

async def performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        res = requests.get(f"{APP_URL}/ai_performance").json()
        text = (
            f"📈 <b>AI Performance</b><br>"
            f"<b>Total Sinyal:</b> {res.get('total_signals')}<br>"
            f"<b>Winrate:</b> {res.get('winrate')}%<br>"
            f"<b>Profit Factor:</b> {res.get('profit_factor')}<br>"
            f"<b>Model:</b> <code>{res.get('model_status')}</code>"
        )
        await update.message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}", parse_mode="HTML")

async def logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        res = requests.get(f"{APP_URL}/logs_summary").json()
        text = (
            f"🧾 <b>Log Terakhir</b><br>"
            f"<b>Pair:</b> {res.get('pair')}<br>"
            f"<b>Signal:</b> {res.get('signal_type')}<br>"
            f"<b>Entry:</b> {res.get('entry')}<br>"
            f"<b>TP1:</b> {res.get('tp1')} | <b>SL:</b> {res.get('sl')}<br>"
            f"<b>Confidence:</b> {res.get('confidence')}<br>"
            f"<b>Reason:</b> {res.get('reasoning')[:400]}..."
        )
        await update.message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}", parse_mode="HTML")

async def retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧠 Melatih ulang model AI... tunggu sebentar.", parse_mode="HTML")
    try:
        res = requests.post(f"{APP_URL}/retrain_learning").json()
        await update.message.reply_text(
            f"✅ <b>Retrain selesai!</b><br>Status: {res.get('status')}<br>Samples: {res.get('samples')}",
            parse_mode="HTML"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error retrain: {e}", parse_mode="HTML")

async def autotune(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚙️ Menjalankan auto-tune SMC parameters...", parse_mode="HTML")
    try:
        res = requests.get(f"{APP_URL}/force_autotune").json()
        tuned = res.get("tuned", [])
        msg = f"✅ Auto-Tune selesai. {len(tuned)} profil disesuaikan."
        await update.message.reply_text(msg, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Error autotune: {e}", parse_mode="HTML")

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
