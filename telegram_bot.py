import os
import telebot
import requests
from telebot.types import InputFile

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
APP_URL = os.getenv("APP_URL")  # AI Agent URL
BACKTEST_URL = os.getenv("BACKTEST_URL")  # Optional

bot = telebot.TeleBot(BOT_TOKEN)

# ================================
# 🔹 /start Command
# ================================
@bot.message_handler(commands=['start'])
def start(message):
    text = (
        "🤖 *Pro Trader AI Bot*\n\n"
        "Kirim pair dan timeframe seperti:\n"
        "`BTCUSDT 15m` atau `XAUUSD 1h`\n\n"
        "📊 Kirim *gambar chart* untuk analisis otomatis.\n"
        "📂 Kirim *file .csv* untuk analisis data historis.\n\n"
        "Perintah lain:\n"
        "• /stats — Lihat performa AI\n"
        "• /status — Cek status model & learning\n"
        "• /backtest <pair> — Jalankan backtest manual\n"
    )
    bot.reply_to(message, text, parse_mode="Markdown")

# ================================
# 🔹 Analisis Pair / Timeframe
# ================================
@bot.message_handler(func=lambda msg: msg.text and msg.text.strip() != "")
def handle_message(message):
    text = message.text.strip().upper()
    parts = text.split()

    if len(parts) >= 2:
        pair = parts[0]
        tf = parts[1]
    else:
        pair = parts[0]
        tf = "15m"

    bot.reply_to(message, f"🔎 Menganalisis {pair} ({tf})...")

    try:
        url = f"{APP_URL}/pro_signal"
        params = {"pair": pair, "tf_main": "1h", "tf_entry": tf, "auto_log": True}
        res = requests.get(url, params=params, timeout=30)

        if res.status_code != 200:
            bot.reply_to(message, f"⚠️ Gagal: {res.text}")
            return

        data = res.json()
        signal = data.get("signal_type", "WAIT")
        entry = data.get("entry", 0)
        tp1 = data.get("tp1", 0)
        sl = data.get("sl", 0)
        conf = data.get("confidence", 0)
        reason = data.get("reasoning", "-")

        reply = (
            f"📊 *Hasil Analisis {pair} ({tf})*\n\n"
            f"💡 Signal: `{signal}`\n"
            f"🎯 Entry: `{entry}`\n"
            f"TP1: `{tp1}` | SL: `{sl}`\n"
            f"📈 Confidence: `{conf}`\n"
            f"🧠 Reason: {reason}"
        )
        bot.send_message(message.chat.id, reply, parse_mode="Markdown")

    except Exception as e:
        bot.reply_to(message, f"❌ Error: {e}")

# ================================
# 🔹 Upload Gambar Chart
# ================================
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.reply_to(message, "🖼️ Menganalisis chart dari gambar...")

    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        files = {'file': ("chart.jpg", downloaded_file, 'image/jpeg')}
        res = requests.post(f"{APP_URL}/analyze_chart", files=files, timeout=60)

        if res.status_code != 200:
            bot.reply_to(message, f"⚠️ Gagal menganalisis chart: {res.text}")
            return

        data = res.json()
        signal = data.get("signal_type", "WAIT")
        entry = data.get("entry", 0)
        tp1 = data.get("tp1", 0)
        sl = data.get("sl", 0)
        conf = data.get("confidence", 0)
        reason = data.get("reasoning", "-")

        reply = (
            f"🖼️ *Analisis Chart Gambar*\n\n"
            f"💡 Signal: `{signal}`\n"
            f"🎯 Entry: `{entry}`\n"
            f"TP1: `{tp1}` | SL: `{sl}`\n"
            f"📈 Confidence: `{conf}`\n"
            f"🧠 Reason: {reason}"
        )
        bot.send_message(message.chat.id, reply, parse_mode="Markdown")

    except Exception as e:
        bot.reply_to(message, f"⚠️ Error: {e}")

# ================================
# 🔹 Upload File CSV (NEW FEATURE)
# ================================
@bot.message_handler(content_types=['document'])
def handle_csv_upload(message):
    try:
        doc = message.document
        if not doc.file_name.lower().endswith('.csv'):
            bot.reply_to(message, "⚠️ Hanya file .csv yang didukung untuk analisis.")
            return

        bot.reply_to(message, "📂 File CSV diterima, sedang dianalisis...")

        file_info = bot.get_file(doc.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        files = {'file': (doc.file_name, downloaded_file, 'text/csv')}
        res = requests.post(f"{APP_URL}/analyze_csv", files=files, timeout=60)

        if res.status_code == 200:
            data = res.json()
            signal = data.get("signal_type", "WAIT")
            entry = data.get("entry", 0)
            tp1 = data.get("tp1", 0)
            sl = data.get("sl", 0)
            conf = data.get("confidence", 0)
            reason = data.get("reasoning", "-")

            reply = (
                f"📊 *Analisis CSV Berhasil*\n\n"
                f"💡 Signal: `{signal}`\n"
                f"🎯 Entry: `{entry}`\n"
                f"TP1: `{tp1}` | SL: `{sl}`\n"
                f"📈 Confidence: `{conf}`\n"
                f"🧠 Reason: {reason}"
            )
        else:
            reply = f"❌ Gagal menganalisis CSV: {res.text}"

        bot.send_message(message.chat.id, reply, parse_mode="Markdown")

    except Exception as e:
        bot.send_message(message.chat.id, f"⚠️ Error: {e}")

# ================================
# 🔹 Command: /stats
# ================================
@bot.message_handler(commands=['stats'])
def stats(message):
    try:
        res = requests.get(f"{APP_URL}/ai_performance", timeout=30)
        if res.status_code != 200:
            bot.reply_to(message, f"⚠️ Gagal: {res.text}")
            return
        data = res.json()
        reply = (
            f"📈 *AI Performance*\n\n"
            f"🏁 Total Signals: {data.get('total_signals')}\n"
            f"✅ TP: {data.get('tp_hits')} | ❌ SL: {data.get('sl_hits')}\n"
            f"🎯 Winrate: {data.get('winrate')}%\n"
            f"💰 Total PnL: {data.get('total_pnl')}\n"
            f"⚖️ Profit Factor: {data.get('profit_factor')}\n"
            f"📉 Max Drawdown: {data.get('max_drawdown')}\n"
            f"🧠 Model: {data.get('model_status')}"
        )
        bot.send_message(message.chat.id, reply, parse_mode="Markdown")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Error: {e}")

# ================================
# 🔹 Command: /status
# ================================
@bot.message_handler(commands=['status'])
def status(message):
    try:
        res = requests.get(f"{APP_URL}/learning_status", timeout=30)
        if res.status_code != 200:
            bot.reply_to(message, f"⚠️ Gagal: {res.text}")
            return
        data = res.json()
        reply = (
            f"📚 *Learning Status*\n\n"
            f"Model Exists: {data.get('model_exists')}\n"
            f"Trade Log Count: {data.get('trade_log_count')}\n"
            f"Features: {data.get('features')}"
        )
        bot.send_message(message.chat.id, reply, parse_mode="Markdown")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Error: {e}")

# ================================
# 🚀 Run Bot
# ================================
print("🤖 Telegram Bot sedang berjalan...")
bot.infinity_polling()
