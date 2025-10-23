# ======================================================
# 🤖 PRO TRADER AI - UNIVERSAL DOCKERFILE (FINAL)
# ======================================================

FROM python:3.10-slim

# ------------------------------------------------------
# 🧰 Install dependencies for OpenCV & OCR (Tesseract)
# ------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------
# 📂 Set working directory
# ------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------
# 📦 Copy dependencies and source code
# ------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# ------------------------------------------------------
# ⚙️ Environment Variables (default)
# Railway akan override secara otomatis
# ------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV TARGET_FILE=main_combined_learning.py

# ------------------------------------------------------
# 🚀 Universal CMD
# Jalankan salah satu dari:
# - main_combined_learning.py (AI Agent)
# - backtester.py (Backtester)
# - telegram_bot.py (Bot Telegram)
# ------------------------------------------------------
CMD bash -c "\
if [ \"$TARGET_FILE\" = 'telegram_bot.py' ]; then \
    echo '💬 Menjalankan Telegram Bot...' && python telegram_bot.py; \
elif [ \"$TARGET_FILE\" = 'backtester.py' ]; then \
    echo '📊 Menjalankan Backtester di port ${PORT}...' && uvicorn backtester:app --host 0.0.0.0 --port ${PORT}; \
else \
    echo '🧠 Menjalankan AI Agent di port ${PORT}...' && uvicorn main_combined_learning:app --host 0.0.0.0 --port ${PORT}; \
fi"
