# ================================================================
# 🚀 PRO TRADER AI - UNIVERSAL DOCKERFILE
# Support: AI Agent + Backtester + Telegram Bot
# ================================================================

# Gunakan Python slim agar ringan & cepat
FROM python:3.10-slim

# ================================================================
# ⚙️ Install System Dependencies
# (dibutuhkan untuk OpenCV, OCR, dan FastAPI)
# ================================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ================================================================
# 📂 Set Working Directory
# ================================================================
WORKDIR /app

# ================================================================
# 📦 Copy dan Install Dependencies
# ================================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ================================================================
# 📂 Copy Semua File Project
# ================================================================
COPY . .

# ================================================================
# ⚙️ Environment Default (Railway override ini otomatis)
# ================================================================
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# ================================================================
# 🧠 Default Command
# (Railway akan override sesuai project-nya)
# ================================================================
# Contoh command di Railway:
# - AI Agent     → uvicorn main_combined_learning:app --host 0.0.0.0 --port $PORT
# - Backtester   → uvicorn backtester:app --host 0.0.0.0 --port $PORT
# - Telegram Bot → python telegram_bot.py
# ================================================================
CMD ["python", "main_combined_learning.py"]
