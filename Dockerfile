# ================================
# 🔧 Base image: Python + dependencies
# ================================
FROM python:3.10-slim

# Install dependencies & system libs (Tesseract, OpenCV runtime)
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ================================
# 📂 Set working directory
# ================================
WORKDIR /app

# ================================
# 📦 Copy project files
# ================================
COPY . .

# ================================
# 📦 Install Python dependencies
# ================================
RUN pip install --no-cache-dir -r requirements.txt

# ================================
# ⚙️ Environment variables
# ================================
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# ================================
# 🚀 Start FastAPI (AI Agent) + Telegram Bot
# ================================
# - Uvicorn menjalankan FastAPI
# - Telegram bot berjalan di background
# - Gunakan shell untuk mengeksekusi keduanya secara paralel
CMD sh -c "uvicorn main_protrader:app --host 0.0.0.0 --port ${PORT:-8000} & python telegram_bot.py"
