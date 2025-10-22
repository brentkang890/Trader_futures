# ================================
# 🔧 Base image: Python + system deps (OCR + OpenCV)
# ================================
FROM python:3.10-slim

# Install dependencies & system libs for OpenCV + Tesseract OCR
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
COPY requirements.txt requirements.txt
COPY main_combined_learning.py .
COPY telegram_bot.py .

# ================================
# 📦 Install Python dependencies
# ================================
RUN pip install --no-cache-dir -r requirements.txt

# ================================
# ⚙️ Environment Variables
# ================================
# (Railway akan override ini lewat "Variables" tab)
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV BOT_TOKEN=""
ENV CHAT_ID=""
ENV APP_URL=""

# ================================
# 🚀 Run FastAPI + Telegram Bot together
# ================================
CMD sh -c "uvicorn main_combined_learning:app --host 0.0.0.0 --port ${PORT:-8000} & python telegram_bot.py"
