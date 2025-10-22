# ========================================================
# 🚀 DOCKERFILE - PRO TRADER AI (FastAPI + Telegram Bot)
# ========================================================

# Gunakan base image ringan
FROM python:3.10-slim

# Install dependencies (termasuk OCR & OpenCV libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja
WORKDIR /app

# Salin semua file proyek ke container
COPY . .

# Install semua dependencies dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment default
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Jalankan FastAPI + Telegram Bot bersamaan
CMD sh -c "uvicorn main_combined_learning:app --host 0.0.0.0 --port ${PORT:-8000} & python telegram_bot.py"
