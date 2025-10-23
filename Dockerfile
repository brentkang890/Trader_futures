# ================================================================
# 🤖 PRO TRADER AI - UNIVERSAL DOCKERFILE (AI + Backtester + Telegram Bot)
# ================================================================

# -------------------------------
# 🐍 Base image Python + system libs
# -------------------------------
FROM python:3.10-slim

# Install dependencies for OCR (Tesseract) & OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# 📂 Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# 📦 Copy dependency list
# -------------------------------
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# 📦 Copy all source code
# -------------------------------
COPY . .

# -------------------------------
# ⚙️ Default Environment Variables
# -------------------------------
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# -------------------------------
# 🚀 Default command
# (Railway akan override sesuai “Start Command”)
# -------------------------------
CMD ["uvicorn", "main_combined_learning:app", "--host", "0.0.0.0", "--port", "8000"]
