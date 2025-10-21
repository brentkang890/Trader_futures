# ================================
# 🔧 Base image: Python + dependencies
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
# 📦 Copy all project files
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
# 🚀 Run both FastAPI (AI) + Telegram Bot together
# ================================
CMD sh -c "uvicorn main_combined_learning:app --host 0.0.0.0 --port ${PORT:-8000} & python telegram_bot.py"
