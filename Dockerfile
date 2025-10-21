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
    supervisor \
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
# 🧠 Supervisor configuration
# ================================
RUN echo "[supervisord]\nnodaemon=true\n\
[program:fastapi]\ncommand=uvicorn main_combined_learning:app --host 0.0.0.0 --port ${PORT:-8000}\nautorestart=true\n\
[program:telegrambot]\ncommand=python telegram_bot.py\nautorestart=true" > /etc/supervisor/conf.d/supervisord.conf

# ================================
# 🚀 Run both services
# ================================
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
