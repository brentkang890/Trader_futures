# Dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Jalankan uvicorn (FastAPI) lalu telegram bot.
CMD sh -c "uvicorn main_combined_learning:app --host 0.0.0.0 --port ${PORT:-8000} & python telegram_bot.py"
