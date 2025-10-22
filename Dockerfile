# Dockerfile
FROM python:3.10-slim

# install system deps for opencv + tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy project files
COPY . .

# install python deps
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# run uvicorn + telegram bot (praktis). Untuk production, jalankan service terpisah.
CMD sh -c "uvicorn main_protrader:app --host 0.0.0.0 --port ${PORT:-8000} & python telegram_bot.py"
