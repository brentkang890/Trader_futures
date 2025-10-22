FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
COPY main_combined_learning.py .

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main_combined_learning:app", "--host", "0.0.0.0", "--port", "8000"]
