FROM python:3.11-slim

# Tesseract (OCR) + poppler-utils (pdf2image's pdftoppm) — this is exactly
# why this service can't run on plain PHP shared hosting: these are native
# binaries, not something you can composer-install.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
