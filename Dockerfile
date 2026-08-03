FROM python:3.11-slim

# System packages: tesseract-ocr (OCR engine) + the libs Playwright's
# Chromium needs to run headless. This step is exactly what Render's native
# Python runtime blocks (no apt-get access there) — Docker gives it back.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    wget \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Downloads Chromium itself (the browser binary Playwright drives).
# --with-deps is skipped since we already installed the needed libs above
# via apt (keeps the image a bit leaner / avoids duplicate installs).
RUN playwright install chromium

COPY . .

ENV PORT=5000
EXPOSE 5000

CMD ["python", "ai_chat.py"]
