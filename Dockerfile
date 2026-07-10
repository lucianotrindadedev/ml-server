FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# WEB_CONCURRENCY controla quantos workers uvicorn rodam em paralelo.
# Default 1 (seguro): cada worker carrega sua própria cópia dos modelos
# (InsightFace + PaddleOCR ~1-1.5GB cada). Só aumente depois de confirmar a RAM
# disponível — regra prática: WEB_CONCURRENCY = min(núcleos, RAM_livre_GB / 1.5).
# Ajuste também ML_NUM_THREADS = núcleos / WEB_CONCURRENCY no painel do Coolify.
CMD uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-1} --timeout-keep-alive 75
