FROM python:3.12-slim

# System deps for xgboost / numpy / psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy project source
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

# Create logs dir (MetricsCollector writes here)
RUN mkdir -p logs

# Render injects $PORT at runtime; default to 8000 locally
ENV PORT=8000
ENV MODEL_PATH=models/advanced_xgboost.pkl

EXPOSE $PORT

CMD uvicorn src.inference.api:app \
        --host 0.0.0.0 \
        --port $PORT \
        --workers 2 \
        --log-level info
