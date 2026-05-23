FROM python:3.11-slim

WORKDIR /app

# System deps for reportlab fonts and pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Use the Render-tuned production requirements (groq, google-generativeai,
# boto3, asyncpg, langfuse, sentry-sdk). The base `requirements.txt` is for
# local dev/test only.
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1
# Render injects PORT; default to 8000 for plain `docker run`.
ENV PORT=8000
EXPOSE 8000

# Shell form so ${PORT} expands at container start.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
