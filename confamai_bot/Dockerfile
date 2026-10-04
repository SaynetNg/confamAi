# Confam AI WhatsApp bot
#
# Build from THIS directory:
#
#     docker build -t confamai-bot confamai_bot
#     docker run -d --name confamai-bot -p 8080:8080 --env-file .env confamai-bot
#
# TLS is terminated by whatever sits in front (Cloudflare Tunnel, Caddy, a PaaS
# ingress), not here. Meta requires a valid CA certificate on the public
# endpoint and rejects self-signed, so do not expose this container's port
# directly to the internet.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# /app is the working directory and the import root. The package is copied to
# /app/confamai_bot rather than /app, because every module uses relative
# imports (`from .config import ...`) and `uvicorn confamai_bot.webhook:app`
# resolves the package name from the working directory.
WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/confamai_bot/

# Sessions live in SQLite by default, which needs a real volume or it is lost on
# every deploy. For a stateless deployment set WHATSAPP_SESSION_BACKEND=supabase
# instead and mount nothing here.
VOLUME ["/app/data"]

ENV WHATSAPP_SQLITE_PATH=/app/data/sessions.db \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8080'), timeout=8)"

CMD ["sh", "-c", "uvicorn confamai_bot.webhook:app --host 0.0.0.0 --port ${PORT:-8080} --no-access-log"]