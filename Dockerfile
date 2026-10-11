FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=5000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 franchiseflow \
    && useradd --uid 10001 --gid franchiseflow --create-home franchiseflow

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=franchiseflow:franchiseflow prototype/franchiseflow/ ./prototype/franchiseflow/
COPY --chown=franchiseflow:franchiseflow spikes/franchiseflow/ ./spikes/franchiseflow/
RUN mkdir -p prototype/franchiseflow/uploads prototype/franchiseflow/outputs/batches/packages prototype/franchiseflow/brand_profiles \
    && chown -R franchiseflow:franchiseflow prototype/franchiseflow

USER franchiseflow
EXPOSE 5000
VOLUME ["/app/prototype/franchiseflow/uploads", "/app/prototype/franchiseflow/outputs", "/app/prototype/franchiseflow/brand_profiles"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','5000')+'/api/health',timeout=8).close()"
CMD ["python", "prototype/franchiseflow/server.py"]
