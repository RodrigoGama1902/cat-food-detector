FROM python:3.12-slim

# System libraries required by OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY detector.py app.py config.py ./
COPY templates ./templates

# Persist calibration to a mounted volume.
ENV CONFIG_PATH=/data/config.json
# Persist uploaded calibration images (empty/medium/full) to the same volume so
# they survive container restarts and rebuilds.
ENV CALIB_DIR=/data/calib

EXPOSE 8000

# --access-logfile - logs every request to stdout (visible via docker logs).
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--access-logfile", "-", "app:app"]
