# Digit Detection API - CPU only, ONNX Runtime, no TensorFlow at runtime.
# Build:  docker build -t digit-api .
# Run:    docker run --rm -p 7860:7860 digit-api          -> http://localhost:7860
# Works unchanged on Hugging Face Spaces (port 7860), Google Cloud Run and Render ($PORT).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/mnist_cnn.onnx \
    OMP_NUM_THREADS=1

# opencv-python-headless and onnxruntime need these two shared libraries on a slim image
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

COPY preprocess.py predictor.py detect.py server.py ./
COPY static ./static
COPY models/mnist_cnn.onnx ./models/mnist_cnn.onnx

# run as an unprivileged user (Hugging Face Spaces expects uid 1000)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
# PORT is injected by Cloud Run / Render; Hugging Face Spaces uses 7860.
# WORKERS: 1 per vCPU is a good default for this CPU-bound service.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860} --workers ${WORKERS:-1} --timeout-keep-alive 30"]
