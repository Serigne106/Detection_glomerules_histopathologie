# API de prédiction sur CPU.
#   docker build -t glomseg .
#   docker run -p 8000:8000 -v $PWD/models:/app/models glomseg
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# PyTorch CPU (image ~4x plus légère que la version CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"
COPY app ./app

RUN useradd --create-home appuser
USER appuser
ENV MODEL_PATH=/app/models/model_deploy.pth OMP_NUM_THREADS=2
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
