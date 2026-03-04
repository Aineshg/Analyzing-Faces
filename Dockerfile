FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Native build/runtime deps for OpenCV, dlib/face-recognition, and scientific stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip setuptools wheel && \
    pip install .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "analyzing_faces.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
