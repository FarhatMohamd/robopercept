# CPU image: simulator, ONNX Runtime and Ultralytics backends, web dashboard.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY robopercept ./robopercept
COPY configs ./configs
COPY scripts ./scripts

ARG EXTRAS="onnx"
RUN pip install ".[${EXTRAS}]"

RUN useradd --create-home robot && mkdir -p /app/runs /app/models && chown -R robot /app
USER robot

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health')"
ENTRYPOINT ["robopercept"]
CMD ["demo", "--scenario", "urban"]
