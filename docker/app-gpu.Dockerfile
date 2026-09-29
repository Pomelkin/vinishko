# syntax=docker/dockerfile:1
# Приложение vinishko на GPU: поиск через TensorRT (группа flash-inference). TensorRT 11 собран под CUDA 13, поэтому база — CUDA 13.0,
# на хосте нужен драйвер ≥ 580 и NVIDIA Container Toolkit. torch и его библиотеки CUDA 12.8 приезжают колёсами из uv.lock,
# системный CUDA им не нужен: образ base, без runtime и cudnn.
# Собирается из корня репозитория: docker compose -f docker-compose.yml -f docker-compose.gpu.override.yml build app

FROM nvidia/cuda:13.0.3-base-ubuntu24.04 AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /bin/uv
# git — для git-источников из pyproject: CLIP и форк open_clip
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Python 3.13 в Ubuntu 24.04 нет — ставит uv, в /opt/python: этот же путь переедет в итоговый образ
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed
RUN uv python install 3.13
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY wheels ./wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group flash-inference --no-install-project

FROM nvidia/cuda:13.0.3-base-ubuntu24.04
# libGL, glib и xcb нужны OpenCV, остальное лежит в колёсах
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /opt/python /opt/python
COPY --from=build /app/.venv /app/.venv
COPY vinishko ./vinishko
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    NORMALIZER_DEV=cuda:0 \
    VIS_SEARCHER_DEV=cuda:0 \
    SOMMELIER_DATA_DIR=/data/sessions
EXPOSE 8000
CMD ["python", "-m", "vinishko.app", "--host", "0.0.0.0", "--port", "8000"]
