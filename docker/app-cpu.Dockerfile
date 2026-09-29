# syntax=docker/dockerfile:1
# Приложение vinishko на CPU: python -m vinishko.app, поиск через OpenVINO (группа cpu-inference), без группы flash-inference.
# Собирается из корня репозитория: docker compose build app

FROM python:3.13-slim-trixie AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /bin/uv
# git — для git-источников из pyproject: CLIP и форк open_clip
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.13
WORKDIR /app
COPY pyproject.toml uv.lock ./
# колёса flash-attn и causal-conv1d: uv проверяет path-источники из uv.lock и тогда, когда их группа не ставится
COPY wheels ./wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group cpu-inference --no-install-project

FROM python:3.13-slim-trixie
# libGL, glib и xcb нужны OpenCV, остальное лежит в колёсах
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY vinishko ./vinishko
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    NORMALIZER_DEV=cpu \
    VIS_SEARCHER_DEV=cpu \
    SOMMELIER_DATA_DIR=/data/sessions
EXPOSE 8000
CMD ["python", "-m", "vinishko.app", "--host", "0.0.0.0", "--port", "8000"]
