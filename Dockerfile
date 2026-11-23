# syntax=docker/dockerfile:1

# ---- builder ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /application
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project || uv sync --no-dev --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev || uv sync --no-dev

# ---- runtime ----
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 application
WORKDIR /application
COPY --from=builder --chown=application:application /application /application
ENV PATH="/application/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER application
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import ml_graph_compiler" || exit 1
CMD ["python", "-m", "ml_graph_compiler", "--help"]
