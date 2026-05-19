FROM python:3.12-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip pip install uv

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY scripts /app/scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --group vision

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

CMD ["uv", "run", "python", "scripts/run_dense_service.py"]
