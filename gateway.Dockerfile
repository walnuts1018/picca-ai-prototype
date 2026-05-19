FROM python:3.12-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY scripts /app/scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

EXPOSE 8000

CMD ["python", "scripts/run_gateway.py"]
