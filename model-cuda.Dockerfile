FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY scripts /app/scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --group vision --python python3

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get -y update && apt-get upgrade -y && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

CMD ["python3", "scripts/run_dense_service.py"]
