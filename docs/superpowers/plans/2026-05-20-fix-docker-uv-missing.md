# Fix Missing UV in Docker Runtime Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `exec: "uv": executable file not found` error by ensuring the `uv` binary is available in the runtime stage of `model.Dockerfile`.

**Architecture:** Copy the `uv` binary from the official `ghcr.io/astral-sh/uv` image into the runtime stage. This is more robust than installing it via pip in the builder stage and hoping it survives or re-installing it.

**Tech Stack:** Docker, uv

---

### Task 1: Fix model.Dockerfile

**Files:**
- Modify: `model.Dockerfile`

- [ ] **Step 1: Update model.Dockerfile to copy uv binary**

```dockerfile
<<<<
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app /app
====
FROM python:3.12-slim AS runtime

# Install uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY --from=builder /app /app
>>>>
```

- [ ] **Step 2: Verify the change (Dry run/Inspection)**
Check that `uv` is now used in a way that matches the official documentation for Dockerizing uv projects.

### Task 2: Fix model-cuda.Dockerfile (Consistency)

**Files:**
- Modify: `model-cuda.Dockerfile`

- [ ] **Step 1: Update model-cuda.Dockerfile to copy uv binary**

```dockerfile
<<<<
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
====
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS runtime

# Install uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
>>>>
```

### Task 3: Fix gateway.Dockerfile (Consistency)

**Files:**
- Modify: `gateway.Dockerfile`

- [ ] **Step 1: Update gateway.Dockerfile to copy uv binary**

```dockerfile
<<<<
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app /app
====
FROM python:3.12-slim AS runtime

# Install uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY --from=builder /app /app
>>>>
```
