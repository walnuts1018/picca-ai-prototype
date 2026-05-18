# Natural Language Image Search Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a uv-managed Python prototype that ingests local images into Qdrant and searches them with Japanese natural language queries.

**Architecture:** Domain types live in `src/picca_search/domain.py`; application workflows compose injected ports in `src/picca_search/application.py`; infrastructure adapters for Qdrant and ML models live under `src/picca_search/infrastructure/`; top-level scripts only parse arguments and wire dependencies. I/O is pushed to the script and infrastructure edges, while validation and workflow shapes are explicit in typed values.

**Tech Stack:** Python 3.12, uv, pytest, qdrant-client, transformers, torch, Pillow, Qdrant docker compose.

---

### Task 1: Project Scaffold and Domain Types

**Files:**
- Create: `pyproject.toml`
- Create: `src/picca_search/__init__.py`
- Create: `src/picca_search/domain.py`
- Test: `tests/test_domain.py`

- [ ] Write failing tests for domain validation.
- [ ] Run `uv run --group dev pytest tests/test_domain.py -q` and verify import/validation failures.
- [ ] Implement newtypes and validation.
- [ ] Run `uv run --group dev pytest tests/test_domain.py -q` and verify pass.

### Task 2: Application Workflows

**Files:**
- Create: `src/picca_search/application.py`
- Test: `tests/test_application.py`

- [ ] Write failing tests using fake encoders and fake index.
- [ ] Run `uv run --group dev pytest tests/test_application.py -q` and verify failures.
- [ ] Implement ingest and search workflows with injected dependencies.
- [ ] Run `uv run --group dev pytest tests/test_application.py -q` and verify pass.

### Task 3: Qdrant and Model Adapters

**Files:**
- Create: `src/picca_search/infrastructure/__init__.py`
- Create: `src/picca_search/infrastructure/qdrant_index.py`
- Create: `src/picca_search/infrastructure/embedding_models.py`
- Create: `docker-compose.yml`
- Test: `tests/test_qdrant_mapping.py`

- [ ] Write failing tests for converting domain vectors to Qdrant models.
- [ ] Run `uv run --group dev pytest tests/test_qdrant_mapping.py -q` and verify failures.
- [ ] Implement Qdrant collection setup, upsert, RRF search, and model adapters.
- [ ] Run `uv run --group dev pytest tests/test_qdrant_mapping.py -q` and verify pass.

### Task 4: Script Entry Points and Documentation

**Files:**
- Create: `scripts/ingest_images.py`
- Create: `scripts/search_images.py`
- Modify: `README.md`

- [ ] Add thin scripts for ingestion and search.
- [ ] Document Qdrant startup and uv commands.
- [ ] Run `uv run --group dev pytest -q`.
- [ ] Run script help commands with `uv run python scripts/ingest_images.py --help` and `uv run python scripts/search_images.py --help`.
