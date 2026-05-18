from __future__ import annotations

import tomllib
from pathlib import Path


def test_transformers_is_pinned_below_next_major_for_florence_remote_code() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())

    transformers_specs = [
        dependency
        for dependency in project["project"]["dependencies"]
        if dependency.startswith("transformers")
    ]

    assert transformers_specs == ["transformers>=4.45.0,<5"]


def test_florence_remote_model_dependencies_are_declared() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())

    assert any(
        dependency.startswith("timm")
        for dependency in project["project"]["dependencies"]
    )
