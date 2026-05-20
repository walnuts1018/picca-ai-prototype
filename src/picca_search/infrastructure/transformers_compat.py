from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from contextlib import contextmanager
from typing import Any, Iterator


OPTIONAL_TRANSFORMERS_IMPORT_PROBES = {"scipy", "sklearn"}
@contextmanager
def _hide_optional_packages(
    *,
    import_package_names: set[str],
    distribution_names: set[str] = frozenset(),
) -> Iterator[None]:
    original_find_spec = importlib.util.find_spec
    original_version = importlib.metadata.version

    def find_spec(name: str, package: str | None = None):
        if name in import_package_names:
            return None
        return original_find_spec(name, package)

    def version(name: str) -> str:
        if name in distribution_names:
            raise importlib.metadata.PackageNotFoundError(name)
        return original_version(name)

    importlib.util.find_spec = find_spec
    importlib.metadata.version = version
    try:
        yield
    finally:
        importlib.util.find_spec = original_find_spec
        importlib.metadata.version = original_version


def import_module_symbols(
    module_name: str,
    *names: str,
    hidden_import_packages: set[str] = frozenset(),
    hidden_distribution_names: set[str] = frozenset(),
) -> tuple[Any, ...]:
    with _hide_optional_packages(
        import_package_names=hidden_import_packages,
        distribution_names=hidden_distribution_names,
    ):
        module = importlib.import_module(module_name)
        return tuple(getattr(module, name) for name in names)


def import_transformers_symbols(*names: str) -> tuple[Any, ...]:
    return import_module_symbols(
        "transformers",
        *names,
        hidden_import_packages=OPTIONAL_TRANSFORMERS_IMPORT_PROBES,
    )


def ort_provider_for_device(device: str, *, require_accelerator: bool = False) -> str:
    preferred_provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    fallback_provider = "CPUExecutionProvider"
    try:
        import onnxruntime as ort

        available_providers = set(ort.get_available_providers())
    except Exception as exc:
        if require_accelerator and device == "cuda":
            raise RuntimeError(
                "MODEL_DEVICE=cuda was requested, but onnxruntime could not be imported. "
                "Install the GPU runtime and verify CUDA libraries are available."
            ) from exc
        return fallback_provider
    if preferred_provider in available_providers:
        return preferred_provider
    if require_accelerator and device == "cuda":
        raise RuntimeError(
            "MODEL_DEVICE=cuda was requested, but CUDAExecutionProvider is unavailable. "
            f"Available providers: {sorted(available_providers)}"
        )
    if fallback_provider in available_providers:
        return fallback_provider
    return preferred_provider


def transformers_pretrained_kwargs(*, prefer_slow: bool) -> dict[str, Any]:
    # Metaspace などの一部の pre-tokenizer は item assignment に非対応であり、
    # fix_mistral_regex=True に設定すると TypeError が発生するため False に設定する。
    kwargs: dict[str, Any] = {"fix_mistral_regex": False}
    if prefer_slow:
        kwargs["use_fast"] = False
    return kwargs

