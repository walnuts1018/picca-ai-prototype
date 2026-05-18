from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from contextlib import contextmanager
from typing import Any, Iterator


OPTIONAL_TRANSFORMERS_IMPORT_PROBES = {"scipy", "sklearn"}
OPTIONAL_PADDLEOCR_IMPORT_PROBES = {"scipy", "sklearn"}
OPTIONAL_PADDLEOCR_DISTRIBUTIONS = {"scikit-learn", "scipy"}


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
