from __future__ import annotations

from pathlib import Path
import sys


def build_manifest_sources(image_name: str, digests_dir: Path) -> list[str]:
    digests = sorted(
        path.name
        for path in digests_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )
    if not digests:
        raise ValueError(f"No digest files found in {digests_dir}")
    return [f"{image_name}@sha256:{digest}" for digest in digests]


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: python scripts/build_manifest_sources.py <image-name> <digests-dir>",
            file=sys.stderr,
        )
        return 2

    image_name = sys.argv[1]
    digests_dir = Path(sys.argv[2])
    for source in build_manifest_sources(image_name, digests_dir):
        print(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
