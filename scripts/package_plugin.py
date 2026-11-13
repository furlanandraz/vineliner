"""Build the installable plugin zip: dist/vineliner-<version>.zip.

    python scripts/package_plugin.py

The zip holds one folder, vineliner/, as QGIS expects ("Plugins > Manage and
Install Plugins > Install from ZIP"). Caches and compiled files are left out.
The version in vineliner/metadata.txt, vineliner/core/__init__.py and
pyproject.toml must agree, or nothing is built.
"""

from __future__ import annotations

import configparser
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "vineliner"
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def versions() -> dict[str, str]:
    meta = configparser.ConfigParser(interpolation=None)
    meta.read(PLUGIN / "metadata.txt", encoding="utf-8")
    core = re.search(r'__version__ = "([^"]+)"', (PLUGIN / "core" / "__init__.py").read_text(encoding="utf-8")).group(1)
    project = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    return {"metadata.txt": meta["general"]["version"], "core/__init__.py": core, "pyproject.toml": project}


def plugin_files() -> list[Path]:
    return sorted(
        p for p in PLUGIN.rglob("*")
        if p.is_file() and not SKIP_DIRS & set(p.relative_to(PLUGIN).parts) and p.suffix not in SKIP_SUFFIXES
    )


def build(out_dir: Path = ROOT / "dist") -> Path:
    found = versions()
    if len(set(found.values())) != 1:
        raise SystemExit(f"versions disagree: {found}")
    version = next(iter(found.values()))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"vineliner-{version}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in plugin_files():
            zf.write(f, Path("vineliner") / f.relative_to(PLUGIN))
    return path


if __name__ == "__main__":
    zip_path = build()
    print(f"{zip_path} ({zip_path.stat().st_size / 1024:.0f} KiB)")
    sys.exit(0)
