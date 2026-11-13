"""The plugin folder as QGIS loads it: metadata, entry point, versions, the zip, and the core/QGIS boundary."""

import configparser
import importlib.util
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "vineliner"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_has_every_mandatory_key_and_no_interpolation_traps():
    meta = configparser.ConfigParser()  # QGIS reads metadata.txt with interpolation on, so '%' would break it
    meta.read(PLUGIN / "metadata.txt", encoding="utf-8")
    general = meta["general"]
    for key in ("name", "qgisMinimumVersion", "description", "about", "version", "author", "email", "repository"):
        assert general.get(key), key
    assert general["hasProcessingProvider"] == "yes" and (PLUGIN / general["icon"]).exists()
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?", general["version"])


def test_versions_agree():
    found = load_script("package_plugin").versions()
    assert len(set(found.values())) == 1, found


def test_entry_point_imports_without_qgis():
    import vineliner

    assert callable(vineliner.classFactory)  # the plugin modules are only imported inside classFactory


def test_core_never_imports_qgis_or_gdal():
    pattern = re.compile(r"^\s*(from|import)\s+(qgis|osgeo|PyQt5|PyQt6)\b", re.M)
    offenders = [p for p in (PLUGIN / "core").rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_coordinates_are_never_transformed_with_qgis():
    """QGIS 3.44 can get stuck handing back untransformed coordinates from a cached QgsCoordinateTransform (in every
    thread, until restart); the plugin transforms with GDAL's OSR instead (qgis_io/transform.py)."""
    pattern = re.compile(r"QgsCoordinateTransform\s*\(|\.transformBoundingBox\s*\(|QgsGeometry\b.*\.transform\s*\(")
    offenders = [f"{p.relative_to(ROOT)}:{i}" for p in PLUGIN.rglob("*.py")
                 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if pattern.search(line) and not line.lstrip().startswith("#")]
    assert offenders == []


def test_zip_holds_one_plugin_folder_without_caches(tmp_path):
    package = load_script("package_plugin")
    path = package.build(tmp_path)
    names = zipfile.ZipFile(path).namelist()
    assert all(n.startswith("vineliner/") for n in names)
    for required in ("vineliner/__init__.py", "vineliner/metadata.txt", "vineliner/icon.png", "vineliner/core/defaults.toml",
                     "vineliner/processing_provider/detect_rows.py", "vineliner/qgis_io/rasters.py"):
        assert required in names, required
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
