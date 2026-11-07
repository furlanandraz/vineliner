"""Print raster and polygon metadata as JSON (PROJECT.md §3, §14).

    python src/describe_inputs.py dof/bordeaux/rgb.tif dof/bordeaux/nir.tif poly/poly.gpkg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import tools  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.io import describe_raster


def describe_vector(path: Path) -> dict:
    import geopandas as gpd
    import pyogrio

    info = {"path": str(path), "layers": []}
    for name, geom_type in pyogrio.list_layers(path):
        gdf = gpd.read_file(path, layer=name)
        info["layers"].append({
            "name": name,
            "geometry_type": geom_type,
            "features": len(gdf),
            "fields": [c for c in gdf.columns if c != "geometry"],
            "crs_epsg": gdf.crs.to_epsg() if gdf.crs else None,
            "crs_name": gdf.crs.name if gdf.crs else None,
            "bounds": [float(v) for v in gdf.total_bounds],
            "areas_m2": [round(float(a), 1) for a in gdf.area],
            "all_valid": bool(gdf.is_valid.all()),
        })
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    out = []
    for p in args.paths:
        if p.suffix.lower() in (".tif", ".tiff"):
            d = describe_raster(p)
            d.pop("crs_wkt", None)
            out.append(d)
        else:
            out.append(describe_vector(p))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
