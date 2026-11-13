"""Put input layers on one projected analysis grid (PROJECT.md §3).

    python src/prepare_inputs.py dof/cremons --crs EPSG:6708 --pixel 0.10 \\
        --layer rgb  <drone>/map/result.tif 1 2 3 \\
        --layer nir  <drone>/map/result_NIR.tif \\
        --layer ndvi <drone>/map/index_map/NDVI.tif

Each --layer is NAME PATH [BAND ...] (1-based bands; default every band).
Every layer is reprojected onto one north-up grid in --crs that covers all the
sources, with pixels of --pixel metres, and written to OUT_DIR/NAME.tif. The
source, its CRS and pixel size, the resampling and the datum transformation are
written into each file's GeoTIFF tags and into OUT_DIR/prepare.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import tools  # noqa: F401  (installed with `pip install -e .`)
except ModuleNotFoundError:  # running from a checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.io import RESAMPLING, SourceLayer, describe_raster, target_grid, warp_to_grid


def parse_layer(values: list[str]) -> SourceLayer:
    if len(values) < 2:
        raise argparse.ArgumentTypeError("--layer needs NAME PATH [BAND ...]")
    name, path, *bands = values
    try:
        numbers = tuple(int(b) for b in bands) or None
    except ValueError:
        raise argparse.ArgumentTypeError(f"--layer {name}: bands must be whole numbers, got {bands}") from None
    return SourceLayer(name, Path(path), numbers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="folder for NAME.tif and prepare.json, e.g. dof/<area>")
    parser.add_argument("--crs", required=True, help="target projected CRS, e.g. EPSG:6708 (RDN2008 / UTM zone 33N)")
    parser.add_argument("--pixel", type=float, required=True, help="analysis pixel size in metres")
    parser.add_argument("--resampling", default="average", choices=sorted(RESAMPLING), help="default: average (area mean)")
    parser.add_argument("--layer", nargs="+", action="append", required=True, metavar="ARG", help="NAME PATH [BAND ...]")
    args = parser.parse_args(argv)
    try:
        layers = [parse_layer(v) for v in args.layer]
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    names = [layer.name for layer in layers]
    if len(set(names)) != len(names):
        parser.error(f"layer names repeat: {names}")

    grid = target_grid([layer.path for layer in layers], args.crs, args.pixel)
    print(f"Grid: {grid.width} x {grid.height} px at {args.pixel} m in {args.crs}, bounds {[round(v, 2) for v in grid.bounds()]}")
    summary = {"crs": args.crs, "pixel_m": args.pixel, "resampling": args.resampling, "grid": {
        "width": grid.width, "height": grid.height, "transform": list(grid.affine), "bounds": list(grid.bounds())}, "layers": {}}
    for layer in layers:
        dst = args.out_dir / f"{layer.name}.tif"
        provenance = warp_to_grid(layer, dst, grid, args.resampling)
        meta = describe_raster(dst)
        summary["layers"][layer.name] = {"path": str(dst), "count": meta["count"], "dtype": meta["dtypes"][0],
                                         "nodata": meta["nodata"], "provenance": provenance}
        print(f"  {layer.name}: {layer.path} bands {provenance['VINELINER_SOURCE_BANDS']} -> {dst}")
    print(f"  transformation: {provenance['VINELINER_TRANSFORMATION']}")
    (args.out_dir / "prepare.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
