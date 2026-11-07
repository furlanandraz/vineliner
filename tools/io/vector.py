"""Vector reading and writing through GeoPandas (PROJECT.md §4, §13).

This is a file adapter. A QGIS plugin can replace it with QgsVectorLayer
reading and writing, as long as it returns the same structures.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from pyproj import CRS


def read_polygons(path: str | Path, layer: str | None, id_field: str | None, target_crs_wkt: str):
    """Read polygons as (ids, geometries, provenance, attribute records), in the raster's CRS.

    Without id_field, ids are 0, 1, ... in file order. A polygon layer in a
    different CRS is reprojected explicitly and the reprojection is recorded.
    """
    path = Path(path)
    gdf = gpd.read_file(path, layer=layer)
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS; assign one explicitly before use (PROJECT.md §3)")
    target = CRS.from_wkt(target_crs_wkt)
    provenance = {"path": str(path), "layer": layer, "id_field": id_field, "features": len(gdf), "reprojected": False}
    if not CRS.from_user_input(gdf.crs).equals(target, ignore_axis_order=True):
        gdf = gdf.to_crs(target)
        provenance["reprojected"] = True
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if not set(gdf.geom_type) <= {"Polygon", "MultiPolygon"}:
        raise ValueError(f"{path} must hold polygons, found {sorted(set(gdf.geom_type))}")
    if id_field is None:
        ids = list(range(len(gdf)))
    else:
        ids = [int(v) for v in gdf[id_field]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: field {id_field!r} has duplicate ids")
    records = gdf.drop(columns="geometry").to_dict("records")
    return ids, list(gdf.geometry), provenance, records


def write_layer(path: str | Path, records: list[dict], geometries: list, crs_wkt: str, layer: str | None = None, driver: str = "GPKG") -> Path:
    """Write one layer. Shapefile field names must already be 10 characters or fewer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=CRS.from_wkt(crs_wkt))
    if driver == "ESRI Shapefile":
        too_long = [c for c in gdf.columns if c != "geometry" and len(c) > 10]
        if too_long:
            raise ValueError(f"shapefile field names longer than 10 characters: {too_long}")
        gdf.to_file(path, driver=driver)
    else:
        gdf.to_file(path, layer=layer, driver=driver)
    return path
