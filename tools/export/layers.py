"""Vector outputs (PROJECT.md §13).

shapefiles/ holds the products for human review in QGIS:
  rows.shp     final row lines, clipped to their block, with review attributes
  blocks.shp   the supplied block polygons with a vertex at every row end, plus counts
  issues.shp   points where checks found something (dangling ends, crossings, spacing)
  bridges.shp  parts of rows drawn without image support
  removed.shp  hanging rows dropped by [export] drop_hanging_rows, with their attributes

vineliner.gpkg holds the same layers plus diagnostics (centres, seeds, chains,
endpoints, joins). Shapefile field names are 10 characters or fewer.
"""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import LineString, Point

from ..io import write_layer
from ..validation.review import ROW_FIELDS


# Schemas for review layers, so an empty one is still written with its fields.
SCHEMAS = {
    "rows": (ROW_FIELDS, "LineString"),
    "removed": (ROW_FIELDS, "LineString"),
    "issues": (("id", "check", "block", "sections", "value", "detail"), "Point"),
    "bridges": (("section", "block", "length_m"), "LineString"),
}


def write_outputs(out_dir: Path, crs_wkt: str, gpkg_layers: dict, shapefiles: dict, shapefile: bool, geopackage: bool) -> dict:
    """layers: name -> (records, geometries).

    Empty review layers (SCHEMAS) are written empty, with their fields; other
    empty layers are skipped and reported.
    """
    written: dict = {"geopackage": None, "shapefiles": {}, "empty": [], "skipped_empty": []}

    def write(path, name, records, geoms, **kw):
        if not records and name not in SCHEMAS:
            written["skipped_empty"].append(name)
            return False
        fields, geom_type = SCHEMAS.get(name, (None, None))
        write_layer(path, records, geoms, crs_wkt, fields=fields, geometry_type=geom_type, **kw)
        if not records:
            written["empty"].append(name)
        return True

    if geopackage:
        gpkg = out_dir / "vineliner.gpkg"
        for name, (records, geoms) in gpkg_layers.items():
            write(gpkg, name, records, geoms, layer=name)
        written["geopackage"] = str(gpkg)
    if shapefile:
        for name, (records, geoms) in shapefiles.items():
            path = out_dir / "shapefiles" / f"{name}.shp"
            if write(path, name, records, geoms, driver="ESRI Shapefile"):
                written["shapefiles"][name] = str(path)
    written["empty"] = sorted(set(written["empty"]))
    return written


def issue_layer(issues) -> tuple[list[dict], list]:
    records, geoms = [], []
    for i, f in enumerate(issues):
        records.append({
            "id": i,
            "check": f.check,
            "block": f.block_id,
            "sections": ",".join(map(str, f.section_ids)),
            "value": round(f.value, 3) if f.value is not None else None,
            "detail": f.detail,
        })
        geoms.append(f.geometry if isinstance(f.geometry, Point) else f.geometry.representative_point())
    return records, geoms


def bridge_layer(sections) -> tuple[list[dict], list]:
    records, geoms = [], []
    for s in sections:
        if s.bridged is not None and not s.bridged.is_empty:
            records.append({"section": s.id, "block": s.block_id, "length_m": round(s.bridged_m, 2)})
            geoms.append(s.bridged)
    return records, geoms


def block_records(blocks, block_reports, sections, reviews) -> list[dict]:
    """Input attributes (names shortened to 10 characters) plus counts per block."""
    by_block = {b["block_id"]: b for b in block_reports}
    records = []
    for b in blocks:
        rec = {}
        for key, value in b.attributes.items():
            short = key[:10]
            while short in rec:
                short = short[:9] + "_"
            rec[short] = value
        mine = [r for r in reviews if r["block"] == b.id]
        extra = {
            "block": b.id,
            "area_m2": round(b.area_m2, 1),
            "rows": len(mine),
            "review": sum(r["review"] for r in mine),
            "removed": by_block[b.id]["validation"].get("removed_hanging", 0),
            "row_ends": sum(len(s.boundary_points()) for s in sections if s.block_id == b.id),
            "length_m": round(sum(r["length_m"] for r in mine), 1),
            "theta_deg": by_block[b.id].get("theta_median_deg"),
            "spacing_m": by_block[b.id].get("spacing_median_m"),
            "search": by_block[b.id].get("search_mode", "full"),
        }
        pt = by_block[b.id].get("centroid_pretest")
        if pt is not None:
            extra.update(pt_theta=pt["theta_deg"], pt_spacing=pt["spacing_m"], pt_score=pt["score"], pt_ring=pt["ring"], pt_reason=pt["reason"])
        for key, value in extra.items():
            rec[key if key not in rec else (key[:8] + "_v")] = value
        records.append(rec)
    return records


def seed_geometry(seed) -> LineString:
    return LineString(seed.endpoints())
