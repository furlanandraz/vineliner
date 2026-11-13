"""Raster layers from the QGIS project onto one analysis grid (PROJECT.md §3).

The core needs every layer on one north-up grid, in a projected CRS in metres,
with square pixels. analysis_grid() chooses it:

  CRS     the CRS given in the dialog; else the first layer's CRS (RGB, NIR-R-G,
          NDVI) when it is projected in metres; else the project's CRS when it
          is; else WGS 84 / UTM at the first layer's centre.
  pixel   the size given in the dialog; else the coarsest layer's own pixel,
          but never finer than PIXELS_PER_ROW_WIDTH per middle row width. The
          search tries every whole-pixel shift across a row width, so finer
          pixels cost quadratically (drone mosaics at 1 cm would not fit in
          memory) without adding row detail.
  extent  the vineyard polygons' bounding box plus a margin for the search
          windows and filters, cut to the first layer's extent. When the CRS
          and pixel size are the first layer's own, the grid sits on that
          layer's pixels, so it is read without resampling.

read_layer() puts one layer on that grid with GDAL: nearest neighbour when the
layer's pixels already match, the area average when pixels get coarser,
bilinear otherwise. The result is float32 with NaN wherever the source has its
NoData value or no coverage.

Coordinates are transformed with GDAL's OSR (transform.py), the engine that
warps the rasters, never with QgsCoordinateTransform: QGIS 3.44 can get stuck
handing back untransformed coordinates. analysis_grid() and plan_read() look at
the QGIS layers, so they run on the main thread (prepareAlgorithm); read_layer()
needs only GDAL and runs on the worker thread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from osgeo import gdal
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsProcessingException,
    QgsRectangle,
    QgsUnitTypes,
)

from ..core.geo import Grid
from .transform import Transform, gdal_srs

PIXELS_PER_ROW_WIDTH = 13  # automatic pixel: at least this many pixels per middle row width (Cremons: 2.6 m -> 0.2 m)
MARGIN_ROW_WIDTHS = 3.0  # extent margin around the polygons, in row widths, plus MARGIN_M
MARGIN_M = 10.0
ALIGN_TOL = 1e-6  # relative pixel-size difference still counted as the same grid
NATIVE_PIXEL_RANGE_M = (0.001, 50.0)  # pixel sizes an image of vine rows can have; outside it the layer's CRS is wrong
MAX_GRID_PIXELS = 50_000_000  # about 16 times Bordeaux (1663 x 1829 px at 0.2 m)
FIX_CRS = ("Remove the layer and add the file again so QGIS reads the file's own CRS, "
           "or set the CRS that matches its coordinates in Layer Properties > Source.")


@dataclass
class AnalysisGrid:
    crs: QgsCoordinateReferenceSystem
    pixel_m: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: int
    height: int
    native_layer: str | None  # the layer whose own pixel grid this is, if any
    wkt: str  # the CRS as WKT and as GDAL takes it, worked out on the main thread (see plan_read)
    gdal_srs: str
    notes: dict = field(default_factory=dict)

    def grid(self) -> Grid:
        return Grid(self.pixel_m, 0.0, self.xmin, 0.0, -self.pixel_m, self.ymax, self.width, self.height, self.wkt)

    def describe(self) -> dict:
        return {
            "crs": self.crs.authid() or self.crs.description(),
            "pixel_m": self.pixel_m,
            "bounds": [self.xmin, self.ymin, self.xmax, self.ymax],
            "size_px": [self.width, self.height],
            "on_native_grid_of": self.native_layer,
        } | self.notes


def layer_problem(key: str, layer) -> str | None:
    """Why a selected raster layer cannot be used in its slot ("rgb", "nir", "ndvi"), or None."""
    name = {"rgb": "RGB", "nir": "NIR-R-G", "ndvi": "NDVI"}[key]
    if not layer.isValid():
        return f"The {name} layer '{layer.name()}' is not valid (its source could not be opened)."
    if layer.providerType() != "gdal":
        return f"The {name} layer '{layer.name()}' is not a raster file GDAL can read (provider '{layer.providerType()}')."
    if not layer.crs().isValid():
        return (
            f"The {name} layer '{layer.name()}' has no coordinate reference system, so where it lies is unknown. "
            "Use a georeferenced raster. (DJI Terra's index_map_color/*_local.tif maps are colour renderings in a "
            "local frame without a CRS; use the GeoTIFFs in map/ and map/index_map/ instead.)"
        )
    bands = layer.bandCount()
    if key == "ndvi" and bands != 1:
        return (
            f"The NDVI layer '{layer.name()}' has {bands} bands. An NDVI layer has one band of NDVI values from -1 to 1; "
            "a 3- or 4-band layer is a colour rendering of NDVI (such as DJI Terra's index_map_color/*.tif), whose values "
            "are colours, and changing its style to grayscale does not change them. Use the single-band NDVI GeoTIFF "
            "(DJI Terra: map/index_map/NDVI.tif)."
        )
    if key != "ndvi" and bands < 3:
        return f"The {name} layer '{layer.name()}' has {bands} band(s); it needs 3 ({'red, green, blue' if key == 'rgb' else 'near-infrared, red, green'})."
    return (coordinates_problem(f"{name} layer '{layer.name()}'", layer.crs(), layer.extent())
            or pixel_problem(f"{name} layer '{layer.name()}'", layer))


def ground_pixel_m(layer) -> float | None:
    """The layer's pixel size on the ground in metres, roughly, from its CRS's units (no transform)."""
    crs = layer.crs()
    px = max(layer.rasterUnitsPerPixelX(), layer.rasterUnitsPerPixelY())
    if crs.isGeographic():
        return px * 111_320.0  # a degree of latitude; the larger side of a square-degree pixel
    factor = QgsUnitTypes.fromUnitToUnitFactor(crs.mapUnits(), Qgis.DistanceUnit.Meters)
    return px * factor if math.isfinite(factor) and factor > 0 else None


def _distance(metres: float) -> str:
    return f"{metres / 1000:.3g} km" if metres >= 1000 else f"{metres:.3g} m" if metres >= 0.01 else f"{metres * 1000:.3g} mm"


def pixel_problem(label: str, layer) -> str | None:
    """Why a raster's pixels, read in its CRS, cannot be an image of vine rows, or None."""
    ground = ground_pixel_m(layer)
    lo, hi = NATIVE_PIXEL_RANGE_M
    if ground is None or lo <= ground <= hi:
        return None
    crs = layer.crs()
    px = max(layer.rasterUnitsPerPixelX(), layer.rasterUnitsPerPixelY())
    units = "degrees" if crs.isGeographic() else QgsUnitTypes.toString(crs.mapUnits())
    message = (
        f"The {label} has pixels {px:.4g} {units} wide in its CRS {crs.authid() or crs.description()}: about "
        f"{_distance(ground)} on the ground, where images of vine rows have {_distance(lo)} to {_distance(hi)}. "
        "A CRS only tells QGIS what the numbers in the file mean; setting one does not move or rescale the image, "
        "so no CRS makes the wrong numbers right."
    )
    if crs.isGeographic() and lo <= px <= hi:
        message += (
            f" {px:.4g} would be a likely pixel size in metres: the file's coordinates are probably metres in a local "
            "frame with no position on Earth, not degrees (DJI Terra's index_map_color/*_local.tif maps are like that, "
            "and so is anything saved from them)."
        )
    return message + " Use a georeferenced raster (DJI Terra: the GeoTIFFs in map/ and map/index_map/, such as map/index_map/NDVI.tif)."


def _within_lon_lat(rect: QgsRectangle) -> bool:
    return -180.0 <= rect.xMinimum() and rect.xMaximum() <= 180.0 and -90.0 <= rect.yMinimum() and rect.yMaximum() <= 90.0


def coordinates_problem(label: str, crs: QgsCoordinateReferenceSystem, extent: QgsRectangle) -> str | None:
    """Why a layer's CRS contradicts its coordinates, or None.

    A CRS set in Layer Properties > Source replaces the file's own without
    changing the coordinates: degrees then read as metres (a vineyard 0.001 m
    across) or metres as degrees (latitudes in the millions).
    """
    if not crs.isValid() or extent.isNull():
        return None
    authid = crs.authid() or crs.description()
    if crs.isGeographic() and not _within_lon_lat(extent):
        return (f"The {label} has the CRS {authid}, in degrees, but its coordinates ({extent.toString(0)}) are not "
                f"longitudes and latitudes; they look like metres. {FIX_CRS}")
    if not crs.isGeographic() and _within_lon_lat(extent) and extent.width() < 1.0 and extent.height() < 1.0:
        return (f"The {label} has the CRS {authid}, in metres, but spans only {extent.width():.2g} x {extent.height():.2g} m "
                f"at coordinates that look like longitudes and latitudes ({extent.toString(5)}): they are degrees. {FIX_CRS}")
    return None


def is_metric(crs: QgsCoordinateReferenceSystem) -> bool:
    return crs.isValid() and not crs.isGeographic() and crs.mapUnits() == Qgis.DistanceUnit.Meters


def utm_crs(layer) -> QgsCoordinateReferenceSystem:
    """WGS 84 / UTM zone of the layer's centre."""
    c = layer.extent().center()
    lon, lat = Transform(layer.crs(), QgsCoordinateReferenceSystem("EPSG:4326"), f"The layer '{layer.name()}'").point(c.x(), c.y())
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return QgsCoordinateReferenceSystem(f"EPSG:{(32600 if lat >= 0 else 32700) + zone}")


def native_pixel_m(layer, crs: QgsCoordinateReferenceSystem) -> float:
    """The layer's pixel size measured in crs, at the layer's centre (the larger of its two sides)."""
    px, py = layer.rasterUnitsPerPixelX(), layer.rasterUnitsPerPixelY()
    if layer.crs() == crs:
        return max(px, py)
    c = layer.extent().center()
    n = 10.0  # measure over 10 pixels for precision
    p0, p1, p2 = Transform(layer.crs(), crs, f"The layer '{layer.name()}'").points(
        [(c.x(), c.y()), (c.x() + n * px, c.y()), (c.x(), c.y() + n * py)])
    return float(max(np.hypot(*(p1 - p0)), np.hypot(*(p2 - p0)))) / n


def _geotransform(layer):
    with gdal.ExceptionMgr(useExceptions=True):
        ds = gdal.Open(layer.source())
        return ds.GetGeoTransform(), ds.RasterCount


def on_grid(layer, target: "AnalysisGrid") -> bool:
    """True when the layer's own pixels are the analysis grid's pixels (same CRS, size and alignment)."""
    if layer.crs() != target.crs:
        return False
    gt, _ = _geotransform(layer)
    px = target.pixel_m
    if gt[2] != 0 or gt[4] != 0 or abs(gt[1] - px) > ALIGN_TOL * px or abs(-gt[5] - px) > ALIGN_TOL * px:
        return False
    dx, dy = (gt[0] - target.xmin) / px, (target.ymax - gt[3]) / px
    return abs(dx - round(dx)) < 1e-4 and abs(dy - round(dy)) < 1e-4


def analysis_grid(layers: dict, polygons_extent: QgsRectangle, polygons_crs, requested_crs, requested_pixel_m: float,
                  middle_width_m: float, max_width_m: float, context) -> AnalysisGrid:
    """layers: {"rgb"/"nir"/"ndvi": QgsRasterLayer} in that order; the first sets the CRS and extent."""
    key, first = next(iter(layers.items()))
    notes = {}
    if requested_crs is not None and requested_crs.isValid():
        if not is_metric(requested_crs):
            raise QgsProcessingException(f"The analysis CRS {requested_crs.authid()} is not projected in metres.")
        crs, notes["crs_from"] = requested_crs, "dialog"
    elif is_metric(first.crs()):
        crs, notes["crs_from"] = first.crs(), f"the {key} layer"
    elif context.project() is not None and is_metric(context.project().crs()):
        crs, notes["crs_from"] = context.project().crs(), "the project"
    else:
        crs, notes["crs_from"] = utm_crs(first), "WGS 84 / UTM at the first layer's centre"

    natives = {}
    for k, layer in layers.items():
        natives[k] = native_pixel_m(layer, crs)
        lo, hi = NATIVE_PIXEL_RANGE_M
        if not (math.isfinite(natives[k]) and lo <= natives[k] <= hi):
            raise QgsProcessingException(
                f"The {k} layer '{layer.name()}' has pixels of {natives[k]:.3g} m in {crs.authid()}; images of vine rows have "
                f"{lo:g} to {hi:g} m. Its CRS ({layer.crs().authid() or layer.crs().description()}) probably does not match "
                f"the coordinates in the file. {FIX_CRS}")
    if requested_pixel_m and requested_pixel_m > 0:
        pixel, notes["pixel_from"] = float(requested_pixel_m), "dialog"
    else:
        coarsest = max(natives.values())
        floor = round(middle_width_m / PIXELS_PER_ROW_WIDTH, 2)
        if coarsest >= floor:
            pixel, notes["pixel_from"] = coarsest, "the coarsest layer's own pixel"
        else:
            pixel = floor
            notes["pixel_from"] = f"1/{PIXELS_PER_ROW_WIDTH} of the middle row width (the layers are finer: {coarsest:.3g} m)"
    notes["native_pixels_m"] = {k: round(v, 6) for k, v in natives.items()}

    extent = Transform(polygons_crs, crs, "The vineyard polygon layer").rectangle(polygons_extent)
    span = max(extent.width(), extent.height())
    if not math.isfinite(span) or span < max_width_m:
        raise QgsProcessingException(
            f"The vineyard polygons span only {span:.3g} m in {crs.authid()}, too little for rows up to {max_width_m:g} m apart. "
            f"If they are larger on the map, the polygon layer's CRS ({polygons_crs.authid() or polygons_crs.description()}) "
            f"probably does not match the coordinates in its file. {FIX_CRS}")
    extent.grow(MARGIN_ROW_WIDTHS * max_width_m + MARGIN_M)
    layer_extent = Transform(first.crs(), crs, f"The {key} layer '{first.name()}'").rectangle(first.extent())
    extent = extent.intersect(layer_extent)
    if extent.isEmpty():
        raise QgsProcessingException(
            f"The vineyard polygons do not overlap the {key} layer '{first.name()}'. Check that they cover the same place, "
            f"and that each layer's CRS matches its coordinates (Layer Properties > Source).")

    native_layer = None
    gt, _ = _geotransform(first)
    north_up = gt[2] == 0 and gt[4] == 0 and gt[1] > 0 and gt[5] < 0
    if first.crs() == crs and north_up and abs(gt[1] - pixel) <= ALIGN_TOL * pixel and abs(-gt[5] - pixel) <= ALIGN_TOL * pixel:
        ox, oy, native_layer = gt[0], gt[3], key  # snap to the first layer's own pixels: no resampling
    else:
        ox, oy = 0.0, 0.0
    xmin = ox + math.floor((extent.xMinimum() - ox) / pixel) * pixel
    xmax = ox + math.ceil((extent.xMaximum() - ox) / pixel) * pixel
    ymax = oy + math.ceil((extent.yMaximum() - oy) / pixel) * pixel
    ymin = oy + math.floor((extent.yMinimum() - oy) / pixel) * pixel
    width, height = int(round((xmax - xmin) / pixel)), int(round((ymax - ymin) / pixel))
    if min(width, height) * pixel < 2 * max_width_m:
        raise QgsProcessingException(
            f"The {key} layer '{first.name()}' covers only {width * pixel:.3g} x {height * pixel:.3g} m around the vineyard "
            f"polygons, too little for rows up to {max_width_m:g} m apart.")
    if width * height > MAX_GRID_PIXELS:
        raise QgsProcessingException(
            f"The analysis grid would be {width} x {height} px ({width * pixel:.0f} x {height * pixel:.0f} m at {pixel:g} m), "
            f"more than {MAX_GRID_PIXELS / 1e6:.0f} million pixels. Run fewer polygons at a time (Selected features only) "
            f"or set a coarser analysis pixel size. If the vineyard polygons are not {_distance(span)} across on the ground, "
            f"the polygon layer's CRS does not match its coordinates (were they drawn over a raster without a real CRS?).")
    return AnalysisGrid(crs, pixel, xmin, ymin, xmax, ymax, width, height, native_layer,
                        wkt=crs.toWkt(), gdal_srs=gdal_srs(crs), notes=notes)


@dataclass
class LayerRead:
    """How read_layer() puts one layer on the analysis grid.

    plan_read() works it out on the main thread, with QGIS's CRS objects;
    read_layer() then needs only GDAL, so it can run on the worker thread.
    """
    key: str
    name: str
    source: str
    crs_label: str
    src_srs: str  # the layer's CRS as QGIS has it (it may be assigned in Layer Properties, over the file's own)
    bands: list[int]
    native_pixel_m: float
    resampling: str


def plan_read(key: str, layer, bands: list[int], target: AnalysisGrid) -> LayerRead:
    native = native_pixel_m(layer, target.crs)
    if on_grid(layer, target):
        resampling = "near"  # the layer's pixels are the grid's: read as they are
    elif native < target.pixel_m * (1 - 1e-3):
        resampling = "average"  # coarser pixels: the area mean of the finer ones
    else:
        resampling = "bilinear"
    crs = layer.crs()
    return LayerRead(key, layer.name(), layer.source(), crs.authid() or crs.description(), gdal_srs(crs),
                     list(bands), native, resampling)


def read_layer(plan: LayerRead, target: AnalysisGrid) -> tuple[np.ndarray, dict]:
    """The planned bands (1-based) on the analysis grid: (bands, rows, cols) float32, NaN for NoData. GDAL only."""
    with gdal.ExceptionMgr(useExceptions=True):
        try:
            ds = gdal.Open(plan.source)
        except RuntimeError as exc:
            raise QgsProcessingException(f"GDAL cannot read the {plan.key} layer '{plan.name}' ({plan.source}): {exc}") from exc
        if ds.RasterCount < max(plan.bands):
            raise QgsProcessingException(
                f"The {plan.key} layer '{plan.name}' has {ds.RasterCount} band(s); it needs {max(plan.bands)}.")
        subset = gdal.Translate("", ds, format="VRT", bandList=list(plan.bands))  # keeps each band's NoData
        out = gdal.Warp(
            "", subset, format="MEM",
            outputBounds=(target.xmin, target.ymin, target.xmax, target.ymax), width=target.width, height=target.height,
            srcSRS=plan.src_srs, dstSRS=target.gdal_srs, resampleAlg=plan.resampling, outputType=gdal.GDT_Float32,
            dstNodata=float("nan"), multithread=True,
        )
        data = out.ReadAsArray().astype(np.float32, copy=False)
        out = subset = ds = None
    if data.ndim == 2:
        data = data[np.newaxis]
    metadata = {
        "layer": plan.name,
        "source": plan.source,
        "crs": plan.crs_label,
        "bands": list(plan.bands),
        "native_pixel_m": round(plan.native_pixel_m, 6),
        "resampling": plan.resampling,
        "valid_share": round(float(np.isfinite(data).all(axis=0).mean()), 4),
    }
    return data, metadata
