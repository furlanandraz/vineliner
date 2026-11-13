# Vineliner

A QGIS plugin that finds the vine rows in aerial or drone imagery and draws each row as a line.

The design brief, decisions and test results are in [PROJECT.md](PROJECT.md).

## Install

Requires QGIS 3.40 or later. The plugin uses NumPy, SciPy and Shapely 2, which QGIS 3.40+ ships with its Python on Windows. If one is missing, the plugin says so in the message bar when it loads.

- **From a zip:** `python scripts/package_plugin.py` builds `dist/vineliner-<version>.zip`. In QGIS, open *Plugins → Manage and Install Plugins → Install from ZIP* and pick it.
- **For development:** `python scripts/link_plugin.py` links `vineliner/` into the default QGIS profile's plugins folder (a junction on Windows). Enable *Vineliner* under *Installed*, then reload it with the Plugin Reloader plugin after editing the code. `--remove` takes the link away.

## Use

*Raster → Vineliner → Detect vine rows*, the Vineliner icon on the Plugins toolbar, or *Processing Toolbox → Vineliner → Detect vine rows*.

### Inputs

| Parameter | |
| --- | --- |
| RGB layer | Bands red, green, blue (an alpha band after them is ignored). Optional |
| NIR-R-G layer | A false-colour layer with bands near-infrared, red, green. Optional |
| NDVI layer | One band with NDVI values from −1 to 1, such as DJI Terra's `map/index_map/NDVI.tif`. Optional |
| Vineyard polygons | One polygon per planting block. Rows are searched and clipped inside each. *Selected features only* works |
| Row width range | Minimum and maximum distance between neighbouring rows, in metres. 21 widths across the range are searched, and rows spaced outside it are flagged |

At least one raster layer is needed, and every layer given is used:

- RGB alone → RGB evidence.
- RGB and NIR-R-G → RGB, NIR, and NDVI computed as (NIR − red) / (NIR + red) from the NIR-R-G layer.
- NIR-R-G alone → NIR and the computed NDVI.
- NDVI alone → NDVI. A supplied NDVI layer is used rather than one computed from NIR-R-G.

When there are several, they compete per vineyard polygon. Each polygon uses the one that shows its rows best: the highest median template score at the polygon's centre and the 8 search boxes around it. The log lists the scores and the choice. Mixing them instead smears the rows, because RGB marks a row as dark (canopy and its shadow) and NIR as bright (canopy only). On the Bordeaux test data an equal mix found 94 of the 313 rows, the per-polygon choice 320.

Separate single-band NIR, red and green files, such as a drone's, become a NIR-R-G layer with *Raster → Miscellaneous → Build Virtual Raster* and *Place each input file into a separate band*, in that order.

Every layer must be georeferenced. DJI Terra's `map/index_map_color/*_local.tif` maps are colour renderings in a local frame with no position on Earth: setting a CRS on one in *Layer Properties* only relabels its coordinates, and a grayscale made from its colours is not NDVI. Layers like that are refused before the run with the reason, as are a CRS that contradicts a layer's coordinates (degrees read as metres or the reverse), pixels outside 1 mm to 50 m on the ground, and NDVI values mostly outside −1..1.

Pick a row width range that contains the rows' spacing: a range that misses it finds rows between rows. Bordeaux's rows are about 1.4 m apart, Cormons' 2.6 m.

Advanced parameters, all optional:

| Parameter | Default |
| --- | --- |
| Analysis CRS | The first layer's CRS if it is projected in metres; else the project's CRS if it is; else WGS 84 / UTM at the first layer's centre |
| Analysis pixel size | The coarsest layer's own pixel, but at least 1/13 of the middle row width. Drone mosaics at 1 cm become 0.2 m for 2.6 m rows |
| Run report | A JSON file with the settings, inputs, per-block search details, checks and timings |

The layers are put on one north-up grid in that CRS, covering the polygons plus a margin. A layer whose pixels already match is read as it is; one with finer pixels is averaged, any other resampled bilinearly. The log says which.

### Outputs

**Vine rows** (lines) carry exactly one attribute:

| Field | Type | Meaning |
| --- | --- | --- |
| `needs_review` | Boolean | True when a person should check the row: an end that stops inside the block, a crossing, more than 15 m or 30 % drawn without image support, spacing outside the range at more than 25 % of samples, support at less than half the samples, shorter than 4 m, an uncertain position in the row order, or a row the missing-row estimate added between rows the image did not show |

**Vineyard row statistics** (polygons) are the input polygons with their own attributes, a vertex wherever a row meets the boundary, and:

| Field | Meaning |
| --- | --- |
| `average_row_length` | Mean row length, m |
| `average_row_direction` | Mean row direction as an azimuth: degrees clockwise from north, 0–180 (length-weighted) |
| `average_row_width` | Mean distance between neighbouring rows, m: the mean of each row's median measured spacing |
| `confidence_interval` | Half-width of the 95 % confidence interval of `average_row_width`, m (Student t over the rows): the width is `average_row_width ± confidence_interval`. Rows the estimate added without image support are left out of both |

Both outputs are in the analysis CRS.

## How it works

The method is described in PROJECT.md (§5–§11). In short, for each polygon:

1. **Evidence.** A row-evidence raster in [0, 1] is built from each layer: a black top-hat of the RGB gray (rows darker than the ground), and white top-hats of NIR and NDVI (rows brighter), each scaled per block. The block uses the one that shows its rows best (see Inputs).
2. **Search.** A five-line template is matched at search centres every 2 m over every rotation (1°), row width and whole-pixel position. A full search at the polygon's centroid first narrows the rotations and widths for the rest.
3. **Rows.** Accepted matches seed chains, which grow into rows, join, bridge faint patches, are smoothed (at most 20° per vertex) and clipped to the polygon.
4. **Missing rows.** Rows that do not reach the boundary at both ends are removed. Rows are estimated where neighbours stand too far apart, and removed pieces lying on an estimated row are merged into it.
5. **Checks and numbering** set `needs_review`.

The settings are fixed in [vineliner/core/defaults.toml](vineliner/core/defaults.toml). They are the final configuration tested on the Bordeaux data (PROJECT.md §18); the dialog sets only the row width range. Kernels are cached in the QGIS profile (`cache/vineliner/kernels`), which saves about 20 s per run after the first.

## Development

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest
```

- **Core tests** (`python -m pytest`, about 235 tests, 30 s) run in this environment. They need no QGIS and no imagery: unit tests per module, and end-to-end runs on a synthetic striped vineyard whose rows are known.
- **QGIS tests** run the plugin through Processing in QGIS's own Python, on synthetic GeoTIFFs:

  ```
  "C:\Program Files\QGIS 3.44.15\bin\python-qgis-ltr.bat" tests/qgis/test_plugin.py -v
  ```

  They use a throwaway QGIS profile. Plain pytest skips them.

### Layout

```
vineliner/                    the plugin folder QGIS loads
  __init__.py                 classFactory()
  metadata.txt, icon.png
  plugin.py                   registers the Processing provider, Raster menu entry and Plugins toolbar icon
  processing_provider/        provider.py, detect_rows.py (the algorithm: parameters, outputs)
  qgis_io/                    rasters.py (layers onto the analysis grid, GDAL), vectors.py (polygons in, geometries out),
                              transform.py (coordinate transforms with GDAL's OSR, PROJECT.md §17 item 49)
  core/                       the method: NumPy, SciPy and Shapely only, no QGIS imports
    defaults.toml             every setting
    config/ geo/ imagery/ vineyards/ template/ search/ reconstruction/ validation/ export/ pipeline/
scripts/                      package_plugin.py (zip), link_plugin.py (development link)
tests/                        mirrors core/; tests/qgis/ for the plugin inside QGIS
dof/, poly/                   local test data: imagery and vineyard polygons (git-ignored)
```

`core/` is independent of QGIS: `qgis_io/` turns layers into NumPy arrays and Shapely polygons, `core.pipeline.run()` returns rows and statistics in memory, and the algorithm writes them to the output layers. A test checks that nothing in `core/` imports `qgis` or `osgeo`.

## Test data

The test data is not in the repository: `dof/` (imagery) and `poly/` (vineyard polygons) are git-ignored and kept locally.

- **Bordeaux** (`dof/bordeaux/`, git-ignored): a 0.20 m RGB orthophoto (`rgb.tif`) and a false-colour NIR, red, green one (`nir.tif`) on the same grid, with 7 hand-drawn blocks in `poly/poly.gpkg`. Rows are about 1.4 m apart: use a row width range of 1.2–1.6 m. PROJECT.md §14 has the details and assumptions.
- **Cremons** (DJI Terra flight of 2025-08-22, outside the repository): `map/result.tif` (RGB), `result_NIR.tif`, `result_Red.tif` and `result_Green.tif` (stack them as NIR-R-G, see above), and `index_map/NDVI.tif`, all EPSG:4326 at about 1 cm. The block is in `poly/cremons.gpkg` (EPSG:6708). Use a row width range of 2.4–2.8 m and, to match earlier results, the analysis CRS EPSG:6708 (RDN2008 / UTM zone 33N). PROJECT.md §14 and §19 have the details.

## Contact

Questions, bug reports and data to try: open an issue at [github.com/furlanandraz/vineliner/issues](https://github.com/furlanandraz/vineliner/issues), or write to [info@furlanandraz.com](mailto:info@furlanandraz.com).

## Licence

GNU General Public License, version 2 or (at your option) any later version, the licence of QGIS itself: see [LICENSE](LICENSE). Copyright (C) 2026 furlanandraz.
