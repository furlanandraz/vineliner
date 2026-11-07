# Vineliner

Finds vineyard rows in aerial orthophotos and exports them as georeferenced polylines.

The design brief, decisions and open questions are in [PROJECT.md](PROJECT.md).

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest
```

## Running

```bash
python src/run_pipeline.py configs/bordeaux.toml               # all blocks
python src/run_pipeline.py configs/bordeaux.toml --blocks 0 3  # some blocks
python src/run_pipeline.py configs/bordeaux.toml --plan        # counts only, no search
python src/describe_inputs.py dof/bordeaux/rgb.tif poly/poly.gpkg
```

A full Bordeaux run (7 blocks, about 7,900 search centres) takes about 7 minutes. Outputs go to `runs/<timestamp>[-name]/`.

**For review in QGIS** (`shapefiles/`, all in the raster's CRS):

| File | Contents |
| --- | --- |
| `rows.shp` | Final row lines, clipped to their block, with the attributes below |
| `blocks.shp` | The supplied block polygons, same shape and attributes, with a vertex inserted wherever a row meets the boundary. Row ends snap exactly to these vertices. Adds `block`, `rows`, `review`, `row_ends`, `length_m`, `theta_deg`, `spacing_m` |
| `issues.shp` | Points where a check found something: `check` = completion (dangling end), cross_row, spacing (narrow/wide), containment, self_intersection |
| `bridges.shp` | Parts of rows drawn without image support, where a row was carried straight through a faint patch |
| `removed.shp` | Only with `drop_hanging_rows = true`: the removed hanging rows, with the same attributes as `rows.shp`. A reference for adding the missing rows by hand |

Every review layer is written even when it's empty, so the set of files stays the same between runs.

**Other outputs**

| File | Contents |
| --- | --- |
| `vineliner.gpkg` | The same four layers plus diagnostics: `centres`, `seeds`, `chains`, `endpoints`, `joins` |
| `overlay.png` | Imagery with rows by confidence (green high, amber medium, red low), bridges white, dangling ends magenta, crossings red, blocks blue |
| `report.json` | Counts per block, issue totals, open choices this run made, timings, input metadata |
| `evidence.tif`, `valid.tif` | Row-evidence raster V actually used, and the valid block mask |
| `scores/`, `centre_profiles.npz` | Score surfaces and per-centre profiles |
| `config.json` | The exact merged configuration |

### Row attributes (`rows.shp`)

Checks never remove or fail a row. They are reported, so a person can decide.

| Field | Meaning |
| --- | --- |
| `review` | 1 = a person should look at this row |
| `conf` | `high`: no notes. `medium`: minor notes only (≥ 1 m bridged, or spacing off at ≥ 5% of samples). `low`: review = 1 |
| `issues` | The reasons in words, e.g. `dangling end end; bridged 18.0 m (35%)` |
| `dangling` | Ends that stop inside the block (0–2) |
| `crossings` | Other rows this row crosses |
| `start_st`, `end_st` | How each end stopped: boundary, unsupported, conflict, limit |
| `bridged_m`, `bridge_pct` | Length and share drawn without image support |
| `ev_mean`, `ev_sup_pct` | Mean template score along the line (1 m samples), and share of samples with support |
| `sp_med_m`, `sp_out_pct` | Median measured distance to neighbouring rows, and share of samples outside the validation range |
| `near_row` | The row this one runs too close to, if any. Of two rows that are too close, only the one with weaker evidence (`ev_mean`) is flagged for it; the other gets a note |
| `pos_rms_m` | RMS distance of the row's seed detections from the final line, a measure of positional consistency |
| `theta_deg`, `spacing_m`, `length_m` | Mean direction, template spacing, length |
| `n_seeds`, `n_joins`, `row_id`, `block`, `id` | Provenance |

### Keeping only complete rows

If the imagery is poor, set this in the area's config:

```toml
[export]
drop_hanging_rows = true
```

Then `rows.shp` keeps only rows that meet their block boundary at exactly two points: their two ends. Rows with a dangling end, and rows that touch the boundary anywhere else, are removed. The checks and attributes are then computed on the kept rows only. The removed rows go to `removed.shp`, so you can add the missing rows by hand in QGIS. The default, `false`, keeps every row.

### When a row is flagged

A row gets `review = 1` when:
- it has a dangling end, crosses another row, crosses itself, or leaves its block, or
- it passes a `[review]` threshold in the config: more than 15 m or 30% bridged, spacing off at more than 25% of samples, support at less than 50% of samples, or shorter than 4 m.

## Layout

`src/` holds only entry files. Importable code lives in the `tools/` package, one subpackage per pipeline stage. Modules inside `tools/` import each other with relative imports only, so the package can later move into a QGIS plugin unchanged.

```
src/                     entry files
  run_pipeline.py        run the pipeline for a config
  describe_inputs.py     print raster/polygon metadata
tools/
  config/                load, merge and validate configs (loader, spacing, settings)
  geo/                   pixel grid <-> world coordinates, angle convention
  io/                    file adapters: rasterio rasters, GeoPandas vectors
  imagery/               valid mask (255 rule), row-evidence raster
  vineyards/             block polygons and masks; classification slot (not implemented)
  template/              five-line template geometry, rasterised kernels
  search/                hypotheses, scoring, batched local search, selection, persistence, storage
  reconstruction/        seeds, chains, growth and joins, final row sections
  validation/            the five geometry checks
  export/                GeoPackage/shapefile layers, report, overlay PNG
  pipeline/              runs the stages in order
tests/                   mirrors tools/, plus an end-to-end run on synthetic stripes
configs/                 default.toml and per-area files
dof/, poly/              imagery (git-ignored) and block polygons
```

Only `io/` and `export/` touch file formats. The other subpackages work on NumPy arrays and Shapely geometries, so a QGIS plugin only needs to replace the file adapters. QGIS ships NumPy and GDAL but usually not rasterio or GeoPandas.

## Configuration

Run defaults are in [configs/default.toml](configs/default.toml). Settings for one area go in their own file, which only needs the keys it changes and is merged on top of the defaults. [configs/bordeaux.toml](configs/bordeaux.toml) is the example.

Settings that PROJECT.md leaves open are unset in the defaults. A run refuses to start until every one of them is set, and it lists all the missing ones at once. In `bordeaux.toml`, each value chosen for an open setting is marked `PICK`, with the reason next to it. These are first-run choices to compare, not validated values. Each run's `report.json` lists them under `open_choices_set_by_this_run`.

### Row spacing (row width)

Row width means the distance between neighbouring rows: the lane the tractors drive in. Rows themselves are reduced to lines, so they have no width. You choose the range of spacings to search in the `[spacing]` table:

```toml
[spacing]
min_m = 1.20    # smallest spacing searched
max_m = 1.60    # largest spacing searched
step_m = 0.02   # 1.20, 1.22, ..., 1.60 m
```

- Both ends are searched.
- The range is fixed before the run.
- `max_m - min_m` must be a whole number of steps. Otherwise the run refuses to start instead of silently skipping `max_m`.

Each extra spacing option adds a full search per location: 180 rotations times every translation. 21 options is the default.

## Data

Put orthophotos in `dof/<area>/`. GeoTIFFs are git-ignored, so imagery stays local. See PROJECT.md §14 for the test pair in `dof/bordeaux/`.

[poly/poly.gpkg](poly/poly.gpkg) holds 7 hand-drawn vineyard blocks for the Bordeaux scene. They stand in for automatic classification, which isn't implemented yet. Reports label them as a supplied fixture.

### Assumptions for the Bordeaux test pair

These are working assumptions. The files don't record them, and the provider hasn't confirmed them.

- **Band order of `nir.tif` is NIR, red, green.** It's a false-colour product. Its bands 2 and 3 closely track the red and green bands of `rgb.tif`. The supplied `ndvi.tif` equals (NIR − red) / (NIR + red) computed from `nir.tif` band 1 and `rgb.tif` band 1, with correlation 1.0. That confirms band 1 is NIR.
- **Band order of `rgb.tif` is red, green, blue.**
- **`ndvi.tif`** is used as an optional evidence component (`[evidence] ndvi_weight`). Its own NoData value (−3.4e38) and any value outside [−1, 1] are masked.
- **The value 255 means "no data or outlier".**
  - The files declare 255 as NoData.
  - Vegetation never reflects that much in any band, so a 255 is water glint, a white roof or similar. It's never row evidence.
  - A pixel is masked when any band of either file is 255. That removes 955 pixels, 0.024 % of the scene.
  - Checked on 2026-11-07: all 19 pixels with NIR at 255 are near-white in RGB (mean 249, 252, 242). None look like vegetation.

The source product, tile and capture date are still unknown. Record them before reporting results.
