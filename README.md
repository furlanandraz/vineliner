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

### Tests

`python -m pytest` runs about 180 tests in roughly 20 seconds. None of them need the orthophotos in `dof/`, so they pass on a fresh clone:

- **Unit tests** check one module each, using small hand-built inputs and mocks. For example, a fake scorer stands in for the image when testing row growth, joins and gaps, and the pipeline is mocked when testing the command line.
- **The end-to-end tests** in `tests/pipeline/` run the whole pipeline on a synthetic striped vineyard, where the true rows are known.

`tests/` mirrors the `tools/` packages; `tests/cli/` covers the entry files in `src/`.

## Running

```bash
python src/run_pipeline.py configs/centroid-pretest-cache.toml               # all blocks
python src/run_pipeline.py configs/centroid-pretest-cache.toml --blocks 0 3  # some blocks
python src/run_pipeline.py configs/full-search.toml --plan                   # counts only, no search
python src/run_pipeline.py configs/multispectral.toml --layers ndvi --name ndvi  # only some input layers
python src/describe_inputs.py dof/bordeaux/rgb.tif poly/poly.gpkg
```

A full Bordeaux run (7 blocks, about 7,900 search centres) takes about 7 minutes; Cremons (1 block, 833 centres) about 20 s. Outputs go to `<runs_dir>/<timestamp>[-name]/`. `runs_dir` is set in `[export]`: `runs/` by default, `runs/cremons/` in `configs/multispectral.toml`.

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
| `conf` | `high`: no notes. `medium`: minor notes only (≥ 1 m bridged, or spacing off at ≥ 5% of samples). `low`: review = 1. `merged` / `estimated`: an estimated row with or without detected pieces squashed in |
| `issues` | The reasons in words, e.g. `dangling end end; bridged 18.0 m (35%)` |
| `dangling` | Ends that stop inside the block (0–2) |
| `crossings` | Other rows this row crosses |
| `start_st`, `end_st` | How each end stopped: boundary, unsupported, conflict, limit |
| `bridged_m`, `bridge_pct` | Length and share drawn without image support |
| `det_pct` | Share of the row drawn from the image: everything not bridged for detected rows, the squashed pieces for estimated rows |
| `ev_mean`, `ev_sup_pct` | Mean template score along the line (1 m samples), and share of samples with support |
| `sp_med_m`, `sp_out_pct` | Median measured distance to neighbouring rows, and share of samples outside the validation range |
| `row_no`, `row_no_ok` | Row number within the block, and whether its neighbours agree on it (with `[row_numbering]`) |
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
  prepare_inputs.py      reproject layers onto one projected analysis grid
  describe_inputs.py     print raster/polygon metadata
tools/
  config/                load, merge and validate configs (loader, spacing, settings)
  geo/                   pixel grid <-> world coordinates, angle convention
  io/                    file adapters: rasterio rasters, GeoPandas vectors, reprojection (warp)
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

Run defaults are in [configs/default.toml](configs/default.toml). Run settings go in their own file, named after what they set. A file only needs the keys it changes, and is merged on top of the defaults. It can also start with `extends = "other.toml"` to build on another file, so variants stay a few lines long:

| Config | What it runs |
| --- | --- |
| [configs/centroid-pretest.toml](configs/centroid-pretest.toml) | The full settings for the test data: inputs, assumptions and first-run picks. Centroid pre-test on, kernels built every run |
| [configs/centroid-pretest-cache.toml](configs/centroid-pretest-cache.toml) | The same, with the kernel cache on. The quickest |
| [configs/full-search.toml](configs/full-search.toml) | The same, without the pre-test: the full grid search baseline, for comparison |
| [configs/centroid-pretest-cache-row-estimate.toml](configs/centroid-pretest-cache-row-estimate.toml) | Pre-test and cache, plus the missing-row estimate (with squash), with hanging rows deleted |
| [configs/centroid-pretest-cache-row-estimate-numbering.toml](configs/centroid-pretest-cache-row-estimate-numbering.toml) | The same, plus row numbering |
| [configs/multispectral.toml](configs/multispectral.toml) | The same method on RGB, NIR and NDVI layers together, for the Cremons drone data. Outputs to `runs/cremons/` |

The test data's inputs live in `centroid-pretest.toml`. The file names don't refer to the git-ignored imagery, so a fresh clone isn't confused by them.

### Input layers: RGB, NIR, NDVI

```toml
[input]
rgb = "dof/cremons/rgb.tif"     # each layer is optional; at least one must be in use
rgb_bands = ["red", "green", "blue"]
use_rgb = true                  # the switch: false runs without the layer and keeps its path
nir = "dof/cremons/nir.tif"
nir_bands = ["nir"]             # one NIR band, or a false-colour file such as ["nir", "red", "green"]
use_nir = true
ndvi = "dof/cremons/ndvi.tif"   # any index in [-1, 1] with green = high works here, e.g. GNDVI
use_ndvi = true

[evidence]
rgb_weight = 1.0                # relative weights, rescaled over the layers in use
nir_weight = 1.0
ndvi_weight = 1.0
```

- **Switches.** A layer is in use when its path is set and its `use_*` switch is on (the default). `--layers ndvi nir` on the command line does the same for one run: only the named layers stay on.
- **Evidence.** Each layer in use with weight > 0 adds one component to the row evidence V, and the weights of those layers are rescaled to sum to 1. So switching a layer off needs no other change. The run log, `settings_notes` and `report.json` (`evidence.weights`) show the shares used.
  - **RGB:** gray from `bands`, rows darker than the ground (`method`: `black_tophat` or `gray_stretch`).
  - **NIR** and **NDVI:** rows brighter than the ground (`nir_method` / `ndvi_method`: `white_tophat` or `stretch`).
  - Only a component that is used needs its method. The defaults (`rgb_weight = 1`, the others 0) are the RGB-only baseline, so the Bordeaux configs behave exactly as before.
- **NoData.** NaN always counts as NoData. `nodata_value` applies to the RGB and NIR files; unset, each file's own declared NoData applies. The NDVI file's own NoData and values outside [−1, 1] are masked.
- **One grid.** Every layer in use must share one projected, north-up grid; the first layer in use (RGB, else NIR, else NDVI) sets it. The overlay is drawn on the RGB, or on the NIR or NDVI in gray when there is no RGB.

### Preparing drone or other unaligned layers

`src/prepare_inputs.py` reprojects layers onto one grid in a projected CRS, with an explicit pixel size and resampling, and records where each came from:

```bash
python src/prepare_inputs.py dof/cremons --crs EPSG:6708 --pixel 0.20 \
    --layer rgb  <flight>/map/result.tif 1 2 3 \
    --layer nir  <flight>/map/result_NIR.tif \
    --layer ndvi <flight>/map/index_map/NDVI.tif
```

- Each `--layer` is `NAME PATH [BAND ...]` and becomes `OUT_DIR/NAME.tif`. The grid covers every source and has its corners on whole pixels.
- Resampling is `average` by default, the area mean, which is right when making pixels coarser. The source's NoData is left out of the average; float outputs use NaN as NoData.
- The source path, its CRS and pixel size, the resampling and PROJ's datum transformation go into each file's GeoTIFF tags, so every `report.json` repeats them, and into `OUT_DIR/prepare.json`.
- The pixel size sets the search cost: the translation box is (row spacing / pixel)² per hypothesis. At 0.20 m and 2.6 m rows that is 169, against 49 for Bordeaux.

Settings that PROJECT.md leaves open are unset in the defaults. A run refuses to start until every one of them is set, and it lists all the missing ones at once. In `centroid-pretest.toml`, each value chosen for an open setting is marked `PICK`, with the reason next to it. These are first-run choices to compare, not validated values. Each run's `report.json` lists them under `open_choices_set_by_this_run`.

### Faster search: centroid pre-test

```toml
[centroid_pretest]
enabled = true              # on in centroid-pretest*.toml, off in the defaults
theta_window_deg = 10.0     # grid rotations within +/- this of the pre-test direction
spacing_window_frac = 0.10  # grid spacings within +/- 10 % of the pre-test spacing
neighbour_rings = 3         # rings of neighbouring boxes tried when the centroid box is weak
```

For each block:
1. One search box on the polygon's centroid is searched in full: every phase, all 180 rotations and all spacings.
2. If that box is weak (best score below `accept_min`), rings of neighbouring boxes around it are tested the same way, spaced like the grid centres. The first ring with an accepted box wins.
3. The winning box's direction and spacing become the block's pre-test. The grid search then still scores every phase at every centre, but only for rotations and spacings within the windows around the pre-test.
4. A block gets the full grid search when its centroid lies outside the polygon, or when no box within `neighbour_rings` is accepted.

`blocks.shp` records each block's search mode and pre-test (`search`, `pt_theta`, `pt_spacing`, `pt_score`, `pt_ring`, `pt_reason`). The GeoPackage layer `pretests` marks the box each pre-test came from. The log warns when many grid winners sit on the edge of the rotation window, which means the pre-test was probably off.

### Smoothing

```toml
[smoothing]
enabled = true      # on by default
max_turn_deg = 20.0 # the largest direction change allowed between consecutive vertices
```

While any vertex of a row turns by more than `max_turn_deg`, the sharpest one is removed, worst first. The two end vertices are never removed, so rows keep ending on the boundary. Bridged segments stay marked as bridged.

It runs on the grown rows and again on the final clipped rows, so the limit holds on the output. It also applies to estimated rows. With vertices about 1 m apart, 20° is a curve radius of about 2.9 m, so real curves within the 5–10 m radius assumption are kept.

Joins also no longer step backwards: when a row piece joins one that starts slightly behind its tip, the overlapping start is dropped.

### Rows at block edges

These rules are fixed, not settings:
- **Template lines outside the block.** A template line that falls outside the block passes its weight to its mirror line on the other side of the centre. With one side outside, the weights are 0.50 centre, 0.40 inner, 0.10 outer, so a template at the edge still checks its neighbouring row.
- **Head-on joins.** Two pieces of one row that meet head-on join if they're less than half a row width apart sideways, even when that's more than the usual join allows. `report.json` counts them per block (`head_on_joins`).
- **Row ends.** A row ends where its own direction meets the boundary, at any angle. Its last step doesn't turn, because the template there is half outside the block.

### Missing-row estimate

```toml
[missing_row_estimate]
enabled = true                  # on in centroid-pretest-cache-row-estimate.toml, off in the defaults
gap_tolerance_frac = 0.50       # neighbouring rows may differ from the median row width by +/- 50 % (drift)
whole_row_tolerance_frac = 0.20 # whole-number check: after filling, rows must be within the width +/- 20 %
station_factor = 5.0            # added rows follow their neighbours, with a vertex every 5 x row width
max_missing_rows = 3            # wider gaps are reported, not filled
```

It adds rows the image didn't show, so there's less to draw by hand:
1. **Row width.** The block's median row width comes from the measured distances between its detected rows. A distance that spans several row intervals is first divided by their number, so missing rows don't inflate the median.
2. **Gaps.** Neighbouring rows further apart than `width × (1 + gap_tolerance_frac)` get `round(gap / width) - 1` rows in between. Rows closer than that count as correctly spaced, even if they drift.
3. **Shape.** Each added row is interpolated between its two neighbours, with a vertex every `station_factor` × width, so it follows their curvature. It runs from boundary to boundary.
4. **Not filled.** A gap is reported in `issues.shp` (check `row_gap`) instead of filled in two cases:
   - it fails the whole-number check, i.e. the new spacing would be more than `whole_row_tolerance_frac` off the width. Rounding never leaves it more than 25% off, so 0.25 or more turns the check off;
   - it would need more than `max_missing_rows`.

**Squash.** Hanging pieces removed earlier often lie on an estimated row: the image showed the row, only broken. Each such piece is squashed into that row:
- **Which pieces:** at least `merge_min_inside_frac` (80%) of the piece must lie within `merge_buffer_frac` × the block's row width on each side (0.5), running within `merge_max_angle_deg` (10°). Half a row width catches drift but never reaches the neighbouring row.
- **Geometry:** the row then follows the detected piece wherever one covers it, and the estimate in between, with short ramps where it switches.
- **Turning it off:** set `merge_pieces = false`.

**Marking.** Added rows have `estimated = 1`. A row rebuilt from pieces gets `conf = merged`, and `det_pct` says how much of it came from the image. It's flagged for review only when less than half is detected, or when another check flags it. A purely interpolated row gets `conf = estimated` and `review = 1`. In the overlay, merged rows are blue and estimated rows cyan.

The estimate only works between complete rows, so it needs `[export] drop_hanging_rows = true`. TOML can't make one setting depend on another, so the code enforces it: if the estimate is on and hanging rows aren't being deleted, deletion is turned on and the run says so in the log and in `report.json` (`settings_notes`).

### Row numbering

```toml
[row_numbering]
enabled = true              # on in centroid-pretest-cache-row-estimate-numbering.toml, off in the defaults
transect_interval_m = 10.0  # a transect across the rows every 10 m, plus one through each row's middle
```

Each row gets a number, `row_no`, within its block:
- **How:** transects across the rows give, for each pair of neighbouring rows, the number of row intervals between them. The numbers follow from those relations, with the most common relation winning where transects disagree.
- **Missing rows:** a missing row leaves its number out, and `blocks.shp` reports `max_row_no` and `missing_no`.
- **Uncertain rows:** a row whose neighbours disagree on its number, such as one that drifts into the next lane, gets `row_no_ok = 0` and `review = 1`.
- **Direction:** numbers increase to the left when facing along the rows. That's northward for rows running east–west, and westward for rows running north–south.
- **Split rows:** two sections of one row in a concave block share a number.

Numbering works best on complete rows, so use it together with the estimate.

### Kernel cache

The 18,900 template kernels take about 20 s to build. They're saved to `.cache/kernels/` (git-ignored) and reused by later runs with the same template geometry, rotations, spacings, pixel size and coverage method. Any change to those builds a new set automatically. The cache is off by default; `[kernels] cache = true` turns it on, as `centroid-pretest-cache.toml` does.

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

[poly/cremons.gpkg](poly/cremons.gpkg) holds the Cremons block (id 1), in EPSG:6708. See "Cremons drone data" below.

### Cremons drone data (`dof/cremons/`)

A DJI Terra multispectral flight of 2025-08-22 over one vineyard block near Cormons (FVG, Italy), about 84 m × 111 m. Details and checks are in PROJECT.md §14.

- **Source.** `Cormons_20250822_MS/map/`: `result.tif` (RGB + alpha, 0.7 cm), `result_NIR.tif`, `result_Red.tif`, `result_Green.tif`, `result_RedEdge.tif` (float reflectance, 1.2 cm) and `index_map/NDVI.tif`, `GNDVI.tif` and others on the same 1.2 cm grid. All are georeferenced in WGS 84 (EPSG:4326) through their GeoTIFF tags, `.prj` and `.tfw`. Their centre is the one in `config.json`. Only the colour maps in `index_map_color/*_local.tif` have no CRS: they are colour renderings in a local metric frame, not measurements, so they are not used.
- **Prepared** with `src/prepare_inputs.py` (command in `configs/multispectral.toml`) onto one 0.20 m grid in RDN2008 / UTM zone 33N (EPSG:6708), the system of the FVG regional data. PROJ transforms WGS 84 to RDN2008 with the null transformation "RDN2008 to WGS 84 (1)", accuracy 1 m. `gndvi.tif` is prepared too, for runs that put it in the `ndvi` slot.
- **Block.** The user's polygon was drawn on the June flight's `NDVI_local.tif` and saved with an EPSG:4326 label over local metres. It was moved through that image's pixels and the June DJI georeferencing, then by the measured shift of the August mosaic against the June one (−0.97 m east, −1.09 m north). The August mosaic matches the user's GCP-corrected June mosaic (`*_modified.tif`) within 0.06 m.
- **Assumptions.** NDVI is (NIR − red) / (NIR + red) from these bands (checked: r = 0.999). The August flight records no GCPs, so its absolute position rests on the drone's GNSS; it agrees with the GCP-corrected June mosaic.

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
