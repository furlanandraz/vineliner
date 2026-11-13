# Vineliner — design brief for Claude Code

Date: 2026-11-06, revised 2026-11-07 and 2026-11-12 (see §17)  
Status: agreed direction for an initial experiment. A first implementation of the full pipeline exists (2026-11-07, §13, §18); its results are experimental. Since 2026-11-12 it is a QGIS plugin (Processing algorithm *Detect vine rows*, §13, §17 items 42–49); the command line is gone.  
Preferred language: Python.  
Target imagery: Slovenian GURS orthophotos. First test data on hand: a 0.20 m Bordeaux RGB + NIR pair in `dof/bordeaux/` (§14). Second: the Cremons multispectral drone flight in `dof/cremons/` (§14, §19).

## 1. Purpose and scope

Vineliner will analyse aerial raster imagery to identify vineyard areas, distinguish vine-row evidence from the background, and reconstruct individual vine rows as georeferenced vector polylines.

The intended end-to-end sequence is:

1. Read georeferenced RGB imagery, optionally supplemented by NIR.
2. Classify vineyard areas and produce separate vector vineyard polygons.
3. Produce an image-derived raster expressing evidence of vine rows inside those polygons.
4. Search for short, approximately parallel row segments using a weighted five-line template.
5. Combine local detections and reconstruct complete, potentially curved rows.
6. Validate geometry and export row vectors, scores, configuration and diagnostics.

This document records the discussion and its working decisions. It does not claim that the method has already been validated. Keep empirical defaults configurable. Do not replace the agreed exhaustive two-dimensional translation search with a one-dimensional phase search without an explicit later decision.

## 2. Working assumptions and accepted limitations

- The first target is row-trained vineyards, initially in Coastal Slovenia.
- Within a classified vineyard, the desired final classes are vine row and background. Background can include soil, grass and other non-vine features; binary output does not imply only two physical materials exist.
- RGB alone is an acceptable starting point. NIR is optional, and its contribution must be tested.
- The user considers July particularly favourable locally: developed vine canopy, dry/yellow inter-row grass and high sun. Prefer acquisition near solar noon when suitable imagery is available. This is a regional acquisition preference, not a claim that every July image is optimal.
- March and October imagery can be harder to interpret and are not the preferred initial cases.
- Intended product positional accuracy is approximately 1 m. This is a design target, not an empirically demonstrated accuracy rating.
- Canopy and shadow may both supply row evidence. Their positional displacement is an accepted limitation to disclose; dedicated shadow correction is not a requirement of the first experiment.
- Typical target row spacing is approximately 1.5–2.5 m; default expected spacing is 2.0 m. These are configurable scope assumptions, not universal viticultural rules.
- Rows within the same planting block should not cross.
- Rows may curve, including changing curvature while following terrain. The user's initial minimum curvature-radius assumption is approximately 5–10 m.
- Dense sampling is deliberately acceptable overhead. The initial priority is investigating behaviour and preserving evidence, rather than prematurely reducing the search.
- No UI, production infrastructure or deployment architecture has been specified.

## 3. Input and coordinate requirements

Use genuine georeferenced imagery, normally GeoTIFF. The Google Earth screenshot used during discussion was illustrative only and has no usable georeferencing supplied.

Read and retain:

- Coordinate reference system (CRS).
- Raster affine transform, pixel dimensions and spatial bounds.
- Band count, band descriptions/order, numeric type and value ranges.
- NoData/valid-data mask.
- Source product, tile identifier and capture date when supplied by the provider. A catalogue/update date is not necessarily the capture date.

Perform distances, rotations, spacing and growth in a projected coordinate system with metre units. For GURS D96/TM data, the expected CRS is EPSG:3794; inspect the actual file rather than silently assigning that CRS. If another CRS is supplied, transform correctly before metric operations. Assigning a CRS label is not reprojection.

Preserve the spatial transform when cropping or generating intermediate rasters. Any resampling must be explicit and recorded.

The analysis raster's pixel size (ground sampling distance) also sets the translation search increment (§7). The search does not test sub-pixel phases: the imagery holds no finer detail, and a template rasterised onto the pixel grid (§6.1) cannot represent them. Resampling to a finer grid adds no detail and must not be used to make the phase search finer.

The first version expects a north-up analysis raster: an affine transform with no rotation or shear terms. Reject other transforms, or resample explicitly to north-up and record it.

If RGB and NIR are combined, align them spatially and document the resampling and band mapping. GURS DOF050IR is a false-colour IR/R/G product, not an ordinary RGB file with a fourth NIR band. Do not infer band semantics solely from array position.

**Implemented** (2026-11-12, plugin): the plugin puts the layers on one north-up analysis grid itself (`vineliner/qgis_io/rasters.py`, GDAL warp). The CRS is the one given, else the first layer's if projected in metres, else the project's, else WGS 84 / UTM. The pixel is the one given, else the coarsest layer's own but at least 1/13 of the middle row width. The extent is the polygons' plus a margin. A layer already on that grid is read with nearest neighbour (no resampling); finer layers are averaged, others resampled bilinearly. The run report records each layer's source, native pixel and resampling. The core still refuses layers on different grids, non-north-up grids and non-square pixels. (Before the plugin, `src/prepare_inputs.py` did this as a separate step.)

## 4. Vineyard polygons and block separation

Vineyard classification is the first intended processing stage. Its result defines the permitted processing area and separates planting blocks that may have different row orientations.

The example screenshot contains four vineyards with two main orientation families, separated by roads. Do not impose one global orientation across all four blocks.

Apply each vineyard polygon to image scoring and row growth. Imagery outside the polygon must not contribute as support for that block. Final vector clipping provides the final boundary geometry.

For isolated row-extraction experiments, an externally supplied or manually outlined vineyard polygon can be used as a test fixture. That does not constitute implementation or validation of automatic vineyard classification. Retain the distinction in reports.

## 5. Raster preprocessing and row evidence

The initial baseline to investigate is RGB converted to grayscale, with contrast enhancement and/or thresholding. In favourable summer imagery, repeated darker canopy/shadow bands may already be sufficient to reveal rows.

Keep the method configurable. Candidate approaches discussed include:

- Grayscale with global or local contrast enhancement.
- Grayscale thresholding.
- RGB-derived vegetation/colour features.
- Edge or ridge filters and texture features.
- NIR-derived features or a custom RGB/NIR combination.
- Rotated matched filters or Gabor filters as comparisons to the explicit template.

No particular filter, threshold or RGB/NIR weighting has been selected. A vegetation index alone should not be assumed to distinguish grass from grapevine.

Prefer retaining a continuous row-evidence raster V in [0,1] for scoring and diagnostics. Binary thresholded imagery can be a configurable experiment. Define polarity explicitly: high V must mean stronger row evidence, even if the original rows are dark in grayscale.

**Implemented** (`tools/imagery/evidence.py`, `[evidence]`). V mixes one component per input layer in use (RGB, NIR, NDVI; each optional, revised 2026-11-12, §17). Each is a raw response scaled to [0,1]:

    V = sum_k w_k * V_k,  w_k = weight_k / sum of the weights of the layers in use

- **RGB component** (`rgb_weight`; gray = mean of the configured bands; rows darker than the ground).
  - `black_tophat`: closing(gray) − gray with a square window. Keeps narrow dark rows, removes broad soil-brightness changes. 0 maps to 0 and the `scale_percentile` maps to 1.
  - `gray_stretch`: inverted gray between two percentiles.
- **NIR component** (`nir_weight`; the band named `nir` of the NIR layer; canopy brighter than the ground) and **NDVI component** (`ndvi_weight`; vines are greener than soil). Both use:
  - `white_tophat`: image − opening(image). Narrow bright features.
  - `stretch`: the image between two percentiles.
- **Layers in use.** Since the plugin (§17 item 45), the components are the layers the user selects: RGB, NIR-R-G (NIR, and NDVI computed from it) and NDVI. The weights are rescaled over those present.
- **Combining** (`combine`, §17 item 48). By default each block uses the single component with the highest median template score at its centroid box and the 8 around it (`best_per_block`); `weighted` uses the weighted mean above.
- **Scaling region** (`scale_region`, open): `per_block` scales each block on its own statistics; `all_blocks` uses one scale for every block.
- Invalid pixels are replaced by the median valid value before filtering, so they cannot leak into neighbours, and V is 0 there.
- The defaults (`rgb_weight = 1`, `nir_weight = ndvi_weight = 0`) are the RGB-only baseline. NDVI's contribution on Bordeaux is tested in §18, the layers on Cremons in §19.

The downscaled fingerprint analogy is useful for recognising vineyard texture. Coarse-scale classification must still preserve the finer imagery needed for row placement. No fixed downsampling factor has been selected.

Height data remain a possible later input. If needed, investigate photogrammetric surface/terrain products from overlapping photographs and/or LiDAR. A height-above-ground cue requires an appropriate terrain reference; raw elevation alone includes the hillside. Neither LiDAR nor height processing is required for the initial RGB experiment.

## 6. Five-line template

Each template contains five short parallel line segments or bands:

| Relative position | Index | Weight |
| --- | ---: | ---: |
| Outer left | -2 | 0.05 |
| Inner left | -1 | 0.20 |
| Centre | 0 | 0.50 |
| Inner right | +1 | 0.20 |
| Outer right | +2 | 0.05 |

The weights sum to 1.00. The later mention of 0.10 for each outer line was clarified back to 0.05; do not accidentally create a total weight of 1.10.

- Expected row spacing d0 defaults to 2.0 m.
- Spacing d is centre-to-centre separation of neighbouring template lines, measured perpendicular to their direction.
- "Row width" means this spacing: the distance between neighbouring rows, where the tractors drive. A vine row itself has no width in this method because it is reduced to a line.
- Template line length L should be configurable. Discussed choices are 1.0 m, 2.0 m, or L equal to the expected row spacing, e.g. 2.2 m. No final choice has been made.
- Band width w is distinct from row spacing and line length. It is the sampling width of each template line's kernel (§6.1), not a physical row width. The appropriate width/soft profile remains an empirical parameter.
- At d = 2.2 m, the outer line centres are 8.8 m apart, even when L is only 1 m. Short length does not make the entire template footprint small.
- One spacing hypothesis uses a common d for all five lines. It does not independently move each neighbouring row. Centre-heavy weights make outer-row mismatches less influential.

For an explicit coordinate convention, use projected easting/northing and let theta be counterclockwise from the positive easting axis. Then u = (cos(theta), sin(theta)) is the along-row direction and n = (-sin(theta), cos(theta)) the across-row direction. Angles are axial: theta and theta + 180 degrees describe the same row orientation. The exact software convention must be recorded and used consistently.

For grid centre p and translation Delta, a point on template line k is:

    q(k,t) = p + Delta + k*d*n + t*u,  with -L/2 <= t <= L/2.

The image is sampled by rasterising the template onto the analysis raster grid, as specified in §6.1. Band width and the coverage method stay configurable and must be recorded.

### 6.1 Template rasterisation and overlap

The template is vector geometry and the row evidence V is a raster. Their overlap is computed by rasterising each template band onto the analysis raster's own pixel grid, then reading V through the resulting weight array (kernel). Do not overlay vector geometry on the raster once per hypothesis.

Zonal-statistics tools that overlay polygons on rasters do exist. For example, rasterstats computes zonal statistics and exactextract computes exact per-pixel polygon coverage. Calling them once per hypothesis, hundreds of thousands of times per search-box centre, is impractical. Use one of them only as an independent check of the kernel rasteriser in tests.

**Kernel construction**

- Each band k is a rectangle: the template line segment of length L, buffered by w/2 on both sides across the row, with flat ends. Rasterisation needs a non-zero band width w. The initial default is one pixel; w remains an empirical parameter (§12, §15).
- Kernels depend only on (theta, d, k, L, w), not on location. Build them once per run, on a local grid with the analysis raster's pixel size and axes and the template centre on a pixel centre, and cache them. The default search needs 180 × 21 × 5 = 18,900 small kernels.
- Kernel value K_k(x) in [0,1] is the fraction of pixel x covered by band k. Default: fractional coverage computed exactly by polygon–pixel clipping (Shapely; `coverage = "exact"`). s × s sub-pixel sampling (`"sampled"`, record s) is the alternative; tests check the two against each other and against a pixel-by-pixel Shapely clip. Binary rasterisation (pixel centre inside, or all-touched) is a configurable comparison only. At 0.20 m pixels and 0.02 m spacing steps, the outer bands move 0.04 m per spacing step. Binary masks of neighbouring spacing hypotheses are then often identical, so those hypotheses would not be distinct.
- Array row indices increase southward, so the northing components of u and n change sign in array coordinates. Implement the world↔array conversion once, in one place, and test it.
- Store kernel provenance (theta, d, L, w, coverage method, s) with the scores.

**Band means**

With c the search-box centre pixel, Δ a translation in whole pixels (§7), and M the validity mask (inside the block polygon and not NoData; §8):

    m_k(Δ)        = Σ_x K_k(x) · M(c+Δ+x) · V(c+Δ+x)  /  Σ_x K_k(x) · M(c+Δ+x)
    coverage_k(Δ) = Σ_x K_k(x) · M(c+Δ+x)  /  Σ_x K_k(x)

Because every translation is a whole-pixel shift, all translations reuse the same kernel. For one (theta, d, k), the full translation surface is the cross-correlation of M·V and of M with K_k over the search window. scipy.ndimage.correlate or an FFT correlation computes it in one call. Its results must equal the direct sums within floating-point tolerance, and it still evaluates every hypothesis (§7).

The denominator above renormalises over valid pixels. Whether to keep that, or keep the full kernel weight, is still the open border-case choice from §8. Store coverage_k either way.

## 7. Exhaustive local search

Keep three separate concepts explicit:

1. Search-box centre spacing: distance between coarse/local analysis locations across a vineyard.
2. Translation increment: distance between hypotheses inside each local search box.
3. Raster pixel size: physical footprint of source pixels.

The translation increment is derived from the raster pixel size: it equals the pixel size on each axis (2026-11-07 decision, §17). It is not a free user parameter. Search-box centre spacing stays independent of both.

### Search-box centres

Choose sufficient centres over each vineyard polygon. Surface area informs the count, but spacing also needs to support local curvature and reconstruction. The user deliberately favours close centres to reduce unsupported gaps. No numeric centre spacing has been fixed.

Place centres on pixel centres of the analysis raster, so that the template centre coincides with the kernel centre (§6.1).

### Translation/phase grid

- Search both horizontal axes, not only the across-row axis.
- The conceptual search box is d by d in metres. The original worked example used (0,0) through (2.2,2.2).
- Prefer a centred representation, approximately [-d/2,d/2) on both axes, with deterministic centre-outward traversal.
- The translation increment equals the analysis raster's pixel size on each axis, read from its affine transform. Examples: 0.20 m for the Bordeaux test data, 0.25 m for DOF025, 0.50 m for DOF050. This replaces the earlier 0.10 m default. Every translation is therefore a whole-pixel shift of the rasterised template (§6.1).
- Evaluate all translation hypotheses. Centre-outward traversal is an ordering preference, not permission to stop after the first good match.
- Record whether the square follows the tested d or stays fixed at nominal d0. Following each tested d is the natural interpretation of the row-spacing-sized box; fixed nominal size is useful for reproducing the original arithmetic. Do not silently switch conventions.
- Endpoints and rounding: with box side s (the tested d or nominal d0) and pixel size p on one axis, the offsets are i·p for every integer i with -s/2 <= i·p < s/2, that is i = ceil(-s/(2p)) … ceil(s/(2p)) - 1. Compute in pixel units with a small tolerance (1e-9) so floating-point error cannot add or drop an offset. Example: s = 2.0 m, p = 0.20 m gives i = -5 … 4, ten offsets from -1.00 m to +0.80 m. Half-open bounds define the finite search grid; do not assume a finite centre-weighted template is periodic in both world axes.

### Rotation

Search theta = 0, 1, ..., 179 degrees: 180 options at a 1-degree increment.

### Centroid pre-test (optional, 2026-11-08)

The user's proposal for a cheaper search is `[centroid_pretest]`, implemented in `tools/search/pretest.py`. It was first called "blueprint".
- **Centroid box.** If a block's centroid lies inside its polygon, one search box on the centroid pixel is searched in full: every translation, all rotations and all spacings.
- **Neighbour backup.** If the centroid box is weak (best S below accept_min, or its pixel is invalid), rings of neighbouring boxes are tested the same way, ring by ring, spaced like the grid centres (`neighbour_rings`, default 3). The first ring with an accepted box wins; within it, the highest score.
- **Pre-test.** The winning box gives the block's pre-test direction θ_pt and spacing d_pt. No assumption is made that the centre has the best data.
- **Grid search.** The grid search keeps every translation (phase) at every centre, but only rotations within `theta_window_deg` of θ_pt (default ±10°) and spacings within `spacing_window_frac` × d_pt (default ±10%).
- **Fallback.** The block gets the full grid search when the centroid is outside the polygon (the user's rule), or when no box within `neighbour_rings` is accepted.
- **Outputs.** Restricted searches report global rotation and spacing indices; unsearched ones are NaN in the profiles. A window-edge share is reported per block: the share of grid winners on the edge of the rotation window, which signals a pre-test that was off.

The full grid search remains the default (`enabled = false`) and the agreed baseline.

### Spacing variation

Spacing (row width, §6) hypotheses come from a predetermined, user-configured range: d_min, d_max and step, fixed before the run. The range is not adapted to the image during a run. Both endpoints are included:

    d = d_min + j * step,  j = 0, 1, ..., round((d_max - d_min) / step)

The range is set in the `[spacing]` table of the run configuration as `min_m`, `max_m` and `step_m`, normally in a per-area file merged over `configs/default.toml`. `vineliner.config.SpacingRange` builds the list. It refuses a range where d_max - d_min is not a whole number of steps (tolerance 1e-6 steps), so d_max is never silently dropped.

The default reproduces the earlier rule of d0 ±10% at 1% of d0. For d0 = 2.0 m that is 1.80, 1.82, …, 2.20 m: 21 options. For the earlier worked example d0 = 2.2 m it is 1.98 through 2.42 m at 0.022 m steps.

Set the range for the target vineyards. The Bordeaux test data appears to have rows about 1.3–1.5 m apart (§14), outside the 2.0 m default range. `configs/centroid-pretest.toml` therefore searches 1.20–1.60 m at 0.02 m, again 21 options. This search range is separate from the final geometric validation range (§11, §12).

### Search size and retained evidence

Per search-box centre, the hypothesis count is translations per spacing × 180 rotations × spacing options. Compute it from the actual configuration. With the default 21 spacing options:

| Pixel size | Translation box | Offsets per axis | Translations | Template scores per centre |
| --- | --- | ---: | ---: | ---: |
| 0.20 m (Bordeaux test data) | fixed 2.0 m | 10 | 100 | 378,000 |
| 0.20 m | fixed 2.2 m | 11 | 121 | 457,380 |
| 0.20 m | tracks tested d (1.80–2.20 m) | 9–11 | varies | 381,600 |
| 0.25 m (DOF025) | fixed 2.2 m | 9 | 81 | 306,180 |
| 0.50 m (DOF050) | fixed 2.2 m | 5 | 25 | 94,500 |
| 0.10 m (superseded default) | fixed 2.2 m | 22 | 484 | 1,829,520 |

The last row reproduces the original arithmetic for the earlier 0.10 m increment.

At one analysis location the score tensor has four search axes:

    S[delta_x, delta_y, orientation, spacing]

Adding the grid-centre index adds a further dimension. If phase counts vary by spacing, storage may be grouped per spacing or use explicit hypothesis records instead of a rectangular tensor.

The user wants all phase scores available. Preserve full local score surfaces, with configuration and coordinate mapping, rather than only the winning hypothesis. Storage format, chunking and whether every surface must remain permanently available are not yet selected. Vectorisation, batching, correlation or parallel computation are possible optimisations only if they preserve the requested hypothesis set and scoring semantics.

## 8. Scoring and acceptance threshold

Let m_k be the mean row evidence in template band k, computed through the rasterised kernel K_k (§6.1). For fully valid template support, the initial base score is:

    S = 0.50*m_0
      + 0.20*m_-1 + 0.20*m_+1
      + 0.05*m_-2 + 0.05*m_+2

Each m_k is in [0,1]. Means, rather than raw sums, make scores comparable across sample counts when coverage is otherwise comparable.

Accept a candidate when S >= 0.50; discard S < 0.50. The threshold is tentative and must be tuned from results.

The user explicitly accepts the following equal totals:

- Perfect central response, no neighbouring response: S = 0.50.
- No central response, perfect response in all four neighbouring bands: S = 0.50.

Do not require a separate central-response minimum in the first baseline. Neighbouring rows may supply the support for a faint central row. Retain individual m_k values to make the result inspectable.

Call this an overlap/matching score. It is not a calibrated probability of positional correctness.

A row-versus-inter-row contrast term was proposed to reject uniformly strong patches. It is an optional comparison, not an agreed subtraction from the base score. Subtracting a penalty changes the acceptance examples above; if explored, retain the base score and the additional term separately and record the decision.

Outside-polygon and NoData samples must not contribute external image support. Coverage normalisation and minimum valid coverage remain open border-case choices. Do not quietly reinterpret missing pixels as observed soil.

**Unavailable bands (decided 2026-11-08).** A band below the minimum valid coverage, for example one outside the block, is unavailable. Its weight moves to its mirror band on the other side of the centre (`tools/search/scoring.py`). This is fixed, not a setting. Weights are rescaled to sum to 1, which matters only when a band and its mirror are both unavailable. The score is undefined when the centre band is unavailable.

| Available bands | Weights (−2, −1, 0, +1, +2) |
| --- | --- |
| All | 0.05, 0.20, 0.50, 0.20, 0.05 |
| −2 outside | –, 0.20, 0.50, 0.20, 0.10 |
| −2 and −1 outside | –, –, 0.50, 0.40, 0.10 |
| Only the centre | –, –, 1.00, –, – |

The centre keeps its share, and the side still inside keeps the neighbour support that holds a template in its lane. Before, the remaining weights were rescaled in proportion, so the centre's share rose to 0.67 with one side outside.

## 9. Directional persistence across translations

The two-axis translation search is intentionally retained because along-row motion can test whether a good match persists over adjacent sections.

For each theta and d, inspect the translation surface S(delta_x, delta_y | theta,d):

- Correct alignment may produce an elongated region/ridge of good scores along the row direction.
- Across-row translation should eventually move the template off the rows.
- For a row direction of 24 degrees, the corresponding along-row translation also follows 24 degrees; strictly eastward translation includes an across-row component.
- A high isolated maximum and a spatially persistent high-score region convey different information.
- Adjacent phases reuse many pixels, so persistence is supporting consistency, not independent probability evidence.
- A uniformly high surface in all directions is ambiguous.

Keep local maxima, competing orientations and diagnostics. The exact persistence statistic, its weight and tie-breaking rule have not been fixed. Do not silently make persistence a new rejection threshold.

Implemented diagnostics (`tools/search/persistence.py`), stored per centre and never used to reject:

- Mean S along the row-direction line and along the across-row line through the winning translation, plus the box mean, minimum and maximum.
- The strongest competing orientation: another local maximum of the per-rotation best score that reaches accept_min at least 10° from the winner.

**Peak selection** (`[selection] method`, open, §15) decides which scored hypothesis a centre reports and seeds from. It does not remove hypotheses from the search.

- `best` reports the unconstrained winner.
- `orientation_consensus` reports the best hypothesis within a window of the local consensus direction. The consensus is the score-weighted axial mean of the unconstrained winners of accepted centres within a radius.
- The unconstrained winner is always reported too, as `raw_*` fields.

## 10. Row reconstruction

The user's proposed mechanism is to lock accepted local line samples, extend free ends in both directions in universal increments (initially 1 m), crop at intersections, mark connections and repeat.

Working refinements discussed for making that mechanism well defined:

1. Select useful accepted local hypotheses and consolidate duplicate samples representing the same physical row. Passing the score threshold does not make every phase/rotation a separate output row.
2. Establish seed segments and preserve their evidence/provenance. Neighbouring templates may repeatedly describe the same row.
3. Grow free endpoints in steps of up to 1 m.
4. Update direction using local detections/orientation and image support as the row bends. Short local straight samples do not justify extending the original tangent indefinitely. At radius 5 m, a 1 m arc changes tangent direction by about 11.5 degrees.
5. Treat intersection, overlap or nearby compatible endpoints as potential joins. Check tangent consistency and row identity before accepting the join. Slightly offset fragments need not geometrically intersect exactly.
6. Snap/trim accepted joins to shared geometry and merge compatible fragments.
7. Stop at the vineyard boundary. Define terminal outcomes for unsupported continuation, conflicts and exhausted limits, so growth cannot run forever waiting for every line to connect.
8. Clip and merge final geometry, split disconnected in-polygon sections, and validate.

The user considers denser analysis-centre placement the primary practical mitigation for connecting the wrong rows, and accepts the additional computation. Keep this parameter tunable.

Represent state per endpoint, for example active, joined, boundary, unsupported or conflict. A whole-line connected flag can be derived, but one flag cannot describe both ends during growth. Terminate when no active endpoints remain; legitimate boundary endpoints need not join another row.

Maximum unsupported gap, join/snap tolerances, curvature/turning limits, tie resolution, growth-score policy and duplicate suppression are not yet empirically selected. Do not force a connection merely to satisfy completion checks.

### First implementation (2026-11-07, `tools/reconstruction/`)

Every threshold named here is an `[reconstruction]` or `[seeds]` config key and is open (§15).

1. **Seeds.** An accepted centre yields one seed per configured template line k (`[seeds] lines`). The line's band must be available, and its own band mean m_k must reach `min_band_mean`.
2. **Chains (duplicate consolidation).** Seeds are linked when their directions differ by at most `link_angle_deg`, and they sit at most `link_lateral_frac` × d apart across the row and at most `link_along_m` apart along it. Each connected component becomes a chain. Seed midpoints are averaged in 1 m bins along the chain's mean direction, weighted by score. Optionally the across-row position is then smoothed over `chain_smoothing_m`.
3. **Growth.** In rounds, each active endpoint takes one `growth_step_m` step.
   - Candidate turns are every whole degree up to step / `min_curvature_radius_m` radians (11° at 1 m and 5 m).
   - Each candidate is scored with the same five-line template and kernels as the search.
   - The best score wins if it reaches `growth_min_score`. Otherwise the step goes straight on, unsupported.
4. **Joins.** A free end of another row that lies within `link_lateral_frac` × d to the side of the growth line, at most `join_distance_frac` × d beyond the step, and facing back within `join_angle_deg`, is joined, and the rows merge.
   - *Head-on joins (2026-11-08).* When a step conflicts with another row, that row may be the same row coming the other way, offset a little more than the join allows. It is joined when it has a free end facing back within `join_angle_deg`, less than 0.5 × d to the side, and at most two steps plus the join reach behind the tip. Beyond 0.5 × d lies the neighbouring row. Both ends move in a round, so they can pass each other before either one sees the other.
5. **Terminal states.**
   - *conflict:* the step crosses another row or lands on its line, and it is not a head-on join.
   - *boundary:* the step leaves the polygon, and is cut at the boundary. If the straight step leaves too, the last step goes straight, not turned: its template hangs over the boundary and can't steer. There is no angle rule, so a row may end at a long edge at any angle.
   - *unsupported:* more than `max_gap_m` of unsupported steps; the unsupported tail is removed.
   - *limit:* `max_row_length_m` grown.
   Growth always terminates.
6. **Sections.** Rows are clipped to the block polygon and split into connected sections. An endpoint within the validation tolerance of the boundary is *boundary*; any other endpoint keeps its growth state.

### Smoothing (2026-11-08)

`[smoothing]` (on by default, `max_turn_deg = 20`), implemented in `tools/reconstruction/smoothing.py`, caps the direction change between consecutive row vertices.
- **Rule.** While any interior vertex turns more than the limit, the sharpest one is removed. The ends are never removed, and a merged segment counts as bridged if either part was.
- **When.** It runs on the grown rows before clipping, and again, through `clean_line`, on the final clipped and snapped rows. The second pass also first drops vertices less than 1 cm from a neighbour, whose tiny segments have no meaningful direction. It applies to estimated rows too.
- **Join fix.** At the source of the worst kinks, a join now drops the start of the absorbed piece that lies behind the joining tip. The join accepts an end up to `join_distance_frac` × d behind, and connecting to it used to make the line step back and forward again.

### Missing-row estimate (optional, 2026-11-08)

`[missing_row_estimate]`, implemented in `tools/reconstruction/missing.py`, adds rows the image did not show. Its purpose is to reduce the manual drawing left over from detection errors that cannot be eliminated; it does not improve detection.
- **Row width d.** Per block: the median distance between neighbouring rows, measured along the normals every 1 m. Each distance is first divided by the number of row intervals it spans, judged with the search spacing. Without that, blocks that lost most rows to hanging-row deletion would report a width of about twice the real one.
- **Gaps.** Neighbours are the rows that normal rays hit most often on each side. A pair further apart than d × (1 + `gap_tolerance_frac`) gets n = round(gap / d) − 1 rows, when:
  - 1 ≤ n ≤ `max_missing_rows`, and
  - gap / (n + 1) lies within `whole_row_tolerance_frac` × d of d (the whole-number check).

  Defaults are 0.50, 0.20 and 3. Rows within ±50% of d count as correctly spaced, since rows drift.
- **Shape.** Each added row is interpolated between its neighbours at stations every 5 × d, so it follows their curvature. It is extended to the block boundary and clipped there, and must meet the boundary at exactly its two ends.
- **Not filled.** Other wide gaps are reported as `row_gap` issues.
- **Marking.** Added rows have `estimated = 1`, `conf = estimated` and `review = 1`.
- **Dependency.** The estimate needs complete rows. When it is on, `[export] drop_hanging_rows` is forced on in `RunSettings` (TOML cannot tie settings together), and a note is logged and stored in `report.json`.

### Squash (2026-11-08)

`tools/reconstruction/squash.py`, settings `merge_*` in `[missing_row_estimate]`. After the estimate, removed hanging pieces are assigned to estimated rows.
- **Assignment.** A piece goes to the estimated row whose buffer of ±0.5 × the block's row width holds at least 80% of the piece, running within 10°. If several qualify, the one holding the most wins.
- **Rebuild.** The row then follows each piece where the piece covers it (its extent projected onto the row), and keeps the estimate elsewhere. Overlaps go to the piece with the better template evidence.
- **Ramps.** Where the row switches between lines that are offset sideways, a straight ramp of length offset / tan(0.75 × the smoothing limit) keeps the switch within the turn limit.
- **Ends and checks.** The estimate's boundary ends are kept. The result must meet the boundary at exactly its two ends; otherwise the plain estimate stays.
- **Attributes.** `det_pct` gives the detected share. Merged rows have `conf = merged` and are reviewed only below 50% detected, or when another check flags them.

### Row numbering (2026-11-08)

`tools/reconstruction/numbering.py`, `[row_numbering]`.
- **Block direction.** θ_b is the length-weighted axial mean of the block's rows.
- **Transects.** Transects run along n_b through every row's midpoint, plus one every `transect_interval_m` along the block.
- **Relations.** On each transect, consecutive crossings give a relation no(b) − no(a) = max(1, round(gap · cos φ / d)), with φ the rows' local deviation from θ_b.
- **Solving.** The most common relation per pair wins. A maximum spanning tree weighted by transect support assigns the numbers.
- **Uncertain rows.** Rows in a relation the numbers break, or not linked to the main group, get `row_no_ok = 0` and `review = 1`.
- **Direction and gaps.** Numbers start at 1 and increase along n_b. Missing rows leave skipped numbers.

## 11. Geometry validation suite

Run these checks on final merged/clipped row sections, not individual short seed samples.

| Check | Failure condition |
| --- | --- |
| Self-intersection | A row section crosses itself. |
| Cross-row intersection | Distinct final rows cross or overlap. Shared vertices inside one merged row are normal. |
| Spacing | Local perpendicular separation of neighbouring rows is outside the configured allowable range. |
| Containment | A final row section extends outside its classified vineyard polygon. |
| Completion | Either endpoint of a final continuous row section is left hanging inside the polygon. |

The user's key completion rule is: every final continuous row section should meet the classified vineyard boundary at both ends. Express this as two endpoints on the boundary with the line interior contained in the polygon, using an explicit numerical tolerance. Counting any two arbitrary boundary intersections is insufficient.

For holes or concave polygons where a row exits and re-enters, split the result into connected in-polygon sections and apply the endpoint rule to each section.

Spacing checks must use local row normals and actual neighbour relationships, rather than horizontal/vertical coordinate differences. Boundary/headland situations and configured tolerances need explicit treatment.

A failed check flags an invalid/incomplete reconstruction for review. These are geometry-consistency checks, not proof of image correctness: missing entire rows can pass them, and an empty output is not a successful extraction.

**Reported, never enforced** (user decision, 2026-11-07). No row is rejected for failing a check. Each check is reported in two places:
- As attributes on the row: dangling ends, crossings, the share of spacing samples outside the range, the measured spacing, and length outside the block.
- As a located point in an issues layer.

Measurements of support are added alongside:
- bridged length;
- template score along the line, sampled every 1 m;
- RMS distance of the seed detections from the line.

A `review` flag and a `conf` class (high / medium / low) then mark the rows a person should check. The `[review]` thresholds set the flag. They only label rows, so being too strict would bury the uncertain rows among false alarms. The validation spacing range for Bordeaux, 1.2–1.6 m, is confirmed by the user.

Also compare representative results against manually inspected or annotated imagery. Useful empirical measures include missed/extra rows, positional deviation, fragmentation, incorrect joins, endpoint failures and execution/storage cost. The ~1 m accuracy target needs a defined evaluation metric before being published as a rating.

## 12. Configuration summary

| Parameter | Working value or status |
| --- | --- |
| Language | Python |
| Image source | GURS orthophotos; RGB first, NIR optional. First test data: Bordeaux 0.20 m (§14) |
| Preferred season | July/dry summer in Coastal Slovenia; near solar noon preferred |
| Target positional accuracy | Approximately 1 m; unvalidated |
| Expected row spacing | 2.0 m default |
| Typical allowable spacing range | 1.5–2.5 m; configurable for the target vineyard |
| Spacing (row width) search | Predetermined range d_min–d_max with step, inclusive, set in `[spacing]` of the run config. Default 1.80–2.20 m at 0.02 m (d0 ±10% at 1%): 21 options. Bordeaux: 1.20–1.60 m at 0.02 m |
| Line count | 5 |
| Weights, left to right | 0.05, 0.20, 0.50, 0.20, 0.05 |
| Line length | Configurable; 1 m, 2 m or nominal spacing discussed |
| Band width/profile | Kernel sampling width, not a row width. Must be > 0 for rasterisation; initial default one pixel; profile to determine |
| NoData / outliers | Per area. Bordeaux: 255, masked per pixel when any band of any input file is 255 (§14) |
| Template sampling | Rasterised kernels on the analysis pixel grid; fractional coverage by default (§6.1) |
| Translation search | Full two-dimensional square, centred, exhaustive |
| Translation increment | Analysis raster pixel size per axis; derived, not user-set (0.20 m on the test data) |
| Traversal order | Centre outward; still evaluate every phase |
| Rotation search | 0–179 degrees, 1-degree increments |
| Acceptance | Base overlap score >= 0.50 |
| Row evidence | One component per input layer in use (RGB gray, NIR, NDVI), relative weights; top-hat or stretch each; scaling region open (§5) |
| Peak selection | Open: unconstrained best or orientation consensus (§9) |
| Seeds | Open: which template lines seed, and the minimum own-line evidence (§10) |
| Search-centre spacing | To determine; favour sufficiently dense placement; centres on pixel centres |
| Growth increment | Up to 1 m initially |
| Minimum curvature radius | Approximate 5–10 m scope assumption |
| Export | `shapefiles/`: rows with review attributes, blocks with row-end vertices, issues, bridges; GeoPackage copy with diagnostics (§13) |
| Checks | Reported as attributes and issue points, never enforced; `[review]` thresholds set the review flag (§11) |
| Hanging rows | Kept by default; `[export] drop_hanging_rows = true` keeps only rows meeting the boundary exactly at their two ends (§17 item 22) |

Do not confuse spacing-search tolerance around d0 with the final geometric validation range; record both separately.

## 13. Outputs and implementation organisation

Suggested outputs for an initial experiment:

- Classified vineyard polygon layer.
- Row-evidence raster with CRS/transform.
- Accepted/deduplicated seed layer and rejected-candidate diagnostics.
- Full local phase score surfaces or equivalent complete hypothesis records.
- Final row polylines and endpoint states.
- Validation failures located on a map, plus a machine-readable report.
- Exact run configuration and source/crop metadata.
- A visual overlay of imagery, vineyard boundary, seeds and final rows.

The user mentioned a shapefile attribute table. Support a shapefile export with its accompanying files. A GeoPackage working/export copy is a suggested convenience, not an imposed replacement. Design attribute names with shapefile limitations in mind; preserve more detailed per-run/per-hypothesis data outside the shapefile if necessary.

Python ecosystem candidates include Rasterio/GDAL for rasters, NumPy/SciPy for scoring, scikit-image/OpenCV for image processing, Shapely/GeoPandas for geometry and PyProj for coordinate transformations. `pyproject.toml` declares NumPy, SciPy, Shapely, Rasterio, GeoPandas, pyogrio and PyProj. Only the file adapters (`tools/io/`) use Rasterio, GeoPandas and pyogrio; the algorithm needs only NumPy, SciPy and Shapely. scikit-image/OpenCV remain candidates for later. No package versions have been pinned.

Keep pipeline stages separable so filters and reconstruction choices can be compared without rewriting coordinate handling. Save useful intermediate results because exhaustive search is expensive. Optimisations should preserve results and the explicitly requested full translation grid.

### Repository layout

Revised 2026-11-12 (§17, items 42–48): the repository is a QGIS plugin. The command line, the per-area config files and the rasterio/GeoPandas file adapters are gone. `vineliner/` is the plugin folder QGIS loads, following the PyQGIS Developer Cookbook: `__init__.py` with `classFactory`, `metadata.txt`, a main plugin class that registers a Processing provider, and the provider with its algorithm. The method is `vineliner/core/` (the former `tools/`, moved unchanged where it could be): NumPy, SciPy and Shapely only, relative imports, no `qgis` or `osgeo` imports (a test checks this). `vineliner/qgis_io/` is the only bridge: it reads QGIS layers into arrays and polygons and writes results back.

    vineliner/                        the repository
    ├── PROJECT.md, README.md
    ├── pyproject.toml                development environment for the core's tests (QGIS ignores it)
    ├── vineliner/                    the plugin folder
    │   ├── __init__.py               classFactory(iface)
    │   ├── metadata.txt, icon.png
    │   ├── plugin.py                 VinelinerPlugin: dependency check, provider, Raster menu entry, toolbar icon
    │   ├── processing_provider/      provider.py (VinelinerProvider), detect_rows.py (DetectRowsAlgorithm)
    │   ├── qgis_io/                  rasters.py (analysis grid, GDAL warp), vectors.py (polygons <-> Shapely)
    │   └── core/
    │       ├── defaults.toml         every setting: the final Bordeaux configuration (§18)
    │       ├── config/               loader.py (defaults + dialog), spacing.py, settings.py (§12)
    │       ├── geo/                  grid.py (world <-> array, once), angles.py (§3, §6)
    │       ├── imagery/              masks.py, evidence.py (V in [0,1], §5)
    │       ├── vineyards/            blocks.py (polygons, masks), classify.py (not implemented) (§4)
    │       ├── template/             geometry.py (§6), kernels.py (§6.1), cache.py
    │       ├── search/               hypotheses.py, scoring.py, local.py, selection.py, persistence.py, pretest.py (§7–9)
    │       ├── reconstruction/       seeds, chains, growth, rows, smoothing, missing, squash, numbering (§10)
    │       ├── validation/           checks.py, review.py (§11)
    │       ├── export/               summary.py (needs_review, block statistics), polygons.py, report.py (§13)
    │       └── pipeline/             inputs.py (layer rules, valid mask), run.py (the run, in memory)
    ├── scripts/                      package_plugin.py (installable zip), link_plugin.py (development link)
    ├── tests/                        mirrors core/; tests/qgis/ runs the algorithm inside QGIS (unittest)
    ├── dof/                          local test imagery, git-ignored (§14)
    └── poly/                         local vineyard polygons, git-ignored: poly.gpkg (Bordeaux blocks), cremons.gpkg (Cremons block)

**Search implementation.**
- For each spacing, every rotation, band and translation is packed into one sparse matrix over a fixed pixel window.
- A batch of centres is then scored with one sparse-dense product per spacing. Tests check the result against the direct sums of §6.1.
- On the Bordeaux settings this takes about 40 ms per centre with 3-px bands. That's 185,220 hypotheses per centre.

**Score storage** (`[storage] surfaces`):
- `all` keeps every surface. That would be about 5.9 GB per Bordeaux run before compression (7,947 centres × 185,220 float32 scores).
- `winning_spacing` keeps every translation × rotation surface of each centre's winning spacing.
- `none` keeps nothing.
- Per-centre best scores per rotation and per spacing are always saved (`centre_profiles.npz`).

## 14. GURS imagery: acquisition and crop workflow

Primary entry point: [Javni geodetski podatki (JGP)](https://ipi.eprostor.gov.si/jgp/data).

GURS products relevant here are RGB DOF025 at 0.25 m/pixel, RGB DOF050 at 0.50 m/pixel, and false-colour DOF050IR at 0.50 m/pixel. Prefer DOF025 where obtainable; DOF050 is a useful readily listed baseline. At 2 m row spacing these correspond to approximately 8 and 4 pixels per spacing respectively. These pixel counts are not extraction-accuracy guarantees.

### Verified access distinction, 2026-11-06

The live JGP catalogue was inspected. The observed route is:

    VSTOPI V APLIKACIJO
    -> DRŽAVNI TOPOGRAFSKI SISTEM
    -> DALJINSKO ZAZNAVANJE
    -> Ortofoto
    -> Prenos podatkov

It lists **List ortofoto DOF050 — Oktober 2025 / GeoTiff**. Its external-link control opens [the JGP map](https://ipi.eprostor.gov.si/jv/?map=jgp). The map imagery did not render in this inspection, so no specific vineyard tile or completed tile download was verified. The catalogue date must not be assumed to be the photography date.

DOF025 and infrared products are documented by GURS, but were not listed as downloadable choices in the inspected JGP panel. GURS's access page directs users to request these products at gic.gu@gov.si, identifying the required area. Follow that route for DOF025 instead of assuming that it appears in the current direct-download catalogue. No request/email has been sent on the user's behalf.

GURS states that JGP orthophoto downloads embed geolocation in the GeoTIFF header. Other distribution routes may use TIFF plus world-file geolocation; preserve any sidecars and verify CRS/transform if using those files. GURS identifies D96/TM as the native orthophoto coordinate system; its EPSG code is 3794.

### Obtaining and preparing a test area

1. Identify a vineyard the user recognises in the selected coastal area. The screenshot refers to Škocjan; its exact coordinates were not supplied, so no tile ID or geographic footprint has been inferred from it.
2. Use the GURS product's tile selector/map to obtain the relevant native-resolution orthophoto sheet. If DOF025 is needed, use the documented GURS request route. The exact tile selection was not completed in this session.
3. Open the downloaded raster in QGIS and inspect its CRS, extent, bands and pixel size before cropping.
4. Use **Raster > Extraction > Clip Raster by Extent** and **Draw on canvas** to select a rectangle containing the complete vineyard and some surrounding context. Save a permanent `.tif` output. Keep the source CRS and native resolution; do not use a screenshot or rendered map image as the source.
5. If clipping to an existing polygon, use **Clip Raster by Mask Layer** and enable **Keep resolution of input raster**, preserving the appropriate target CRS. Keep the polygon separately for algorithm testing.
6. Check the crop's CRS and transform after export, and confirm its pixel size matches the source. Cropping changes the raster origin/extent while retaining georeferencing.
7. Retain one crop with straight rows, one with curved rows and one with neighbouring blocks of different orientations if available. Include imagery provenance and any manually annotated reference rows.

Do not use the convenient outer rectangle of a crop as a substitute for the actual classified vineyard boundary in the completion tests.

### Test imagery: `dof/bordeaux/` (local, git-ignored)

The first test data is not GURS imagery. Its file headers and pixel statistics were inspected on 2026-11-07, without GDAL. Re-check these facts with Rasterio/GDAL once the dependencies are installed.

| Property | `rgb.tif` | `nir.tif` |
| --- | --- | --- |
| Size | 2000 × 2000 px, 3 bands, uint8, uncompressed, pixel-interleaved, no overviews | same |
| Pixel size | 0.20 × 0.20 m | same |
| Upper-left corner | 449350.5, 6426545.4 | same grid, so no resampling is needed to combine them |
| Extent | 400 m × 400 m | same |
| CRS | Lambert Conformal Conic parameters matching Lambert-93, with citation "EPSG:2154". Stored as a user-defined projected CRS with an incomplete datum/ellipsoid definition, not as the formal EPSG code | same |
| NoData | 255 in every band | same |
| Band descriptions | none recorded | none recorded |

Observations and cautions:

- The scene contains several vineyard blocks with different row orientations, separated by roads, plus a bare parcel and buildings. That suits the block-separation and orientation tests (§4).
- Rough row-spacing estimate: about 1.3–1.5 m, roughly 7 pixels, from autocorrelation on five crops. This is unvalidated, and it lies outside the 1.80–2.20 m default spacing range and below the 1.5–2.5 m Slovenian scope assumption. `configs/centroid-pretest.toml` sets the search range to 1.20–1.60 m and the validation range to the same values as a starting point.
- **Band order assumption (2026-11-07):** `nir.tif` is a false-colour product with bands NIR, red, green; `rgb.tif` is red, green, blue. The files do not record this. Evidence: `nir.tif` bands 2 and 3 closely track the RGB red and green bands but are not identical (mean absolute difference ≈ 2.3 DN in a sample). The order is set explicitly in `configs/centroid-pretest.toml` ([input] `rgb_bands`, `nir_bands`), not inferred from array position (§3). It is recorded as an assumption in README.md.
- **NoData and outliers (2026-11-07):** 255 is treated as "no data or outlier". Vegetation never reflects that strongly in any band, so 255 values are water glint, white roofs and similar, never row evidence. A pixel is masked when any band of either file equals 255 (`nodata_rule = "any_band"`). Counts on the full scene:

  | Band | Pixels at 255 |
  | --- | ---: |
  | `rgb.tif` red / green / blue | 390 / 253 / 168 |
  | `nir.tif` NIR / red / green | 19 / 603 / 623 |
  | Any band of `rgb.tif` | 513 |
  | Any band of `nir.tif` | 821 |
  | **Any band of either file (masked)** | **955 (0.024%)** |

  All 19 pixels with NIR = 255 are near-white in RGB (mean 249, 252, 242); none are vegetation-like. 50 RGB pixels are 255 in all three bands. The scene has no large NoData region, so in practice the mask removes saturated outliers.
- Source product, tile identifier and capture date are not recorded in the files. Record them from the provider before reporting results.
- **`ndvi.tif` (added by the user, 2026-11-07).**
  - Single band, float32, NoData −3.4e38 (about 0.01% of pixels are NoData or non-finite). Same grid and CRS as `rgb.tif`.
  - It equals (NIR − red) / (NIR + red), with NIR from `nir.tif` band 1 and red from `rgb.tif` band 1 (correlation 1.0). That confirms the band-order assumption above.
  - Inside the blocks NDVI is low (median −0.08, 99th percentile 0.18): mostly bare soil with vine rows.
  - It separates vegetation from soil clearly at scene scale, for example the grass verges. At full resolution it is smoother than the RGB: its pixel-to-pixel gradient relative to its contrast is half that of the red band. Rows show only faintly in some blocks.
- **Blocks.** `poly/poly.gpkg` (supplied 2026-11-07) holds 7 hand-drawn blocks, 0.2–0.8 ha each and 3.2 ha in total, in the raster's own CRS. Its `id` field is empty, so blocks are numbered 0–6 in file order.

### Drone data: `dof/cremons/` (local, git-ignored; 2026-11-12)

A DJI Terra 4.4 multispectral flight of 2025-08-22, `Cormons_20250822_MS`, supplied by the user as "Cremons". It covers one row-trained block in FVG, Italy, of about 84 m × 111 m. The provider was said to have no original RGB and NIR channels, but the delivered `map/` folder holds them. They were inspected with rasterio on 2026-11-12:

| File | Content | Grid |
| --- | --- | --- |
| `result.tif` | RGB + alpha, uint8, NoData 0 | 11647 × 15467 px, 0.7 cm |
| `result_NIR.tif`, `_Red`, `_Green`, `_RedEdge` | float32 reflectance, NoData NaN | 6862 × 9112 px, 1.2 cm |
| `index_map/NDVI.tif`, `GNDVI.tif`, `NDRE`, `LCI`, `OSAVI` | float32 index, NoData 0 | same 1.2 cm grid |
| `index_map_color/*_local.tif` | colour rendering (RGBA), no CRS, local metric frame | 1.2 cm |
| `dsm.tif`, `segment.tif` | surface model; DJI semantic classes (tree, building, ...) | not used |

- **Georeferencing.** Every analysis raster is in WGS 84 (EPSG:4326), in its GeoTIFF tags, its `.prj` and its `.tfw`. The DJI log records "Output geo descriptor EPSG:4326" and "GCP is empty". The raster centre (13.460973, 45.971100) equals `config.json`'s centre. Only the `_local` colour maps lack a CRS; they are renderings, not measurements, and are not used.
- **Target CRS.** RDN2008 / UTM zone 33N (EPSG:6708, N-E axis order, coordinates as EPSG:7792), the system declared by the FVG regional datasets on the INSPIRE geoportal. EPSG:6708 round-trips through GeoTIFF, GeoPackage and shapefile with x as easting. PROJ uses the null transformation "RDN2008 to WGS 84 (1)" (accuracy 1 m).
- **Analysis grid: 0.20 m.** At 1.2 cm, a 2.6 m translation box would be 26² = 676 phases with kernels 2.7× larger, about 1.6 billion sparse weights. At 0.20 m it is 169 phases, about 150 million weights (1.2 GB). Three pixels of band width are then 0.6 m, the canopy width seen in the NIR. All layers are averaged onto one 431 × 566 px grid (`src/prepare_inputs.py`, command in `configs/multispectral.toml`).
- **Band check.** NDVI = (NIR − red) / (NIR + red) from `result_NIR` and `result_Red`: r = 0.9992, mean absolute difference 0.002. The bands are what their names say.
- **Rows.** 15 rows at about 115° (24° west of north), 2.57–2.64 m apart (across-row NIR profiles in 15 × 15 m patches), with grassed inter-rows. The rightmost row runs partly under a tree verge.
- **Profiles across the rows.** NIR has one sharp canopy peak about 0.6 m wide. NDVI and GNDVI peak on the canopy, but the inter-row grass is also high (NDVI 0.74 against 0.89). RGB gray is darker on the canopy than on the grass (as on Bordeaux); blue has the strongest RGB contrast.
- **Row contrast by layer** (folded profiles in 18 patches of 12 × 12 m; (canopy mean − mean ≥ 0.7 m away) / std, median): NIR 1.9, GNDVI 1.8, NDVI 1.7, RGB gray 0.8, blue alone 1.1. Top-hat windows of 1.4–2.0 m and stretches score alike; at 0.6 m all are worse.
- **Block.** The user drew the block (`Cormons_20250606/map/index_map/polygon.gpkg`) on the June flight's `NDVI_local.tif`. Its file says EPSG:4326, but its coordinates are local metres (−11..66, −78..13). It was moved in three steps, recorded in `poly/cremons.gpkg` (id 1, 3365 m²):
  1. Local metres to June image pixels (`NDVI_local.tif` transform), then to lon/lat with the June DJI georeferencing (`NDVI.tif`), then to EPSG:6708.
  2. The August mosaic's offset from June, measured by phase correlation of the two NDVI mosaics on the 0.20 m grid: −0.97 m east, −1.09 m north.
  3. Cross-check: against the user's GCP-corrected June mosaic (`NDVI_modified.tif`, QGIS Georeferencer, 15 GCPs) the August offset is 0.01 m / 0.06 m. So the August mosaic is aligned with the corrected June one without GCPs of its own. An affine fit to the GCPs (RMS 0.26 m) moves the vertices up to 0.75 m differently from the measured shift; the shift was used, as it is measured against the August imagery.
- **The block's edges.** It runs about 0.5 m beyond the row tops and about 2 m beyond the row ends at the bottom, takes in the verge row on the right, passes a little outside the August data on the left, and includes a parked car in the bottom-right corner (380800–380803, 5091951–5091956). Rows are bridged to the boundary across the headland strips.
- Capture date 2025-08-22 (folder name and log). Absolute position rests on the drone's GNSS; the mosaic agrees with the GCP-corrected June mosaic.

### Sentinel suitability

Sentinel-2 is georeferenced, but its highest-resolution RGB and broad NIR bands have 10 m pixels. That is inadequate for directly resolving rows separated by approximately 2 m. Resampling to a finer output grid cannot recover those row details. Sentinel may be useful for larger-scale context or parcel-level monitoring; it is not the initial row-extraction test source.

## 15. Unresolved empirical choices

The following remain deliberately open rather than being silently settled by an implementer:

- Automatic vineyard classification method and separation of adjoining planting blocks.
- Grayscale/colour/NIR evidence function, contrast settings and any threshold.
- Band width/profile and line length.
- Kernel coverage method (fractional by default, binary as a comparison) and, if sampled, the sub-pixel sample count s.
- Search-box centre spacing and its possible adaptation to curvature/area.
- Whether the translation square tracks each tested d or stays at nominal d0. The increment and rounding are now fixed by §7.
- Spacing search range for each test area. Bordeaux starts at 1.20–1.60 m (§14); confirm or narrow it from results.
- Provenance (product, tile, capture date) of the Bordeaux test imagery. Its band order and 255 masking are working assumptions (§14).
- Partially supported lines and polygon-border normalisation.
- Exact directional-persistence statistic, tie-breaking and peak selection.
- Duplicate detection and whether neighbouring template lines become separate seeds.
- Join rules, turning limits, unsupported-gap limit, growth thresholds and stopping safeguards.
- Geometric tolerances, boundary/headland spacing policy and evaluation metrics.
- Full-score storage format and practical runtime/memory limits.

Use small empirical comparisons to choose these. Preserve the user-approved baseline and document changes to scoring or hypothesis coverage.

Every open choice is an unset key in `configs/default.toml`, and a run refuses to start without it. `configs/centroid-pretest.toml` sets first-run PICKs for all of them, each with its reason. They are listed in §18, and every run's `report.json` repeats them under `open_choices_set_by_this_run`. Two further open choices arose during implementation:

- Peak selection (`[selection]`, §9).
- Seed filtering and chain smoothing (`[seeds] min_band_mean`, `[reconstruction] chain_smoothing_m`, §10).

## 16. Sources and verification notes

The core algorithm and parameter choices above come from the design discussion, not from copying an existing published algorithm. The following primary sources ground imagery facts and provide research comparisons. The only georeferenced test input so far is the Bordeaux pair in `dof/bordeaux/`. Its headers and basic statistics have been inspected (§14), but no row extraction has been run on it.

1. [GURS — Daljinsko zaznavanje / orthophoto specifications and embedded geolocation](https://www.e-prostor.gov.si/podrocja/drzavni-topografski-sistem/daljinsko-zaznavanje/).
2. [GURS — Orthophoto FAQ: JGP GeoTIFF and D96/TM](https://www.e-prostor.gov.si/podrocja/drzavni-topografski-sistem/drugo/razno/pogosta-vprasanja/).
3. [GURS — Access to geodetic data, including DOF025/infrared request route](https://www.e-prostor.gov.si/en/access-to-geodetic-data/).
4. [GURS — Horizontal coordinate system and EPSG:3794](https://www.e-prostor.gov.si/podrocja/drzavni-koordinatni-sistem/horizontalna-sestavina/).
5. [QGIS 3.44 — Clip raster by extent and mask layer](https://doc.qgis.org/3.44/en/docs/user_manual/processing_algs/gdal/rasterextraction.html).
6. [Copernicus SentiWiki — Sentinel-2 MSI spatial resolutions](https://sentiwiki.copernicus.eu/web/s2-mission).
7. [Comba et al. (2015), Vineyard detection from unmanned aerial systems images](https://doi.org/10.1016/j.compag.2015.03.011): grayscale aerial-image row extraction as a comparison.
8. [Delenne et al. (2008), A non-supervised approach using Gabor filters for vine-plot detection in aerial images](https://www.sciencedirect.com/science/article/abs/pii/S0168169907002487): periodic-pattern detection, orientation and spacing.
9. [Padua et al. (2024), Versatile method for grapevine row detection in challenging vineyard terrains using aerial imagery](https://doi.org/10.1016/j.compag.2024.109372): irregular/curved rows with image and elevation information.
10. [Poblete-Echeverria et al. (2017), Detection and Segmentation of Vine Canopy in Ultra-High Spatial Resolution RGB Imagery](https://www.mdpi.com/2072-4292/9/3/268): canopy segmentation and spectral/background ambiguity.
11. [Padua et al. (2018), Multi-Temporal Vineyard Monitoring through UAV-Based RGB Imagery](https://www.mdpi.com/2072-4292/10/12/1907): seasonal canopy/inter-row changes.

12. [rasterstats — zonal statistics of vector geometries over rasters](https://github.com/perrygeo/python-rasterstats) and [exactextract — exact per-pixel polygon coverage](https://github.com/isciences/exactextract): candidate independent checks for the kernel rasteriser (§6.1), not the search's scoring path.

Access routes were checked on 2026-11-06. Recheck current product availability and metadata when obtaining actual test tiles.

## 17. Decision log

### 2026-11-07

1. **Translation increment from pixel size.** The translation increment equals the analysis raster's pixel size on each axis, replacing the 0.10 m default. There is no reason to test 0.10 m phases on 0.20 m imagery. This reduces hypothesis coverage, for example from 484 to 100–121 translations per (theta, d) on 0.20 m data (§7). The exhaustive two-dimensional translation search is kept.
2. **Predetermined spacing range.** Spacing hypotheses come from a user-configured range fixed before the run (d_min, d_max, step). The default is unchanged: 21 options, d0 ±10% at 1%.
3. **Rasterised template.** Template overlap is computed through kernels rasterised on the analysis pixel grid, with fractional coverage by default (§6.1). Translations are whole-pixel shifts of those kernels.
4. **Repository set-up.** `src/` layout, `pyproject.toml` with setuptools, pytest, one module per pipeline stage (§13). Imagery (`*.tif`, `*.tiff`) is git-ignored; `dof/.gitkeep` keeps the input folder.
5. **Row width means row spacing.** "Row width" is the distance between neighbouring rows, where the tractors drive. Rows are reduced to lines and have no width of their own; the template band width w is only a kernel sampling width (§6).
6. **Spacing range in the config.** Users set the searched spacing range in `[spacing]` (`min_m`, `max_m`, `step_m`) of a per-area config merged over `configs/default.toml`. `vineliner.config` loads, merges and validates it (§7). The Bordeaux range is 1.20–1.60 m at 0.02 m.
7. **Bordeaux band order assumed.** `nir.tif` is NIR, red, green and `rgb.tif` is red, green, blue, set in `configs/centroid-pretest.toml` and recorded in README.md (§14).
8. **255 is NoData or outlier on the Bordeaux pair.** Vegetation never reaches 255, so 255 values are water, white roofs and similar. A pixel is masked when any band of either file is 255: 955 pixels, 0.024% (§14).
9. **Layout: entry files in `src/`, importable modules in `tools/`.** Supersedes the `src/vineliner/` layout of item 4. Subpackages are nested by stage; internal imports are relative, so `tools/` can move into a future QGIS plugin; file formats are confined to `tools/io/` and `tools/export/` (§13).
10. **Block polygons from `poly/poly.gpkg`.** The user supplied 7 hand-drawn Bordeaux blocks in the raster's own CRS. Automatic classification stays unimplemented; reports call the polygons a fixture (§4).
11. **Exact fractional coverage by default.** Kernels use exact polygon–pixel clipping. This removes the open sub-pixel sample count s from the default path (§6.1).
12. **Peak selection becomes an explicit option.** `[selection] method` = `best` or `orientation_consensus` (§9). It was added because a 2 m template separates neighbouring rotations weakly: in block 0, 17% of raw winners were off the 40° rows, scoring barely above the 40° hypothesis at the same centre. All hypotheses are still scored and the raw winner is still reported.
13. **Seed filter and chain smoothing.** Two new open keys: `[seeds] min_band_mean`, a line's own evidence needed to seed, and `[reconstruction] chain_smoothing_m` (§10).
14. **Score storage modes.** `all`, `winning_spacing` or `none` (§13). Bordeaux uses `winning_spacing`, because `all` would be about 5.9 GB per run.
15. **First-run PICKs for Bordeaux** are recorded in `configs/centroid-pretest.toml` and §18. They include two changes found by trial on block 0, recorded there:
    - Translation box `nominal_d0` instead of `tested_d`. With `tested_d`, d = 1.60 m gets 64 translations against 49 for the other spacings, and the larger max pulled 10% of centres onto that range edge.
    - Band width 3 px instead of 1. A 1 px band scores flat across a 3–4 px row stripe, so neighbouring centres placed one row up to 0.6 m apart.
16. **Evidence scaled per block** (`[evidence] scale_region`, open; Bordeaux PICK `per_block`). Scaling over all blocks let the high-contrast block 1 set the scale: block 0 accepted 234 centres in the full run but 545 when run alone. Per-block scaling makes a block's result independent of the other blocks in the run.
17. **NDVI as an optional evidence component** (`[input] ndvi`, `[evidence] ndvi_weight`, `ndvi_method`). Added at the user's request to augment the RGB evidence. The default weight 0 keeps the RGB-only baseline; the comparison is in §18.
18. **Checks are reported, not enforced** (user). Every row is kept. Per-row attributes, a `review` flag and a `conf` class expose uncertainty for later human correction (§11). The thresholds are in `[review]`.
19. **Bridge gaps** (user). In faint patches, growth continues straight along the row's direction over the last `bridge_tangent_m` (Bordeaux 10 m), for up to `max_gap_m` (Bordeaux 50 m). Every segment records whether it had image support. Bridged parts are reported per row (`bridged_m`, `bridge_pct`) and exported as `bridges.shp`.
20. **Polygon output with row vertices** (user). `blocks.shp` keeps each supplied polygon's shape and attributes and inserts a vertex wherever a row meets its boundary. Row ends within the boundary tolerance are snapped onto the boundary first: extended along the row when they stop just short, otherwise projected. Row ends and polygon vertices are then identical coordinates.
21. **Reference rows deferred** (user). Accuracy against hand-digitised rows (§11) waits until the user has reviewed the outputs.
22. **Optional removal of hanging rows** (user; `[export] drop_hanging_rows`, default `false`). When on, a row is kept only if it meets its block boundary at exactly two points, its two ends: both ends within the validation tolerance of the boundary, and no other contact. All other rows are removed before the checks, so the kept rows' attributes describe the cleaned set. The removed rows are written to `removed.shp` with the attributes they had among all rows. This is for poor imagery: keep only probably-accurate rows and add the rest by hand.

### 2026-11-08

23. **Centroid pre-test** (user proposal; `[centroid_pretest]`, §7; first called "blueprint"). A full search of one box at each block's centroid narrows the grid search to ±10° and ±10% around its winner, keeping every phase. The full grid search is used when the centroid is outside the polygon. On for Bordeaux on the `feature/centroid-blueprint` branch; results in §18.
24. **Neighbouring boxes back up a weak centroid box** (user). Rings of neighbouring boxes, spaced like the grid centres and up to `neighbour_rings`, are tried before falling back to the full grid search.
25. **Kernel cache** (user). Kernels are saved to `.cache/kernels/` under a key of everything they depend on (template geometry, rotations, spacings, pixel size, coverage method, cache format version) and reused by later runs (`[kernels] cache`, `cache_dir`). Off by default; on in `configs/centroid-pretest-cache.toml`.
26. **Config files named after their settings** (user). `configs/bordeaux.toml` became `configs/centroid-pretest.toml`, so no config name refers to the git-ignored imagery. Variants build on it with a new top-level key, `extends = "other.toml"`: `centroid-pretest-cache.toml` (kernel cache on) and `full-search.toml` (pre-test off). The kernel cache is now off by default, so each file's name says what it turns on.
27. **Missing-row estimate** (user; `[missing_row_estimate]`, §10). Fills gaps between complete neighbouring rows with rows interpolated every 5 × the median row width. It forces `drop_hanging_rows` on. Config `centroid-pretest-cache-row-estimate.toml`. Bordeaux result in §18.
28. **Estimate tolerances are settings** (user). `gap_tolerance_frac` (default 0.50, first 0.10: too tight for drifting rows), `whole_row_tolerance_frac` (default 0.20) and `max_missing_rows` (default 3) are all in `[missing_row_estimate]`.
29. **Row smoothing** (user; `[smoothing]`, §10). Rows came out jagged: 28 vertices doubled back (turns over 90°, up to 177°) where row pieces were joined, and 121 turned more than 20°. Joins no longer step back, and a maximum direction change between vertices (20° by default) is enforced on the final rows. Results in §18.
30. **Squash hanging pieces into estimated rows** (user; `merge_*` in `[missing_row_estimate]`, §10). Buffer ±0.5 × the block's median row width (user choice: catches drift, not the neighbouring row), ≥ 80% inside, ≤ 10°. On Bordeaux the length drawn from the image rose from 15.35 to 18.21 km and rows for review fell from 70 to 49 (§18).
31. **Row numbering** (user; `[row_numbering]`, §10). Built only after the squash showed a clear improvement. Numbering is consecutive in clean blocks, skipped numbers mark missing rows, and uncertain numbers flag rows that drift between lanes.
32. **Mirrored weights at block edges** (user; fixed, §8). The weight of a template line outside the block moves to its mirror line: 0.50 / 0.40 / 0.10 with one side outside, instead of 0.67 / 0.27 / 0.07. The `[scoring] unavailable_bands` option is gone, and `[template] indices` must be symmetric about 0. Equal weights of 0.2, tried as config, were much worse (§18).
33. **Head-on joins** (§10). Most rows lost after mirroring broke where two pieces of one row met head-on, 0.37–0.55 m apart sideways: just over the join's 0.36 m. Both stopped as a conflict and were removed as hanging. A conflict with a free end of the same row now joins.
34. **Rows end along their own direction** (§10). Following the edge was tried first: when a step would leave the block, it took a supported turn that stayed inside. The user ruled out an angle rule, since rows really do end at long edges. Edge following made 15 steps on Bordeaux with no measurable effect, and it bent synthetic row ends by up to 0.45 m along the boundary. It was replaced by the opposite: the last step to the boundary goes straight.

### 2026-11-12

35. **Input layers are optional, each with a switch** (user). RGB, NIR and NDVI are each optional, and at least one must be in use. `[input] use_rgb`, `use_nir`, `use_ndvi` (default true) switch a layer off without removing its path; `--layers` does it for one run. A switched-off layer is not read at all. The first layer in use (RGB, else NIR, else NDVI) sets the grid, and the others must share it. Before, RGB was required.
36. **Relative evidence weights and a NIR component** (§5). `rgb_weight`, `nir_weight` and `ndvi_weight` are rescaled over the layers in use, so a switch needs no other edit. The NIR component uses the band named `nir`, with `white_tophat` or `stretch` (canopy bright). Only components in use need their method, so `[evidence] method` is required only when the RGB component is used. `ndvi_weight` was a share (V = (1 − w) V_gray + w V_ndvi) and is now a relative weight. Every committed config has it at 0, so every earlier result is unchanged: the Bordeaux run `20261112-120024-multispectral-check` gives the same 313 rows, geometries and attributes as `20261108-220040-edge-final`. The comment in `centroid-pretest.toml` now says how to repeat the 0.3 test.
37. **NoData per file.** With `nodata_value` unset, each RGB and NIR file's own declared NoData applies; before, the RGB file's applied to both. NaN (and infinity) is always NoData, since `x == NaN` never matches.
38. **Preparation step** (§3, user: "use the Italian CRS for FVG"). `src/prepare_inputs.py` puts layers on one projected grid with explicit resampling and recorded provenance. The Cremons layers go to RDN2008 / UTM zone 33N (EPSG:6708) at 0.20 m, averaged (§14).
39. **Run outputs per area** (user: Cremons runs in `runs/cremons/`). `[export] runs_dir`, default `runs`.
40. **Cremons block from the user** (§14). The user's polygon was supplied in the June flight's local frame under an EPSG:4326 label. It was moved into the August mosaic's frame by the measured June-to-August shift. A provisional block drawn from the NIR before it arrived was discarded.
41. **Config name.** The Cremons config is `configs/multispectral.toml`, named after what it sets (item 26): the three layers together. It extends `centroid-pretest-cache-row-estimate-numbering.toml` and replaces the inputs and area-specific picks.

### 2026-11-12 (QGIS plugin)

42. **A QGIS plugin, no command line** (user). The repository is reworked into a QGIS plugin following the PyQGIS Developer Cookbook (3.44): `vineliner/` with `classFactory`, `metadata.txt`, a main class that registers a Processing provider, and one algorithm, *Detect vine rows*. The user dropped the requirement that the method also run from the command line. `src/`, `configs/` and the rasterio/GeoPandas adapters are removed; items 35–41 now apply through the plugin's inputs instead of config keys.
43. **The method stays independent of QGIS.** `vineliner/core/` (the former `tools/`) uses NumPy, SciPy and Shapely only and has no QGIS imports (tested). `qgis_io/` is the only bridge, and `core.pipeline.run()` works in memory. The core's tests run outside QGIS; a second set runs the algorithm inside QGIS. Curve fitting, tried the same day, was rejected after the visual check and is not part of it.
44. **Settings: the final Bordeaux configuration, fixed** (user: "the final version of the Bordeaux config is the best possible option"). `core/defaults.toml` holds the PICKs of §18 with the centroid pre-test, kernel cache, missing-row estimate with squash and row numbering on. The dialog sets only the row width range: 21 widths across it, and the same range for the spacing check. The kernel cache lives in the QGIS profile.
45. **Inputs: RGB, NIR-R-G, NDVI layers and vineyard polygons** (user). At least one raster layer. RGB alone gives RGB evidence; with NIR-R-G it also gives NIR and NDVI, computed as (NIR − red) / (NIR + red) from the NIR-R-G layer; NDVI alone gives NDVI. A supplied NDVI layer is used rather than a computed one. An NDVI layer whose values are mostly outside −1..1 is refused.
46. **Outputs** (user). Row lines have exactly one attribute, `needs_review` (Boolean), the review flag of §11 including numbering uncertainty and estimated rows. The vineyard polygons keep their attributes and gain `average_row_length`, `average_row_direction` (azimuth from north, 0–180), `average_row_width` (mean of the rows' median spacings) and `confidence_interval` (half-width of the 95 % Student-t interval of that width). Rows interpolated without image support are left out of the width and its interval. The polygons keep a vertex at every row end (item 20).
47. **Analysis grid chosen by the plugin** (§3). CRS, pixel size and extent as in §3's implementation note; the CRS and pixel are advanced parameters. The automatic pixel is at least 1/13 of the middle row width, so 1 cm drone mosaics become 0.2 m, which kept the Cremons search at about 150 million kernel weights.
48. **Per-block choice of evidence, not a mix** (`[evidence] combine = "best_per_block"`, §5, §20). With RGB and NIR-R-G given, the user's rule is to use both and compute NDVI. An equal mix of the three found 94 of Bordeaux's 313 rows; even RGB 1 : NIR 0.25 : NDVI 0.25 found 274 with 138 for review. Mixing smears rows: RGB marks canopy and shadow dark, NIR the canopy bright. Instead every component is scored alone at the same 9 boxes (the centroid and the ring around it, a full search each), and the block uses the one with the highest median score. All layers still take part. On Bordeaux this chose NIR for 3 of 7 blocks and found 320 rows with 39 for review, more and fewer than RGB alone (313, 44); on Cremons it chose NDVI (16 rows, 1 for review). `combine = "weighted"` keeps the weighted mean.
49. **Coordinates are transformed with GDAL's OSR, not QGIS** (2026-11-13). In QGIS 3.44.15 the cached `QgsCoordinateTransform` for a pair of CRSs can get stuck handing the coordinates back unchanged, in every thread, without an error, until QGIS restarts. Reproduced after map rendering once the Processing thread pool replaced an expired thread: EPSG:4326 → EPSG:32633 returned degrees as they were, which put a user's vineyard 0.001 m across on a 1 × 1 px grid on the second run of a session. GDAL's OSR gave the right result in that state, and GDAL warps the rasters anyway, so `qgis_io/transform.py` transforms the polygons, extents and pixel sizes with it (a test forbids `QgsCoordinateTransform` in the plugin). Layer access happens in `prepareAlgorithm` on the main thread; `processAlgorithm` on the worker thread needs only GDAL and the core. Implausible layers are refused before the run: a CRS that contradicts the coordinates (degrees read as metres or the reverse), pixels outside 1 mm to 50 m on the ground (such as a DJI local-frame map saved with WGS 84 attached), polygons narrower than a row width, and grids over 50 million pixels.
50. **Test data stays out of the repository** (user, 2026-11-13). `dof/` (imagery) and `poly/` (vineyard polygons) are data, not code, and are git-ignored; the files are kept locally. The per-area configs went with the command line (item 42): the only settings file is `vineliner/core/defaults.toml`, which the plugin reads.

## 18. First Bordeaux results (2026-11-07)

These are the first experimental results, from `configs/centroid-pretest.toml` on the 7 blocks of `poly/poly.gpkg` (3.2 ha).

**Search size.**
- 7,947 search centres at 2 m spacing.
- 185,220 hypotheses per centre: 180 rotations × 21 spacings (1.20–1.60 m) × 49 translations. That's 1.47 billion template scores per run.
- About 6.5 minutes per run: 20 s for kernels, the rest searching (about 40 ms per centre with 3-px bands).

**First-run PICKs for the open choices.**

| Key | PICK | Reason |
| --- | --- | --- |
| `evidence.method`, `tophat_size_m` | `black_tophat`, 1.4 m | Rows are thin dark lines on bare soil; removes soil-brightness patches |
| `evidence.scale_region` | `per_block` | All-block scaling let block 1 set the scale (block 0: 234 vs 545 accepted centres) |
| `evidence.ndvi_weight` | 0.0 | NDVI at 0.3 did not help (below) |
| `template.line_length_m` | 2.0 | Steadier orientation than 1 m |
| `template.band_width_px` | 3 | About the row-stripe width; 1 px left rows ambiguous by up to 0.6 m |
| `translation.box` | `nominal_d0` | Equal translation count for every spacing; `tested_d` favoured d = 1.60 m |
| `centres.spacing_m` | 2.0 | About 1.4 rows apart; with seed lines −1..1 every row gets seeds |
| `scoring.normalisation`, `min_valid_coverage` | `renormalise_valid`, 0.5 | Missing pixels are not read as soil. Unavailable bands were `renormalise`; they are now mirrored (§8, 2026-11-08) |
| `selection.method` | `orientation_consensus`, 6 m, ±5° | 17% of raw winners were off-angle in block 0 |
| `seeds.lines`, `min_band_mean` | [−1, 0, 1], 0.4 | Inner lines seed neighbouring rows; drops lines on inter-row soil |
| `reconstruction.*` | link 10°, 0.25 d, 4 m; smoothing 4 m; growth support 0.4; gap 6 m; join 0.35 d, 20°; limit 1000 m | See comments in `centroid-pretest.toml` |
| `validation.tolerance_m` | 0.3 | Well inside the ~1 m target |
| `storage.surfaces` | `winning_spacing` | `all` would be about 5.9 GB per run |

**Evidence variants** (all 7 blocks):

| Run | Row sections | Pass all checks | Row length | Spacing / completion failures |
| --- | ---: | ---: | ---: | ---: |
| RGB, scaled over all blocks | 428 | 94 | 16.0 km | 2,768 / 384 |
| **RGB, scaled per block (chosen)** | 444 | 101 | 20.7 km | 2,980 / 291 |
| + NDVI stretch, weight 0.3 | 539 | 100 | 18.7 km | 3,682 / 458 |
| + NDVI white top-hat, weight 0.3 | 516 | 92 | 16.8 km | 4,024 / 519 |

Spacing failures are counted per 1 m sample on each side of a row; completion failures per endpoint.

**Per block** (chosen run):

| Block | Accepted centres | Direction | Spacing | Sections | Pass all | Row length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 545 / 963 | 40° | 1.46 m | 49 | 1 | 2.2 km |
| 1 | 1,402 / 1,423 | 71° | 1.40 m | 77 | 71 | 4.1 km |
| 2 | 377 / 572 | 43° | 1.38 m | 58 | 8 | 1.6 km |
| 3 | 1,372 / 1,953 | 40° | 1.50 m | 80 | 2 | 5.2 km |
| 4 | 489 / 529 | 62° | 1.40 m | 33 | 19 | 1.5 km |
| 5 | 914 / 1,581 | 77° | 1.46 m | 78 | 0 | 4.0 km |
| 6 | 362 / 926 | 41° | 1.40 m | 69 | 0 | 2.2 km |

**Observations.**
- **Synthetic test.** On synthetic stripes (`tests/pipeline/`), all 38 rows are found once each, end on the boundary at both ends, pass every check, and lie within 0.5 m of the truth (typically within 0.1 m).
- **Block 1** is reconstructed almost completely: 71 of 77 sections pass and every endpoint is on the boundary.
- **Blocks 3 and 5** look dense and parallel on the overlay, but almost no section passes. A section fails if any one of its 1 m spacing samples is out of range, and nearly every 50–60 m row has one. The failure points show where.
- **Pass rates.** 25% of row length passes every check. Another 19% fails only on spacing samples between 1.0 and 1.9 m, just outside the 1.2–1.6 m validation range.
- **Completion failures** are row ends that stop inside a block. They concentrate in faint, lighter-soil patches where rows are hard to see even by eye, especially in blocks 0, 3, 5 and 6. By the §10 rule they are left unjoined rather than forced.
- **NDVI** at weight 0.3 gave shorter, more fragmented rows with both methods, though slightly better pass counts in blocks 0 and 2. At 0.2 m it is softer than the RGB (§14). It may serve vineyard classification (§4) better than row placement.
- The geometry checks cannot show positional accuracy, missed rows outside detected areas, or wrong joins. That needs reference rows.

**Suggested next steps.**
- Hand-digitise reference rows for one or two blocks (e.g. 1 and 3) and measure positional deviation, missed and extra rows, and fragmentation (§11).
- Decide the section pass rule: should one out-of-range 1 m spacing sample fail a whole row? Also decide the Bordeaux validation range.
- Decide how faint patches should end: longer bridging (`max_gap_m`) against stopping inside the block.
- Speed: kernel caching across runs (20 s), and vectorised growth scoring.

Run outputs: `runs/20261107-142834-all-blocks`, `runs/20261107-143844-rgb-perblock` (chosen), `runs/20261107-143847-ndvi-stretch`, `runs/20261107-143848-ndvi-tophat`. Block-0 trials behind the §17 decisions: `runs/20261107-14*-block0*`.

### Final run for human review (2026-11-07, `runs/20261107-171413-final`)

This run uses the same settings, plus the user decisions §17 items 18–20:
- checks reported, not enforced;
- gaps bridged up to 50 m;
- block polygons with a vertex at every row end.

| | Rows | Length | Share of length |
| --- | ---: | ---: | ---: |
| `conf = high` | 100 | 5.3 km | 23% |
| `conf = medium` (notes only) | 131 | 9.2 km | 40% |
| `conf = low` (`review = 1`) | 182 | 8.4 km | 37% |
| **Total** | 413 | 22.9 km | |

- **Bridged length:** 1.7 km, 7% of all row length.
- **Dangling ends:** 176.
- **Crossings:** 2.
- **Reasons for `review = 1`** (a row can have several):

  | Reason | Rows |
  | --- | ---: |
  | Spacing off at more than 25% of samples | 168 |
  | Bridged more than 15 m or 30% | 157 |
  | Dangling end(s) | 157 |
  | Weak evidence | 80 |
  | Crossings | 4 |
  | Short | 4 |

| Block | Rows | high / medium / low | Dangling ends |
| ---: | ---: | --- | ---: |
| 0 | 50 | 2 / 28 / 20 | 18 |
| 1 | 77 | 73 / 4 / 0 | 0 |
| 2 | 55 | 12 / 25 / 18 | 13 |
| 3 | 72 | 2 / 13 / 57 | 59 |
| 4 | 33 | 10 / 23 / 0 | 0 |
| 5 | 70 | 1 / 27 / 42 | 37 |
| 6 | 56 | 0 / 11 / 45 | 49 |

Blocks 1 and 4 need little or no review. In blocks 3, 5 and 6 the flags concentrate in faint patches, and those flags are the work list for the user's review of the product.

### Centroid pre-test against the full search (2026-11-08, first version without neighbour backup)

The same Bordeaux settings were run with and without the pre-test (then called `[blueprint]`), one after the other on the same machine:
`runs/20261108-115437-cmp-blueprint` and `runs/20261108-115610-cmp-full`.

| | Blueprint | Full search |
| --- | ---: | ---: |
| Total run time | 92 s | 362 s |
| Search time | 50 s | 319 s |
| Template scores | 0.37 billion | 1.47 billion |
| Rows | 408 | 413 |
| Row length | 22.8 km | 22.9 km |
| high / medium / low | 103 / 132 / 173 | 100 / 131 / 182 |
| Dangling ends | 165 | 176 |

| Block | Mode | Blueprint | Hypotheses per centre | Search s (bp / full) | Rows (bp / full) | Row length agreeing* |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | blueprint | 41°, 1.44 m, S 0.64 | 15,435 | 2.2 / 28.2 | 50 / 50 | 99% |
| 1 | blueprint | 72°, 1.38 m, S 0.59 | 13,377 | 2.7 / 40.9 | 77 / 77 | 100% |
| 2 | full | centroid box S 0.39 < 0.50 | 185,220 | 13.9 / 15.2 | 55 / 55 | 100% |
| 3 | blueprint | 36°, 1.52 m, S 0.74 | 12,348 | 3.5 / 159.5 | 67 / 72 | 98–99% |
| 4 | blueprint | 62°, 1.42 m, S 0.60 | 15,435 | 1.3 / 13.7 | 33 / 33 | 100% |
| 5 | blueprint | 76°, 1.46 m, S 0.56 | 15,435 | 3.3 / 39.9 | 70 / 70 | 100% |
| 6 | full | centroid box S 0.45 < 0.50 | 185,220 | 22.7 / 21.8 | 56 / 56 | 100% |

\* Share of one run's row length lying within 0.3 m of the other run's rows, both ways.

- **Speed.** The blueprint search is about 12 times cheaper per centre. The run as a whole is 3.9 times faster.
- **Remaining time.** It is now mostly the 20 s kernel build and the two fallback blocks.
- **Accuracy.** Results agree with the full search to within 0.3 m along 98–100% of row length.
- **Block 3.** The blueprint direction (36°) differs from the block's median direction (40°) but lies well inside the window. Only 1.8% of the grid winners sat on the window edge.
- **Fallback blocks.** Blocks 2 and 6 fell back because their centroid boxes landed on weak data.

### Centroid pre-test with neighbour backup and kernel cache (2026-11-08)

Runs `runs/20261108-121223-pretest-cold` (kernel cache empty) and `runs/20261108-121327-pretest-warm` (cache filled) are compared with the full search `runs/20261108-115610-cmp-full` from the comparison above. The full-search code is unchanged since then.

| | Pre-test, cold cache | Pre-test, warm cache | Full search |
| --- | ---: | ---: | ---: |
| Total run time | 63 s | 41 s | 362 s |
| Kernels | 20.3 s (built, saved) | 0.1 s (loaded) | 20 s (built) |
| Search time | 17 s | 17 s | 319 s |
| Template scores | 0.11 billion | 0.11 billion | 1.47 billion |
| Rows | 407 | 407 | 413 |
| Row length | 22.8 km | 22.8 km | 22.9 km |
| high / medium / low | 102 / 135 / 170 | 102 / 135 / 170 | 100 / 131 / 182 |
| Dangling ends / crossings | 164 / 0 | 164 / 0 | 176 / 2 |

| Block | Pre-test from | Direction, spacing, S | Search s (pre-test / full) | Rows (pre-test / full) | Row length agreeing* |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | centroid box | 41°, 1.44 m, 0.64 | 2.5 / 28.2 | 50 / 50 | 99% |
| 1 | centroid box | 72°, 1.38 m, 0.59 | 2.7 / 40.9 | 77 / 77 | 100% |
| 2 | ring-1 box (9 tested) | 43°, 1.38 m, 0.64 | 1.5 / 15.2 | 56 / 55 | 97–98% |
| 3 | centroid box | 36°, 1.52 m, 0.74 | 3.4 / 159.5 | 67 / 72 | 98–99% |
| 4 | centroid box | 62°, 1.42 m, 0.60 | 1.5 / 13.7 | 33 / 33 | 100% |
| 5 | centroid box | 76°, 1.46 m, 0.56 | 3.4 / 39.9 | 70 / 70 | 100% |
| 6 | ring-1 box (9 tested) | 42°, 1.40 m, 0.56 | 2.3 / 21.8 | 54 / 56 | 99% |

\* Share of one run's row length within 0.3 m of the other run's rows, both ways.

- **Neighbour backup.** Blocks 2 and 6, which fell back to the full search without it, now take their pre-test from a neighbouring box one ring out.
- **Speed.** The search is 19 times faster than the full search. With a warm kernel cache the whole run is 8.8 times faster.
- **Remaining time.** It is now mostly growth, the checks and export, about 20 s.
- **Accuracy.** Results agree with the full search to within 0.3 m along 97–100% of row length.

### Missing-row estimate on Bordeaux (2026-11-08)

`configs/centroid-pretest-cache-row-estimate.toml` with the default tolerances (gap 50%, whole-number 20%, at most 3 rows per gap), `runs/20261108-132900-row-estimate-50`, 63 s.

| Block | Row width (raw median) | Hanging rows removed | Gaps too wide | Rows added | Gaps not filled |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | 1.44 m (1.45) | 16 | 8 | 8 | 1: an interpolated row would not span the block |
| 1 | 1.40 m (1.40) | 0 | 0 | 0 | |
| 2 | 1.37 m (1.38) | 15 | 5 | 8 | |
| 3 | 1.50 m (1.53) | 42 | 8 | 7 | 2: 7.45 m gaps need 4 rows (> `max_missing_rows` 3) |
| 4 | 1.40 m (1.40) | 0 | 0 | 0 | |
| 5 | 1.45 m (1.47) | 34 | 11 | 14 | |
| 6 | 1.40 m (**2.21**) | 40 | 7 | 9 | |

- **Totals.** 306 rows (19.8 km): 260 detected and complete, plus 46 estimated. No dangling ends, no crossings.
- **Rows for review.** 69 in total: the 46 estimated rows plus 23 detected rows.
- **Tolerances.** With the first 10% tolerances (`runs/20261108-131001-row-estimate`), gaps of 1.83 m and 1.96 m were reported as holding no whole row, and block 6's 2.48 m gap failed the whole-number check. At the 50% / 20% defaults the first two count as drift and the third gets its row.
- **Width correction.** Block 6's raw median distance (2.21 m) spans more than one row interval, because it lost most of its rows to hanging-row deletion. The interval-corrected width (1.40 m) matches the search spacing.
- **Placement.** On the overlay the estimated rows fall evenly between the detected rows.
- **Evidence.** The template score along estimated rows (median 0.41) is not near zero, because four of the template's five lines sit on the real neighbouring rows.

### Row smoothing on Bordeaux (2026-11-08)

Same config (`centroid-pretest-cache-row-estimate.toml`), before (`runs/20261108-132900-row-estimate-50`) and after (`runs/20261108-143124-smoothing-final`, 57 s):

| Vertex turns in rows.shp | Before | After |
| --- | ---: | ---: |
| over 20° | 121 | 0 |
| over 45° | 46 | 0 |
| over 90° (line doubles back) | 28 | 0 |
| sharpest | 177° | 20.0° |

Rows, row length and estimates are unchanged (306 rows, 19.8 km, 46 estimated). 85 vertices were removed before clipping. The worst former kink, a hairpin at a join, is now a straight line with a slight bend.

### Squash and row numbering on Bordeaux (2026-11-08)

Squash (`centroid-pretest-cache-row-estimate.toml`, `runs/20261108-201902-squash-ramps`, 54 s) against the run before it (`runs/20261108-143124-smoothing-final`):

| | Before | Squash |
| --- | ---: | ---: |
| Row length drawn from the image | 15.35 km | 18.21 km |
| Purely interpolated length | 3.60 km | 0.76 km |
| Estimated rows: pure / merged | 46 / 0 | 5 / 41 (median 92% detected) |
| Rows for review | 70 | 49 |
| Hanging pieces left over | 147 (6.6 km) | 79 (3.4 km) |
| Dangling ends / crossings / turns over 20° | 0 / 0 / 0 | 0 / 0 / 0 |

68 pieces were squashed. That matches the earlier test of the buffer at ±0.5 × width.

Numbering (`centroid-pretest-cache-row-estimate-numbering.toml`, `runs/20261108-203818-numbering`, 47 s):

| Block | Rows | Numbers | Missing numbers | Uncertain |
| ---: | ---: | --- | --- | ---: |
| 0 | 42 | 1–41 | none | 4 |
| 1 | 77 | 1–77 | none | 0 |
| 2 | 49 | 1–49 | none | 0 |
| 3 | 32 | 1–37 | 7–10, 16–19 | 5 |
| 4 | 33 | 1–33 | none | 0 |
| 5 | 50 | 1–50 | none | 8 |
| 6 | 23 | 1–28 | 20–24 | 4 |

- **Clean blocks.** Blocks 1, 2 and 4 are numbered without gaps or uncertainty.
- **Block 3.** Its skipped numbers are the two 4-row gaps the estimate leaves unfilled (`max_missing_rows` = 3).
- **Block 6.** Its 5-row gap lies beyond the estimate's neighbour search, which reaches (3 + 2.5) × the row width. The numbering found it anyway.
- **Uncertain rows.** All 21 also have other issues. Their median share of spacing samples out of range is 31.5%, against 1.5% for the other rows, so they mark real lane drift. They raise the rows for review from 49 to 57.
- **Shared numbers.** Two numbers are shared, each between a long outermost row and short fragments at the block edge.

### Edge rows on Bordeaux (2026-11-08)

All runs use `centroid-pretest-cache-row-estimate-numbering.toml`. Rows are grouped by their row number's distance from the first and last number in their block.

| | Before (`…203818-numbering`) | Mirrored | Equal weights 0.2 | Mirrored + head-on joins (`…220040-edge-final`) |
| --- | ---: | ---: | ---: | ---: |
| Rows | 306 | 295 | 252 | 313 |
| Length drawn from the image | 18.21 km | 18.00 km | 15.96 km | 19.64 km |
| Rows for review | 57 | 41 | 40 | 45 |
| Uncertain / missing row numbers | 21 / 13 | 11 / 21 | 23 / 51 | 13 / 5 |
| Hanging pieces left over | 79 | 87 | 192 | 47 |
| Outermost rows: review | 50% | 33% | 33% | 33% |
| Outermost rows: mean length | 29 m | 38 m | 39 m | 38 m |
| Outermost rows: ends grazing the edge (< 25°) | 42% | 27% | 27% | 27% |
| Third rows from the edge: review | 43% | 14% | – | 14% |

- **Mirroring.** It fixes the edge rows, but it lost 11 rows. Two blocks each got a 4-row gap, which the estimate doesn't fill (`max_missing_rows` = 3).
- **Lost rows.** Block 2 lost them in a weak patch inside the block. Block 5 lost them at a long edge the rows meet at a sharp angle. Both times the pieces of each row were found, but they met head-on just too far apart to join.
- **Head-on joins.** 37 joins put those rows back and more. 14 estimated rows are now detected directly (33 estimated, from 46). Six of seven blocks are numbered without gaps. Block 6 keeps the 5-row gap it had before, beyond the estimate's reach.
- **Review.** Rows for review rose from 41 to 45 against mirroring alone, but with 18 more rows: the extra rows run through weak patches and are flagged as such.
- **Geometry.** No dangling ends, crossings or turns over 20° in any run.
- **Synthetic test.** Every true row is within 0.3 m (Hausdorff) of its detected row. Before the last-step rule, ends at a 25° edge overshot by up to 0.55 m.


## 19. First Cremons results (2026-11-12)

All runs use `configs/multispectral.toml` on the user's block (3365 m², 833 centres) at 0.20 m, in `runs/cremons/`. They differ only in `--layers` (and, for GNDVI, `ndvi = "dof/cremons/gndvi.tif"`). Each takes about 20 s with cached kernels. "Off the NIR rows" is the distance from 1 m samples along each run's rows to the nearest row of the NIR run.

| Layers (`--layers`) | Rows | Row numbers | Length | High / medium / low | For review | Bridged | Off the NIR rows (mean / 95%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB + NIR + NDVI (`…110012-all-layers`) | 16 | 1–15 | 1.34 km | 5 / 8 / 3 | 3 | 52 m | 0.11 / 0.37 m |
| NIR (`…110132-nir`) | 16 | 1–15 | 1.32 km | 4 / 9 / 3 | 3 | 56 m | – |
| NDVI (`…110154-ndvi`) | 16 | 1–15 | 1.29 km | 8 / 7 / 1 | 1 | 19 m | 0.07 / 0.18 m |
| NIR + NDVI (`…110258-nir-ndvi`) | 16 | 1–15 | 1.30 km | 5 / 8 / 3 | 3 | 68 m | 0.04 / 0.16 m |
| GNDVI in the NDVI slot (`…110237-gndvi`) | 17 | 1–16 | 1.31 km | 3 / 10 / 4 | 5 | 64 m | 0.11 / 0.25 m |
| RGB (`…110216-rgb`) | 15 | 1–15 | 1.19 km | 0 / 11 / 3 (+1 estimated) | 5 | 56 m | 0.32 / 0.78 m |

- **Same rows.** Every layer finds the block's 15 rows, and every row reaches the boundary at both ends. No run has crossings or dangling ends. The pre-test finds 115°, 2.60 m on the centroid box in every run.
- **NIR, NDVI and their mix agree within 0.1 m** on average. Two sections share number 1: the leftmost row, broken where it runs along the August data edge.
- **NDVI has the fewest flags** (1 for review, 19 m bridged). Part of that is grass: NDVI is high on grassed headlands and canopy gaps, so it supports growth there where NIR reports a bridge (NIR: 19 m bridged at the boundary and 37 m inside; NDVI: 6 m and 12 m).
- **RGB works alone, but is the weakest.** No high-confidence rows, rows wobble, and they sit 0.3 m from the NIR rows on average. That fits dark-row evidence that includes the canopy's shadow (§2). At a third of the weight it pulls the mixed rows by 0.1 m.
- **The verge row** (row 15) is low confidence in every NIR-based run: about half of it lies under the tree verge.
- **GNDVI** adds one spurious 12 m row beyond the left data edge (91% bridged, low confidence, flagged), and finds only 14 m of the verge row.
- **Open.** The weights in `multispectral.toml` are equal first-run picks. NIR + NDVI or NDVI alone may be better choices; deciding needs reference rows (item 21).

## 20. The plugin on Bordeaux and Cremons (2026-11-12)

Runs through the Processing algorithm, headless in QGIS 3.44.15 (`runs/20261112-*-plugin-*`, GeoPackages and a JSON report each), with the packaged defaults (§17 item 44).

| Data and layers | Combine | Rows | Length | Need review | Time |
| --- | --- | ---: | ---: | ---: | ---: |
| Bordeaux, RGB (`…155043-plugin-bordeaux-rgb`) | – | 313 | 21.11 km | 44 | 100 s |
| Bordeaux, RGB + NIR-R-G | equal weights | 94 | 6.58 km | 67 | 274 s |
| Bordeaux, RGB + NIR-R-G | RGB 1, NIR 0.25, NDVI 0.25 | 274 | 17.98 km | 138 | 175 s |
| Bordeaux, NIR-R-G (NIR + computed NDVI) | equal weights | 139 | 13.15 km | 37 | 161 s |
| Bordeaux, RGB + NIR-R-G (`…161240-plugin-bordeaux-rgb-nrg`) | best per block | 320 | 21.64 km | 39 | 116 s |
| Cremons, RGB + NIR-R-G + NDVI | equal weights | 17 | 1.34 km | 4 | 58 s |
| Cremons, RGB + NIR-R-G + NDVI (`…161240-plugin-cremons-rgb-nrg-ndvi`) | best per block | 16 | 1.29 km | 1 | 49 s |

- **Same rows as the command line.** Bordeaux RGB through the plugin gives the 313 rows of `20261112-120024-multispectral-check` with identical geometry. Three review flags differ, all at a threshold (support exactly 50 %, bridging and spacing at their limits): the older run also masked pixels where the unused NIR and NDVI files had NoData, which shifts evidence slightly.
- **Bordeaux choices.** Blocks 1, 6 and 7 used NIR (median box scores 0.63 / 0.59 / 0.54 against RGB 0.61 / 0.47 / 0.46), the others RGB. NDVI never won: it scores 0.20–0.53, as soft as §14 found.
- **Cremons.** NDVI 0.72, NIR 0.70, RGB 0.60; with NDVI the width is 2.584 ± 0.026 m (the equal mix gave 2.527 ± 0.107), matching the 2.57–2.64 m measured by hand (§14). The analysis CRS was set to EPSG:6708 to match §19; the grid became 0.2 m automatically from the 1 cm mosaics.
- **Statistics.** Bordeaux widths are 1.37–1.50 m with 95 % intervals of ±0.004–0.031 m; directions agree with the pre-test (e.g. block 1: 49.3° from north = 40.7° from east).

