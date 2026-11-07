"""Vineliner core: the importable modules behind the entry files in src/.

Modules inside tools/ import each other only with relative imports, so the
package can later be moved under a QGIS plugin package unchanged.

Subpackages, in pipeline order (PROJECT.md §1):

    config          run configuration, merging and validation
    geo             pixel grid <-> world coordinates, angle convention
    io              raster and vector file adapters (rasterio, GeoPandas)
    imagery         valid-data mask and the row-evidence raster
    vineyards       vineyard block polygons
    template        five-line template and its rasterised kernels
    search          hypotheses, scoring, exhaustive local search, persistence
    reconstruction  seeds, chains, growth, joins, final row sections
    validation      geometry checks on final rows
    export          vector layers, rasters, report, overlay image
    pipeline        runs the stages in order and saves intermediates

Only io/ and export/ touch file formats. The other subpackages work on NumPy
arrays and Shapely geometries, so a QGIS plugin can swap the file adapters.
"""

__version__ = "0.1.0"
