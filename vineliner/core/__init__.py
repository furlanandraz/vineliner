"""Vineliner core: the row-finding method behind the QGIS plugin, with no QGIS imports.

Modules import each other only with relative imports, and use only NumPy,
SciPy and Shapely, so the core runs and is tested outside QGIS too.

Subpackages, in pipeline order (PROJECT.md §1):

    config          settings: packaged defaults plus the dialog's row width range
    geo             pixel grid <-> world coordinates, angle convention
    imagery         valid-data mask and the row-evidence raster
    vineyards       vineyard block polygons and their pixel masks
    template        five-line template and its rasterised kernels
    search          hypotheses, scoring, exhaustive local search, persistence
    reconstruction  seeds, chains, growth, joins, final rows, missing-row estimate
    validation      geometry checks and review flags on final rows
    export          the outputs' attributes: needs_review, block statistics, report
    pipeline        input assembly and the run, in memory
"""

__version__ = "0.2.0"
