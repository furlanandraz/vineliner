"""QGIS side of the plugin: raster layers onto the analysis grid, polygons in, output features out.

This package and processing_provider/ are the only parts that import qgis or
osgeo; core/ works on NumPy arrays and Shapely geometries.
"""

from .rasters import AnalysisGrid, LayerRead, analysis_grid, coordinates_problem, layer_problem, plan_read, read_layer
from .vectors import BLOCK_OUTPUT_FIELDS, read_blocks, to_qgs_geometry

__all__ = ["BLOCK_OUTPUT_FIELDS", "AnalysisGrid", "LayerRead", "analysis_grid", "coordinates_problem", "layer_problem", "plan_read", "read_blocks", "read_layer", "to_qgs_geometry"]
