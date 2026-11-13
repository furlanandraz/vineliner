"""Automatic vineyard classification (PROJECT.md §4): not implemented yet.

The method is an open choice (PROJECT.md §15). Until one is chosen, runs take
block polygons from a file ([vineyards] source), such as the hand-drawn
poly/poly.gpkg. Those are test fixtures, not classification results, and
reports must say so.
"""

from __future__ import annotations


def classify_blocks(*args, **kwargs):
    raise NotImplementedError(
        "Automatic vineyard classification is still open (PROJECT.md §4, §15). "
        "Set [vineyards] source to a polygon file instead."
    )
