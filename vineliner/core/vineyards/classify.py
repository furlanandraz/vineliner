"""Automatic vineyard classification (PROJECT.md §4): not implemented yet.

The method is an open choice (PROJECT.md §15). Until one is chosen, the user
supplies the vineyard polygons as a layer (the algorithm's Vineyard polygons
input), such as the hand-drawn Bordeaux blocks. Those are test fixtures, not
classification results, and reports must say so.
"""

from __future__ import annotations


def classify_blocks(*args, **kwargs):
    raise NotImplementedError(
        "Automatic vineyard classification is still open (PROJECT.md §4, §15). "
        "Set [vineyards] source to a polygon file instead."
    )
