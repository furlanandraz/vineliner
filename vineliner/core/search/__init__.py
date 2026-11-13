"""Exhaustive local search: hypotheses, scoring, persistence diagnostics."""

from .hypotheses import SearchSpace, centre_outward_order, search_centres, translation_offsets
from .local import CentreResults, LocalSearch
from .persistence import RIDGE_FIELDS, competing_orientations, ridge_statistics
from .pretest import Pretest, box_scores, ring_offsets, run_pretest, window_edge_share
from .scoring import PointScorer, band_statistics, base_score, mirrored_weights, score_hypothesis
from .selection import Selection, select_best, select_orientation_consensus

__all__ = [
    "RIDGE_FIELDS",
    "Pretest",
    "box_scores",
    "ring_offsets",
    "run_pretest",
    "window_edge_share",
    "CentreResults",
    "LocalSearch",
    "PointScorer",
    "SearchSpace",
    "Selection",
    "band_statistics",
    "base_score",
    "centre_outward_order",
    "competing_orientations",
    "mirrored_weights",
    "ridge_statistics",
    "score_hypothesis",
    "search_centres",
    "select_best",
    "select_orientation_consensus",
    "translation_offsets",
]
