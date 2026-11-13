"""Geometry checks on final rows, reported as attributes for human review."""

from .checks import CHECKS, Issue, SectionStats, check_block, meets_boundary_twice, split_hanging
from .review import (
    CONF_CLASSES,
    ROW_FIELDS,
    direction_deg,
    evidence_along,
    mark_estimated,
    mark_numbered,
    mean_evidence,
    review_section,
    seed_rms,
)

__all__ = [
    "CHECKS",
    "CONF_CLASSES",
    "ROW_FIELDS",
    "Issue",
    "SectionStats",
    "check_block",
    "direction_deg",
    "evidence_along",
    "mark_estimated",
    "mark_numbered",
    "mean_evidence",
    "meets_boundary_twice",
    "review_section",
    "seed_rms",
    "split_hanging",
]
