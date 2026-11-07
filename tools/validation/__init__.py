"""Geometry checks on final rows, reported as attributes for human review."""

from .checks import CHECKS, Issue, SectionStats, check_block
from .review import ROW_FIELDS, direction_deg, evidence_along, mean_evidence, review_section, seed_rms

__all__ = [
    "CHECKS",
    "ROW_FIELDS",
    "Issue",
    "SectionStats",
    "check_block",
    "direction_deg",
    "evidence_along",
    "mean_evidence",
    "review_section",
    "seed_rms",
]
