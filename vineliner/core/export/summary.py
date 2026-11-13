"""Plugin outputs: one boolean per row line, and statistics per vineyard polygon (PROJECT.md §13).

Row lines carry exactly one attribute, needs_review: true when a person should
check the row (a dangling end, a crossing, too much bridging, spacing off,
weak evidence, a short row, an uncertain row number, or a row the image did
not show and the missing-row estimate added).

Each vineyard polygon carries:
  average_row_length     mean length of its rows, in metres
  average_row_direction  mean row direction as an azimuth: degrees clockwise
                         from north, 0-180 (rows are lines, so 30 and 210 are
                         the same direction); a length-weighted axial mean
  average_row_width      mean distance between neighbouring rows, in metres:
                         the mean of each row's median measured spacing
  confidence_interval    half-width of the 95 % confidence interval of
                         average_row_width, in metres (Student t over the
                         rows): the width is average_row_width +/- this.
                         Rows the estimate interpolated without image support
                         are left out of the width and its interval, since
                         they are spaced to the width by construction.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as st

from ..geo import axial_mean

BLOCK_FIELDS = ("average_row_length", "average_row_direction", "average_row_width", "confidence_interval")
ROW_FIELDS = ("needs_review",)
CONFIDENCE = 0.95


def row_record(review: dict) -> dict:
    """The one attribute of an output row line."""
    return {"needs_review": bool(review["review"])}


def azimuth_deg(theta_deg: float) -> float:
    """Row direction theta (counter-clockwise from east, axial) as an azimuth from north, clockwise, in [0, 180)."""
    return float((90.0 - theta_deg) % 180.0)


def block_statistics(sections, reviews, stats_by_id) -> dict:
    """The four polygon attributes of one block, plus the counts behind them (for the run report).

    sections: the block's final rows; reviews: their review records (same order);
    stats_by_id: validation.SectionStats by section id (spacing samples).
    """
    out = {k: None for k in BLOCK_FIELDS} | {"rows": len(sections), "rows_needing_review": 0, "rows_in_width": 0}
    if not sections:
        return out
    lengths = np.array([s.length_m for s in sections])
    out["average_row_length"] = round(float(lengths.mean()), 2)
    directions, weights = [], []
    for s in sections:
        c = np.asarray(s.geometry.coords)
        v = c[-1] - c[0]
        if np.hypot(*v) > 0:
            directions.append(math.degrees(math.atan2(v[1], v[0])))
            weights.append(s.length_m)
    if directions:
        out["average_row_direction"] = round(azimuth_deg(axial_mean(directions, weights)), 2)
    out["rows_needing_review"] = int(sum(bool(r["review"]) for r in reviews))

    widths = []
    for s, r in zip(sections, reviews):
        interpolated = r.get("estimated") and not r.get("det_pct")
        values = stats_by_id[s.id].spacing_values if s.id in stats_by_id else []
        if values and not interpolated:
            widths.append(float(np.median(values)))
    out["rows_in_width"] = len(widths)
    if widths:
        w = np.array(widths)
        out["average_row_width"] = round(float(w.mean()), 3)
        if len(w) >= 2:
            t = float(st.t.ppf(0.5 + CONFIDENCE / 2, df=len(w) - 1))
            out["confidence_interval"] = round(t * float(w.std(ddof=1)) / math.sqrt(len(w)), 4)
    return out
