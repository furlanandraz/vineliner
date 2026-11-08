"""Per-row attributes for human review (PROJECT.md §11, §13).

Nothing here removes a row. Each final section gets measurements of how well
the image supports it and what the checks found, then:

  review = 1  when a human should look at it:
              a dangling end, a crossing, a self-intersection, leaving the block,
              or a threshold in [review] exceeded (bridged length, spacing
              samples out of range, weak evidence along the line, very short).
  conf        "low" when review = 1; "medium" when only minor notes remain
              (at least NOTE_BRIDGED_M of bridging, or spacing off at
              NOTE_SPACING_FRAC of samples or more); else "high".
  issues      the reasons, in words, for the attribute table.

Rows added by the missing-row estimate (tools/reconstruction/missing.py) get
estimated = 1, conf = "estimated" and review = 1: they are drawn from their
neighbours, not seen in the image. Their ev_* fields still show whether the
image supports them.

Two rows that run too close both measure the same narrow spacing, but usually
only one is wrong. Narrow samples therefore count towards a row's review flag
only against a neighbour with stronger evidence along it (higher ev_mean; ties
count for both). Against a weaker neighbour they are a note. near_row names
the neighbour a row runs closest to, so the pair is easy to find.

Field names are 10 characters or fewer so the shapefile keeps them.
"""

from __future__ import annotations

import numpy as np

from ..config.settings import ReviewSettings
from ..geo import axial_mean
from .checks import SectionStats

# Below these, bridging or spacing noise is recorded in the numbers but not noted.
NOTE_BRIDGED_M = 1.0  # one growth step
NOTE_SPACING_FRAC = 0.05

ROW_FIELDS = (
    "id", "block", "row_id", "length_m", "theta_deg", "spacing_m", "sp_med_m", "sp_out_pct",
    "bridged_m", "bridge_pct", "ev_mean", "ev_sup_pct", "pos_rms_m", "n_seeds", "n_joins",
    "start_st", "end_st", "dangling", "crossings", "near_row", "estimated", "review", "conf", "issues",
)
CONF_CLASSES = ("high", "medium", "low", "estimated")


def mean_evidence(evidence: np.ndarray) -> float | None:
    finite = np.isfinite(evidence)
    return float(evidence[finite].mean()) if finite.any() else None


def evidence_along(section, scorer, step_m: float = 1.0) -> np.ndarray:
    """Template score sampled every step_m along the section, in the local direction."""
    line = section.geometry
    length = line.length
    scores = []
    for d in np.arange(step_m / 2, length, step_m):
        a = np.array(line.interpolate(max(d - 0.5, 0.0)).coords[0])
        b = np.array(line.interpolate(min(d + 0.5, length)).coords[0])
        p = np.array(line.interpolate(d).coords[0])
        theta = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) % 180.0
        scores.append(scorer(p[0], p[1], theta, section.spacing_index))
    return np.array(scores, dtype=float)


def direction_deg(line) -> float:
    c = np.asarray(line.coords)
    seg = np.diff(c, axis=0)
    lengths = np.hypot(seg[:, 0], seg[:, 1])
    ok = lengths > 0
    if not ok.any():
        return float("nan")
    return axial_mean(np.degrees(np.arctan2(seg[ok, 1], seg[ok, 0])), weights=lengths[ok])


def seed_rms(section, seeds_by_id) -> float | None:
    """RMS distance of the row's seed midpoints from the final line (seeds on this section only)."""
    from shapely.geometry import Point

    d = [section.geometry.distance(Point(seeds_by_id[i].x, seeds_by_id[i].y)) for i in section.seed_ids if i in seeds_by_id]
    d = [v for v in d if v < 0.5 * section.spacing_m]
    return float(np.sqrt(np.mean(np.square(d)))) if d else None


def review_section(
    section,
    stats: SectionStats,
    evidence: np.ndarray,
    pos_rms: float | None,
    growth_min_score: float,
    settings: ReviewSettings,
    neighbour_ev: dict[int, float | None] | None = None,
) -> dict:
    """neighbour_ev: ev_mean of other sections by id, to decide which of two close rows is suspect."""
    length = section.length_m
    bridged = section.bridged_m
    bridge_frac = bridged / length if length > 0 else 0.0
    finite = np.isfinite(evidence)
    ev_mean = mean_evidence(evidence)
    support_frac = float(np.mean(np.where(finite, evidence, -1.0) >= growth_min_score)) if evidence.size else 0.0
    sp_out_frac = stats.spacing_out / stats.spacing_samples if stats.spacing_samples else 0.0
    # Narrow samples against a weaker neighbour do not count towards this row's flag.
    neighbour_ev = neighbour_ev or {}
    own = ev_mean if ev_mean is not None else -1.0

    def blamed(pid: int) -> bool:  # unknown neighbours count, to stay on the cautious side
        other = neighbour_ev.get(pid)
        return other is None or other >= own

    blamed_narrow = sum(n for pid, n in stats.narrow_partners.items() if blamed(pid))
    blamed_frac = (stats.spacing_wide + blamed_narrow) / stats.spacing_samples if stats.spacing_samples else 0.0
    near_row = max(stats.narrow_partners, key=stats.narrow_partners.get) if stats.narrow_partners else None

    reasons, notes = [], []
    if stats.dangling:
        reasons.append("dangling " + " and ".join(stats.dangling) + " end")
    if stats.crossings:
        reasons.append(f"crosses {stats.crossings} row(s)")
    if stats.self_intersection:
        reasons.append("crosses itself")
    if stats.outside_m > 0:
        reasons.append(f"{stats.outside_m:.1f} m outside block")
    if bridged > settings.max_bridged_m or bridge_frac > settings.max_bridged_frac:
        reasons.append(f"bridged {bridged:.1f} m ({100 * bridge_frac:.0f}%)")
    elif bridged >= NOTE_BRIDGED_M:
        notes.append(f"bridged {bridged:.1f} m")
    near = f", closest to row {near_row}" if near_row is not None else ""
    if blamed_frac > settings.max_spacing_out_frac:
        reasons.append(f"spacing off at {100 * sp_out_frac:.0f}% of samples{near}")
    elif sp_out_frac >= NOTE_SPACING_FRAC:
        notes.append(f"spacing off at {100 * sp_out_frac:.0f}% of samples{near}")
    if support_frac < settings.min_support_frac:
        reasons.append(f"weak evidence ({100 * support_frac:.0f}% supported)")
    if length < settings.min_length_m:
        reasons.append(f"short ({length:.1f} m)")

    review = int(bool(reasons))
    conf = "low" if review else ("medium" if notes else "high")
    values = stats.spacing_values
    return {
        "id": section.id,
        "block": section.block_id,
        "row_id": section.row_id,
        "length_m": round(length, 2),
        "theta_deg": round(direction_deg(section.geometry), 1),
        "spacing_m": round(section.spacing_m, 3),
        "sp_med_m": round(float(np.median(values)), 3) if values else None,
        "sp_out_pct": round(100 * sp_out_frac, 1),
        "bridged_m": round(bridged, 2),
        "bridge_pct": round(100 * bridge_frac, 1),
        "ev_mean": round(ev_mean, 3) if ev_mean is not None else None,
        "ev_sup_pct": round(100 * support_frac, 1),
        "pos_rms_m": round(pos_rms, 3) if pos_rms is not None else None,
        "n_seeds": section.n_seeds,
        "n_joins": section.n_joins,
        "start_st": section.start_state,
        "end_st": section.end_state,
        "dangling": len(stats.dangling),
        "crossings": stats.crossings,
        "near_row": near_row,
        "estimated": 0,
        "review": review,
        "conf": conf,
        "issues": "; ".join(reasons + notes)[:254],
    }


def mark_estimated(record: dict, section) -> dict:
    """Label a review record as an estimated row (in place)."""
    between = next((f for f in section.flags if f.startswith("between rows")), "between neighbouring rows")
    record["estimated"] = 1
    record["review"] = 1
    record["conf"] = "estimated"
    record["issues"] = ("estimated " + between + ("; " + record["issues"] if record["issues"] else ""))[:254]
    return record
