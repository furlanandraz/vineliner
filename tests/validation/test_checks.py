"""Geometry checks (PROJECT.md §11) are reported, never failed; review attributes follow them."""

import numpy as np
from shapely.geometry import LineString, box

from vineliner.core.config.settings import ReviewSettings, ValidationSettings
from vineliner.core.reconstruction import RowSection
from vineliner.core.validation import check_block, review_section

POLY = box(0, 0, 20, 10)
SETTINGS = ValidationSettings(spacing_min_m=1.2, spacing_max_m=1.6, tolerance_m=0.05, sample_step_m=1.0)
REVIEW = ReviewSettings(max_bridged_m=15.0, max_bridged_frac=0.3, max_spacing_out_frac=0.25, min_support_frac=0.5, min_length_m=4.0)


def section(i, coords, start="boundary", end="boundary", bridged=None):
    return RowSection(i, i, 0, LineString(coords), start, end, 1.4, 0, [], 0, bridged=bridged)


def parallel_rows(ys):
    return [section(i, [(0, y), (20, y)]) for i, y in enumerate(ys)]


def run(sections):
    issues, stats, summary = check_block(sections, POLY, SETTINGS, 0)
    return issues, stats, summary["issues"]


def good_evidence(s):
    return np.full(int(s.length_m), 0.8)


def test_good_parallel_rows_have_no_issues_and_high_confidence():
    rows = parallel_rows([1.0, 2.4, 3.8, 5.2])
    issues, stats, _ = run(rows)
    assert issues == []
    for s in rows:
        rec = review_section(s, stats[s.id], good_evidence(s), 0.05, 0.4, REVIEW)
        assert (rec["review"], rec["conf"], rec["issues"]) == (0, "high", "")
        assert rec["sp_med_m"] == 1.4


def test_missing_row_is_reported_as_a_wide_gap():
    _, stats, counts = run(parallel_rows([1.0, 2.4, 5.2]))  # row at 3.8 missing -> 2.8 m gap
    assert counts["spacing"] > 0
    assert stats[1].spacing_wide > 0


def test_rows_too_close():
    _, stats, counts = run(parallel_rows([1.0, 1.8]))
    assert counts["spacing"] > 0 and stats[0].spacing_narrow > 0


def test_of_two_close_rows_only_the_weaker_is_flagged():
    rows = parallel_rows([1.0, 1.8])  # 0.8 m apart: one of them is probably a duplicate
    _, stats, _ = run(rows)
    assert stats[0].narrow_partners == {1: stats[0].spacing_narrow}
    ev = {0: 0.8, 1: 0.5}
    strong = review_section(rows[0], stats[0], np.full(20, 0.8), None, 0.4, REVIEW, ev)
    weak = review_section(rows[1], stats[1], np.full(20, 0.5), None, 0.4, REVIEW, ev)
    assert weak["review"] == 1 and "closest to row 0" in weak["issues"]
    assert strong["review"] == 0 and strong["conf"] == "medium" and strong["near_row"] == 1


def test_crossing_rows_are_counted_on_both():
    issues, stats, counts = run([section(0, [(0, 1), (20, 5)]), section(1, [(0, 5), (20, 1)])])
    assert counts["cross_row"] == 1
    assert stats[0].crossings == stats[1].crossings == 1


def test_dangling_end_flags_review_but_keeps_the_row():
    rows = [section(0, [(0, 5), (12, 5)], end="unsupported")]
    _, stats, counts = run(rows)
    assert counts["completion"] == 1 and stats[0].dangling == ["end"]
    rec = review_section(rows[0], stats[0], good_evidence(rows[0]), None, 0.4, REVIEW)
    assert rec["review"] == 1 and rec["conf"] == "low" and rec["dangling"] == 1
    assert "dangling end end" in rec["issues"]


def test_outside_polygon_is_reported():
    _, stats, counts = run([section(0, [(-3, 5), (20, 5)])])
    assert counts["containment"] == 1 and stats[0].outside_m > 2.9


def test_self_intersection():
    _, stats, counts = run([section(0, [(0, 2), (10, 2), (10, 6), (5, 0)])])
    assert counts["self_intersection"] == 1 and stats[0].self_intersection


def test_some_bridging_is_a_note_long_bridging_flags_review():
    short = section(0, [(0, 5), (20, 5)], bridged=LineString([(5, 5), (8, 5)]))
    long = section(1, [(0, 3), (20, 3)], bridged=LineString([(2, 3), (18, 3)]))
    _, stats, _ = run([short])
    rec = review_section(short, stats[0], good_evidence(short), None, 0.4, REVIEW)
    assert (rec["review"], rec["conf"], rec["bridged_m"]) == (0, "medium", 3.0)
    _, stats, _ = run([long])
    rec = review_section(long, stats[1], good_evidence(long), None, 0.4, REVIEW)
    assert rec["review"] == 1 and "bridged 16.0 m" in rec["issues"]


def test_weak_evidence_flags_review():
    s = parallel_rows([5.0])[0]
    _, stats, _ = run([s])
    rec = review_section(s, stats[0], np.full(20, 0.1), None, 0.4, REVIEW)
    assert rec["review"] == 1 and "weak evidence" in rec["issues"]


def test_attribute_names_fit_a_shapefile():
    s = parallel_rows([5.0])[0]
    _, stats, _ = run([s])
    rec = review_section(s, stats[0], good_evidence(s), None, 0.4, REVIEW)
    assert all(len(k) <= 10 for k in rec)


def test_meets_boundary_twice():
    from vineliner.core.validation import meets_boundary_twice, split_hanging

    complete = LineString([(0, 5), (20, 5)])
    dangling = LineString([(0, 5), (12, 5)])
    floating = LineString([(4, 5), (12, 5)])
    touching = LineString([(0, 2), (10, 0), (20, 2)])  # touches the boundary mid-way: three points
    along_edge = LineString([(0, 0), (20, 0)])  # runs along the boundary
    assert meets_boundary_twice(complete, POLY, 0.05)
    for line in (dangling, floating, touching, along_edge):
        assert not meets_boundary_twice(line, POLY, 0.05)
    rows = [section(i, list(g.coords)) for i, g in enumerate((complete, dangling, touching))]
    kept, hanging = split_hanging(rows, POLY, 0.05)
    assert [s.id for s in kept] == [0] and [s.id for s in hanging] == [1, 2]


def test_empty_result_is_flagged():
    _, _, summary = check_block([], POLY, SETTINGS, 0)
    assert summary["empty"] is True
