"""Missing-row estimate on hand-built rows (no imagery)."""

import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from vineliner.core.config import ConfigError, RunSettings, plugin_config
from vineliner.core.config.settings import ValidationSettings
from vineliner.core.reconstruction import RowSection, fill_gaps, find_gaps, interpolate_rows, row_width
from vineliner.core.validation import check_block, mark_estimated, meets_boundary_twice

FIELD = box(0, 0, 30, 10)
CHECK = ValidationSettings(spacing_min_m=1.3, spacing_max_m=1.7, tolerance_m=0.1, sample_step_m=1.0)


def rows_at(*ys, polygon=FIELD):
    x0, _, x1, _ = polygon.bounds
    return [RowSection(i, i, 0, LineString([(x0, y), (x1, y)]), "boundary", "boundary", 1.5, 0, [], 0) for i, y in enumerate(ys)]


def width_of(sections, search_spacing=1.5):
    _, stats, _ = check_block(sections, FIELD, CHECK, 0)
    return row_width(sections, stats, search_spacing)


def gaps_of(sections, width=1.5, max_missing=3, gap_tol=0.50, whole_tol=0.20):
    """Defaults as in default.toml."""
    return find_gaps(sections, width, gap_tol, whole_tol, max_missing, 1.0)


# -- row width -------------------------------------------------------------------------


def test_row_width_from_complete_rows():
    width, info = width_of(rows_at(1.0, 2.5, 4.0, 5.5, 7.0))
    assert width == pytest.approx(1.5) and info["raw_median_m"] == pytest.approx(1.5)


def test_row_width_is_not_inflated_when_most_rows_are_missing():
    # Only every second row survived: every measured distance is 3 m.
    width, info = width_of(rows_at(1.0, 4.0, 7.0))
    assert info["raw_median_m"] == pytest.approx(3.0)
    assert width == pytest.approx(1.5)  # each 3 m distance spans two 1.5 m intervals


def test_row_width_without_neighbours():
    width, info = width_of(rows_at(5.0))
    assert width is None and info["samples"] == 0


# -- gaps -------------------------------------------------------------------------------


def test_gap_holding_one_row_is_found():
    (gap,) = gaps_of(rows_at(1.0, 2.5, 5.5, 7.0))  # row at 4.0 missing
    assert (gap.a, gap.b, gap.missing, gap.filled) == (1, 2, 1, True)
    assert gap.gap_m == pytest.approx(3.0) and gap.samples >= 3


def test_neighbours_within_tolerance_have_no_gap():
    assert gaps_of(rows_at(1.0, 2.6, 4.1, 5.5)) == []  # 1.6, 1.5, 1.4 m


def test_drifting_rows_are_not_a_gap_with_the_default_tolerance():
    drifted = rows_at(1.0, 2.5, 4.5, 6.0)  # 2.0 m is 33 % wider than 1.5 m: drift, not a missing row
    assert gaps_of(drifted) == []
    (gap,) = gaps_of(drifted, gap_tol=0.10)  # a tight tolerance would call it a gap...
    assert not gap.filled and "no whole row" in gap.reason  # ...that holds no row


def test_whole_number_check_is_configurable():
    sections = rows_at(1.0, 2.5, 5.05)  # 2.55 m: one row would leave 1.275 m spacings, 15 % off
    (strict,) = gaps_of(sections, whole_tol=0.10)
    assert not strict.filled and "whole number" in strict.reason
    (default,) = gaps_of(sections)  # 20 % by default
    assert default.filled and default.missing == 1


def test_gap_wider_than_max_missing_rows_is_reported():
    (gap,) = gaps_of(rows_at(1.0, 8.5), max_missing=3)  # 7.5 m: four rows would fit
    assert not gap.filled and gap.missing == 4 and "max_missing_rows" in gap.reason


# -- interpolation ----------------------------------------------------------------------


def test_filled_rows_sit_between_the_neighbours_and_span_the_block():
    sections = rows_at(1.0, 2.5, 7.0)  # 4.5 m gap: two rows missing, at 4.0 and 5.5
    gaps = gaps_of(sections)
    added = fill_gaps(gaps, sections, 1.5, 5.0, FIELD, 0.1, (1.4, 1.5, 1.6), 100, meets_boundary_twice)
    assert [s.id for s in added] == [100, 101]
    for s, y in zip(added, (4.0, 5.5)):
        assert s.geometry.hausdorff_distance(LineString([(0, y), (30, y)])) < 1e-6
        assert "estimated" in s.flags and (s.start_state, s.end_state) == ("boundary", "boundary")
        assert meets_boundary_twice(s.geometry, FIELD, 0.1) and s.spacing_index == 1


def test_interpolated_row_follows_the_curvature():
    centre = Point(0, 0)
    arc = lambda r: LineString([(r * np.cos(a), r * np.sin(a)) for a in np.linspace(0.1, 1.4, 60)])
    field = box(-1, -1, 30, 30).intersection(centre.buffer(26).difference(centre.buffer(18)))
    (line,) = interpolate_rows(arc(20.0), arc(23.0), 1, 1.5, 5.0, field, 0.1)
    radii = np.hypot(*np.array(line.coords)[1:-1].T)  # interior vertices, every 5 x 1.5 m
    assert len(radii) >= 3
    assert np.all(np.abs(radii - 21.5) < 0.05)


def test_mark_estimated():
    rec = {"estimated": 0, "review": 0, "conf": "high", "issues": ""}
    sec = rows_at(4.0)[0]
    sec.flags = ["estimated", "between rows 1 and 2 (1 of 1)"]
    assert mark_estimated(rec, sec) == {"estimated": 1, "review": 1, "conf": "estimated", "issues": "estimated between rows 1 and 2 (1 of 1)"}


# -- settings -------------------------------------------------------------------------


def test_estimate_turns_hanging_row_deletion_on():
    s = RunSettings.from_config(plugin_config(1.2, 1.6, {"reconstruction": {"drop_hanging_rows": False}}))
    assert s.missing_row_estimate.enabled and s.drop_hanging_rows
    assert len(s.notes) == 1 and "drop_hanging_rows" in s.notes[0]
    off = RunSettings.from_config(plugin_config(1.2, 1.6, {
        "missing_row_estimate": {"enabled": False}, "reconstruction": {"drop_hanging_rows": False}}))
    assert not off.drop_hanging_rows and off.notes == ()


def test_estimate_defaults():
    m = RunSettings.from_config(plugin_config(1.2, 1.6)).missing_row_estimate
    assert (m.enabled, m.gap_tolerance_frac, m.whole_row_tolerance_frac, m.station_factor, m.max_missing_rows) == (True, 0.50, 0.20, 5.0, 3)


@pytest.mark.parametrize("bad", [{"gap_tolerance_frac": 1.5}, {"gap_tolerance_frac": 0.0}, {"whole_row_tolerance_frac": 0.6},
                                 {"station_factor": 0.0}, {"max_missing_rows": 0}])
def test_bad_estimate_settings_are_refused(bad):
    cfg = plugin_config(1.2, 1.6, {"missing_row_estimate": {"enabled": True, **bad}})
    with pytest.raises(ConfigError, match="missing_row_estimate"):
        RunSettings.from_config(cfg)
