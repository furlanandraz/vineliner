"""Row numbering on hand-built rows."""

import numpy as np
import pytest
from shapely.geometry import LineString

from tools.config import CONFIG_DIR, ConfigError, RunSettings, load_config, merge
from tools.reconstruction import RowSection, number_rows
from tools.validation import mark_numbered


def section(i, coords):
    return RowSection(i, i, 0, LineString(coords), "boundary", "boundary", 1.5, 0, [], 0)


def straight(*ys, x0=0.0, x1=30.0):
    return [section(10 + k, [(x0, y), (x1, y)]) for k, y in enumerate(ys)]


def numbers(sections, width=1.5, interval=10.0):
    return number_rows(sections, width, interval)


def test_parallel_rows_are_numbered_in_order_across_the_block():
    rows = straight(5.5, 1.0, 7.0, 2.5, 4.0)  # given out of order on purpose
    n = numbers(rows)
    by_y = {s.geometry.coords[0][1]: n.numbers[s.id] for s in rows}
    assert [by_y[y] for y in (1.0, 2.5, 4.0, 5.5, 7.0)] == [1, 2, 3, 4, 5]  # rows run east: numbers increase north
    assert all(n.ok.values()) and n.missing == [] and n.direction_deg == pytest.approx(0.0, abs=1e-6)


def test_a_missing_row_leaves_a_skipped_number():
    rows = straight(1.0, 2.5, 5.5, 7.0)  # the row at 4.0 is missing
    n = numbers(rows)
    assert sorted(n.numbers.values()) == [1, 2, 4, 5] and n.missing == [3]
    assert all(n.ok.values())


def test_rows_of_different_lengths_are_all_linked():
    rows = [section(1, [(0, 1.0), (30, 1.0)]), section(2, [(20, 2.5), (30, 2.5)]),  # a short corner row
            section(3, [(0, 4.0), (30, 4.0)]), section(4, [(0, 5.5), (8, 5.5)])]
    n = numbers(rows)
    assert [n.numbers[i] for i in (1, 2, 3, 4)] == [1, 2, 3, 4] and all(n.ok.values())


def test_curved_rows():
    arc = lambda r: [(r * np.cos(a), r * np.sin(a)) for a in np.linspace(0.2, 1.2, 40)]
    rows = [section(i, arc(r)) for i, r in enumerate((20.0, 21.5, 23.0, 24.5))]
    n = numbers(rows)
    order = [n.numbers[i] for i in range(4)]
    assert order in ([1, 2, 3, 4], [4, 3, 2, 1]) and all(n.ok.values())


def test_a_row_that_drifts_into_the_next_lane_is_uncertain():
    # Lanes at 1.0, 2.5, 4.0, 5.5: row 3 runs from lane 2.5 into lane 4.0 along the block.
    rows = [section(1, [(0, 1.0), (60, 1.0)]), section(2, [(0, 5.5), (60, 5.5)]), section(3, [(0, 2.5), (60, 4.0)])]
    n = numbers(rows)
    assert not n.ok[3]
    assert not (n.ok[1] and n.ok[2])  # at least one neighbour relation is broken too


def test_empty_block():
    n = numbers([])
    assert n.numbers == {} and n.describe()["rows_numbered"] == 0


def test_mark_numbered():
    rec = {"review": 0, "issues": "", "row_no": None, "row_no_ok": None}
    assert mark_numbered(dict(rec), 7, True) == {"review": 0, "issues": "", "row_no": 7, "row_no_ok": 1}
    bad = mark_numbered(dict(rec), 7, False)
    assert bad["review"] == 1 and bad["row_no_ok"] == 0 and "row number uncertain" in bad["issues"]


def test_numbering_settings_and_config():
    base = RunSettings.from_config(load_config(CONFIG_DIR / "centroid-pretest.toml")).row_numbering
    assert (base.enabled, base.transect_interval_m) == (False, 10.0)
    s = RunSettings.from_config(load_config(CONFIG_DIR / "centroid-pretest-cache-row-estimate-numbering.toml"))
    assert s.row_numbering.enabled and s.missing_row_estimate.enabled and s.missing_row_estimate.merge_pieces
    with pytest.raises(ConfigError, match="row_numbering"):
        RunSettings.from_config(merge(load_config(CONFIG_DIR / "centroid-pretest.toml"), {"row_numbering": {"transect_interval_m": 0}}))
