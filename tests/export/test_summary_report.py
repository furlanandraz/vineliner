"""The plugin's output attributes (needs_review, block statistics) and the strict JSON run report."""

import json
import math

import numpy as np
import pytest
from scipy import stats
from shapely.geometry import LineString

from vineliner.core.export import BLOCK_FIELDS, ROW_FIELDS, azimuth_deg, block_statistics, row_record, write_report
from vineliner.core.reconstruction import RowSection
from vineliner.core.validation import SectionStats


def section(i, coords):
    return RowSection(i, i, 0, LineString(coords), "boundary", "boundary", 1.5, 0, [], 0)


def review(i, flag=0, estimated=0, det_pct=100.0):
    return {"id": i, "review": flag, "estimated": estimated, "det_pct": det_pct}


def test_rows_have_exactly_one_boolean_attribute():
    assert ROW_FIELDS == ("needs_review",)
    assert row_record({"review": 1, "conf": "low", "issues": "x"}) == {"needs_review": True}
    assert row_record({"review": 0}) == {"needs_review": False}


def test_azimuth_is_clockwise_from_north_and_axial():
    assert azimuth_deg(90.0) == 0.0  # a row running north-south
    assert azimuth_deg(0.0) == 90.0  # east-west
    assert azimuth_deg(45.0) == 45.0  # north-east
    assert azimuth_deg(135.0) == 135.0  # north-west, i.e. south-east
    assert azimuth_deg(-45.0) == azimuth_deg(135.0)


def test_block_statistics_with_a_confidence_interval_of_the_width():
    # Three parallel rows running north-east, 1.5 m apart, 100 m long.
    u, n = np.array([1, 1]) / math.sqrt(2), np.array([-1, 1]) / math.sqrt(2)
    secs = [section(i, [tuple(i * 1.5 * n), tuple(i * 1.5 * n + 100 * u)]) for i in range(3)]
    widths = {0: [1.48, 1.52], 1: [1.49, 1.51, 1.50], 2: [1.55, 1.53]}
    st = {i: SectionStats(spacing_samples=len(v), spacing_values=v) for i, v in widths.items()}
    s = block_statistics(secs, [review(0), review(1, flag=1), review(2)], st)
    assert set(BLOCK_FIELDS) <= set(s)
    assert s["average_row_length"] == pytest.approx(100.0)
    assert s["average_row_direction"] == pytest.approx(45.0)
    medians = np.array([1.50, 1.50, 1.54])
    assert s["average_row_width"] == pytest.approx(medians.mean(), abs=1e-3)
    half = stats.t.ppf(0.975, 2) * medians.std(ddof=1) / math.sqrt(3)
    assert s["confidence_interval"] == pytest.approx(half, abs=1e-4)
    assert (s["rows"], s["rows_needing_review"], s["rows_in_width"]) == (3, 1, 3)


def test_interpolated_rows_are_left_out_of_the_width_but_not_out_of_length_and_direction():
    secs = [section(0, [(0, 0), (0, 50)]), section(1, [(1.5, 0), (1.5, 60)]), section(2, [(3.0, 0), (3.0, 70)])]
    st = {0: SectionStats(spacing_values=[1.5]), 1: SectionStats(spacing_values=[9.9]), 2: SectionStats(spacing_values=[1.6])}
    s = block_statistics(secs, [review(0), review(1, flag=1, estimated=1, det_pct=0.0), review(2)], st)
    assert s["average_row_width"] == pytest.approx(1.55) and s["rows_in_width"] == 2
    assert s["average_row_length"] == pytest.approx(60.0) and s["average_row_direction"] == pytest.approx(0.0)
    merged = block_statistics(secs, [review(0), review(1, estimated=1, det_pct=40.0), review(2)], st)
    assert merged["rows_in_width"] == 3  # rows partly drawn from the image count


def test_empty_and_single_row_blocks():
    empty = block_statistics([], [], {})
    assert all(empty[k] is None for k in BLOCK_FIELDS) and empty["rows"] == 0
    one = block_statistics([section(0, [(0, 0), (10, 0)])], [review(0)], {0: SectionStats(spacing_values=[1.5])})
    assert one["average_row_width"] == 1.5 and one["confidence_interval"] is None  # one row: no interval
    assert one["average_row_direction"] == 90.0


def test_report_is_strict_json(tmp_path):
    report = {"n": np.int64(3), "x": np.float32(0.5), "nan": np.float64("nan"), "inf": float("inf"),
              "nested": {"arr": np.array([1.0, np.nan]), "ok": np.bool_(True)}, "a": np.arange(3), "p": tmp_path}
    path = write_report(tmp_path / "sub" / "r.json", report)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text  # strict parsers reject those tokens

    def strict(token):
        raise ValueError(token)

    data = json.loads(text, parse_constant=strict)
    assert data["n"] == 3 and data["x"] == 0.5 and data["nan"] is None and data["inf"] is None
    assert data["nested"] == {"arr": [1.0, None], "ok": True} and data["a"] == [0, 1, 2]
    assert data["p"] == str(tmp_path)
