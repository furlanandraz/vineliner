"""Report helpers, the PNG writer and the review-layer builders."""

import json
import struct
import zlib

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, box

from tools.export import block_records, bridge_layer, issue_layer, open_choices, write_outputs, write_png, write_report
from tools.reconstruction import RowSection
from tools.validation import Issue
from tools.vineyards import make_blocks


def read_png(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h, depth, colour = struct.unpack(">IIBB", body[:10])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    channels = 3 if colour == 2 else 1
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * channels)
    assert (raw[:, 0] == 0).all()  # no scanline filters
    return raw[:, 1:].reshape((h, w, 3) if channels == 3 else (h, w))


def test_png_round_trip(tmp_path):
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    rgb = np.random.default_rng(0).integers(0, 255, (5, 6, 3), dtype=np.uint8)
    assert np.array_equal(read_png(write_png(tmp_path / "g.png", gray)), gray)
    assert np.array_equal(read_png(write_png(tmp_path / "c.png", rgb)), rgb)
    with pytest.raises(ValueError):
        write_png(tmp_path / "bad.png", np.zeros((2, 2, 4), np.uint8))


def test_open_choices_lists_keys_the_defaults_leave_unset():
    default = {"a": {"x": 1}, "b": {}}
    run = {"a": {"x": 2, "y": 3}, "b": {"z": "on"}, "c": {"w": 4}, "flag": True}
    assert open_choices(default, run) == {"a.y": 3, "b.z": "on", "c.w": 4}


def test_report_is_strict_json(tmp_path):
    report = {"n": np.int64(3), "x": np.float32(0.5), "nan": np.float64("nan"), "inf": float("inf"),
              "nested": {"arr": np.array([1.0, np.nan]), "ok": np.bool_(True)}, "a": np.arange(3), "p": tmp_path}
    path = write_report(tmp_path / "r.json", report)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text  # strict parsers reject those tokens

    def strict(token):
        raise ValueError(token)

    data = json.loads(text, parse_constant=strict)
    assert data["n"] == 3 and data["x"] == 0.5 and data["nan"] is None and data["inf"] is None
    assert data["nested"] == {"arr": [1.0, None], "ok": True} and data["a"] == [0, 1, 2]
    assert data["p"] == str(tmp_path)


def section(i, coords, start="boundary", end="boundary", bridged=None):
    return RowSection(i, i, 0, LineString(coords), start, end, 1.4, 0, [], 0, bridged=bridged)


def test_block_records_shorten_input_names_and_count_rows():
    (b,) = make_blocks([0], [box(0, 0, 20, 10)], "file", records=[{"vineyard_owner": "x", "vineyard_area": 5}])
    sections = [section(0, [(0, 2), (20, 2)]), section(1, [(0, 4), (12, 4)], end="unsupported")]
    reviews = [{"block": 0, "review": 0, "length_m": 20.0}, {"block": 0, "review": 1, "length_m": 12.0}]
    report = {"block_id": 0, "validation": {"removed_hanging": 2}, "theta_median_deg": 0.0, "spacing_median_m": 2.0}
    (rec,) = block_records([b], [report], sections, reviews)
    assert all(len(k) <= 10 for k in rec)
    assert rec["vineyard_o"] == "x" and rec["vineyard_a"] == 5  # truncated, still distinct
    assert (rec["rows"], rec["review"], rec["row_ends"], rec["removed"], rec["length_m"]) == (2, 1, 3, 2, 32.0)


def test_issue_and_bridge_layers():
    issues = [Issue("completion", 0, (3,), Point(1, 2), 0.7, "end"), Issue("containment", 0, (4,), LineString([(0, 0), (2, 0)]))]
    records, geoms = issue_layer(issues)
    assert [r["check"] for r in records] == ["completion", "containment"]
    assert all(g.geom_type == "Point" for g in geoms)  # lines are reduced to a point for the issues layer
    records, geoms = bridge_layer([section(0, [(0, 0), (9, 0)], bridged=LineString([(2, 0), (5, 0)])), section(1, [(0, 1), (9, 1)])])
    assert records == [{"section": 0, "block": 0, "length_m": 3.0}]


def test_write_outputs_writes_empty_review_layers_and_skips_others(tmp_path):
    wkt = CRS.from_epsg(3794).to_wkt()
    layers = {"issues": ([], []), "joins": ([], [])}
    written = write_outputs(tmp_path, wkt, layers, {"issues": ([], [])}, shapefile=True, geopackage=True)
    assert (tmp_path / "shapefiles" / "issues.shp").exists()
    assert "joins" in written["skipped_empty"] and "issues" in written["empty"]
    assert len(gpd.read_file(tmp_path / "vineliner.gpkg", layer="issues")) == 0
