"""Valid mask (255 rule) and row-evidence components (PROJECT.md §5, §14)."""

import numpy as np
import pytest

from tools.imagery import combine, gray_response, ndvi_response, row_evidence, scale_response, valid_mask


def test_any_band_255_masks_the_pixel_in_every_file():
    rgb = np.full((3, 4, 4), 100, np.uint8)
    nir = np.full((3, 4, 4), 100, np.uint8)
    rgb[1, 0, 0] = 255  # one band of one file
    nir[0, 2, 3] = 255
    valid = valid_mask([rgb, nir], 255, "any_band")
    assert not valid[0, 0] and not valid[2, 3]
    assert valid.sum() == 14


def test_all_bands_rule_only_masks_full_255():
    rgb = np.full((3, 2, 2), 100, np.uint8)
    rgb[:, 0, 0] = 255
    rgb[0, 1, 1] = 255
    valid = valid_mask([rgb], 255, "all_bands")
    assert not valid[0, 0] and valid[1, 1]


def stripes(bright_rows: bool):
    img = np.full((60, 60), 150.0, np.float32)
    img[:, 10::7] = 250.0 if bright_rows else 50.0  # 1-px lines every 7 px
    return img


def test_black_tophat_keeps_dark_rows():
    valid = np.ones((60, 60), bool)
    V, info = row_evidence(stripes(False), valid, valid, "black_tophat", 0.2, tophat_size_m=1.4)
    assert V[:, 10::7].mean() > 0.9
    assert V[:, 12::7].mean() < 0.1  # inter-row
    assert info["tophat_size_px"] == 7


def test_white_tophat_keeps_bright_ndvi_rows():
    valid = np.ones((60, 60), bool)
    resp = ndvi_response(stripes(True) / 250.0, valid, "white_tophat", 0.2, 1.4)
    V, _ = scale_response(resp, valid, 99.0, (2.0, 98.0))
    assert V[:, 10::7].mean() > 0.9 and V[:, 12::7].mean() < 0.1


def test_invalid_pixels_do_not_leak_into_neighbours():
    img = stripes(False)
    valid = np.ones_like(img, bool)
    img[30, 30] = 255.0  # saturated outlier
    valid[30, 30] = False
    resp = gray_response(img, valid, "black_tophat", 0.2, 1.4)
    V, _ = scale_response(resp, valid, 99.0, (2.0, 98.0))
    V = combine([(V, 1.0)], valid)
    assert V[30, 30] == 0
    clean = scale_response(gray_response(stripes(False), np.ones_like(valid), "black_tophat", 0.2, 1.4), valid, 99.0, (2.0, 98.0))[0]
    assert np.abs(V - clean)[valid].max() == pytest.approx(0.0, abs=1e-6)


def test_combine_weights():
    valid = np.ones((2, 2), bool)
    a, b = np.full((2, 2), 1.0, np.float32), np.zeros((2, 2), np.float32)
    assert combine([(a, 0.7), (b, 0.3)], valid) == pytest.approx(np.full((2, 2), 0.7))


def test_nan_counts_as_nodata_without_a_nodata_value():
    nir = np.full((1, 3, 3), 900.0, np.float32)
    nir[0, 1, 1] = np.nan
    nir[0, 2, 0] = np.inf
    valid = valid_mask([nir], None, "any_band")
    assert valid.sum() == 7 and not valid[1, 1] and not valid[2, 0]
    assert valid_mask([nir], float("nan"), "any_band").sum() == 7  # a NaN NoData value is the same


def test_all_bands_rule_applies_to_nan_too():
    data = np.ones((2, 2, 2), np.float32)
    data[:, 0, 0] = np.nan
    data[0, 1, 1] = np.nan
    valid = valid_mask([data], None, "all_bands")
    assert not valid[0, 0] and valid[1, 1]


def test_unknown_rule_is_refused_even_without_nodata():
    with pytest.raises(ValueError, match="nodata rule"):
        valid_mask([np.ones((1, 2, 2), np.uint8)], None, "some_bands")


def test_bright_response_keeps_bright_nir_rows():
    from tools.imagery import bright_response

    valid = np.ones((60, 60), bool)
    resp = bright_response(stripes(True) * 8.0, valid, "white_tophat", 0.2, 1.4, "nir")  # reflectance-like values
    V, _ = scale_response(resp, valid, 99.0, (2.0, 98.0))
    assert resp.name == "nir" and V[:, 10::7].mean() > 0.9 and V[:, 12::7].mean() < 0.1
    with pytest.raises(ValueError, match="unknown nir evidence method"):
        bright_response(stripes(True), valid, "black_tophat", 0.2, 1.4, "nir")
