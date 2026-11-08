"""On-disk kernel cache."""

import numpy as np

from tools.template import Template, cache_key, load_or_build
from tools.template.cache import cache_path

T = Template((-2, -1, 0, 1, 2), (0.05, 0.20, 0.50, 0.20, 0.05), 2.0, 0.6)
THETAS = (0.0, 30.0, 45.0, 117.0)
SPACINGS = (1.4, 1.5)


def build(cache_dir, template=T):
    return load_or_build(template, THETAS, SPACINGS, 0.2, 0.2, "exact", None, cache_dir)


def same(a, b):
    for t in range(len(THETAS)):
        for j in range(len(SPACINGS)):
            for ka, kb in zip(a.get(t, j), b.get(t, j)):
                assert np.array_equal(ka.drow, kb.drow) and np.array_equal(ka.dcol, kb.dcol) and np.array_equal(ka.weight, kb.weight)


def test_miss_then_hit_gives_identical_kernels(tmp_path):
    built, info = build(tmp_path)
    assert info["cache"].startswith("miss")
    loaded, info2 = build(tmp_path)
    assert info2["cache"] == "hit"
    same(built, loaded)
    assert loaded.radius == built.radius
    assert loaded.get(1, 0)[2] is loaded.get(1, 1)[2]  # line 0 shared across spacings, as in a fresh build


def test_cache_off_builds(tmp_path):
    bank, info = build(None)
    assert info["cache"] == "off" and not any(tmp_path.iterdir())


def test_key_follows_geometry_not_weights():
    other_weights = Template(T.indices, (0.1, 0.2, 0.4, 0.2, 0.1), T.line_length_m, T.band_width_m)
    wider = Template(T.indices, T.weights, T.line_length_m, 0.2)
    k = cache_key(T, THETAS, SPACINGS, 0.2, 0.2, "exact", None)
    assert k == cache_key(other_weights, THETAS, SPACINGS, 0.2, 0.2, "exact", None)
    assert k != cache_key(wider, THETAS, SPACINGS, 0.2, 0.2, "exact", None)
    assert k != cache_key(T, THETAS, SPACINGS, 0.25, 0.25, "exact", None)


def test_damaged_cache_file_is_rebuilt(tmp_path):
    build(tmp_path)
    path = cache_path(tmp_path, cache_key(T, THETAS, SPACINGS, 0.2, 0.2, "exact", None))
    path.write_bytes(b"not a cache file")
    bank, info = build(tmp_path)
    assert info["cache"].startswith("miss")
    _, info2 = build(tmp_path)
    assert info2["cache"] == "hit"
