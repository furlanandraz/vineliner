"""Config loading, merging and RunSettings validation."""

import math
from pathlib import Path

import pytest

from tools.config import CONFIG_DIR, PROJECT_ROOT, ConfigError, RunSettings, load_config, merge, read_toml

PRETEST = CONFIG_DIR / "centroid-pretest.toml"
DEFAULT = read_toml(CONFIG_DIR / "default.toml")


def test_config_dir_is_the_repo_configs_folder():
    assert CONFIG_DIR == Path(__file__).resolve().parents[2] / "configs"


def test_default_template_matches_project_md():
    t = DEFAULT["template"]
    assert t["indices"] == [-2, -1, 0, 1, 2]
    assert math.isclose(sum(t["weights"]), 1.0)  # not 1.10 (PROJECT.md §6)
    assert t["weights"] == t["weights"][::-1]


def test_default_rotation_gives_180_options():
    r = DEFAULT["rotation"]
    assert len(range(r["start_deg"], r["stop_deg"], r["step_deg"])) == 180


def test_translation_increment_is_not_configurable():
    assert "increment_m" not in DEFAULT["translation"]


def test_defaults_alone_are_refused_and_every_open_key_is_listed():
    with pytest.raises(ConfigError) as exc:
        RunSettings.from_config(load_config())
    message = str(exc.value)
    for key in ("[input] rgb", "[vineyards] source", "[evidence] method", "[template] line_length_m",
                "[translation] box", "[centres] spacing_m", "[scoring] normalisation", "[storage] surfaces"):
        assert key in message


def test_centroid_pretest_settings_build():
    s = RunSettings.from_config(load_config(PRETEST))
    assert s.input.rgb == PROJECT_ROOT / "dof" / "bordeaux" / "rgb.tif"
    assert s.input.nir_bands == ("nir", "red", "green")
    assert s.input.rgb_bands == ("red", "green", "blue")
    assert (s.input.nodata_value, s.input.nodata_rule) == (255, "any_band")
    assert s.spacing.values()[0] == 1.20
    assert len(s.rotation.angles()) == 180
    assert s.template.weights == (0.05, 0.20, 0.50, 0.20, 0.05)  # untouched defaults survive the merge


def test_bad_weights_are_refused():
    cfg = merge(load_config(PRETEST), {"template": {"weights": [0.10, 0.20, 0.50, 0.20, 0.10]}})
    with pytest.raises(ConfigError, match="sum to"):
        RunSettings.from_config(cfg)


def test_seed_lines_must_be_template_lines():
    cfg = merge(load_config(PRETEST), {"seeds": {"lines": [0, 3]}})
    with pytest.raises(ConfigError, match="seeds"):
        RunSettings.from_config(cfg)


def test_bad_choice_is_refused():
    cfg = merge(load_config(PRETEST), {"translation": {"box": "huge"}})
    with pytest.raises(ConfigError, match="choose one of"):
        RunSettings.from_config(cfg)


def test_later_override_wins(tmp_path):
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text("[spacing]\nmin_m = 1.0\nmax_m = 1.5\nstep_m = 0.1\n")
    b.write_text("[spacing]\nmax_m = 1.2\n")
    cfg = load_config(a, b)
    assert (cfg["spacing"]["min_m"], cfg["spacing"]["max_m"]) == (1.0, 1.2)
    assert cfg["spacing"]["nominal_m"] == 2.0  # from default.toml


def test_merge_does_not_modify_inputs():
    base = {"t": {"x": 1, "y": 2}}
    merged = merge(base, {"t": {"y": 3}})
    assert merged == {"t": {"x": 1, "y": 3}}
    assert base == {"t": {"x": 1, "y": 2}}


def test_variants_extend_the_pretest_config():
    base = RunSettings.from_config(load_config(PRETEST))
    cached = RunSettings.from_config(load_config(CONFIG_DIR / "centroid-pretest-cache.toml"))
    full = RunSettings.from_config(load_config(CONFIG_DIR / "full-search.toml"))
    assert base.centroid_pretest.enabled and base.kernels.cache_dir is None
    assert cached.centroid_pretest.enabled and cached.kernels.cache_dir == PROJECT_ROOT / ".cache" / "kernels"
    assert not full.centroid_pretest.enabled and full.kernels.cache_dir is None
    # Everything else is inherited unchanged.
    assert cached.input == base.input == full.input
    assert cached.spacing == base.spacing == full.spacing
    assert "extends" not in load_config(CONFIG_DIR / "centroid-pretest-cache.toml")


def test_extends_chain_is_relative_to_the_file_and_cycles_are_refused(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.toml").write_text("[spacing]\nmin_m = 1.0\nmax_m = 1.4\nstep_m = 0.2\n")
    (tmp_path / "sub" / "b.toml").write_text('extends = "../a.toml"\n[spacing]\nmax_m = 1.2\n')
    (tmp_path / "c.toml").write_text('extends = "sub/b.toml"\n[centres]\nspacing_m = 3.0\n')
    cfg = load_config(tmp_path / "c.toml")
    assert (cfg["spacing"]["min_m"], cfg["spacing"]["max_m"], cfg["centres"]["spacing_m"]) == (1.0, 1.2, 3.0)
    (tmp_path / "x.toml").write_text('extends = "y.toml"\n')
    (tmp_path / "y.toml").write_text('extends = "x.toml"\n')
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "x.toml")
