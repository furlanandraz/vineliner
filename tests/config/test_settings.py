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
    for key in ("[input] rgb, nir or ndvi", "[vineyards] source", "[template] line_length_m",
                "[translation] box", "[centres] spacing_m", "[scoring] normalisation", "[storage] surfaces"):
        assert key in message


def test_rgb_component_needs_its_method():
    cfg = load_config(PRETEST)
    del cfg["evidence"]["method"]
    with pytest.raises(ConfigError, match=r"\[evidence\] method"):
        RunSettings.from_config(cfg)


def test_layers_can_be_switched_off_without_other_changes():
    cfg = merge(load_config(PRETEST), {"evidence": {"ndvi_weight": 1.0}})  # ndvi_method is already set
    s = RunSettings.from_config(cfg)
    assert s.input.layers == ("rgb", "nir", "ndvi") and s.evidence.weights == {"rgb": 0.5, "ndvi": 0.5}
    off = RunSettings.from_config(merge(cfg, {"input": {"use_rgb": False}}))
    assert off.input.rgb is None and off.input.layers == ("nir", "ndvi") and off.input.switched_off == ("rgb",)
    assert off.evidence.weights == {"ndvi": 1.0}  # weights are rescaled over the layers in use
    assert off.evidence.method == "black_tophat"  # set in the file, unused now
    assert any("switched off: rgb" in n and "ndvi 1.00" in n for n in off.notes)


def test_weights_are_relative_and_only_layers_in_use_need_a_method():
    cfg = merge(load_config(PRETEST), {"evidence": {"rgb_weight": 1.0, "nir_weight": 1.0, "nir_method": "stretch", "ndvi_weight": 2.0}})
    s = RunSettings.from_config(cfg)
    assert s.evidence.weights == {"rgb": 0.25, "nir": 0.25, "ndvi": 0.5}
    no_nir_method = load_config(PRETEST)
    no_nir_method["evidence"]["nir_weight"] = 1.0
    with pytest.raises(ConfigError, match=r"\[evidence\] nir_method"):
        RunSettings.from_config(no_nir_method)
    no_nir_method["input"]["use_nir"] = False  # off: its weight and missing method no longer matter
    assert RunSettings.from_config(no_nir_method).evidence.weights == {"rgb": 1.0}


def test_bad_layer_combinations_are_refused():
    base = load_config(PRETEST)
    none_on = merge(base, {"input": {"use_rgb": False, "use_nir": False, "use_ndvi": False}})
    with pytest.raises(ConfigError, match=r"at least one input layer"):
        RunSettings.from_config(none_on)
    only_ndvi_without_weight = merge(base, {"input": {"use_rgb": False, "use_nir": False}})
    with pytest.raises(ConfigError, match=r"every layer in use \(ndvi\) has weight 0"):
        RunSettings.from_config(only_ndvi_without_weight)
    nir_without_nir_band = merge(base, {"input": {"nir_bands": ["a", "red", "green"]},
                                        "evidence": {"nir_weight": 1.0, "nir_method": "stretch"}})
    with pytest.raises(ConfigError, match=r"band named 'nir'"):
        RunSettings.from_config(nir_without_nir_band)
    with pytest.raises(ConfigError, match=r">= 0"):
        RunSettings.from_config(merge(base, {"evidence": {"ndvi_weight": -1.0}}))


def test_ndvi_alone_needs_no_rgb_settings(tmp_path):
    cfg = load_config(PRETEST)
    for key in ("rgb", "rgb_bands", "nir", "nir_bands"):
        del cfg["input"][key]
    del cfg["evidence"]["method"], cfg["evidence"]["tophat_size_m"]
    cfg = merge(cfg, {"evidence": {"ndvi_weight": 0.3}, "export": {"runs_dir": str(tmp_path / "area")}})
    s = RunSettings.from_config(cfg)
    assert s.input.layers == ("ndvi",) and s.input.rgb_bands is None and s.evidence.weights == {"ndvi": 1.0}
    assert s.evidence.method is None and s.input.switched_off == ()
    assert s.export.runs_dir == tmp_path / "area"


def test_runs_dir_defaults_to_the_project_runs_folder():
    assert RunSettings.from_config(load_config(PRETEST)).export.runs_dir == PROJECT_ROOT / "runs"


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
