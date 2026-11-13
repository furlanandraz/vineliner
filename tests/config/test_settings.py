"""Packaged defaults, the dialog's row width range, and RunSettings validation."""

import math

import pytest

from vineliner.core.config import WIDTH_OPTIONS, ConfigError, RunSettings, defaults, merge, plugin_config


def test_defaults_are_the_final_bordeaux_configuration():
    s = RunSettings.from_config(plugin_config(1.2, 1.6))
    assert s.centroid_pretest.enabled and s.kernels.cache
    assert s.missing_row_estimate.enabled and s.missing_row_estimate.merge_pieces
    assert s.row_numbering.enabled and s.drop_hanging_rows and s.smoothing.enabled
    assert s.evidence.method == "black_tophat" and s.evidence.tophat_size_m == 1.4 and s.evidence.scale_region == "per_block"
    assert s.template.line_length_m == 2.0 and s.template.band_width_px == 3
    assert s.selection.method == "orientation_consensus" and s.seed_lines == (-1, 0, 1)
    assert (s.reconstruction.max_gap_m, s.reconstruction.join_distance_frac) == (50.0, 0.35)
    assert s.notes == ()  # nothing had to be adjusted


def test_template_defaults_match_project_md():
    t = defaults()["template"]
    assert t["indices"] == [-2, -1, 0, 1, 2]
    assert math.isclose(sum(t["weights"]), 1.0) and t["weights"] == t["weights"][::-1]
    assert len(RunSettings.from_config(plugin_config(1.8, 2.2)).rotation.angles()) == 180


def test_row_width_range_sets_the_searched_widths_and_the_spacing_check():
    s = RunSettings.from_config(plugin_config(1.2, 1.6))
    assert s.spacing.count == WIDTH_OPTIONS == 21
    assert s.spacing.values()[0] == 1.2 and s.spacing.values()[-1] == 1.6 and s.spacing.values()[1] == 1.22
    assert s.spacing_nominal_m == pytest.approx(1.4)
    assert (s.validation.spacing_min_m, s.validation.spacing_max_m) == (1.2, 1.6)
    single = RunSettings.from_config(plugin_config(2.0, 2.0))
    assert single.spacing.values() == (2.0,)
    for lo, hi in ((0.0, 1.0), (2.0, 1.5), (-1.0, 2.0)):
        with pytest.raises(ValueError, match="row width range"):
            plugin_config(lo, hi)


def test_evidence_weights_rescale_over_the_components_present():
    ev = RunSettings.from_config(plugin_config(1.8, 2.2)).evidence
    assert ev.weights_for(("rgb",)) == {"rgb": 1.0}
    assert ev.weights_for(("ndvi",)) == {"ndvi": 1.0}
    three = ev.weights_for(("rgb", "nir", "ndvi"))
    assert list(three) == ["rgb", "nir", "ndvi"] and all(w == pytest.approx(1 / 3) for w in three.values())
    no_nir = RunSettings.from_config(plugin_config(1.8, 2.2, {"evidence": {"nir_weight": 0.0}})).evidence
    assert no_nir.weights_for(("rgb", "nir", "ndvi")) == {"rgb": 0.5, "ndvi": 0.5}
    with pytest.raises(ConfigError, match="weight 0"):
        no_nir.weights_for(("nir",))


def test_bad_values_are_refused_with_every_problem_listed():
    with pytest.raises(ConfigError, match="sum to"):
        RunSettings.from_config(plugin_config(1.8, 2.2, {"template": {"weights": [0.1, 0.2, 0.5, 0.2, 0.1]}}))
    with pytest.raises(ConfigError, match="seeds"):
        RunSettings.from_config(plugin_config(1.8, 2.2, {"seeds": {"lines": [0, 3]}}))
    with pytest.raises(ConfigError, match="choose one of"):
        RunSettings.from_config(plugin_config(1.8, 2.2, {"translation": {"box": "huge"}}))
    with pytest.raises(ConfigError, match=r">= 0, and one of them > 0"):
        RunSettings.from_config(plugin_config(1.8, 2.2, {"evidence": {"rgb_weight": 0.0, "nir_weight": 0.0, "ndvi_weight": 0.0}}))
    broken = plugin_config(1.8, 2.2)
    del broken["scoring"]["normalisation"], broken["centres"]["spacing_m"]
    with pytest.raises(ConfigError) as exc:
        RunSettings.from_config(broken)
    assert "[scoring] normalisation" in str(exc.value) and "[centres] spacing_m" in str(exc.value)


def test_estimate_turns_hanging_row_deletion_on():
    s = RunSettings.from_config(plugin_config(1.8, 2.2, {"reconstruction": {"drop_hanging_rows": False}}))
    assert s.drop_hanging_rows and "drop_hanging_rows is turned on" in s.notes[0]


def test_merge_does_not_modify_inputs_and_defaults_are_fresh_copies():
    base = {"t": {"x": 1, "y": 2}}
    assert merge(base, {"t": {"y": 3}}) == {"t": {"x": 1, "y": 3}} and base == {"t": {"x": 1, "y": 2}}
    d = defaults()
    d["spacing"]["min_m"] = 99
    assert defaults()["spacing"]["min_m"] == 1.8
