"""SpacingRange: the row-spacing (row width) search range."""

import pytest

from vineliner.core.config import SpacingRange, defaults, plugin_config


def test_default_spacing_range():
    spacings = SpacingRange.from_config(defaults()).values()
    assert len(spacings) == 21
    assert spacings[0] == 1.80
    assert spacings[3] == 1.86  # rounded, no float noise
    assert spacings[-1] == 2.20  # both ends searched


def test_dialog_range_spacings():
    spacings = SpacingRange.from_config(plugin_config(1.2, 1.6)).values()
    assert (spacings[0], spacings[-1], len(spacings)) == (1.20, 1.60, 21)
    spacings = SpacingRange.from_config(plugin_config(2.4, 2.8)).values()  # Cremons
    assert (spacings[0], spacings[1], spacings[-1]) == (2.40, 2.42, 2.80)


def test_single_spacing():
    assert SpacingRange(1.5, 1.5, 0.02).values() == (1.5,)


def test_integer_values_accepted():
    assert SpacingRange(1, 2, 1).values() == (1, 2)


@pytest.mark.parametrize(
    ("min_m", "max_m", "step_m", "message"),
    [
        (1.80, 2.21, 0.02, "whole number"),  # 2.21 is not on the step grid
        (2.20, 1.80, 0.02, "below min_m"),
        (1.80, 2.20, 0.0, "step_m must be > 0"),
        (1.80, 2.20, -0.02, "step_m must be > 0"),
        (0.0, 2.20, 0.02, "min_m must be > 0"),
        (1.80, float("nan"), 0.02, "finite"),
        (1.80, "2.20", 0.02, "finite"),
        (True, 2.20, 0.02, "finite"),
    ],
)
def test_invalid_ranges_are_refused(min_m, max_m, step_m, message):
    with pytest.raises(ValueError, match=message):
        SpacingRange(min_m, max_m, step_m)
