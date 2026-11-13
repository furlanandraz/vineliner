"""The row-spacing (row width) search range (PROJECT.md §6-7)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Spacings are rounded to the micrometre so that 1.80 + 3 * 0.02 is 1.86,
# not 1.8599999999999999.
_DECIMALS = 6
# How far (in steps) max_m may sit from the step grid and still count as on it.
_STEP_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SpacingRange:
    """Row-spacing hypotheses searched by every template.

    Spacing is the row width in the field sense: the distance between
    neighbouring rows, where the tractors drive. Rows are reduced to lines,
    so this is the line-to-line distance measured perpendicular to the rows.

    The range is fixed before a run and both endpoints are searched:
    min_m, min_m + step_m, ..., max_m.
    """

    min_m: float
    max_m: float
    step_m: float

    def __post_init__(self) -> None:
        for name in ("min_m", "max_m", "step_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"spacing {name} must be a finite number, got {value!r}")
        if self.min_m <= 0:
            raise ValueError(f"spacing min_m must be > 0, got {self.min_m}")
        if self.step_m <= 0:
            raise ValueError(f"spacing step_m must be > 0, got {self.step_m}")
        if self.max_m < self.min_m:
            raise ValueError(f"spacing max_m ({self.max_m}) is below min_m ({self.min_m})")
        steps = (self.max_m - self.min_m) / self.step_m
        if abs(steps - round(steps)) > _STEP_TOLERANCE:
            raise ValueError(
                f"spacing range {self.min_m}-{self.max_m} m is not a whole number of "
                f"{self.step_m} m steps, so max_m would never be searched; "
                "adjust max_m or step_m"
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SpacingRange:
        s = config["spacing"]
        return cls(min_m=s["min_m"], max_m=s["max_m"], step_m=s["step_m"])

    @property
    def count(self) -> int:
        return round((self.max_m - self.min_m) / self.step_m) + 1

    def values(self) -> tuple[float, ...]:
        """Every spacing to search, in metres, smallest first."""
        return tuple(round(self.min_m + j * self.step_m, _DECIMALS) for j in range(self.count))
