"""Exercise checked probe input types without requiring a target container."""

from pathlib import Path

import pytest

from tests.transpile.test_marginalize_numeric import (
    _ANALYTIC,
    _FIXTURE_SOURCES,
    _POINTS,
    _qvr_log_densities,
)


@pytest.mark.parametrize("name", tuple(_FIXTURE_SOURCES))
def test_marginal_probe_respects_checked_observation_types(
    name: str, tmp_path: Path
) -> None:
    points = _POINTS[name]()
    actual = _qvr_log_densities(_FIXTURE_SOURCES[name], name, points, tmp_path)
    expected = [_ANALYTIC[name](point.params, point.data) for point in points]
    assert actual == pytest.approx(expected, abs=1e-10)
