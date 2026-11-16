"""Ordered observations keep their integer carrier at the reference boundary."""

import math

import pytest
import torch

from quivers.continuous._ordered import OrderedLogistic, OrderedProbit
from quivers.qiec.reference_families import DENSITIES
from quivers.qiec.reference_support import DistributionError


@pytest.mark.parametrize(
    "family,native,parameter",
    [
        ("OrderedLogistic", OrderedLogistic, "eta"),
        ("OrderedProbit", OrderedProbit, "eta"),
    ],
)
@pytest.mark.parametrize(
    "value", [-1, -0.5, 0.5, 1.5, 3, math.nan, math.inf, -math.inf, 2**80]
)
def test_invalid_categories_are_not_floored_or_clamped(
    family, native, parameter, value
):
    arguments = {parameter: 0.2, "cutpoints": (-0.4, 0.7)}
    assert DENSITIES[family](arguments, value) == -math.inf
    distribution = native(
        torch.tensor(0.2), torch.tensor([-0.4, 0.7]), validate_args=False
    )
    assert distribution.log_prob(torch.tensor(float(value))).item() == -math.inf


@pytest.mark.parametrize(
    "family,parameter", [("OrderedLogistic", "eta"), ("OrderedProbit", "eta")]
)
@pytest.mark.parametrize("value", [False, True, "0", None])
def test_noninteger_carriers_are_rejected(family, parameter, value):
    with pytest.raises(DistributionError):
        DENSITIES[family]({parameter: 0.2, "cutpoints": (-0.4, 0.7)}, value)


@pytest.mark.parametrize(
    "family,parameter", [("OrderedLogistic", "eta"), ("OrderedProbit", "eta")]
)
def test_integral_float_categories_preserve_the_normalized_law(family, parameter):
    arguments = {parameter: 0.2, "cutpoints": (-0.4, 0.7)}
    masses = [math.exp(DENSITIES[family](arguments, category)) for category in range(3)]
    assert sum(masses) == pytest.approx(1.0, abs=2e-15)
    for category in range(3):
        assert DENSITIES[family](arguments, float(category)) == DENSITIES[family](
            arguments, category
        )
