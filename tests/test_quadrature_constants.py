"""Shared quadrature tables retain Gaussian polynomial exactness."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from quivers import _quadrature_constants as rules


@pytest.mark.parametrize("order", [8, 16, 32])
def test_gaussian_rule_matches_independent_generator_and_polynomial_moments(order):
    nodes = np.asarray(getattr(rules, f"_NODES_{order}"))
    weights = np.asarray(getattr(rules, f"_WEIGHTS_{order}"))
    expected_nodes, expected_weights = leggauss(order)
    np.testing.assert_allclose(nodes, expected_nodes, rtol=0, atol=3e-15)
    np.testing.assert_allclose(weights, expected_weights, rtol=0, atol=3e-15)
    assert (weights > 0).all()
    for degree in range(2 * order):
        expected = 0 if degree % 2 else 2 / (degree + 1)
        assert weights @ nodes**degree == pytest.approx(
            expected, abs=order * np.finfo(float).eps
        )
