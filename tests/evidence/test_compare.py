"""The comparison rule rejects every kind of difference it names."""

from __future__ import annotations

import math

import pytest
import torch

from tests.evidence.compare import (
    Accuracy,
    Mismatch,
    compare_gradients,
    compare_number,
    compare_tensor,
    gamma,
)

EXACT = Accuracy(domain="linear", operations=1)


def test_the_rounding_factor_is_highams() -> None:
    """Compare the factor with its definition."""
    u = 2.0**-53
    assert gamma(10, u) == 10 * u / (1 - 10 * u)
    with pytest.raises(ValueError):
        gamma(2**53, u)


def test_tolerances_follow_their_derivation() -> None:
    """Require a log tolerance to scale with magnitude and condition."""
    small = Accuracy(domain="log", operations=100)
    large = small.with_(magnitude=50.0, condition=4.0)
    assert large.tolerance(0.0) == 200 * small.tolerance(0.0)
    assert EXACT.tolerance(4.0) == 4 * EXACT.tolerance(1.0)
    single = EXACT.with_(dtype="float32")
    assert single.tolerance(1.0) == gamma(1, 2.0**-24)


@pytest.mark.parametrize(
    ("value", "reference", "reason"),
    [
        (math.nan, math.nan, "not-a-number"),
        (1.0, math.nan, "not-a-number"),
        (math.inf, -math.inf, "infinite"),
        (-math.inf, -1e308, "infinite"),
        (1.0 + 1e-15, 1.0, "value"),
    ],
)
def test_numbers_that_differ_are_rejected(
    value: float, reference: float, reason: str
) -> None:
    """Require each kind of difference to be named.

    Parameters
    ----------
    value : float
        The computed number.
    reference : float
        The reference.
    reason : str
        The kind of difference.
    """
    with pytest.raises(Mismatch) as caught:
        compare_number(value, reference, EXACT, "number")
    assert caught.value.reason == reason


def test_equal_infinities_and_rounding_pass() -> None:
    """Require equal infinities and one rounding to pass."""
    compare_number(-math.inf, -math.inf, EXACT, "infinity")
    compare_number(1.0 + 2.0**-53, 1.0, EXACT, "rounding")


def test_tensors_compare_shape_type_and_entries() -> None:
    """Require shapes and types to be equal and every entry close."""
    reference = torch.zeros(2, dtype=torch.float64)
    for value, reason in (
        (torch.zeros(3, dtype=torch.float64), "shape"),
        (torch.zeros(2, dtype=torch.float32), "dtype"),
        (torch.tensor([0.0, 1e-300], dtype=torch.float64), "value"),
    ):
        with pytest.raises(Mismatch) as caught:
            compare_tensor(value, reference, EXACT, "tensor")
        assert caught.value.reason == reason


def test_a_missing_gradient_fails_even_where_the_reference_is_zero() -> None:
    """Require a missing derivative and a missing name to be rejected."""
    with pytest.raises(Mismatch) as caught:
        compare_gradients({"a": None}, {"a": 0.0}, EXACT, "gradient")
    assert caught.value.reason == "missing"
    with pytest.raises(Mismatch) as caught:
        compare_gradients({"a": torch.tensor(0.0)}, {"a": 0.0, "b": 0.0}, EXACT, "g")
    assert caught.value.reason == "support"
    compare_gradients({"a": torch.tensor(0.0)}, {"a": 0.0}, EXACT, "gradient")
