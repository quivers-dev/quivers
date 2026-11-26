"""Work bounds on growing families, each with a control of the next order."""

from __future__ import annotations

import math

import pytest

from tests.evidence.compare import Mismatch
from tests.evidence.conformance import parameters, run
from tests.evidence.scaling import FAMILIES, REPEATED, accept


@pytest.mark.parametrize("name", parameters("work"))
def test_family(name: str) -> None:
    """Run one family and its control.

    Parameters
    ----------
    name : str
        The check of the family.
    """
    run(name)


def test_every_family_spans_a_factor_of_sixteen_in_five_sizes() -> None:
    """Require the shape of every family."""
    for family in (*FAMILIES, *REPEATED):
        assert len(family.sizes) >= 5, family.name
        assert max(family.sizes) >= 16 * min(family.sizes), family.name


def test_every_counted_part_is_covered() -> None:
    """Require a family for every part of a computation that is counted."""
    parts = {family.part for family in (*FAMILIES, *REPEATED)}
    assert parts >= {
        "construction",
        "validation",
        "forward",
        "backward",
        "saved",
        "provenance",
        "memory",
        "refit",
    }
    varies = {family.varies for family in FAMILIES}
    assert varies >= {
        "sparsity",
        "skew",
        "width",
        "cycles",
        "grammar",
        "batch",
        "accuracy",
    }


SIZES = (2, 4, 8, 16, 32)


@pytest.mark.parametrize(
    ("work", "constant", "reason"),
    [
        pytest.param([2.0 * n for n in SIZES], 1.5, "above", id="above-the-constant"),
        pytest.param([n * math.log(n) for n in SIZES], 10.0, "grow", id="a-log-factor"),
        pytest.param([1.0, 2.0, 4.0, 9.0, 16.0], 10.0, "grow", id="one-step-up"),
    ],
)
def test_accept_rejects(work: list[float], constant: float, reason: str) -> None:
    """Require the bound check to reject counters that break a linear bound.

    Parameters
    ----------
    work : list of float
        The counters.
    constant : float
        The stated constant.
    reason : str
        ``"above"`` or ``"grow"``.
    """
    with pytest.raises(Mismatch) as caught:
        accept(SIZES, work, [float(n) for n in SIZES], constant, "control")
    assert reason in str(caught.value)


def test_accept_takes_a_bound_that_holds() -> None:
    """Require the bound check to accept counters below a linear bound."""
    accept(SIZES, [n - 0.5 for n in SIZES], [float(n) for n in SIZES], 1.0, "linear")


def test_accept_refuses_too_few_sizes() -> None:
    """Require five sizes over a factor of sixteen."""
    with pytest.raises(Mismatch):
        accept((2, 4, 8, 16), [1.0] * 4, [1.0] * 4, 1.0, "short")
    with pytest.raises(Mismatch):
        accept((2, 3, 4, 5, 6), [1.0] * 5, [1.0] * 5, 1.0, "narrow")
