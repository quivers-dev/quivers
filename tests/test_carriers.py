"""Statistics carriers: construction, refusals, and claims."""

from __future__ import annotations

import pytest
import torch

from quivers.core import (
    ExpectationSemiring,
    ProductSemiring,
    SecondOrderExpectationSemiring,
)
from quivers.core.algebras import (
    COUNTING,
    LOG_PROB,
    MAX_PLUS,
    REAL,
    FreeSemimodule,
    MatrixSemiring,
)
from quivers.core.laws import TIMES_COMMUTATIVE, LawStatus


def test_construction_is_refused_without_a_commutative_scalar_base() -> None:
    """Require commutative scalar weights and vector statistics."""
    with pytest.raises(TypeError):
        ExpectationSemiring(FreeSemimodule(MatrixSemiring(COUNTING, 2), 2))
    with pytest.raises(TypeError):
        SecondOrderExpectationSemiring(MatrixSemiring(COUNTING, 2), 1, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ProductSemiring([COUNTING])
    with pytest.raises(TypeError):
        ProductSemiring([COUNTING, MatrixSemiring(COUNTING, 2)])
    with pytest.raises(ValueError):
        SecondOrderExpectationSemiring(REAL, -1, 1)


def test_claims_and_subjects() -> None:
    """Require commutativity claimed, proved over established bases."""
    carrier = ExpectationSemiring(FreeSemimodule(LOG_PROB, 2))
    assert TIMES_COMMUTATIVE in carrier.laws()
    assert carrier.claim(TIMES_COMMUTATIVE).status is LawStatus.PROVED
    assert carrier.subject.name.endswith("ExpectationSemiring(Expectation(LogProb^2))")
    assert ProductSemiring([LOG_PROB, MAX_PLUS]).element_shape == (2,)


def test_shapes_follow_the_index() -> None:
    """Require zero and unit to have the index shape, dtype and device asked for."""
    carrier = SecondOrderExpectationSemiring(REAL, 2, 3)
    like = torch.zeros(1, dtype=torch.float32)
    zero = carrier.zero_like((4,), like)
    unit = carrier.unit_like((4,), like)
    assert zero.shape == (4, 12) and zero.dtype == torch.float32
    assert bool((unit[..., 0] == 1).all()) and bool((unit[..., 1:] == 0).all())
    with pytest.raises(ValueError):
        carrier.join(torch.zeros(4, 3), 0)
