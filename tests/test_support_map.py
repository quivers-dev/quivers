"""Support preserves the declared algebra operations on their carriers."""

import itertools

import pytest
import torch

from quivers.core import (
    BOOLEAN,
    COUNTING,
    GODEL,
    LOG_PROB,
    LUKASIEWICZ,
    MARKOV,
    MAX_PLUS,
    PROBABILITY,
    PRODUCT_FUZZY,
    REAL,
    Algebra,
    AlgebraHomomorphism,
    BooleanAlgebra,
    FinSet,
    SupportMap,
    Threshold,
    identity,
    lookup_carrier_map,
    observed,
)


_SOURCES = (BOOLEAN, PRODUCT_FUZZY, GODEL, PROBABILITY, COUNTING, MARKOV)


def _grid(algebra: Algebra) -> torch.Tensor:
    """Return exact, finite carrier points including both identities.

    Parameters
    ----------
    algebra
        One of the supported source algebras.

    Returns
    -------
    torch.Tensor
        Carrier points in float64.
    """
    if algebra is BOOLEAN:
        values = [0.0, 1.0]
    elif algebra is COUNTING:
        values = [0.0, 1.0, 2.0, 4.0]
    elif algebra is MARKOV:
        values = [0.0, 0.125, 1.0, 2.0]
    else:
        values = [0.0, 0.125, 0.5, 1.0]
    return torch.tensor(values, dtype=torch.float64)


@pytest.mark.parametrize("algebra", _SOURCES, ids=lambda q: q.name)
def test_support_preserves_product_join_and_identities(algebra: Algebra) -> None:
    """Check every product pair and join triple on the carrier grid.

    Parameters
    ----------
    algebra
        Supported source whose operations are checked.
    """
    support = SupportMap(algebra)
    assert isinstance(support, AlgebraHomomorphism)
    assert support.source is algebra and support.target is BOOLEAN
    grid = _grid(algebra)
    left, right = torch.meshgrid(grid, grid, indexing="ij")
    expected_product = ((left > 0) & (right > 0)).double()
    torch.testing.assert_close(
        support.apply(algebra.tensor_op(left, right)), expected_product, rtol=0, atol=0
    )
    triples = torch.tensor(
        list(itertools.product(grid.tolist(), repeat=3)), dtype=grid.dtype
    )
    torch.testing.assert_close(
        support.apply(algebra.join(triples, dim=1)),
        (triples > 0).any(dim=1).double(),
        rtol=0,
        atol=0,
    )
    for value, expected in ((algebra.zero, 0.0), (algebra.unit, 1.0)):
        assert support.apply(torch.tensor(value)).item() == expected
    empty = torch.empty(3, 0, dtype=grid.dtype)
    torch.testing.assert_close(
        support.apply(algebra.join(empty, dim=1)),
        torch.zeros(3, dtype=grid.dtype),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("algebra", _SOURCES, ids=lambda q: q.name)
def test_support_preserves_matrix_composition_and_identity(algebra: Algebra) -> None:
    """Compare change of base with independent Boolean path existence.

    Parameters
    ----------
    algebra
        Supported algebra defining the binary matrix contraction.
    """
    a, b, c = (
        FinSet(name=name, cardinality=n) for name, n in (("A", 2), ("B", 3), ("C", 2))
    )
    left = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float64)
    right = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    f, g = observed(a, b, left, algebra=algebra), observed(b, c, right, algebra=algebra)
    support = SupportMap(algebra)
    expected = ((left[:, :, None] > 0) & (right[None, :, :] > 0)).any(dim=1).double()
    torch.testing.assert_close(
        (f >> g).change_base(support).tensor, expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        (f.change_base(support) >> g.change_base(support)).tensor,
        expected,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        identity(a, algebra=algebra).change_base(support).tensor,
        torch.eye(2),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("algebra", _SOURCES, ids=lambda q: q.name)
@pytest.mark.parametrize("bad", (-1.0, float("nan"), float("inf")))
def test_support_rejects_invalid_carrier_entries(algebra: Algebra, bad: float) -> None:
    """Reject invalid values instead of interpreting them as absence.

    Parameters
    ----------
    algebra
        Supported source carrier.
    bad
        Invalid entry to reject.
    """
    with pytest.raises(ValueError, match="carrier"):
        SupportMap(algebra).apply(torch.tensor([bad]))


@pytest.mark.parametrize(
    "algebra,bad",
    (
        (BOOLEAN, 0.5),
        (PRODUCT_FUZZY, 2.0),
        (GODEL, 2.0),
        (PROBABILITY, 2.0),
        (COUNTING, 0.5),
    ),
)
def test_support_checks_each_specific_carrier(algebra: Algebra, bad: float) -> None:
    """Enforce Boolean, interval and integer carrier restrictions.

    Parameters
    ----------
    algebra
        Carrier with a restriction beyond nonnegativity.
    bad
        Nonnegative value outside that carrier.
    """
    with pytest.raises(ValueError, match="carrier"):
        SupportMap(algebra).apply(torch.tensor([bad]))


@pytest.mark.parametrize(
    "algebra", (REAL, LUKASIEWICZ, LOG_PROB, MAX_PLUS), ids=lambda q: q.name
)
def test_support_refuses_undeclared_sources(algebra: Algebra) -> None:
    """Do not infer homomorphism laws from sampled values.

    Parameters
    ----------
    algebra
        Algebra for which literal positivity is not declared a homomorphism.
    """
    with pytest.raises(TypeError, match="declared homomorphism"):
        SupportMap(algebra)


class _CustomBoolean(BooleanAlgebra):
    """A custom subtype whose operations carry no built-in guarantee."""


def test_support_refuses_custom_subclasses_and_complex_values() -> None:
    """Exact source types and real entries delimit the promised laws."""
    with pytest.raises(TypeError, match="declared homomorphism"):
        SupportMap(_CustomBoolean())
    with pytest.raises(TypeError, match="real"):
        SupportMap(MARKOV).apply(torch.tensor([1.0 + 0.0j]))


def test_support_and_interior_threshold_have_different_laws() -> None:
    """An interior threshold fails product preservation on valid values."""
    value = torch.tensor(0.6, dtype=torch.float64)
    threshold = Threshold(0.5)
    support = SupportMap(PRODUCT_FUZZY)
    product = PRODUCT_FUZZY.tensor_op(value, value)
    assert (
        threshold.apply(value).item() == 1.0 and threshold.apply(product).item() == 0.0
    )
    assert support.apply(value).item() == support.apply(product).item() == 1.0
    assert support.apply(torch.tensor(1e-12)).item() == 1.0
    registered = lookup_carrier_map(PRODUCT_FUZZY, BOOLEAN)
    assert isinstance(registered, Threshold)
    assert registered.apply(torch.tensor([0.4, 0.6])).tolist() == [0.0, 1.0]


def test_standard_mv_has_zero_divisors_and_real_addition_cancels() -> None:
    """Independent counterexamples justify two excluded source algebras."""
    value = torch.tensor(0.4)
    assert LUKASIEWICZ.tensor_op(value, value).item() == 0.0
    assert bool(value > 0)
    assert REAL.join(torch.tensor([-1.0, 1.0]), dim=0).item() == 0.0


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.float64, torch.int64, torch.bool)
)
def test_support_preserves_shape_device_and_declared_dtype(dtype: torch.dtype) -> None:
    """Boolean outputs remain compatible with tensor-valued morphisms.

    Parameters
    ----------
    dtype
        Input tensor representation.
    """
    values = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    result = SupportMap(BOOLEAN).apply(values)
    assert result.shape == values.shape and result.device == values.device
    assert result.dtype == (dtype if values.is_floating_point() else torch.float32)
    assert result.tolist() == [[0.0, 1.0], [1.0, 0.0]]
