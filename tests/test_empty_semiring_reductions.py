"""Empty finite reductions use the algebra's declared identities."""

import pytest
import torch

from quivers.categorical import EMPTY
from quivers.core import BOOLEAN, GODEL, LOG_PROB, MAX_PLUS, Algebra, FinSet, observed


@pytest.mark.parametrize(
    "algebra", (BOOLEAN, GODEL, LOG_PROB, MAX_PLUS), ids=lambda q: q.name
)
@pytest.mark.parametrize("dims", ((1, 3), (-3, -1)))
def test_empty_join_returns_zero_with_shape_and_gradient(
    algebra: Algebra, dims: tuple[int, ...]
) -> None:
    """Reduce empty axes to the additive identity, with zero input gradient.

    Parameters
    ----------
    algebra
        Source semiring.
    dims
        Positive or negative spellings of the two empty axes.
    """
    values = torch.empty(2, 0, 3, 0, dtype=torch.float64, requires_grad=True)
    result = algebra.join(values, dim=dims)
    expected = torch.full((2, 3), algebra.zero, dtype=values.dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.sum().backward()
    assert values.grad is not None and values.grad.shape == values.shape
    assert bool(torch.isfinite(values.grad).all())


@pytest.mark.parametrize("algebra", (BOOLEAN, GODEL), ids=lambda q: q.name)
def test_empty_meet_returns_top(algebra: Algebra) -> None:
    """The meet of no lattice elements is the greatest element.

    Parameters
    ----------
    algebra
        Bounded lattice defining the meet.
    """
    values = torch.empty(2, 0, dtype=torch.float64)
    torch.testing.assert_close(
        algebra.meet(values, dim=1), torch.ones(2, dtype=values.dtype), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "algebra", (BOOLEAN, GODEL, LOG_PROB, MAX_PLUS), ids=lambda q: q.name
)
def test_composition_through_empty_object_has_no_paths(algebra: Algebra) -> None:
    """No intermediate element yields the zero morphism.

    Parameters
    ----------
    algebra
        Semiring defining matrix composition.
    """
    a = FinSet(name="A", cardinality=2)
    empty = EMPTY
    b = FinSet(name="B", cardinality=3)
    f = observed(a, empty, torch.empty(2, 0), algebra=algebra)
    g = observed(empty, b, torch.empty(0, 3), algebra=algebra)
    torch.testing.assert_close(
        (f >> g).tensor, torch.full((2, 3), algebra.zero), rtol=0, atol=0
    )
