"""The six partial trace axioms for the declared coproduct matrix categories."""

import math

import pytest
import torch

from quivers.categorical import EMPTY, DivergentClosureError, KleeneStarTrace
from quivers.core import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
    MARKOV,
    MAX_PLUS,
    CoproductSet,
    FinSet,
    Semiring,
    SetObject,
    identity,
)
from tests.test_kleene_star_trace import SEMIRINGS, feedback_morphism


def weights(algebra: Semiring, rows: int, columns: int) -> torch.Tensor:
    """Construct deterministic carrier weights with bounded row sums.

    Parameters
    ----------
    algebra
        Semiring in which the matrix is interpreted.
    rows, columns
        Matrix dimensions.

    Returns
    -------
    torch.Tensor
        Double-precision matrix of carrier values.
    """
    values = (torch.arange(rows * columns).reshape(rows, columns) % 3 + 1).double()
    if algebra is BOOLEAN:
        return (values != 2).double()
    if algebra is COUNTING:
        return values
    if algebra is MAX_PLUS:
        return -values
    linear = values / (8 * max(rows, columns))
    return linear.log() if algebra is LOG_PROB else linear


def block_sum(
    left: torch.Tensor, right: torch.Tensor, algebra: Semiring
) -> torch.Tensor:
    """Form the coproduct matrix using the declared additive identity.

    Parameters
    ----------
    left, right
        Two matrices whose direct sum is required.
    algebra
        Semiring determining the absent edges.

    Returns
    -------
    torch.Tensor
        Block diagonal matrix.
    """
    result = torch.full(
        (left.shape[0] + right.shape[0], left.shape[1] + right.shape[1]),
        algebra.zero,
        dtype=torch.float64,
    )
    result[: left.shape[0], : left.shape[1]] = left
    result[left.shape[0] :, left.shape[1] :] = right
    return result


def feedback_matrix(algebra: Semiring, a: int, b: int, u: int) -> torch.Tensor:
    """Construct a matrix with finite feedback for the selected semiring.

    Parameters
    ----------
    algebra
        Semiring whose closure domain is respected.
    a, b, u
        External domain, external codomain, and feedback dimensions.

    Returns
    -------
    torch.Tensor
        Full matrix with acyclic Counting feedback and bounded other loops.
    """
    result = weights(algebra, a + u, b + u)
    if algebra is COUNTING:
        result[a:, b:] = torch.triu(result[a:, b:], diagonal=1)
    return result


def traced(
    matrix: torch.Tensor,
    algebra: Semiring,
    a: SetObject,
    b: SetObject,
    u: SetObject,
) -> torch.Tensor:
    """Evaluate one side of a trace axiom with explicit object types.

    Parameters
    ----------
    matrix
        Row-domain matrix with final feedback block.
    algebra
        Declared semiring.
    a, b, u
        External domain, external codomain, and feedback objects.

    Returns
    -------
    torch.Tensor
        External matrix obtained by feedback closure.
    """
    return (
        KleeneStarTrace(algebra)
        .trace(feedback_morphism(matrix, algebra, a, b, u), u, a, b)
        .tensor
    )


@pytest.mark.parametrize("algebra", SEMIRINGS, ids=lambda q: q.name)
def test_naturality_under_external_pre_and_postcomposition(algebra: Semiring) -> None:
    """External composition commutes with feedback.

    Parameters
    ----------
    algebra
        Supported trace semiring.
    """
    a, b, u, c, d = (
        FinSet(name=name, cardinality=n)
        for name, n in (("A", 2), ("B", 3), ("U", 2), ("C", 3), ("D", 2))
    )
    f = feedback_matrix(algebra, a.size, b.size, u.size)
    pre, post = weights(algebra, c.size, a.size), weights(algebra, b.size, d.size)
    unit = identity(u, algebra=algebra).tensor.double()
    enlarged = algebra.compose(
        algebra.compose(block_sum(pre, unit, algebra), f, 1),
        block_sum(post, unit, algebra),
        1,
    )
    lhs = traced(enlarged, algebra, c, d, u)
    rhs = algebra.compose(algebra.compose(pre, traced(f, algebra, a, b, u), 1), post, 1)
    torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("algebra", SEMIRINGS, ids=lambda q: q.name)
def test_dinaturality_moves_a_rectangular_map_around_feedback(
    algebra: Semiring,
) -> None:
    """Tracing f;(id+g) equals tracing (id+g);f for distinct feedback objects.

    Parameters
    ----------
    algebra
        Supported trace semiring.
    """
    a, b, u, v = (
        FinSet(name=name, cardinality=n)
        for name, n in (("A", 2), ("B", 3), ("U", 2), ("V", 3))
    )
    f = weights(algebra, a.size + u.size, b.size + v.size)
    g = weights(algebra, v.size, u.size)
    if algebra is COUNTING:
        f[a.size :, b.size :] = 0
        f[a.size, b.size + 1] = 2
        f[a.size, b.size + 2] = 1
        g.zero_()
        g[0, 0], g[1, 1], g[2, 1] = 1, 3, 2
    lhs_matrix = algebra.compose(
        f, block_sum(identity(b, algebra=algebra).tensor.double(), g, algebra), 1
    )
    rhs_matrix = algebra.compose(
        block_sum(identity(a, algebra=algebra).tensor.double(), g, algebra), f, 1
    )
    lhs = traced(lhs_matrix, algebra, a, b, u)
    rhs = traced(rhs_matrix, algebra, a, b, v)
    torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "algebra", (COUNTING, MARKOV, LOG_PROB, MAX_PLUS), ids=lambda q: q.name
)
def test_dinaturality_preserves_undefinedness(algebra: Semiring) -> None:
    """Sliding a rectangular map preserves a divergent feedback cycle.

    Parameters
    ----------
    algebra
        Semiring whose closure can diverge.
    """
    a, b, u = (FinSet(name=name, cardinality=1) for name in ("A", "B", "U"))
    v = FinSet(name="V", cardinality=2)
    f = torch.full((2, 3), algebra.zero, dtype=torch.float64)
    g = torch.full((2, 1), algebra.zero, dtype=torch.float64)
    f[1, 1] = 1.0 if algebra is MAX_PLUS else algebra.unit
    g[0, 0] = algebra.unit
    left = algebra.compose(
        f, block_sum(identity(b, algebra=algebra).tensor.double(), g, algebra), 1
    )
    right = algebra.compose(
        block_sum(identity(a, algebra=algebra).tensor.double(), g, algebra), f, 1
    )
    with pytest.raises(DivergentClosureError):
        traced(left, algebra, a, b, u)
    with pytest.raises(DivergentClosureError):
        traced(right, algebra, a, b, v)


@pytest.mark.parametrize("algebra", SEMIRINGS, ids=lambda q: q.name)
def test_vanishing_one_removes_empty_feedback(algebra: Semiring) -> None:
    """Trace over the monoidal unit leaves the original matrix.

    Parameters
    ----------
    algebra
        Supported trace semiring.
    """
    a, b = FinSet(name="A", cardinality=2), FinSet(name="B", cardinality=3)
    f = weights(algebra, a.size, b.size)
    torch.testing.assert_close(traced(f, algebra, a, b, EMPTY), f, rtol=0, atol=0)


@pytest.mark.parametrize("algebra", SEMIRINGS, ids=lambda q: q.name)
def test_vanishing_two_agrees_in_both_feedback_orders(algebra: Semiring) -> None:
    """Simultaneous and either order of nested feedback agree when defined.

    Parameters
    ----------
    algebra
        Supported trace semiring.
    """
    a, b, u, v = (
        FinSet(name=name, cardinality=n)
        for name, n in (("A", 2), ("B", 3), ("U", 2), ("V", 1))
    )
    uv = CoproductSet(components=(u, v))
    au, bu = CoproductSet(components=(a, u)), CoproductSet(components=(b, u))
    f = feedback_matrix(algebra, a.size, b.size, uv.size)
    simultaneous = traced(f, algebra, a, b, uv)
    inner = traced(f, algebra, au, bu, v)
    nested = traced(inner, algebra, a, b, u)
    torch.testing.assert_close(nested, simultaneous, rtol=1e-12, atol=1e-12)
    rows = (
        list(range(a.size)) + [a.size + u.size] + list(range(a.size, a.size + u.size))
    )
    columns = (
        list(range(b.size)) + [b.size + u.size] + list(range(b.size, b.size + u.size))
    )
    av, bv = CoproductSet(components=(a, v)), CoproductSet(components=(b, v))
    reversed_inner = traced(f[rows][:, columns], algebra, av, bv, u)
    reversed_nested = traced(reversed_inner, algebra, a, b, v)
    torch.testing.assert_close(reversed_nested, simultaneous, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "algebra", (COUNTING, MARKOV, LOG_PROB, MAX_PLUS), ids=lambda q: q.name
)
@pytest.mark.parametrize("inner_diverges", (False, True))
def test_vanishing_two_preserves_undefinedness(
    algebra: Semiring, inner_diverges: bool
) -> None:
    """An inner or cross-block divergent cycle makes the full trace undefined.

    Parameters
    ----------
    algebra
        Semiring with partial closure.
    inner_diverges
        Whether divergence already occurs inside the last feedback block.
    """
    a, b, u, v = (FinSet(name=name, cardinality=1) for name in ("A", "B", "U", "V"))
    uv = CoproductSet(components=(u, v))
    au, bu = CoproductSet(components=(a, u)), CoproductSet(components=(b, u))
    f = torch.full((3, 3), algebra.zero, dtype=torch.float64)
    if inner_diverges:
        f[2, 2] = 0.0 if algebra is LOG_PROB else 1.0
    else:
        forward, backward = (2.0, 0.5) if algebra in (MARKOV, LOG_PROB) else (1.0, 1.0)
        if algebra is LOG_PROB:
            forward, backward = math.log(forward), math.log(backward)
        elif algebra is MAX_PLUS:
            backward = 0.0
        f[1, 2], f[2, 1] = forward, backward
    with pytest.raises(DivergentClosureError):
        traced(f, algebra, a, b, uv)
    if inner_diverges:
        with pytest.raises(DivergentClosureError):
            traced(f, algebra, au, bu, v)
    else:
        inner = traced(f, algebra, au, bu, v)
        with pytest.raises(DivergentClosureError):
            traced(inner, algebra, a, b, u)


@pytest.mark.parametrize("algebra", SEMIRINGS, ids=lambda q: q.name)
def test_superposing_an_independent_matrix(algebra: Semiring) -> None:
    """A disjoint external matrix is unaffected by feedback.

    Parameters
    ----------
    algebra
        Supported trace semiring.
    """
    a, b, u, c, d = (
        FinSet(name=name, cardinality=n)
        for name, n in (("A", 2), ("B", 3), ("U", 2), ("C", 3), ("D", 2))
    )
    f, g = (
        feedback_matrix(algebra, a.size, b.size, u.size),
        weights(algebra, c.size, d.size),
    )
    ca, db = CoproductSet(components=(c, a)), CoproductSet(components=(d, b))
    lhs = traced(block_sum(g, f, algebra), algebra, ca, db, u)
    rhs = block_sum(g, traced(f, algebra, a, b, u), algebra)
    torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("algebra", SEMIRINGS, ids=lambda q: q.name)
@pytest.mark.parametrize("multiaxis", (False, True))
def test_yanking_the_coproduct_symmetry(algebra: Semiring, multiaxis: bool) -> None:
    """Feedback through the symmetry is the identity on the external object.

    Parameters
    ----------
    algebra
        Supported trace semiring.
    multiaxis
        Whether the external object has two tensor axes.
    """
    u = FinSet(name="U", cardinality=2)
    if multiaxis:
        u = u * FinSet(name="V", cardinality=3)
    tracer = KleeneStarTrace(algebra)
    swap = tracer.monoidal.braiding(u, u)
    actual = tracer.trace(swap, u, u, u)
    torch.testing.assert_close(
        actual.tensor, identity(u, algebra=algebra).tensor, rtol=0, atol=0
    )
    assert tracer.verify_yanking(u)
