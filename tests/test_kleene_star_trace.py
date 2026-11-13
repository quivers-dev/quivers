"""Independent path oracles and boundary contracts for coproduct feedback."""

import itertools
import math

import pytest
import torch

from quivers.categorical import (
    EMPTY,
    DivergentClosureError,
    KleeneStarTrace,
    TracedMonoidal,
)
from quivers.core import (
    BOOLEAN,
    COUNTING,
    GODEL,
    LOG_PROB,
    MARKOV,
    MAX_PLUS,
    PRODUCT_FUZZY,
    REAL,
    Algebra,
    CoproductSet,
    FinSet,
    MarkovAlgebra,
    ObservedMorphism,
    Semiring,
    SetObject,
    observed,
)


SEMIRINGS = (BOOLEAN, COUNTING, MARKOV, LOG_PROB, MAX_PLUS)


def feedback_morphism(
    matrix: torch.Tensor, algebra: Algebra, a: SetObject, b: SetObject, u: SetObject
) -> ObservedMorphism:
    """Bind a matrix to the exact coproduct objects under test.

    Parameters
    ----------
    matrix
        Row-domain, column-codomain weights.
    algebra
        Declared carrier operations.
    a, b, u
        External domain, external codomain, and feedback object.

    Returns
    -------
    ObservedMorphism
        Matrix with its coproduct type recorded.
    """
    return observed(
        CoproductSet(components=(a, u)),
        CoproductSet(components=(b, u)),
        matrix,
        algebra=algebra,
    )


def simple_path_oracle(
    matrix: torch.Tensor, algebra: Algebra, a: int, b: int, n: int
) -> torch.Tensor:
    """Enumerate every path with distinct internal vertices.

    This is complete for acyclic counting graphs, Boolean reachability,
    and maximum weights when cycles have nonpositive total weight.

    Parameters
    ----------
    matrix
        Full matrix including external and feedback blocks.
    algebra
        Boolean, Counting or MaxPlus operations.
    a, b, n
        External row count, external column count, feedback count.

    Returns
    -------
    torch.Tensor
        Independently enumerated external path weights.
    """
    rows = []
    for i in range(a):
        row = []
        for j in range(b):
            paths = [float(matrix[i, j])]
            for length in range(1, n + 1):
                for vertices in itertools.permutations(range(n), length):
                    edges = [float(matrix[i, b + vertices[0]])]
                    edges.extend(
                        float(matrix[a + left, b + right])
                        for left, right in zip(vertices, vertices[1:])
                    )
                    edges.append(float(matrix[a + vertices[-1], j]))
                    paths.append(
                        sum(edges) if algebra is MAX_PLUS else math.prod(edges)
                    )
            value = (
                max(paths)
                if algebra is MAX_PLUS
                else (float(any(paths)) if algebra is BOOLEAN else math.fsum(paths))
            )
            row.append(value)
        rows.append(row)
    return torch.tensor(rows, dtype=matrix.dtype)


@pytest.mark.parametrize("algebra", (BOOLEAN, COUNTING, MAX_PLUS), ids=lambda q: q.name)
@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_trace_matches_independent_simple_path_enumeration(
    algebra: Semiring, order: tuple[int, ...]
) -> None:
    """Check weighted paths under every internal vertex order.

    Parameters
    ----------
    algebra
        Semiring whose finite path oracle is used.
    order
        Permutation of the three feedback vertices.
    """
    a = FinSet(name="A", cardinality=2)
    b = FinSet(name="B", cardinality=2)
    u = FinSet(name="U", cardinality=3)
    matrix = torch.tensor(
        [
            [0, 1, 2, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 2, 1],
            [0, 3, 0, 0, 3],
            [2, 1, 0, 0, 0],
        ],
        dtype=torch.float64,
    )
    if algebra is BOOLEAN:
        matrix = (matrix > 0).double()
        # A feedback cycle is harmless for reachability.
        matrix[4, 2] = 1.0
    elif algebra is MAX_PLUS:
        matrix = torch.where(matrix > 0, -matrix, torch.full_like(matrix, -torch.inf))
        matrix[4, 2] = -1.0
        matrix[2, 2] = 0.0  # Zero cycles are finite under maximum.
    permutation = [0, 1] + [2 + i for i in order]
    matrix = matrix[permutation][:, permutation]
    actual = (
        KleeneStarTrace(algebra)
        .trace(feedback_morphism(matrix, algebra, a, b, u), u, a, b)
        .tensor
    )
    expected = simple_path_oracle(matrix, algebra, 2, 2, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("algebra", (MARKOV, LOG_PROB), ids=lambda q: q.name)
def test_trace_matches_resolvent_and_gradient(algebra: Semiring) -> None:
    """Compare closure with a linear system over the exit weights.

    Parameters
    ----------
    algebra
        Linear or logarithmic representation of nonnegative weights.
    """
    a = FinSet(name="A", cardinality=2)
    b = FinSet(name="B", cardinality=3)
    u = FinSet(name="U", cardinality=2)
    generator = torch.Generator().manual_seed(61)
    weights = torch.rand(4, 5, generator=generator, dtype=torch.float64) / 8 + 0.01
    values = (weights.log() if algebra is LOG_PROB else weights).requires_grad_()
    actual = (
        KleeneStarTrace(algebra)
        .trace(feedback_morphism(values, algebra, a, b, u), u, a, b)
        .tensor
    )
    linear = values.exp() if algebra is LOG_PROB else values
    expected = linear[:2, :3] + linear[:2, 3:] @ torch.linalg.solve(
        torch.eye(2, dtype=torch.float64) - linear[2:, 3:], linear[2:, :3]
    )
    if algebra is LOG_PROB:
        expected = expected.log()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(actual.sum(), values, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), values)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-11)


def test_geometric_feedback_rejects_finite_unrolling() -> None:
    """A convergent nonzero loop requires the complete geometric sum."""
    obj = FinSet(name="X", cardinality=1)
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.5]], dtype=torch.float64)
    tracer = KleeneStarTrace(MARKOV)
    assert not isinstance(tracer, TracedMonoidal)
    result = tracer.trace(
        feedback_morphism(matrix, MARKOV, obj, obj, obj), obj, obj, obj
    )
    assert result.tensor.item() == 2.0
    wrong = sum(0.5**k for k in range(4))
    assert abs(result.tensor.item() - wrong) > 0.1


def test_log_trace_preserves_tiny_paths_and_gradients() -> None:
    """A finite log edge stays present without exponentiation."""
    obj = FinSet(name="X", cardinality=1)
    edge = torch.tensor(-1000.0, dtype=torch.float64, requires_grad=True)
    absent = torch.full_like(edge, -torch.inf)
    matrix = torch.stack((absent, edge, torch.zeros_like(edge), absent)).reshape(2, 2)
    result = (
        KleeneStarTrace(LOG_PROB)
        .trace(feedback_morphism(matrix, LOG_PROB, obj, obj, obj), obj, obj, obj)
        .tensor
    )
    assert result.item() == -1000.0
    assert torch.autograd.grad(result.sum(), edge)[0].item() == 1.0


@pytest.mark.parametrize(
    "algebra", (COUNTING, MARKOV, LOG_PROB, MAX_PLUS), ids=lambda q: q.name
)
def test_unreachable_divergent_feedback_is_undefined(algebra: Semiring) -> None:
    """The entire feedback closure must exist, even off external paths.

    Parameters
    ----------
    algebra
        Semiring with divergent feedback cycles.
    """
    obj = FinSet(name="X", cardinality=1)
    loop = 0.0 if algebra is LOG_PROB else 1.0
    matrix = torch.full((2, 2), algebra.zero, dtype=torch.float64)
    matrix[1, 1] = loop
    with pytest.raises(DivergentClosureError):
        KleeneStarTrace(algebra).trace(
            feedback_morphism(matrix, algebra, obj, obj, obj), obj, obj, obj
        )


@pytest.mark.parametrize("algebra", SEMIRINGS, ids=lambda q: q.name)
def test_empty_feedback_restores_multiaxis_external_shapes(algebra: Semiring) -> None:
    """Vanishing at the empty object preserves arbitrary external axes.

    Parameters
    ----------
    algebra
        Supported trace semiring.
    """
    a = FinSet(name="A1", cardinality=2) * FinSet(name="A2", cardinality=3)
    b = FinSet(name="B1", cardinality=2) * FinSet(name="B2", cardinality=2)
    matrix = torch.full((6, 4), algebra.unit, dtype=torch.float64)
    result = KleeneStarTrace(algebra).trace(
        feedback_morphism(matrix, algebra, a, b, EMPTY), EMPTY, a, b
    )
    assert result.domain == a and result.codomain == b
    torch.testing.assert_close(
        result.tensor, matrix.reshape(2, 3, 2, 2), rtol=0, atol=0
    )


@pytest.mark.parametrize("algebra", SEMIRINGS, ids=lambda q: q.name)
def test_empty_external_objects_have_empty_outputs(algebra: Semiring) -> None:
    """Defined feedback also admits the zero-sized external matrix.

    Parameters
    ----------
    algebra
        Supported trace semiring.
    """
    u = FinSet(name="U", cardinality=2)
    matrix = torch.full((2, 2), algebra.zero, dtype=torch.float64)
    result = KleeneStarTrace(algebra).trace(
        feedback_morphism(matrix, algebra, EMPTY, EMPTY, u), u, EMPTY, EMPTY
    )
    assert result.tensor.shape == (0, 0)


class _CustomMarkov(MarkovAlgebra):
    """A subtype outside the declared closure guarantee."""


@pytest.mark.parametrize(
    "algebra", (REAL, PRODUCT_FUZZY, GODEL, _CustomMarkov()), ids=lambda q: q.name
)
def test_trace_rejects_undeclared_algebras(algebra: Algebra) -> None:
    """Reject signed and unproved carriers at construction.

    Parameters
    ----------
    algebra
        Source outside the supported closure family.
    """
    with pytest.raises(TypeError, match="declared"):
        KleeneStarTrace(algebra)


@pytest.mark.parametrize("bad", (-1.0, float("nan"), float("inf")))
def test_trace_rejects_invalid_nonnegative_weights(bad: float) -> None:
    """Invalid carrier values cannot enter a feedback calculation.

    Parameters
    ----------
    bad
        Invalid nonnegative-carrier entry.
    """
    obj = FinSet(name="X", cardinality=1)
    matrix = torch.tensor([[bad, 0.0], [0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="carrier"):
        KleeneStarTrace(MARKOV).trace(
            feedback_morphism(matrix, MARKOV, obj, obj, obj), obj, obj, obj
        )


@pytest.mark.parametrize(
    "dtype", (torch.int64, torch.bool, torch.float16, torch.complex64)
)
def test_trace_requires_explicit_supported_precision(dtype: torch.dtype) -> None:
    """Avoid implicit casts that could change represented weights.

    Parameters
    ----------
    dtype
        Representation outside the float32/float64 contract.
    """
    obj = FinSet(name="X", cardinality=1)
    matrix = torch.zeros(2, 2, dtype=dtype)
    with pytest.raises(TypeError, match="float32 or float64"):
        KleeneStarTrace(MARKOV).trace(
            feedback_morphism(matrix, MARKOV, obj, obj, obj), obj, obj, obj
        )


def test_trace_checks_objects_and_algebra() -> None:
    """Equal cardinalities do not make differently named objects equal."""
    a = FinSet(name="A", cardinality=1)
    b = FinSet(name="B", cardinality=1)
    u = FinSet(name="U", cardinality=1)
    morph = feedback_morphism(torch.zeros(2, 2), MARKOV, a, b, u)
    with pytest.raises(ValueError, match="objects"):
        KleeneStarTrace(MARKOV).trace(morph, u, b, a)
    with pytest.raises(ValueError, match="same algebra"):
        KleeneStarTrace(COUNTING).trace(morph, u, a, b)
