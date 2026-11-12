"""The composition laws of every shipped algebra, against independent statements.

Each shipped algebra is paired with a reference ``(⊕, ⊗, 0, 1)`` written
here in plain Python floats, independently of the library. The tests
check that

1. the library's ``join`` and ``tensor_op`` agree with the reference
   operations on a grid of carrier values;
2. every algebra that claims to be a `Semiring` satisfies the
   commutative-semiring laws (Golan 1999) on that grid, and every
   `IdempotentSemiring` also satisfies ``a ⊕ a = a``;
3. composition under a `Semiring` agrees with the reference matrix
   product and is associative and unital on random matrices;
4. every `Algebra` composes unitally with its identity tensor; and
5. the algebras that do not claim to be semirings really fail a
   semiring law, with an explicit counterexample, so that the
   classification rejects what it should.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import pytest
import torch

from quivers.core.algebras import (
    BOOLEAN,
    BOOLEAN_DUAL,
    COUNTING,
    DUAL_GODEL,
    DUAL_LUKASIEWICZ,
    GODEL,
    LOG_PROB,
    LUKASIEWICZ,
    MARKOV,
    MAX_PLUS,
    PROBABILITY,
    PRODUCT_FUZZY,
    REAL,
    REICHENBACH,
    TROPICAL,
    Algebra,
    IdempotentSemiring,
    Magmoid,
    Semigroupoid,
    Semiring,
    material_implication,
)

_Binary = Callable[[float, float], float]


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi = max(a, b)
    return hi + math.log(math.exp(a - hi) + math.exp(b - hi))


def _noisy_or(a: float, b: float) -> float:
    return a + b - a * b


def _luk_tnorm(a: float, b: float) -> float:
    return max(a + b - 1.0, 0.0)


def _bounded_sum(a: float, b: float) -> float:
    return min(1.0, a + b)


#: name -> (algebra, reference ⊕, reference ⊗, reference 0, reference 1, grid)
_REFERENCE: dict[str, tuple[Algebra, _Binary, _Binary, float, float, list[float]]] = {
    "Boolean": (BOOLEAN, max, lambda a, b: a * b, 0.0, 1.0, [0.0, 1.0]),
    "Godel": (GODEL, max, min, 0.0, 1.0, [0.0, 0.25, 0.5, 1.0]),
    "Tropical": (
        TROPICAL,
        min,
        lambda a, b: a + b,
        math.inf,
        0.0,
        [0.0, 1.0, 2.5, math.inf],
    ),
    "MaxPlus": (
        MAX_PLUS,
        max,
        lambda a, b: a + b,
        -math.inf,
        0.0,
        [-math.inf, -1.0, 0.0, 2.0],
    ),
    "LogProb": (
        LOG_PROB,
        _logaddexp,
        lambda a, b: a + b,
        -math.inf,
        0.0,
        [-math.inf, -1.0, 0.0, 1.5],
    ),
    "Real": (
        REAL,
        lambda a, b: a + b,
        lambda a, b: a * b,
        0.0,
        1.0,
        [-2.0, 0.0, 1.0, 3.0],
    ),
    "Counting": (
        COUNTING,
        lambda a, b: a + b,
        lambda a, b: a * b,
        0.0,
        1.0,
        [0.0, 1.0, 2.0, 3.0],
    ),
    "Markov": (
        MARKOV,
        lambda a, b: a + b,
        lambda a, b: a * b,
        0.0,
        1.0,
        [0.0, 0.5, 1.0, 2.0],
    ),
    "BooleanDual": (BOOLEAN_DUAL, min, max, 1.0, 0.0, [0.0, 1.0]),
    "GodelDual": (DUAL_GODEL, min, max, 1.0, 0.0, [0.0, 0.25, 0.5, 1.0]),
    "ProductFuzzy": (
        PRODUCT_FUZZY,
        _noisy_or,
        lambda a, b: a * b,
        0.0,
        1.0,
        [0.0, 0.25, 0.5, 0.75],
    ),
    "Lukasiewicz": (
        LUKASIEWICZ,
        _bounded_sum,
        _luk_tnorm,
        0.0,
        1.0,
        [0.0, 0.25, 0.5, 0.75, 1.0],
    ),
    "Probability": (
        PROBABILITY,
        _bounded_sum,
        lambda a, b: a * b,
        0.0,
        1.0,
        [0.0, 0.25, 0.5, 1.0],
    ),
    "Reichenbach": (
        REICHENBACH,
        lambda a, b: a * b,
        _noisy_or,
        1.0,
        0.0,
        [0.0, 0.25, 0.5, 0.75],
    ),
    "LukasiewiczDual": (
        DUAL_LUKASIEWICZ,
        _luk_tnorm,
        _bounded_sum,
        1.0,
        0.0,
        [0.0, 0.25, 0.5, 0.75, 1.0],
    ),
}

_SEMIRINGS = [
    "Boolean",
    "Godel",
    "Tropical",
    "MaxPlus",
    "LogProb",
    "Real",
    "Counting",
    "Markov",
    "BooleanDual",
    "GodelDual",
]
_IDEMPOTENT = ["Boolean", "Godel", "Tropical", "MaxPlus", "BooleanDual", "GodelDual"]
_NON_SEMIRINGS = [
    "ProductFuzzy",
    "Lukasiewicz",
    "Probability",
    "Reichenbach",
    "LukasiewiczDual",
]

_ATOL = 1e-9


def _eq(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= _ATOL * max(1.0, abs(a), abs(b))


def _lib_plus(alg: Algebra, a: float, b: float) -> float:
    t = torch.tensor([a, b], dtype=torch.float64)
    return float(alg.join(t, dim=0))


def _lib_times(alg: Algebra, a: float, b: float) -> float:
    return float(
        alg.tensor_op(
            torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
        )
    )


def test_every_shipped_algebra_is_classified() -> None:
    assert sorted(_SEMIRINGS + _NON_SEMIRINGS) == sorted(_REFERENCE)


@pytest.mark.parametrize("name", sorted(_REFERENCE))
def test_library_operations_match_the_reference(name: str) -> None:
    alg, plus, times, zero, one, grid = _REFERENCE[name]
    assert _eq(alg.zero, zero)
    assert _eq(alg.unit, one)
    for a, b in itertools.product(grid, repeat=2):
        assert _eq(_lib_plus(alg, a, b), plus(a, b)), (a, b)
        assert _eq(_lib_times(alg, a, b), times(a, b)), (a, b)


def _semiring_law_failures(
    plus: _Binary, times: _Binary, zero: float, one: float, grid: list[float]
) -> list[str]:
    """Every commutative-semiring law that fails on ``grid``."""
    failures: list[str] = []
    for a in grid:
        if not _eq(plus(a, zero), a):
            failures.append(f"additive identity at {a}")
        if not _eq(times(a, one), a) or not _eq(times(one, a), a):
            failures.append(f"multiplicative identity at {a}")
        if not _eq(times(a, zero), zero) or not _eq(times(zero, a), zero):
            failures.append(f"annihilation at {a}")
    for a, b in itertools.product(grid, repeat=2):
        if not _eq(plus(a, b), plus(b, a)):
            failures.append(f"additive commutativity at {(a, b)}")
        if not _eq(times(a, b), times(b, a)):
            failures.append(f"multiplicative commutativity at {(a, b)}")
    for a, b, c in itertools.product(grid, repeat=3):
        if not _eq(plus(plus(a, b), c), plus(a, plus(b, c))):
            failures.append(f"additive associativity at {(a, b, c)}")
        if not _eq(times(times(a, b), c), times(a, times(b, c))):
            failures.append(f"multiplicative associativity at {(a, b, c)}")
        if not _eq(times(a, plus(b, c)), plus(times(a, b), times(a, c))):
            failures.append(f"distributivity at {(a, b, c)}")
    return failures


@pytest.mark.parametrize("name", _SEMIRINGS)
def test_semirings_satisfy_the_semiring_laws(name: str) -> None:
    alg, plus, times, zero, one, grid = _REFERENCE[name]
    assert isinstance(alg, Semiring)
    assert _semiring_law_failures(plus, times, zero, one, grid) == []


@pytest.mark.parametrize("name", _IDEMPOTENT)
def test_idempotent_semirings_have_idempotent_addition(name: str) -> None:
    alg, plus, _, _, _, grid = _REFERENCE[name]
    assert isinstance(alg, IdempotentSemiring)
    for a in grid:
        assert _eq(plus(a, a), a)


@pytest.mark.parametrize("name", [n for n in _SEMIRINGS if n not in _IDEMPOTENT])
def test_non_idempotent_semirings_are_not_declared_idempotent(name: str) -> None:
    alg, plus, _, _, _, grid = _REFERENCE[name]
    assert not isinstance(alg, IdempotentSemiring)
    assert any(not _eq(plus(a, a), a) for a in grid)


@pytest.mark.parametrize("name", _NON_SEMIRINGS)
def test_non_semirings_fail_a_semiring_law_and_do_not_claim_one(name: str) -> None:
    alg, plus, times, zero, one, grid = _REFERENCE[name]
    assert not isinstance(alg, Semigroupoid)
    failures = _semiring_law_failures(plus, times, zero, one, grid)
    assert any(f.startswith("distributivity") for f in failures), failures


def _sample(name: str, shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    grid = _REFERENCE[name][5]
    idx = torch.randint(len(grid), shape, generator=gen)
    return torch.tensor(grid, dtype=torch.float64)[idx]


def _reference_compose(name: str, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    _, plus, times, zero, _, _ = _REFERENCE[name]
    n, m = f.shape
    _, p = g.shape
    out = torch.empty(n, p, dtype=torch.float64)
    for i in range(n):
        for k in range(p):
            acc = zero
            for j in range(m):
                acc = plus(acc, times(float(f[i, j]), float(g[j, k])))
            out[i, k] = acc
    return out


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.allclose(a, b, atol=1e-9, rtol=1e-9, equal_nan=False))


@pytest.mark.parametrize("name", _SEMIRINGS)
def test_semiring_composition_is_the_reference_matrix_product(name: str) -> None:
    alg = _REFERENCE[name][0]
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        f = _sample(name, (3, 4), gen)
        g = _sample(name, (4, 2), gen)
        assert _close(alg.compose(f, g, 1), _reference_compose(name, f, g))


@pytest.mark.parametrize("name", _SEMIRINGS)
def test_semiring_composition_is_associative(name: str) -> None:
    alg = _REFERENCE[name][0]
    gen = torch.Generator().manual_seed(1)
    for _ in range(20):
        f = _sample(name, (2, 3), gen)
        g = _sample(name, (3, 3), gen)
        h = _sample(name, (3, 2), gen)
        left = alg.compose(alg.compose(f, g, 1), h, 1)
        right = alg.compose(f, alg.compose(g, h, 1), 1)
        assert _close(left, right)


@pytest.mark.parametrize("name", sorted(_REFERENCE))
def test_every_algebra_composes_unitally(name: str) -> None:
    alg = _REFERENCE[name][0]
    gen = torch.Generator().manual_seed(2)
    ident3 = alg.identity_tensor((3,)).to(torch.float64)
    ident4 = alg.identity_tensor((4,)).to(torch.float64)
    for _ in range(20):
        f = _sample(name, (3, 4), gen)
        assert torch.allclose(alg.compose(ident3, f, 1), f, atol=1e-6)
        assert torch.allclose(alg.compose(f, ident4, 1), f, atol=1e-6)


@pytest.mark.parametrize("name", _NON_SEMIRINGS)
def test_non_semiring_composition_depends_on_bracketing(name: str) -> None:
    """A concrete triple whose two bracketings differ."""
    alg = _REFERENCE[name][0]
    gen = torch.Generator().manual_seed(3)
    worst = 0.0
    for _ in range(50):
        f = _sample(name, (2, 3), gen)
        g = _sample(name, (3, 3), gen)
        h = _sample(name, (3, 2), gen)
        left = alg.compose(alg.compose(f, g, 1), h, 1)
        right = alg.compose(f, alg.compose(g, h, 1), 1)
        worst = max(worst, float((left - right).abs().max()))
    assert worst > 1e-2


def test_material_implication_is_a_magmoid_that_is_not_associative() -> None:
    rule = material_implication()
    assert isinstance(rule, Magmoid)
    assert not isinstance(rule, Semigroupoid)
    f = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    g = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64)
    h = torch.tensor([[0.5], [0.5]], dtype=torch.float64)
    left = rule.compose(rule.compose(f, g, 1), h, 1)
    right = rule.compose(f, rule.compose(g, h, 1), 1)
    assert not torch.allclose(left, right, atol=1e-3)


def test_product_fuzzy_zero_is_neutral_for_its_join() -> None:
    """The noisy-OR of many zeros is exactly zero, so the identity law
    holds on large objects and is not eroded by a clamping margin."""
    zeros = torch.zeros(4096)
    assert float(PRODUCT_FUZZY.join(zeros, dim=0)) == 0.0
    n = 512
    eye = torch.eye(n)
    composed = PRODUCT_FUZZY.compose(eye, eye, 1)
    off_diagonal = composed[~torch.eye(n, dtype=torch.bool)]
    assert float(off_diagonal.abs().max()) == 0.0


def test_dual_algebra_composes_matrices() -> None:
    """The Reichenbach dual composes ``∏_j (f + g - f g)``."""
    gen = torch.Generator().manual_seed(4)
    f = torch.rand(2, 3, generator=gen, dtype=torch.float64)
    g = torch.rand(3, 4, generator=gen, dtype=torch.float64)
    expected = torch.empty(2, 4, dtype=torch.float64)
    for i in range(2):
        for k in range(4):
            expected[i, k] = math.prod(
                _noisy_or(float(f[i, j]), float(g[j, k])) for j in range(3)
            )
    assert _close(REICHENBACH.compose(f, g, 1), expected)


def _lib_meet(alg: Algebra, a: float, b: float) -> float:
    return float(alg.meet(torch.tensor([a, b], dtype=torch.float64), dim=0))


@pytest.mark.parametrize("name", _IDEMPOTENT)
def test_idempotent_semirings_are_bounded_lattices(name: str) -> None:
    """``meet`` is the infimum and ``top`` the greatest element of the
    order ``a ≤ b ⟺ a ⊕ b = b``; Kan extensions rely on both."""
    alg, plus, _, _, _, grid = _REFERENCE[name]
    assert isinstance(alg, IdempotentSemiring)

    def leq(x: float, y: float) -> bool:
        return _eq(plus(x, y), y)

    for a in grid:
        assert leq(a, alg.top)
    for a, b in itertools.product(grid, repeat=2):
        m = _lib_meet(alg, a, b)
        assert leq(m, a) and leq(m, b)
        for c in grid:
            if leq(c, a) and leq(c, b):
                assert leq(c, m)


@pytest.mark.parametrize("name", _IDEMPOTENT)
def test_residuum_is_right_adjoint_to_the_tensor(name: str) -> None:
    """``v ⊗ w ≤ x ⟺ v ≤ [w, x]`` for every ``v, w, x`` on the grid."""
    alg, plus, times, _, _, grid = _REFERENCE[name]
    assert isinstance(alg, IdempotentSemiring)

    def leq(a: float, b: float) -> bool:
        return _eq(plus(a, b), b)

    for v, w, x in itertools.product(grid, repeat=3):
        r = float(
            alg.residuum(
                torch.tensor(w, dtype=torch.float64), torch.tensor(x, dtype=torch.float64)
            )
        )
        assert leq(times(v, w), x) == leq(v, r), (v, w, x, r)
