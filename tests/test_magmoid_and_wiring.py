"""Tests for :class:`Magmoid` (non-associative composition) and
:class:`WiringRule` / :class:`EinsumWiring` (operadic n-ary
contractions).

These cover the two design points the composition-rules note
flagged: a level of the hierarchy below ``Semigroupoid`` for
explicitly non-associative tensor contractions, and an operad
surface for tensor networks beyond binary composition.
"""

from __future__ import annotations

import pytest
import torch

from quivers.core.algebras import (
    BOOLEAN,
    PRODUCT_FUZZY,
    REAL,
    Magmoid,
    CompositionRule,
    CustomMagmoid,
    Algebra,
    Semigroupoid,
    magmoid,
)
from quivers.core.wiring import (
    EinsumWiring,
    contract,
    einsum_wiring,
)


# ---------------------------------------------------------------------------
# Magmoid hierarchy
# ---------------------------------------------------------------------------


def test_magmoid_is_composition_rule_not_semigroupoid() -> None:
    assert issubclass(Magmoid, CompositionRule)
    assert not issubclass(Magmoid, Semigroupoid)
    assert not issubclass(Magmoid, Algebra)


def _signed_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise mean — explicitly non-associative.

    ``mean(mean(a, b), c) = (a + b + 2c) / 4``
    ``mean(a, mean(b, c)) = (2a + b + c) / 4``
    """
    return 0.5 * (a + b)


def _sum_reduce(t: torch.Tensor, dim) -> torch.Tensor:
    if isinstance(dim, int):
        dim = (dim,)
    return t.sum(dim=dim)


def test_magmoid_factory_builds_instance() -> None:
    mg = magmoid("SignedDot", _signed_dot, _sum_reduce)
    assert isinstance(mg, Magmoid)
    assert isinstance(mg, CompositionRule)
    assert not isinstance(mg, Semigroupoid)
    assert not isinstance(mg, Algebra)
    assert mg.name == "SignedDot"


def test_magmoid_compose_works() -> None:
    mg = magmoid("SignedDot", _signed_dot, _sum_reduce)
    a = torch.tensor([[0.3, 0.5], [0.7, 0.2]])
    b = torch.tensor([[0.6, 0.1], [0.4, 0.8]])
    out = mg.compose(a, b, n_contract=1)
    assert tuple(out.shape) == (2, 2)


def test_magmoid_non_associative_in_practice() -> None:
    """Confirm that the example op really isn't associative —
    documents *why* Magmoid exists."""
    mg = magmoid("SignedDot", _signed_dot, _sum_reduce)
    a, b, c = torch.tensor(0.3), torch.tensor(0.5), torch.tensor(0.7)
    left = mg.tensor_op(mg.tensor_op(a, b), c)
    right = mg.tensor_op(a, mg.tensor_op(b, c))
    assert not torch.allclose(left, right)


def test_magmoid_skips_associativity_check() -> None:
    """A non-associative op constructs because Magmoid makes no
    associativity promise."""
    mg = CustomMagmoid("SignedDot", _signed_dot, _sum_reduce)
    assert mg.name == "SignedDot"


def test_magmoid_lacks_algebra_operations() -> None:
    mg = magmoid("SignedDot", _signed_dot, _sum_reduce)
    with pytest.raises(AttributeError):
        _ = mg.unit
    with pytest.raises(AttributeError):
        _ = mg.zero
    with pytest.raises(AttributeError):
        mg.identity_tensor((3,))


# ---------------------------------------------------------------------------
# WiringRule / EinsumWiring
# ---------------------------------------------------------------------------


def test_einsum_wiring_binary_matches_compose() -> None:
    """``EinsumWiring(rule, "ij, jk -> ik")`` reproduces binary
    composition under the same rule."""
    wiring = einsum_wiring(PRODUCT_FUZZY, "ij, jk -> ik")
    a = torch.tensor([[0.3, 0.5, 0.2], [0.7, 0.4, 0.1]])
    b = torch.tensor([[0.6, 0.4], [0.1, 0.9], [0.7, 0.2]])
    via_wiring = wiring.apply(a, b)
    via_compose = PRODUCT_FUZZY.compose(a, b, n_contract=1)
    assert torch.allclose(via_wiring, via_compose, atol=1e-6)


def test_einsum_wiring_ternary_contraction() -> None:
    """Three-input contraction: two argument tensors and a kernel
    folded under a shared reduction.

    Inputs: ``arg1 : (S, P)``, ``arg2 : (S, Q)``,
    ``kernel : (P, Q, O)``. Output: ``(S, O)`` after contraction
    of ``P, Q`` under ProductFuzzyAlgebra noisy-OR.
    """
    wiring = einsum_wiring(PRODUCT_FUZZY, "sp, sq, pqo -> so")
    torch.manual_seed(0)
    arg1 = torch.rand(4, 3)
    arg2 = torch.rand(4, 5)
    kernel = torch.rand(3, 5, 2)
    out = wiring.apply(arg1, arg2, kernel)
    assert tuple(out.shape) == (4, 2)
    assert (out >= 0).all() and (out <= 1).all()


def test_einsum_wiring_ternary_against_manual_computation() -> None:
    """Manual unrolling cross-check on small shapes."""
    wiring = einsum_wiring(REAL, "sp, sq, pqo -> so")
    arg1 = torch.tensor([[1.0, 2.0]])  # (S=1, P=2)
    arg2 = torch.tensor([[3.0, 4.0, 5.0]])  # (S=1, Q=3)
    kernel = torch.rand(2, 3, 4)  # (P=2, Q=3, O=4)
    out = wiring.apply(arg1, arg2, kernel)
    # Manual: out[s, o] = sum_{p, q} arg1[s,p] * arg2[s,q] * kernel[p,q,o]
    expected = torch.zeros(1, 4)
    for o in range(4):
        for p in range(2):
            for q in range(3):
                expected[0, o] += arg1[0, p] * arg2[0, q] * kernel[p, q, o]
    assert torch.allclose(out, expected, atol=1e-5)


def test_einsum_wiring_with_boolean_algebra() -> None:
    """Boolean (AND, OR) wiring reproduces relational composition
    on three inputs."""
    wiring = einsum_wiring(BOOLEAN, "ij, jk, kl -> il")
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    c = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = wiring.apply(a, b, c)
    assert tuple(out.shape) == (2, 2)
    # a @ b @ c under Boolean: a is identity-ish, b is swap, c is
    # identity-ish. Result is the swap.
    assert torch.allclose(out, torch.tensor([[0.0, 1.0], [1.0, 0.0]]), atol=1e-6)


def test_einsum_wiring_input_arity() -> None:
    wiring = einsum_wiring(PRODUCT_FUZZY, "ab, bc, cd -> ad")
    assert wiring.input_arity == 3


def test_einsum_wiring_composition_rule_accessor() -> None:
    wiring = einsum_wiring(PRODUCT_FUZZY, "ij, jk -> ik")
    assert wiring.composition_rule is PRODUCT_FUZZY


def test_einsum_wiring_rejects_non_composition_rule() -> None:
    with pytest.raises(TypeError, match="CompositionRule"):
        EinsumWiring("not_a_rule", "ij, jk -> ik")  # type: ignore[arg-type]


def test_einsum_wiring_rejects_bad_spec() -> None:
    with pytest.raises(ValueError, match="'->'"):
        einsum_wiring(PRODUCT_FUZZY, "ij, jk ik")


def test_einsum_wiring_rejects_repeated_input_letter() -> None:
    """A diagonal/trace contraction (``ii`` in one input) is
    refused at construction time rather than silently producing
    the wrong shape."""
    with pytest.raises(ValueError, match="repeats an axis letter"):
        einsum_wiring(PRODUCT_FUZZY, "ii, jk -> ik")


def test_einsum_wiring_rejects_repeated_output_letter() -> None:
    with pytest.raises(ValueError, match="repeats an axis letter"):
        einsum_wiring(PRODUCT_FUZZY, "ij, jk -> iik")


def test_einsum_wiring_rejects_output_letter_not_in_inputs() -> None:
    """Output axis must appear in at least one input."""
    with pytest.raises(ValueError, match="not present in any input"):
        einsum_wiring(PRODUCT_FUZZY, "ij, jk -> il")


def test_einsum_wiring_wrong_arity_raises() -> None:
    wiring = einsum_wiring(PRODUCT_FUZZY, "ij, jk -> ik")
    a = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="expects 2 inputs, got 1"):
        wiring.apply(a)


def test_einsum_wiring_wrong_tensor_ndim_raises() -> None:
    wiring = einsum_wiring(PRODUCT_FUZZY, "ij, jk -> ik")
    a = torch.zeros(2, 3, 4)  # 3-D, spec wants 2-D
    b = torch.zeros(3, 5)
    with pytest.raises(ValueError, match="declares 2 axes"):
        wiring.apply(a, b)


def test_contract_helper_works() -> None:
    """The ``contract(rule, *tensors)`` helper matches
    ``rule.apply(...)``."""
    wiring = einsum_wiring(PRODUCT_FUZZY, "ij, jk -> ik")
    a = torch.tensor([[0.3, 0.5], [0.7, 0.2]])
    b = torch.tensor([[0.6, 0.4], [0.1, 0.9]])
    via_method = wiring.apply(a, b)
    via_helper = contract(wiring, a, b)
    assert torch.allclose(via_method, via_helper)


# ---------------------------------------------------------------------------
# WiringRule + non-Algebra rule integration
# ---------------------------------------------------------------------------


def test_einsum_wiring_with_semigroupoid() -> None:
    """An ``EinsumWiring`` works with a non-algebra composition
    rule (here a Semigroupoid)."""
    from quivers.core.algebras import material_implication

    mi = material_implication()
    wiring = einsum_wiring(mi, "ij, jk -> ik")
    a = torch.tensor([[0.3, 0.5], [0.7, 0.2]])
    b = torch.tensor([[0.6, 0.4], [0.1, 0.9]])
    out = wiring.apply(a, b)
    via_compose = mi.compose(a, b, n_contract=1)
    assert torch.allclose(out, via_compose, atol=1e-6)


def test_einsum_wiring_with_magmoid() -> None:
    """An ``EinsumWiring`` works with a Magmoid rule too."""
    mg = magmoid("SignedDot", _signed_dot, _sum_reduce)
    wiring = einsum_wiring(mg, "ij, jk -> ik")
    a = torch.tensor([[0.3, 0.5], [0.7, 0.2]])
    b = torch.tensor([[0.6, 0.4], [0.1, 0.9]])
    out = wiring.apply(a, b)
    via_compose = mg.compose(a, b, n_contract=1)
    assert torch.allclose(out, via_compose, atol=1e-6)
