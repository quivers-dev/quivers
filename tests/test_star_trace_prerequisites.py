"""Coherence and closure requirements for coproduct feedback."""

import pytest
import torch

from quivers.categorical.monoidal import CoproductMonoidal
from quivers.categorical.traced import TracedMonoidal
from quivers.core.algebras import BOOLEAN, LOG_PROB
from quivers.core.morphisms import Morphism, observed
from quivers.core.objects import FinSet, SetObject
from quivers.stochastic._kleene import DivergentClosureError, kleene_star
from quivers.stochastic.semiring import LogProbSemiring


@pytest.mark.parametrize("side", ["left", "right"])
def test_coproduct_unitors_use_the_algebra_zero(side: str) -> None:
    obj = FinSet(name="A", cardinality=2)
    morph = getattr(CoproductMonoidal(LOG_PROB), f"{side}_unitor")(obj)
    expected = torch.tensor([[0.0, -torch.inf], [-torch.inf, 0.0]])
    torch.testing.assert_close(morph.tensor, expected, rtol=0, atol=0)


@pytest.mark.parametrize("side", ["left", "right"])
def test_coproduct_unitors_flatten_multidimensional_objects(side: str) -> None:
    obj = FinSet(name="A", cardinality=2) * FinSet(name="B", cardinality=3)
    morph = getattr(CoproductMonoidal(BOOLEAN), f"{side}_unitor")(obj)
    torch.testing.assert_close(
        morph.tensor, torch.eye(6).reshape(6, 2, 3), rtol=0, atol=0
    )


class _ZeroTrace(TracedMonoidal):
    def trace(
        self,
        morph: Morphism,
        feedback: SetObject,
        domain: SetObject,
        codomain: SetObject,
    ) -> Morphism:
        return observed(
            domain,
            codomain,
            torch.zeros(*domain.shape, *codomain.shape),
            algebra=BOOLEAN,
        )


def test_yanking_verification_rejects_an_incorrect_coproduct_trace() -> None:
    bad = _ZeroTrace(CoproductMonoidal(BOOLEAN), BOOLEAN)
    assert not bad.verify_yanking(FinSet(name="U", cardinality=2))


def test_log_kleene_star_retains_a_tiny_nonzero_path() -> None:
    weights = torch.tensor(
        [[-torch.inf, -1000.0], [-torch.inf, -torch.inf]], dtype=torch.float64
    )
    expected = torch.tensor([[0.0, -1000.0], [-torch.inf, 0.0]], dtype=torch.float64)
    actual = kleene_star(weights, LogProbSemiring())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_log_star_matches_linear_resolvent_and_derivative() -> None:
    generator = torch.Generator().manual_seed(17)
    logs = (torch.rand(4, 4, generator=generator, dtype=torch.float64) / 8).log()
    logs.requires_grad_()
    actual = kleene_star(logs, LogProbSemiring())
    eye = torch.eye(4, dtype=torch.float64)
    expected = torch.linalg.solve(eye - logs.exp(), eye).log()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(actual.sum(), logs, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logs)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-11)


def test_log_star_tiny_path_has_finite_gradient() -> None:
    edge = torch.tensor(-1000.0, dtype=torch.float64, requires_grad=True)
    absent = torch.full_like(edge, -torch.inf)
    matrix = torch.stack((absent, edge, absent, absent)).reshape(2, 2)
    result = kleene_star(matrix, LogProbSemiring())
    gradient = torch.autograd.grad(result[0, 1], edge)[0]
    torch.testing.assert_close(gradient, torch.ones_like(edge), rtol=0, atol=0)


@pytest.mark.parametrize("loop", [0.0, 0.5])
def test_log_star_rejects_divergent_return_weights(loop: float) -> None:
    matrix = torch.tensor(
        [[-torch.inf, -1000.0], [1000.0 + loop, -torch.inf]], dtype=torch.float64
    )
    with pytest.raises(DivergentClosureError):
        kleene_star(matrix, LogProbSemiring())


def test_log_star_handles_empty_feedback() -> None:
    matrix = torch.empty(0, 0, dtype=torch.float64)
    torch.testing.assert_close(kleene_star(matrix, LogProbSemiring()), matrix)
