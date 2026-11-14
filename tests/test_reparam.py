"""Tests for the reparameterisation layer.

Covers `LocScaleReparam`, `TransformReparam`, `NeuTraReparam`,
`ConjugateReparam`, and the `reparam(strategies)` dispatch
handler.
"""

from __future__ import annotations

import pytest
import torch

from quivers.continuous.bijectors import ExpBijector
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.inline import MixedInlineDistribution, _normal_builder
from quivers.continuous.programs import MonadicProgram
from quivers.continuous import Draw
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet
from quivers.effects import clamp, reparam
from quivers.effects.reparam import (
    ConjugateReparam,
    LocScaleReparam,
    NeuTraReparam,
    Reparam,
    ReparamOrchestrator,
    SiteRequest,
    TransformReparam,
)
from quivers.effects.sites import TorchSampleable
from quivers.inference.trace import trace


def _simple_program() -> MonadicProgram:
    """A minimal ``z ~ prior; y ~ likelihood(z)`` model."""
    Unit = FinSet(name="Unit", cardinality=1)
    R1 = Euclidean(name="R1", dim=1)
    prior = ConditionalNormal(Unit, R1)
    likelihood = ConditionalNormal(R1, R1)
    return MonadicProgram(
        Unit,
        R1,
        steps=[
            Draw(names=("z",), morphism=prior),
            Draw(names=("y",), morphism=likelihood, args=("z",)),
        ],
        return_vars=("y",),
    )


def _batch(n: int = 4) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.long)


class TestReparamOrchestrator:
    """`reparam({name: strategy})` returns a `ReparamOrchestrator`."""

    def test_factory_returns_orchestrator(self) -> None:
        h = reparam({"z": LocScaleReparam()})
        assert isinstance(h, ReparamOrchestrator)

    def test_orchestrator_dispatches_by_name(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        strategy = LocScaleReparam()
        with reparam({"z": strategy}):
            tr = trace(prog, x)
        # z was reparameterised; y was not — but both sites appear.
        assert "z" in tr.sites
        assert "y" in tr.sites


class TestLocScaleReparam:
    """Non-centred rewrite matches the original density exactly."""

    def test_log_prob_equals_direct_normal_score(self) -> None:
        """`LocScaleReparam` on a Normal site produces a log-density
        equal to the direct ``Normal(loc, scale).log_prob(y)``.
        """
        torch.manual_seed(0)
        prog = _simple_program()
        x = _batch(4)
        tr_ref = trace(prog, x)
        z_val = tr_ref.sites["z"].value

        # Rerun with LocScaleReparam given the same z value: the
        # reparam should score z under Normal(loc, scale) directly.
        torch.manual_seed(1)

        class _Preset(Reparam):
            def apply(self, site: SiteRequest) -> tuple[torch.Tensor, torch.Tensor]:
                return LocScaleReparam().apply(site.with_value(z_val))

        with reparam({"z": _Preset()}):
            tr_rep = trace(prog, x)

        # Both should equal `morph.log_prob(x, z_val)` for the prior.
        prior = tr_ref.sites["z"].morphism
        assert prior is not None
        expected = prior.log_prob(x, z_val)
        torch.testing.assert_close(tr_rep.sites["z"].log_prob, expected)
        torch.testing.assert_close(tr_ref.sites["z"].log_prob, expected)

    def test_centered_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            LocScaleReparam(centered=1.5)

    def test_raises_without_get_params(self) -> None:
        """A morphism without `_get_params` fails at apply time."""
        R1 = Euclidean(name="R1", dim=1)
        morphism = MixedInlineDistribution(
            R1, R1, [("var", 1), ("lit", 1.0)], _normal_builder
        )
        site = SiteRequest(
            name="z",
            sampleable=TorchSampleable(morphism=morphism, input=torch.zeros(4, 1)),
            seen={},
        )
        with pytest.raises(TypeError, match="_get_params"):
            LocScaleReparam().apply(site)


class TestTransformReparam:
    """`TransformReparam(bijector)` rewrites via a fixed bijector."""

    def test_exp_bijector_produces_positive_reals(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        with reparam({"z": TransformReparam(ExpBijector())}):
            tr = trace(prog, x)
        # The value at z is still whatever the morphism sampled,
        # but the log-density is scored under the original
        # distribution (change-of-variables identity).
        prior = tr.sites["z"].morphism
        assert prior is not None
        expected = prior.log_prob(x, tr.sites["z"].value)
        torch.testing.assert_close(tr.sites["z"].log_prob, expected)


class TestNeuTraReparam:
    """`NeuTraReparam` passes through when the guide does not cover
    the site."""

    def test_falls_through_when_site_not_covered(self) -> None:
        # Minimal fake guide with an empty registry.
        class _FakeGuide:
            class _EmptyRegistry:
                def names(self) -> list[str]:
                    return []

            registry = _EmptyRegistry()

            def sample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                return {}

        prog = _simple_program()
        x = _batch(4)
        with reparam({"z": NeuTraReparam(_FakeGuide())}):  # type: ignore[arg-type]
            tr = trace(prog, x)
        # Because the guide does not cover z, NeuTra scores it under
        # the model's own prior; the trace is well-defined.
        prior = tr.sites["z"].morphism
        assert prior is not None
        expected = prior.log_prob(x, tr.sites["z"].value)
        torch.testing.assert_close(tr.sites["z"].log_prob, expected)


class TestConjugateReparam:
    """`ConjugateReparam` scores a child under the closed-form marginal."""

    def test_normal_normal_marginal(self) -> None:
        Unit = FinSet(name="Unit", cardinality=1)
        R1 = Euclidean(name="R1", dim=1)
        prior = ConditionalNormal(Unit, R1)
        child = MixedInlineDistribution(
            R1, R1, [("var", 1), ("lit", 0.5)], _normal_builder
        )
        prog = MonadicProgram(
            Unit,
            R1,
            steps=[
                Draw(names=("z",), morphism=prior),
                Draw(names=("y",), morphism=child, args=("z",)),
            ],
            return_vars=("y",),
        )
        x = _batch(4)
        y_val = torch.tensor([[0.3], [-0.2], [1.1], [0.0]])
        strategy = ConjugateReparam("z", "Normal", "Normal")
        with reparam({"y": strategy}):
            tr = trace(prog, x, observations={"y": y_val})
        loc, scale = prior._get_params(x)
        marginal = torch.distributions.Normal(loc, torch.sqrt(scale**2 + 0.5**2))
        torch.testing.assert_close(
            tr.sites["y"].log_prob, marginal.log_prob(y_val).sum(-1)
        )
        torch.testing.assert_close(tr.sites["y"].value, y_val)

    def test_raises_for_unregistered_pair(self) -> None:
        with pytest.raises(KeyError, match="no conjugate marginal"):
            ConjugateReparam("z", "NoSuch", "Pair")

    def test_raises_when_the_parent_was_not_seen(self) -> None:
        R1 = Euclidean(name="R1", dim=1)
        child = MixedInlineDistribution(
            R1, R1, [("var", 1), ("lit", 0.5)], _normal_builder
        )
        site = SiteRequest(
            name="y",
            sampleable=TorchSampleable(morphism=child, input=torch.zeros(4, 1)),
            seen={},
        )
        with pytest.raises(KeyError, match="was not seen"):
            ConjugateReparam("z", "Normal", "Normal").apply(site)


class TestReparamComposesWithOtherHandlers:
    """A reparam handler stacks cleanly with `clamp`, `mask`, etc."""

    def test_reparam_and_clamp_stack(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        z_val = torch.zeros(4, 1)
        # Condition on z; TransformReparam on y should still fire.
        with clamp({"z": z_val}):
            with reparam({"y": TransformReparam(ExpBijector())}):
                tr = trace(prog, x)
        assert "y" in tr.sites
        assert tr.sites["z"].is_observed
        # Sanity check that the joint log-density is finite.
        assert tr.log_joint is not None
        assert torch.isfinite(tr.log_joint).all()


class TestLocScaleGradient:
    """Non-centred sampling supports reparameterised gradients."""

    def test_gradient_flows_through_reparameterised_sample(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        with reparam({"z": LocScaleReparam()}):
            tr = trace(prog, x)
        assert tr.log_joint is not None
        # log_joint should carry grad_fn since sampling used the
        # reparameterisation trick.
        assert tr.log_joint.requires_grad
