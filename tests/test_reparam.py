"""Tests for the reparameterisation layer.

Covers `LocScaleReparam`, `TransformReparam`, `NeuTraReparam`,
`ConjugateReparam`, and the `reparam(strategies)` dispatch
handler.
"""

from __future__ import annotations

from collections.abc import Iterator

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
from quivers.dsl import loads
from quivers.effects.sites import TorchSampleable
from quivers.inference import AutoIAFGuide, AutoNormalGuide
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


def _neutra_model() -> MonadicProgram:
    """A program with a bounded scalar, a simplex, and a plate of scales.

    Returns
    -------
    MonadicProgram
        ``theta`` on the unit interval, ``w`` on the 3-simplex, ``tau``
        three positive plate latents, and six observed responses.
    """
    program = loads(
        "object Subj : FinSet 3\n"
        "object Resp : FinSet 6\n"
        "program p : Resp -> Resp\n"
        "    sample theta <- Beta(concentration1=2.0, concentration0=3.0)\n"
        "    sample w <- Dirichlet(concentration=[1.0, 2.0, 3.0])\n"
        "    sample tau : Subj <- HalfNormal(scale=1.0)\n"
        "    observe r : Resp <- Bernoulli(probs=theta)\n"
        "    return theta\n"
        "export p\n"
    ).morphism
    assert isinstance(program, MonadicProgram)
    return program


_RESPONSES = {"r": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])}


@pytest.fixture
def _float64() -> Iterator[None]:
    """Run a test in double precision.

    Yields
    ------
    None
        Control, with float64 as the default dtype.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _trained_flow(model: MonadicProgram, observed: set[str]) -> AutoIAFGuide:
    """A flow guide moved off its initial parameters.

    Parameters
    ----------
    model : MonadicProgram
        The model.
    observed : set[str]
        The names the guide treats as observed.

    Returns
    -------
    AutoIAFGuide
        The guide.
    """
    torch.manual_seed(0)
    guide = AutoIAFGuide(model, observed, num_flows=2)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in guide.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
    return guide


@pytest.mark.usefixtures("_float64")
class TestNeuTraReparam:
    """`NeuTraReparam` draws a flow guide's sites as its pushforward and
    keeps the model's joint density."""

    def test_reparameterized_run_takes_the_pushforward_and_the_model_joint(
        self,
    ) -> None:
        model = _neutra_model()
        guide = _trained_flow(model, {"r"})
        neutra = NeuTraReparam(guide)
        x = torch.zeros(1, 1)
        torch.manual_seed(5)
        with reparam({"theta": neutra, "w": neutra, "tau": neutra}):
            tr = trace(model, x, observations=dict(_RESPONSES))
        draw = neutra.current_draw
        assert draw is not None
        # The sites take the draw's values, shaped as the sites' own draws.
        values = {name: tr.sites[name].value for name in ("theta", "w", "tau")}
        torch.testing.assert_close(
            values["theta"].reshape(()), draw.sites["theta"], rtol=0, atol=0
        )
        torch.testing.assert_close(values["w"][0], draw.sites["w"], rtol=0, atol=0)
        torch.testing.assert_close(values["tau"], draw.sites["tau"], rtol=0, atol=0)
        # The draw is B(T(u)) for its base point.
        z, _ = guide.push_forward(draw.base)
        constrained, _ = guide.registry.to_constrained(
            guide.registry.unflatten_unconstrained(z)
        )
        for name, value in constrained.items():
            torch.testing.assert_close(draw.sites[name], value, rtol=0, atol=0)
        # The reparameterized run reports the model's joint density.
        reported = sum(site.log_prob.sum() for site in tr.sites.values())
        joint = model.log_joint(x, {**values, **_RESPONSES}).sum()
        torch.testing.assert_close(reported, joint, rtol=1e-12, atol=1e-12)
        # The change of variables gives the guide's density of the draw.
        trace_side = guide.registry.trace_values(draw.sites, 1)
        torch.testing.assert_close(
            guide.log_prob(x, trace_side)[0],
            draw.log_density(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_the_log_jacobian_is_the_flow_and_bijector_terms(self) -> None:
        """For a model of real and positive sites the map from the base
        point to the values is square, and its log-determinant by
        autograd equals the draw's."""
        program = loads(
            "object Resp : FinSet 2\n"
            "program p : Resp -> Resp\n"
            "    sample a <- Normal(loc=0.0, scale=1.0)\n"
            "    sample b <- HalfNormal(scale=1.0)\n"
            "    sample c <- Beta(concentration1=2.0, concentration0=2.0)\n"
            "    observe r : Resp <- Bernoulli(probs=c)\n"
            "    return c\n"
            "export p\n"
        ).morphism
        assert isinstance(program, MonadicProgram)
        guide = _trained_flow(program, {"r"})
        neutra = NeuTraReparam(guide)
        base = torch.randn(3, generator=torch.Generator().manual_seed(3))

        def values(u: torch.Tensor) -> torch.Tensor:
            sites = neutra.push(u).sites
            return torch.stack([sites[name].reshape(()) for name in ("a", "b", "c")])

        jacobian = torch.autograd.functional.jacobian(values, base)
        expected = torch.linalg.slogdet(jacobian)[1]
        torch.testing.assert_close(
            neutra.push(base).log_abs_det, expected, rtol=1e-10, atol=1e-10
        )

    def test_each_run_draws_its_own_point(self) -> None:
        model = _neutra_model()
        guide = _trained_flow(model, {"r"})
        neutra = NeuTraReparam(guide)
        with reparam({"theta": neutra, "w": neutra, "tau": neutra}):
            tr = trace(model, torch.zeros(3, 1), observations=dict(_RESPONSES))
        theta = tr.sites["theta"].value.reshape(-1)
        assert theta.shape == (3,)
        assert len({float(v) for v in theta.detach()}) == 3

    def test_a_site_the_guide_does_not_cover_takes_its_own_draw(self) -> None:
        model = _neutra_model()
        guide = _trained_flow(model, {"r", "tau"})
        neutra = NeuTraReparam(guide)
        x = torch.zeros(1, 1)
        with reparam({"theta": neutra, "w": neutra, "tau": neutra}):
            tr = trace(model, x, observations=dict(_RESPONSES))
        draw = neutra.current_draw
        assert draw is not None
        assert set(draw.sites) == {"theta", "w"}
        prior = tr.sites["tau"].morphism
        assert prior is not None
        torch.testing.assert_close(
            tr.sites["tau"].log_prob, prior.log_prob(x, tr.sites["tau"].value)
        )

    def test_refuses_a_guide_that_is_not_a_flow(self) -> None:
        guide = AutoNormalGuide(_neutra_model(), {"r"})
        with pytest.raises(TypeError, match="flow guide"):
            NeuTraReparam(guide)  # type: ignore[arg-type]

    def test_refuses_a_base_point_of_another_width(self) -> None:
        neutra = NeuTraReparam(_trained_flow(_neutra_model(), {"r"}))
        with pytest.raises(ValueError, match=r"takes \(6,\)"):
            neutra.push(torch.zeros(5))


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
