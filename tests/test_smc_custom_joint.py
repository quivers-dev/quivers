"""Opaque joint callbacks cannot silently lose factors during tempering."""

import types

import pytest
import torch

from quivers.continuous import MonadicProgram
from quivers.dsl import loads
from quivers.effects.joint_plan import JointPlan, _has_custom_joint, prepare_joint
from quivers.inference import SMC, TemperedPotential
from quivers.inference.mcmc import MetropolisKernel, RandomWalkProposal
from quivers.inference.mcmc import kernel as kernel_module
from quivers.inference.mcmc.kernel import PotentialFn
from quivers.inference.registry import LatentRegistry
from quivers.inference import smc as smc_module

SOURCE = """object Out:Real 1
program p:Out->Out
    sample mu <- Normal(loc=0.0,scale=1.0)
    observe y <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""
X = torch.zeros(1, 1)
DATA = {"y": torch.tensor(0.7)}


class ExtraJoint(MonadicProgram):
    def log_joint(self, x, intermediates):
        return super().log_joint(x, intermediates) - 2.5


class InheritedJoint(MonadicProgram):
    pass


def _model(cls=MonadicProgram):
    base = loads(SOURCE).morphism
    return cls(base.domain, base.codomain, base.steps, base._return_vars)


@pytest.mark.parametrize("kind", ["subclass", "instance"])
def test_opaque_override_is_refused_before_any_execution(kind, monkeypatch):
    model = _model(ExtraJoint if kind == "subclass" else MonadicProgram)
    registry = LatentRegistry.from_model(model, {"y"})
    if kind == "instance":
        model.log_joint = types.MethodType(ExtraJoint.log_joint, model)
    assert _has_custom_joint(model)

    def unreachable(*args, **kwargs):
        raise AssertionError("opaque programs must be refused before execution")

    monkeypatch.setattr(smc_module, "trace", unreachable)
    monkeypatch.setattr(smc_module, "_check_observed_support", unreachable)
    monkeypatch.setattr(kernel_module, "_check_observed_support", unreachable)
    with pytest.raises(
        ValueError, match="does not expose prior/likelihood factor roles"
    ):
        SMC(MetropolisKernel(RandomWalkProposal(1.0))).run(model, X, DATA)
    with pytest.raises(
        ValueError, match="does not expose prior/likelihood factor roles"
    ):
        TemperedPotential(model, registry, X, DATA)


def test_override_installed_after_potential_construction_is_refused_without_calling_it():
    model = _model()
    potential = TemperedPotential(
        model, LatentRegistry.from_model(model, {"y"}), X, DATA
    )

    def unreachable(*args, **kwargs):
        raise AssertionError("the new override must not be invoked")

    model.log_joint = unreachable
    with pytest.raises(ValueError, match="overridden log_joint"):
        potential.log_prior_and_likelihood(torch.tensor([0.4]))
    with pytest.raises(ValueError, match="overridden log_joint"):
        potential.log_density(torch.tensor([0.4]))


@pytest.mark.parametrize("cls", [MonadicProgram, InheritedJoint])
def test_canonical_inherited_method_keeps_one_prepared_execution(cls):
    model = _model(cls)
    assert not _has_custom_joint(model)
    registry = LatentRegistry.from_model(model, {"y"})
    potential = TemperedPotential(model, registry, X, DATA)
    z = torch.tensor([0.4])
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(z).sum()
    expected += torch.distributions.Normal(z, 1.0).log_prob(DATA["y"]).sum()
    torch.testing.assert_close(potential.log_density(z), expected)
    assert potential._plans.prepared == 1
    assert isinstance(prepare_joint(model, X, DATA | {"mu": z}), JointPlan)


def test_shared_guard_preserves_exact_custom_joint_fallback_for_ordinary_potential():
    model = _model(ExtraJoint)
    registry = LatentRegistry.from_model(model, {"y"})
    z = torch.tensor([0.4])
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(z).sum()
    expected += torch.distributions.Normal(z, 1.0).log_prob(DATA["y"]).sum() - 2.5
    potential = PotentialFn(model, registry, X, DATA)
    torch.testing.assert_close(potential.log_density(z), expected)
    assert (
        prepare_joint(model, X, DATA | {"mu": z})
        == "the program's log_joint is overridden"
    )
