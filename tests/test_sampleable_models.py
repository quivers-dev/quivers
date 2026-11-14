"""Live morphism and tensor references in effect-site records."""

import copy

import didactic.api as dx
import pytest
import torch

from quivers.continuous.inline import MixedInlineDistribution, _normal_builder
from quivers.continuous.spaces import Euclidean
from quivers.effects.reparam import SiteRequest
from quivers.effects.sites import TorchSampleable
from quivers.effects.trace_types import SampleSite, Trace


def test_sampleable_updates_keep_declared_morphism_and_density_gradient() -> None:
    """The declared morphism persists when a drawing morphism is rebound."""
    real = Euclidean(name="R", dim=1)
    morphism = MixedInlineDistribution(
        real, real, [("var", 1), ("lit", 1.0)], _normal_builder
    )
    rebound = MixedInlineDistribution(
        real, real, [("var", 1), ("lit", 2.0)], _normal_builder
    )
    inputs = torch.tensor([[0.3], [-0.2]], requires_grad=True)
    sampleable = TorchSampleable(morphism=morphism, input=inputs)
    assert sampleable.declared is morphism
    updated = sampleable.with_(morphism=rebound)
    assert updated.declared is morphism
    assert updated.morphism is rebound
    assert updated.input is inputs
    assert copy.copy(sampleable) == sampleable
    assert hash(copy.copy(sampleable)) == hash(sampleable)
    value = torch.tensor([[0.7], [0.4]])
    expected = torch.distributions.Normal(inputs, 2.0).log_prob(value).sum(-1)
    torch.testing.assert_close(updated.log_prob(value), expected)
    updated.log_prob(value).sum().backward()
    torch.testing.assert_close(inputs.grad, (value - inputs.detach()) / 4)
    with pytest.raises(dx.ValidationError, match="ContinuousMorphism"):
        sampleable.with_(declared=None)
    with pytest.raises(dx.ValidationError, match="torch.Tensor"):
        sampleable.with_(input=3)


def test_site_request_update_retains_sampleable_and_given_identity() -> None:
    """Fixing a site value preserves its sampling context and live tensor."""
    real = Euclidean(name="R", dim=1)
    morphism = MixedInlineDistribution(
        real, real, [("var", 1), ("lit", 1.0)], _normal_builder
    )
    sampleable = TorchSampleable(morphism=morphism, input=torch.zeros(1, 1))
    request = SiteRequest(name="z", sampleable=sampleable)
    assert request.seen == {}
    assert request.seen is not SiteRequest(name="z", sampleable=sampleable).seen
    value = torch.ones(1, 1, requires_grad=True)
    updated = request.with_value(value)
    assert updated.sampleable is sampleable
    assert updated.given is value
    assert updated.seen is request.seen
    assert request.given is None
    with pytest.raises(dx.ValidationError, match="tensors or None"):
        request.with_(given=3)
    with pytest.raises(dx.ValidationError, match="map names"):
        request.with_(seen={"bad": 3})


def test_trace_finalization_preserves_live_site_values() -> None:
    """Finalization retains site storage and the exact Gaussian score gradient."""
    value = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    density = torch.distributions.Normal(0.0, 1.0).log_prob(value)
    site = SampleSite(name="z", morphism=None, value=value, log_prob=density)
    pending = Trace(sites={"z": site})
    complete = pending.with_(output=value, log_joint=density)
    assert pending.output is None
    assert pending.log_joint is None
    assert complete.sites is pending.sites
    assert complete.output is value
    assert complete.log_joint is density
    assert complete.latent_sites == {"z": site}
    assert copy.copy(site) == site
    assert hash(copy.copy(site)) == hash(site)
    assert site.with_(value=value.detach().clone()) != site
    assert Trace().sites is not Trace().sites
    density.sum().backward()
    torch.testing.assert_close(value.grad, -value.detach(), rtol=0, atol=0)
    with pytest.raises(dx.ValidationError, match="torch.Tensor"):
        site.with_(log_prob=0.0)
    with pytest.raises(dx.ValidationError, match="SampleSite"):
        pending.with_(sites={"z": 3})
    with pytest.raises(dx.ValidationError, match="trace output"):
        complete.with_(output={"z": 3})
