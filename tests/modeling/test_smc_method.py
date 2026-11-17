"""Checked bindings retain SMC particles and independently verified evidence."""

import copy
import math

import pytest
import torch

from quivers.inference import SMC, SMCResult
from quivers.inference.mcmc import MetropolisKernel, RandomWalkProposal
from quivers.modeling import Model, SMCMethod

SOURCE = """object Row:FinSet _
object Out:Real 1
program regression(y:Tensor[Real]([Row])):Row->Out
    sample mu <- Normal(loc=0.0,scale=1.0)
    observe y:Row <- Normal(loc=mu,scale=1.0)
    return mu
export regression
"""


def _method(particles=512, seed=7):
    return SMCMethod(
        sampler=SMC(
            MetropolisKernel(RandomWalkProposal(0.5)),
            num_particles=particles,
            num_mcmc_steps=2,
            num_pilot_steps=0,
            seed=seed,
        )
    )


@pytest.mark.parametrize("observations", [[0.7], [-0.2, 0.7, 1.1]])
def test_checked_smc_matches_normal_conjugate_evidence_and_posterior(observations):
    y = torch.tensor(observations, dtype=torch.float64)
    before = torch.random.get_rng_state()
    result = Model(SOURCE).fit({"y": y}, method=_method())
    assert isinstance(result, SMCResult)
    assert torch.equal(before, torch.random.get_rng_state())
    n = len(observations)
    # det(I + 11') = n+1 and its inverse is I - 11'/(n+1).
    expected = -0.5 * (
        n * math.log(2 * math.pi)
        + math.log(n + 1)
        + float(y.square().sum() - y.sum().square() / (n + 1))
    )
    assert result.log_evidence == pytest.approx(expected, abs=0.1)
    values = result.samples["mu"].reshape(-1).double()
    assert values.mean().item() == pytest.approx(y.sum().item() / (n + 1), abs=0.12)
    assert values.var().item() == pytest.approx(1 / (n + 1), abs=0.1)
    assert result.num_particles == 512
    assert result.temperatures[0] == 0 and result.temperatures[-1] == 1
    assert result.log_evidence_increments.sum().item() == pytest.approx(
        result.log_evidence
    )


def test_method_reuses_driver_and_preserves_result_and_random_stream():
    bound = Model(SOURCE).bind({"y": torch.tensor([0.3], dtype=torch.float64)})
    method = _method(particles=8)
    assert copy.copy(method).sampler is method.sampler
    direct = method.sampler.run(
        bound.require_tensor_program(), bound.input, bound.tensor_data()
    )
    adapted = bound.infer(method)
    assert adapted.log_evidence == direct.log_evidence
    assert adapted.num_evaluations == direct.num_evaluations
    for name in direct.samples:
        torch.testing.assert_close(
            adapted.samples[name], direct.samples[name], rtol=0, atol=0
        )
    torch.testing.assert_close(
        adapted.temperatures, direct.temperatures, rtol=0, atol=0
    )


def test_invalid_binding_is_refused_before_sampling(monkeypatch):
    method = _method(particles=8)

    def unexpected(*args, **kwargs):
        raise AssertionError("sampler ran before checked input validation")

    monkeypatch.setattr(method.sampler, "run", unexpected)
    with pytest.raises(TypeError, match="dtype"):
        Model(SOURCE).fit({"y": torch.tensor([1], dtype=torch.int64)}, method=method)
