"""Latent response assembly retains one exact joint law in both executors."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from quivers.dsl import lower_qvr_to_qiec, parse
from quivers.effects.clamp import ClampHandler
from quivers.formulas._missing import bind_latent_response
from quivers.inference import trace
from quivers.modeling import Model
from quivers.qiec.program_runtime import run_program


MISSING_SOURCE = """object Row:FinSet 4
object Seen:FinSet 2
object Missing:FinSet 2
program p(x_seen:Tensor[Real]([Seen]),y:Tensor[Real]([Row]),order:Tensor[Int]([Row]),a:Real,b:Real):Row->Row
    observe x_seen:Seen <- Normal(loc=a,scale=1.2)
    sample x_missing:Missing <- Normal(loc=a,scale=1.2)
    let packed = concat(x_seen,x_missing)
    let x = factor i:Row in packed[order[i]]
    let mu=b*x
    observe y:Row <- Normal(loc=mu,scale=0.7)
    return x_missing
export p
"""

MEASUREMENT_SOURCE = """object Row:FinSet 4
object Truth:FinSet 3
object Seen:FinSet 2
program p(x_seen:Tensor[Real]([Seen]),error:Tensor[Real]([Seen]),seen:Tensor[Int]([Seen]),row_index:Tensor[Int]([Row]),y:Tensor[Real]([Row]),a:Real,b:Real):Row->Row
    sample x:Truth <- Normal(loc=a,scale=1.2)
    let measured = factor i:Seen in x[seen[i]]
    observe x_seen:Seen <- Normal(loc=measured,scale=error)
    let row_x = factor i:Row in x[row_index[i]]
    let mu=b*row_x
    observe y:Row <- Normal(loc=mu,scale=0.7)
    return x
export p
"""


def _host(value):
    if isinstance(value, torch.Tensor):
        return _host(value.detach().tolist())
    if isinstance(value, list):
        return tuple(_host(item) for item in value)
    return value


def _joint(source, data, latent):
    bound = Model(source).bind(data)
    with ClampHandler(latent):
        actual = trace(
            bound.require_tensor_program(), bound.input, bound.tensor_data()
        ).log_joint.sum()
    reference = run_program(
        lower_qvr_to_qiec(parse(source)),
        "p",
        data={key: _host(value) for key, value in data.items()},
        sites={key: _host(value) for key, value in latent.items()},
    )
    assert reference.log_joint == pytest.approx(float(actual.detach()), abs=2e-6)
    return actual


def test_missing_joint_has_only_missing_latents_and_original_observed_units():
    plan = bind_latent_response("x", [0.4, np.nan, -0.1, np.nan])
    a = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    b = torch.tensor(-0.2, dtype=torch.double, requires_grad=True)
    missing = torch.tensor([0.6, -0.5], dtype=torch.double, requires_grad=True)
    seen = torch.tensor(plan.observed.data.copy())
    y = torch.tensor([0.2, -0.3, 0.5, 0.1], dtype=torch.double)
    data = {
        "x_seen": seen,
        "y": y,
        "order": torch.tensor(plan.full_to_combined),
        "a": a,
        "b": b,
    }
    actual = _joint(MISSING_SOURCE, data, {"x_missing": missing})
    truth = torch.cat((seen, missing))[list(plan.full_to_combined)]
    normal = torch.distributions.Normal
    expected = (
        normal(a, a.new_tensor(1.2)).log_prob(seen).sum()
        + normal(a, a.new_tensor(1.2)).log_prob(missing).sum()
        + normal(b * truth, a.new_tensor(0.7)).log_prob(y).sum()
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=4e-7)
    for got, wanted in zip(
        torch.autograd.grad(actual, (a, b, missing), retain_graph=True),
        torch.autograd.grad(expected, (a, b, missing)),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted, rtol=0, atol=5e-7)
    oracle = (
        norm.logpdf(truth.detach().numpy(), float(a.detach()), 1.2).sum()
        + norm.logpdf(y.numpy(), (b * truth).detach().numpy(), 0.7).sum()
    )
    assert float(actual.detach()) == pytest.approx(oracle, abs=4e-7)


def test_shared_measurement_joint_scores_each_measurement_once_and_propagates_uncertainty():
    a = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    b = torch.tensor(-0.2, dtype=torch.double, requires_grad=True)
    truth = torch.tensor([0.6, -0.5, 0.2], dtype=torch.double, requires_grad=True)
    seen = torch.tensor([0.4, -0.1], dtype=torch.double)
    error = torch.tensor([0.2, 0.4], dtype=torch.double)
    positions = torch.tensor([0, 2])
    rows = torch.tensor([2, 0, 2, 1])
    y = torch.tensor([0.2, -0.3, 0.5, 0.1], dtype=torch.double)
    data = {
        "x_seen": seen,
        "error": error,
        "seen": positions,
        "row_index": rows,
        "y": y,
        "a": a,
        "b": b,
    }
    actual = _joint(MEASUREMENT_SOURCE, data, {"x": truth})
    normal = torch.distributions.Normal
    expected = (
        normal(a, a.new_tensor(1.2)).log_prob(truth).sum()
        + normal(truth[positions], error).log_prob(seen).sum()
        + normal(b * truth[rows], a.new_tensor(0.7)).log_prob(y).sum()
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=4e-7)
    for got, wanted in zip(
        torch.autograd.grad(actual, (a, b, truth), retain_graph=True),
        torch.autograd.grad(expected, (a, b, truth)),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted, rtol=0, atol=5e-7)
