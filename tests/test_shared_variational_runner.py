"""Both fitting frontends share finite optimization and posterior draw contracts."""

import math

import pandas as pd
import pytest
import torch

from quivers.dsl import loads
from quivers.formulas import fit
from quivers.inference import AutoNormalGuide, ELBO, Objective, SVI
from quivers.inference.svi import _unique_parameters
from quivers.modeling import Model, VariationalMethod


SOURCE = """
object Unit : FinSet 1
object Value : Real 1
program model : Unit -> Value
    sample theta <- Normal(loc=0.0, scale=1.0)
    observe y <- Normal(loc=theta, scale=1.0)
    return theta
export model
"""


def test_shared_runner_recomputes_bindings_and_reports_each_loss():
    program = loads(SOURCE).morphism
    guide = AutoNormalGuide(program, observed_names={"y"})
    optimizer = torch.optim.Adam(
        _unique_parameters(program.parameters(), guide.parameters()), lr=0.01
    )
    svi = SVI(program, guide, optimizer, ELBO())
    supplied, reported = [], []

    def data(index):
        supplied.append(index)
        return {"y": torch.tensor(float(index))}

    losses = svi.run(
        torch.zeros(1, 1),
        data,
        num_steps=3,
        callback=lambda index, loss: reported.append((index, loss)),
    )
    assert supplied == [0, 1, 2]
    assert reported == list(enumerate(losses))
    assert len(losses) == 3 and all(math.isfinite(loss) for loss in losses)


def test_nonfinite_objective_is_rejected_before_optimizer_mutates_parameters():
    class InvalidObjective(Objective):
        def forward(self, model, guide, x, observations):
            return next(guide.parameters()).sum() * math.nan

    program = loads(SOURCE).morphism
    guide = AutoNormalGuide(program, observed_names={"y"})
    optimizer = torch.optim.Adam(guide.parameters())
    original = [parameter.detach().clone() for parameter in guide.parameters()]
    svi = SVI(program, guide, optimizer, InvalidObjective())
    with pytest.raises(ValueError, match="step 0.*nonfinite"):
        svi.run(torch.zeros(1, 1), {"y": torch.zeros(1)}, num_steps=2)
    for actual, expected in zip(guide.parameters(), original, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not optimizer.state


@pytest.mark.parametrize("invalid", [True, False, -1, 1.5])
def test_shared_runner_validates_iteration_budget(invalid):
    program = loads(SOURCE).morphism
    guide = AutoNormalGuide(program, observed_names={"y"})
    svi = SVI(program, guide, torch.optim.Adam(guide.parameters()), ELBO())
    with pytest.raises(ValueError, match="nonnegative integer"):
        svi.run(torch.zeros(1, 1), {"y": torch.zeros(1)}, num_steps=invalid)


def test_zero_step_runner_preserves_initialization_without_reading_bindings():
    program = loads(SOURCE).morphism
    guide = AutoNormalGuide(program, observed_names={"y"})
    svi = SVI(program, guide, torch.optim.Adam(guide.parameters()), ELBO())
    parameters = [value.detach().clone() for value in guide.parameters()]
    state = torch.random.get_rng_state().clone()

    def forbidden(*args):
        raise AssertionError(
            "zero optimization steps must not execute callbacks or bindings"
        )

    assert svi.run(torch.zeros(1, 1), forbidden, num_steps=0, callback=forbidden) == ()
    assert torch.equal(torch.random.get_rng_state(), state)
    for actual, expected in zip(guide.parameters(), parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not svi.optim.state


def test_source_and_formula_fits_retain_finite_loss_history_and_seeded_draws():
    source = (
        Model(SOURCE).bind({"y": 0.2}).infer(VariationalMethod(num_steps=3, seed=7))
    )
    formula = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": [0.2]}),
        fixed_parameters={"sigma": 1.0},
        method="svi",
        num_steps=3,
        seed=7,
    )
    for result in (source, formula):
        assert len(result.losses) == 3
        assert all(math.isfinite(loss) for loss in result.losses)
        state = torch.random.get_rng_state().clone()
        first = result.draws(num_samples=4, seed=93)
        assert torch.equal(torch.random.get_rng_state(), state)
        second = result.draws(num_samples=4, seed=93)
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        with pytest.raises(ValueError, match="positive integer"):
            result.draws(num_samples=True)
