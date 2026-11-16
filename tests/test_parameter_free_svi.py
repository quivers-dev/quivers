"""Parameter-free fits retain the ordinary objective and result contracts."""

import math

import pytest
import torch

from quivers.inference import AutoNormalGuide, ELBO, SVI
from quivers.inference.svi import _prepare_optimizer
from quivers.modeling import Model, VariationalMethod

SOURCE = """object Row : FinSet _
program model(y:Tensor[Real]([Row])) : Row -> Row
    observe y : Row <- Normal(loc=0.25,scale=0.7)
    return y
export model
"""


@pytest.mark.parametrize("steps", [0, 3])
def test_source_parameter_free_fit_retains_raw_density_and_callbacks(steps):
    y = torch.tensor([-0.3, 0.4, 1.2], dtype=torch.float64)
    calls = []
    fitted = Model(SOURCE).fit(
        {"y": y},
        method=VariationalMethod(
            num_steps=steps, callback=lambda i, v: calls.append((i, v))
        ),
    )
    expected = (
        -0.5 * ((y - 0.25) / 0.7).square() - math.log(0.7) - 0.5 * math.log(2 * math.pi)
    )
    assert len(fitted.losses) == steps and [i for i, _ in calls] == list(range(steps))
    for value in fitted.losses:
        assert value == pytest.approx(-expected.sum().item())
    torch.testing.assert_close(
        fitted.log_likelihood(num_samples=2)["y"], expected.expand(1, 2, 3)
    )
    assert not fitted.analysis(num_samples=2).samples


def test_formula_parameter_free_fit_uses_the_same_svi_policy():
    import pandas as pd
    from quivers.formulas import fit

    y = torch.tensor([-0.3, 0.4, 1.2], dtype=torch.float32)
    fitted = fit(
        "y ~ 0",
        data=pd.DataFrame({"y": y.numpy()}),
        fixed_parameters={"sigma": 1.0},
        method="svi",
        num_steps=2,
    )
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(y).sum()
    assert len(fitted.losses) == 2
    assert fitted.losses[0] == pytest.approx(-expected.item())
    assert not fitted.draws(num_samples=2)


def test_empty_optimizer_policy_excludes_frozen_parameters_and_deduplicates_trainable():
    frozen = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
    calls = []
    assert (
        _prepare_optimizer(
            [frozen], learning_rate=0.1, factory=lambda p: calls.append(p)
        )
        is None
    )
    assert not calls
    active = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = _prepare_optimizer([active, frozen], [active], learning_rate=0.1)
    assert optimizer is not None and optimizer.param_groups[0]["params"] == [active]


def test_omitting_optimizer_for_a_trainable_guide_is_refused():
    source = SOURCE.replace(
        "    observe y", "    sample theta <- Normal(loc=0.,scale=1.)\n    observe y"
    )
    bound = Model(source).bind({"y": torch.zeros(2)})
    program = bound.require_tensor_program()
    guide = AutoNormalGuide(program, observed_names={"y"})
    with pytest.raises(ValueError, match="optimizer is required"):
        SVI(program, guide, None, ELBO())
