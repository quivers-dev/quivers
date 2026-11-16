"""Discrimination regressions retain positive support and every native gradient."""

import math

import pandas as pd
import pytest
import torch

from quivers.dsl import loads
from quivers.formulas import families, formula_to_qvr
from quivers.inference.trace import trace
from quivers.qiec.execution import ExecutionFailure


@pytest.mark.parametrize(
    "kind", ("cumulative", "acat", "sratio", "cratio", "hurdle_cumulative")
)
def test_rowwise_discrimination_matches_independent_density_and_all_head_gradients(
    kind,
):
    hurdle = kind == "hurdle_cumulative"
    data = pd.DataFrame(
        {
            "y": [0, 1, 2, 3] if hurdle else [0, 1, 2, 1],
            "x": [-1.0, 0.0, 0.7, 1.8],
            "z": [0.2, -0.3, 0.5, 1.1],
        }
    )
    source = formula_to_qvr(
        "y~x",
        data=data,
        family=kind,
        distributional={"disc": "~z"},
        fixed_parameters={"hu": 0.2} if hurdle else None,
    )
    raw = {
        "intercept": 0.3,
        "beta_x": -0.4,
        "dpar_disc_intercept": math.log(1.7),
        "dpar_disc_beta_z": 0.15,
    }
    raw[
        "cutpoint_log_spacing"
        if kind in {"cumulative", "hurdle_cumulative"}
        else "cutpoint_offset"
    ] = (math.log(1.4),) if kind in {"cumulative", "hurdle_cumulative"} else (1.4,)
    values = {
        name: torch.tensor(value, dtype=torch.double, requires_grad=True)
        for name, value in raw.items()
    }
    x = torch.tensor(data.x.to_numpy(), dtype=torch.double)
    z = torch.tensor(data.z.to_numpy(), dtype=torch.double)
    y = torch.tensor(data.y.to_numpy())
    compiled = loads(source)
    recorded = trace(
        compiled.morphism,
        torch.zeros(1, 1),
        {"x": x, "dpar_disc_z": z, "y": y, **values},
    )
    eta = values["intercept"] + values["beta_x"] * x
    disc = (values["dpar_disc_intercept"] + values["dpar_disc_beta_z"] * z).exp()
    threshold = next(
        value for name, value in values.items() if name.startswith("cutpoint")
    )
    width = (
        threshold.exp() if kind in {"cumulative", "hurdle_cumulative"} else threshold
    )
    cuts = torch.cat((-width / 2, width / 2))
    contrast = disc[:, None] * (cuts[None, :] - eta[:, None])
    if kind in {"cumulative", "hurdle_cumulative"}:
        cdf = torch.sigmoid(contrast)
        masses = torch.cat((cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]), -1)
    elif kind == "acat":
        odds = torch.exp(-contrast)
        masses = torch.stack(
            (torch.ones(4, dtype=torch.double), odds[:, 0], odds.prod(-1)), -1
        )
        masses = masses / masses.sum(-1, keepdim=True)
    else:
        stop = torch.sigmoid(contrast)
        masses = torch.stack(
            (stop[:, 0], (1 - stop[:, 0]) * stop[:, 1], (1 - stop).prod(-1)), -1
        )
    if hurdle:
        masses = torch.cat(
            (torch.full((4, 1), 0.2, dtype=torch.double), 0.8 * masses), -1
        )
    expected = masses.log().gather(-1, y[:, None]).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    params = tuple(values.values())
    observed = torch.autograd.grad(actual, params, retain_graph=True)
    oracle = torch.autograd.grad(expected, params)
    for a, b in zip(observed, oracle, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-7)
    reference = compiled.run(
        "model",
        data={"x": tuple(data.x), "dpar_disc_z": tuple(data.z), "y": tuple(data.y)},
        sites=raw,
    )
    assert reference.log_joint == pytest.approx(recorded.log_joint.item(), rel=2e-6)


@pytest.mark.parametrize("value", (-1.0, 0.0))
@pytest.mark.parametrize("configuration", ("scalar", "regression"))
def test_nonpositive_discrimination_is_rejected_before_observation_scoring(
    value, configuration
):
    data = pd.DataFrame({"y": [0, 1, 2], "x": [0.0, 1.0, 2.0]})
    if configuration == "scalar":
        source = formula_to_qvr(
            "y~x", data=data, family="acat", priors={"disc": "Normal(0.0,1.0)"}
        )
        sites = {"disc": value}
    else:
        source = formula_to_qvr(
            "y~x",
            data=data,
            family=families["acat"].with_links(disc="identity"),
            distributional={"disc": "~1"},
        )
        sites = {"dpar_disc_intercept": value}
    sites.update({"intercept": 0.3, "beta_x": -0.4, "cutpoint_offset": (1.4,)})
    compiled = loads(source)
    observations = {
        "x": torch.tensor(data.x.to_numpy()),
        "y": torch.tensor(data.y.to_numpy()),
        **{name: torch.tensor(item) for name, item in sites.items()},
    }
    with pytest.raises(ValueError, match="positive"):
        trace(compiled.morphism, torch.zeros(1, 1), observations)
    with pytest.raises(ExecutionFailure, match="positive"):
        compiled.run(
            "model", data={"x": tuple(data.x), "y": tuple(data.y)}, sites=sites
        )
