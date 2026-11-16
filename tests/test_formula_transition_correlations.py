"""Mixed-width group blocks retain one joint Cholesky transform across predictors."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.emit import module_to_source
from quivers.formulas import FormulaToQVRModule, families, formula_from_data
from quivers.inference.trace import trace


def _log_probs(family, contrasts):
    if family == "acat":
        return torch.cat(
            (
                torch.zeros(len(contrasts), 1, dtype=contrasts.dtype),
                (-contrasts).cumsum(-1),
            ),
            -1,
        ).log_softmax(-1)
    if family == "cumulative":
        cdf = contrasts.sigmoid()
        return torch.cat(
            (cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]), -1
        ).log()
    stop = contrasts.sigmoid()
    return torch.stack(
        (stop[:, 0], (1 - stop[:, 0]) * stop[:, 1], (1 - stop).prod(-1)), -1
    ).log()


def _reference(value):
    return (
        value.item()
        if value.ndim == 0
        else tuple(_reference(row) for row in value.detach())
    )


@pytest.mark.parametrize("family", ["acat", "sratio", "cratio"])
@pytest.mark.parametrize("shared_auxiliary", [False, True])
@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_mixed_width_correlations_match_independent_joint_transform_and_every_gradient(
    family, shared_auxiliary, parameterization
):
    data = pd.DataFrame(
        {"y": [0, 1, 2, 1], "x": [0.1, -0.2, 0.3, -0.1], "g": ["a", "b", "a", "b"]}
    )
    formula = formula_from_data(
        "y ~ 1 + (0 + cs(x) | joint | g)"
        if shared_auxiliary
        else "y ~ 1 + (1 + cs(x) | g)",
        data,
    )
    distributional = (
        {"disc": formula_from_data("y ~ 1 + (1 | joint | g)", data)}
        if shared_auxiliary
        else {}
    )
    lens = FormulaToQVRModule(
        families[family], distributional=distributional, reparameterize=parameterization
    )
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    compiled = loads(module_to_source(module))
    intercept = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    raw = torch.tensor([0.2, -0.3, 0.1], dtype=torch.double, requires_grad=True)
    correlation = torch.distributions.transforms.CorrCholeskyTransform()(raw)
    cs_sigma = torch.tensor([0.7, 0.9], dtype=torch.double, requires_grad=True)
    scalar_sigma = torch.tensor(0.4, dtype=torch.double, requires_grad=True)
    cs_z = torch.tensor(
        [[1.0, -0.3], [0.3, 0.7]], dtype=torch.double, requires_grad=True
    )
    scalar_z = torch.tensor([0.2, -0.5], dtype=torch.double, requires_grad=True)
    spacing = torch.tensor(
        [math.log(1.4) if family == "cumulative" else 1.4],
        dtype=torch.double,
        requires_grad=True,
    )
    sites = {
        "intercept": intercept,
        "sigma_g_cs_x": cs_sigma,
        "z_g_cs_x": cs_z.reshape(-1),
        "cor_shared_joint" if shared_auxiliary else "cor_g": correlation,
        "cutpoint_log_spacing"
        if family == "cumulative"
        else "cutpoint_offset": spacing,
    }
    if shared_auxiliary:
        sites.update(
            {
                "dpar_disc_intercept": torch.tensor(0.2, dtype=torch.double),
                "dpar_disc_sigma_g_Intercept": scalar_sigma,
                "dpar_disc_z_g_Intercept": scalar_z,
            }
        )
        innovations = torch.cat((cs_z, scalar_z[:, None]), -1)
        scales = torch.cat((cs_sigma, scalar_sigma[None]))
    else:
        sites.update({"sigma_g_Intercept": scalar_sigma, "z_g_Intercept": scalar_z})
        innovations = torch.cat((scalar_z[:, None], cs_z), -1)
        scales = torch.cat((scalar_sigma[None], cs_sigma))
    coefficients = (innovations @ correlation.T) * scales
    if parameterization == "centered":
        sites = {name: value for name, value in sites.items() if "z_g_" not in name}
        sites[
            "group_coefficients_shared_joint"
            if shared_auxiliary
            else "group_coefficients_g"
        ] = coefficients
    inputs = lens.fixed_column_observations(formula)
    inputs["g_idx"] = torch.tensor(formula.group_indices["g"])
    inputs["y"] = torch.tensor(data.y.to_numpy())
    recorded = trace(compiled.morphism, torch.zeros(1, 1), inputs | sites)
    groups = inputs["g_idx"]
    if shared_auxiliary:
        eta = (
            intercept
            + torch.tensor(data.x.to_numpy())[:, None] * coefficients[groups, :2]
        )
        disc = (0.2 + coefficients[groups, 2]).exp()
    else:
        eta = (
            intercept
            + coefficients[groups, 0, None]
            + torch.tensor(data.x.to_numpy())[:, None] * coefficients[groups, 1:]
        )
        disc = torch.ones(len(data), dtype=torch.double)
    gap = spacing.exp() if family == "cumulative" else spacing
    cuts = torch.cat((-gap / 2, gap / 2))
    contrasts = disc[:, None] * (cuts - eta)
    expected = _log_probs(family, contrasts).gather(-1, inputs["y"][:, None]).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    parameters = (intercept, raw, cs_sigma, scalar_sigma, cs_z, scalar_z, spacing)
    got = torch.autograd.grad(actual, parameters, retain_graph=True)
    want = torch.autograd.grad(expected, parameters)
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-7)
    reference = compiled.run(
        "model",
        data={name: _reference(value) for name, value in inputs.items()},
        sites={name: _reference(value) for name, value in sites.items()},
    )
    assert reference.log_joint == pytest.approx(recorded.log_joint.item(), rel=2e-6)


@pytest.mark.parametrize(
    "formula", ["y ~ cs(1)", "y ~ cs(x)", "y ~ 1 + (0 + cs(1) || g)"]
)
@pytest.mark.parametrize("family", ["acat", "sratio", "cratio"])
def test_scalar_proportional_predictor_does_not_acquire_cs_row_axes(formula, family):
    data = pd.DataFrame(
        {"y": [0, 1, 2, 1], "x": [0.1, -0.2, 0.3, -0.1], "g": ["a", "b", "a", "b"]}
    )
    parsed = formula_from_data(formula, data)
    lens = FormulaToQVRModule(families[family])
    module, _ = lens.forward(parsed)
    compiled = loads(module_to_source(module))
    inputs = lens.fixed_column_observations(parsed)
    inputs["y"] = torch.tensor(data.y.to_numpy())
    if parsed.group_levels:
        inputs["g_idx"] = torch.tensor(parsed.group_indices["g"])
    recorded = trace(compiled.morphism, torch.zeros(1, 1), inputs)
    assert np.isfinite(recorded.log_joint.detach().numpy()).all()
    assert recorded.sites["ordinal_contrasts"].value.shape[-2:] == (4, 2)
