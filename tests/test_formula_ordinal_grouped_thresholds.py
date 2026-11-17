"""Grouped hurdle thresholds retain category and group conditioning in prediction."""

import math

import pandas as pd
import pytest
import torch

from quivers.formulas import fit, formula_to_qvr
from quivers.inference.mcmc.driver import MCMCResult


def _fit():
    data = pd.DataFrame(
        {
            "y": [0, 1, 2, 3, 0, 2],
            "x": [-1.0, 0.0, 1.0, 2.0, 0.4, 0.8],
            "g": ["a", "a", "a", "b", "b", "b"],
        }
    )
    fitted = fit(
        "y~x+(1|g)",
        data=data,
        family="hurdle_cumulative",
        thresholds_by="g",
        fixed_parameters={"hu": 0.2},
        method="svi",
        num_steps=1,
        seed=8,
    )
    values = {
        "intercept": torch.tensor(0.3),
        "beta_x": torch.tensor(-0.4),
        "sigma_g_Intercept": torch.tensor(0.5),
        "z_g_Intercept": torch.tensor([0.2, -0.3]),
        "cutpoint_log_spacing_mean": torch.tensor([math.log(1.4)]),
        "cutpoint_log_spacing_scale": torch.tensor([0.4]),
        "cutpoint_log_spacing_noise": torch.tensor([-0.5, 0.8]),
    }
    return fitted.with_(
        posterior=MCMCResult(
            samples={
                name: value.expand(1, 8, *value.shape).clone()
                for name, value in values.items()
            },
            log_densities=torch.zeros(1, 8),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=8,
        )
    )


def _oracle(x, group, *, exclude=False):
    adjustments = torch.tensor([0.1, -0.15])
    widths = 1.4 * torch.exp(torch.tensor([-0.2, 0.32]))
    eta = 0.3 - 0.4 * x + (0.0 if exclude else adjustments[group])
    width = widths[group]
    bounds = torch.stack((-width / 2, width / 2), -1) - eta[..., None]
    cdf = torch.sigmoid(bounds)
    base = torch.stack((cdf[..., 0], cdf[..., 1] - cdf[..., 0], 1 - cdf[..., 1]), -1)
    return torch.cat((torch.full((*base.shape[:-1], 1), 0.2), 0.8 * base), -1)


def test_grouped_hurdle_known_subset_repeated_rows_and_excluded_effects():
    result = _fit()
    fresh = pd.DataFrame({"x": [0.5, -1.0, 2.0], "g": ["b", "b", "b"]})
    x = torch.tensor(fresh.x.to_numpy(), dtype=torch.float32)
    expected = _oracle(x, torch.ones(3, dtype=torch.long))
    torch.testing.assert_close(
        result.posterior_epred(new_data=fresh), expected.expand(1, 8, 3, 4)
    )
    excluded = _oracle(x, torch.ones(3, dtype=torch.long), exclude=True)
    torch.testing.assert_close(
        result.posterior_epred(new_data=fresh, re_formula="~0"),
        excluded.expand(1, 8, 3, 4),
    )
    assert result.category_labels == (0, 1, 2, 3)
    assert result.posterior_predict(new_data=fresh, seed=3).shape == (1, 8, 3)
    with pytest.raises(ValueError, match="missing|group"):
        result.posterior_epred(new_data=fresh.drop(columns="g"), re_formula="~0")


@pytest.mark.parametrize("policy", ("old_levels", "uncertainty"))
def test_new_hurdle_group_jointly_reuses_threshold_and_location_donor(policy):
    result = _fit()
    fresh = pd.DataFrame({"x": [0.0, 0.0, 0.0], "g": ["fresh", "fresh", "a"]})
    predicted = result.posterior_epred(
        new_data=fresh, allow_new_levels=True, sample_new_levels=policy, seed=31
    )
    alternatives = torch.stack(
        (
            _oracle(torch.tensor(0.0), torch.tensor(0)),
            _oracle(torch.tensor(0.0), torch.tensor(1)),
        )
    )
    assert (
        torch.isclose(predicted[..., 0, None, :], alternatives, atol=1e-6, rtol=1e-6)
        .all(-1)
        .any(-1)
        .all()
    )
    torch.testing.assert_close(predicted[..., 0, :], predicted[..., 1, :])
    torch.testing.assert_close(predicted[..., 2, :], alternatives[0].expand(1, 8, 4))
    excluded = result.posterior_epred(
        new_data=fresh,
        re_formula="~0",
        allow_new_levels=True,
        sample_new_levels=policy,
        seed=31,
    )
    threshold_alternatives = torch.stack(
        (
            _oracle(torch.tensor(0.0), torch.tensor(0), exclude=True),
            _oracle(torch.tensor(0.0), torch.tensor(1), exclude=True),
        )
    )
    assert (
        torch.isclose(
            excluded[..., 0, None, :], threshold_alternatives, atol=1e-6, rtol=1e-6
        )
        .all(-1)
        .any(-1)
        .all()
    )
    torch.testing.assert_close(
        excluded[..., 2, :], threshold_alternatives[0].expand(1, 8, 4)
    )


@pytest.mark.parametrize("label", (10**12, float(2**63), -float(2**64)))
def test_invalid_large_category_codes_fail_without_enumerating_the_label_range(label):
    frame = pd.DataFrame({"y": [0, label], "x": [0.0, 1.0]})
    with pytest.raises(ValueError, match="contiguous|range|Int64|integer"):
        formula_to_qvr("y~x", data=frame, family="acat")
