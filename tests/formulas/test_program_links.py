"""The link a replaced parameter takes from its support, and no Jacobian.

Oracles: the support of each law (the real line, a half-line, an interval, a
simplex) and the inverse link that maps the real line onto it, checked at
points; and the log density of a program with a replaced site, written out in
torch. The replaced site is a deterministic function of its predictor, so its
density contributes no term: adding the log Jacobian of the link changes the
value, which the comparison would see.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.distributions import constraints

from quivers.formulas import NonlinearExpression, fit, get_prior
from quivers.formulas._program_kernel import support_law

from tests.formulas._program_cases import (
    log_joint,
    BOUNDED,
    GAUSS,
    HALFWIDTH,
    SIMPLEX,
    frame,
    latent_names,
    pin,
    run,
)


def _inverse(text: str, eta: float) -> float:
    """Evaluate an inverse link written over ``{eta}`` in Python."""
    return float(
        eval(  # noqa: S307 - the expression is a declared link of this library
            text.format(eta=repr(eta)),
            {"exp": math.exp, "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x))},
        )
    )


@pytest.mark.parametrize(
    ("support", "lower", "upper"),
    [
        (constraints.real, None, None),
        (constraints.positive, 0.0, None),
        (constraints.greater_than(1.5), 1.5, None),
        (constraints.less_than(-2.0), None, -2.0),
        (constraints.unit_interval, 0.0, 1.0),
        (constraints.interval(-2.0, 3.0), -2.0, 3.0),
    ],
)
def test_each_scalar_support_takes_a_link_onto_it(support, lower, upper) -> None:
    law = support_law("theta", support)
    assert law.obstacle is None and law.link is not None
    values = [
        _inverse(law.link.inverse_expr, eta) for eta in (-30.0, -1.0, 0.0, 2.0, 30.0)
    ]
    assert values in (sorted(values), sorted(values, reverse=True))
    for value in values:
        assert lower is None or value >= lower
        assert upper is None or value <= upper
    # The image is the whole support: its ends are approached.
    if lower is not None:
        assert min(values) - lower < 1e-6 * max(1.0, abs(lower))
    if upper is not None:
        assert upper - max(values) < 1e-6 * max(1.0, abs(upper))
    if lower is None and upper is None:
        assert values[0] == -30.0 and values[-1] == 30.0


def test_a_simplex_takes_a_softmax_with_its_first_coordinate_fixed() -> None:
    law = support_law("weights", constraints.simplex)
    assert law.link is not None and law.link.name == "softmax"
    assert law.reference == 0


@pytest.mark.parametrize(
    ("support", "reason"),
    [
        (constraints.nonnegative_integer, "discrete"),
        (constraints.boolean, "discrete"),
        (constraints.corr_cholesky, "not free"),
        (constraints.positive_definite, "not free"),
        (constraints.lower_cholesky, "not free"),
        (constraints.interval(torch.tensor([0.0, 1.0]), 2.0), "differs between"),
    ],
)
def test_supports_without_a_link_are_refused_by_name(support, reason) -> None:
    law = support_law("theta", support)
    assert law.link is None
    assert law.obstacle is not None and reason in law.obstacle


def test_a_replaced_site_adds_no_density_term() -> None:
    fitted = fit(
        program=BOUNDED,
        parameters={"lapse": "~ 1"},
        response="accept",
        data=frame(),
        method="svi",
        num_steps=1,
        seed=0,
    )
    priors = get_prior(
        program=BOUNDED, parameters={"lapse": "~ 1"}, response="accept", data=frame()
    )
    assert priors["dpar_lapse_intercept"] == "Normal(loc=0.0, scale=5.0)"
    assert set(latent_names(fitted)) == {"dpar_lapse_intercept", "theta"}
    intercept = torch.tensor(0.7, dtype=torch.float64)
    theta = torch.tensor(-0.3, dtype=torch.float64)
    joint = log_joint(run(fitted, {"dpar_lapse_intercept": intercept, "theta": theta}))
    lapse = 0.5 * torch.sigmoid(intercept)
    p = lapse + (1.0 - 2.0 * lapse) * torch.sigmoid(theta)
    accept = torch.as_tensor(frame()["accept"].to_numpy(), dtype=torch.float64)
    expected = (
        torch.distributions.Normal(0.0, 5.0).log_prob(intercept)
        + torch.distributions.Normal(0.0, 1.0).log_prob(theta)
        + torch.distributions.Bernoulli(probs=p).log_prob(accept).sum()
    )
    assert joint == pytest.approx(float(expected), rel=1e-10)
    jacobian = torch.log(
        0.5 * torch.sigmoid(intercept) * (1.0 - torch.sigmoid(intercept))
    )
    assert abs(joint - float(expected + jacobian)) > 1e-2


def test_a_positive_scale_takes_the_log_link() -> None:
    fitted = fit(
        program=GAUSS,
        parameters={"sigma": "~ 1"},
        response="y",
        data=frame(),
        method="svi",
        num_steps=1,
        seed=0,
    )
    assert "let sigma = exp(dpar_sigma_eta)" in fitted.qvr_source


def test_a_program_simplex_keeps_its_reference_coordinate() -> None:
    # Binary fractions keep the Float32 response carrier exact.
    rows = frame().assign(y=np.arange(-6, 6) / 8)
    fitted = fit(
        program=SIMPLEX,
        parameters={
            "weights": NonlinearExpression(expression={1: "lo", 2: "hi"}),
            "lo": "~ 1",
            "hi": "~ 1",
        },
        response="y",
        data=rows,
        data2={"halfwidth": HALFWIDTH},
        method="svi",
        num_steps=1,
        seed=0,
    )
    logits = torch.tensor([0.0, -0.5, 1.7], dtype=torch.float64)
    sigma = 0.9
    actual = (
        pin(
            fitted,
            {
                "nlpar_lo_intercept": np.array([float(logits[1])]),
                "nlpar_hi_intercept": np.array([float(logits[2])]),
                "sigma": np.array([sigma]),
            },
        )
        .log_likelihood()["y"]
        .reshape(-1)
    )
    response = torch.tensor(rows["y"].to_numpy(), dtype=torch.float64)
    conditional = torch.distributions.Normal(torch.tensor(HALFWIDTH), sigma).log_prob(
        response[:, None]
    )
    expected = torch.logsumexp(conditional + logits.log_softmax(0), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=0.0)
    wrong = torch.logsumexp(conditional + torch.roll(logits, 1).log_softmax(0), dim=-1)
    assert float((actual - wrong).abs().max()) > 1e-3
