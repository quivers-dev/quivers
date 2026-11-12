"""The continuous Bernoulli family scores its logits in log space.

The family's distribution, as the registry builds it for
`ConditionalContinuousBernoulli` and for a QVR draw, is checked in single
precision against the closed form in double precision,
``x log sigmoid(l) + (1 - x) log sigmoid(-l) + log(l / tanh(l / 2))``, at
logits from -30 to 30, with a tolerance of 64 float32 ulps fixed from
single-precision resolution. `torch.distributions.ContinuousBernoulli`,
which forms its normalizer from the probability, is shown to fall outside
that tolerance at a large logit. A distribution built from its probability
scores as `torch.distributions` does, and an expanded distribution scores
as the one it was expanded from.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributions as D

from quivers.continuous.family_spec import FAMILY_REGISTRY

_FLOAT32_TOLERANCE = 2.0**-18
_POINTS = (0.0, 0.1, 0.45, 0.8, 1.0)


def _exact(logit: float, point: float) -> float:
    """The log density in double precision, from the logit.

    Parameters
    ----------
    logit : float
        The logit.
    point : float
        The point of the unit interval.

    Returns
    -------
    float
        The log density.
    """
    value = torch.tensor(logit, dtype=torch.float64)
    normalizer = (
        math.log(2.0)
        if logit == 0.0
        else float(torch.log(value.abs() / torch.tanh(value.abs() / 2.0)))
    )
    return float(
        point * torch.nn.functional.logsigmoid(value)
        + (1.0 - point) * torch.nn.functional.logsigmoid(-value)
        + normalizer
    )


def _family() -> type[D.ContinuousBernoulli]:
    """The distribution class the registry builds the family with.

    Returns
    -------
    type[D.ContinuousBernoulli]
        The class.
    """
    return FAMILY_REGISTRY["ContinuousBernoulli"].dist_class


@pytest.mark.parametrize("logit", [-30.0, -15.0, -0.05, 0.0, 0.05, 15.0, 30.0])
def test_logits_score_the_closed_form_in_single_precision(logit: float) -> None:
    distribution = _family()(logits=torch.full((len(_POINTS),), logit))
    scored = distribution.log_prob(torch.tensor(_POINTS))
    for point, value in zip(_POINTS, scored.tolist(), strict=True):
        expected = _exact(logit, point)
        assert value == pytest.approx(
            expected, rel=_FLOAT32_TOLERANCE, abs=_FLOAT32_TOLERANCE
        )


def test_the_probability_normalizer_falls_outside_the_tolerance() -> None:
    scored = float(
        D.ContinuousBernoulli(logits=torch.tensor(30.0)).log_prob(torch.tensor(0.1))
    )
    expected = _exact(30.0, 0.1)
    assert abs(scored - expected) > _FLOAT32_TOLERANCE * abs(expected)


def test_probabilities_score_as_torch_does() -> None:
    probs = torch.tensor([0.2, 0.5, 0.7])
    points = torch.tensor([0.3, 0.9, 0.05])
    scored = _family()(probs=probs).log_prob(points)
    expected = D.ContinuousBernoulli(probs=probs).log_prob(points)
    assert torch.equal(scored, expected)


def test_an_expanded_distribution_scores_as_its_source() -> None:
    distribution = _family()(logits=torch.tensor([30.0, -12.0]))
    expanded = distribution.expand(torch.Size([3, 2]))
    points = torch.tensor([0.25, 0.6])
    assert torch.equal(
        expanded.log_prob(points.expand(3, 2)),
        distribution.log_prob(points).expand(3, 2),
    )
