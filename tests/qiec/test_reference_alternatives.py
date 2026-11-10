"""The reference machine honors the parameterization a source writes.

Every registry family that accepts alternative parameterizations, the
covariance of a multivariate normal or a Wishart as ``scale_tril``,
``covariance_matrix``, or ``precision_matrix`` and the masses of a
discrete or relaxed family as ``probs`` or ``logits``, is scored under
each alternative as `torch.distributions` scores the same construction,
and samples the same draw from the same seed under each alternative.
A one-hot draw is a vector of the integers its `Int` element names.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import random
from typing import cast

import pytest
import torch
import torch.distributions as D

from quivers.qiec.distributions import ReferenceBackend
from quivers.qiec.families import FAMILIES

type Host = float | int | bool | tuple[object, ...]
"""A host value: a number, a boolean, or nested tuples of numbers."""

type Oracle = Callable[..., D.Distribution]
"""The `torch.distributions` class, constructed from keyword arguments."""

_L = ((1.0, 0.0), (0.5, 1.0))
_COVARIANCE = ((1.0, 0.5), (0.5, 1.25))
_PRECISION = ((1.25, -0.5), (-0.5, 1.0))
_PROBS = (0.2, 0.3, 0.5)
_LOGITS = tuple(math.log(p) for p in _PROBS)
_P = 0.3
_LOGIT = math.log(_P / (1.0 - _P))

#: Each family: its fixed arguments, one argument set per alternative,
#: the value scored, and the torch construction from the arguments.
_CASES: dict[str, tuple[dict[str, Host], dict[str, dict[str, Host]], Host, Oracle]] = {
    "MultivariateNormal": (
        {"loc": (0.0, 1.0)},
        {
            "scale_tril": {"scale_tril": _L},
            "covariance_matrix": {"covariance_matrix": _COVARIANCE},
            "precision_matrix": {"precision_matrix": _PRECISION},
        },
        (0.3, 0.4),
        D.MultivariateNormal,
    ),
    "Wishart": (
        {"df": 4.0},
        {
            "scale_tril": {"scale_tril": _L},
            "covariance_matrix": {"covariance_matrix": _COVARIANCE},
            "precision_matrix": {"precision_matrix": _PRECISION},
        },
        ((2.0, 0.3), (0.3, 1.0)),
        D.Wishart,
    ),
    "Bernoulli": (
        {},
        {"probs": {"probs": _P}, "logits": {"logits": _LOGIT}},
        True,
        D.Bernoulli,
    ),
    "Binomial": (
        {"total_count": 5},
        {"probs": {"probs": _P}, "logits": {"logits": _LOGIT}},
        2,
        D.Binomial,
    ),
    "Geometric": (
        {},
        {"probs": {"probs": _P}, "logits": {"logits": _LOGIT}},
        2,
        D.Geometric,
    ),
    "NegativeBinomial": (
        {"total_count": 5.0},
        {"probs": {"probs": _P}, "logits": {"logits": _LOGIT}},
        2,
        D.NegativeBinomial,
    ),
    "ContinuousBernoulli": (
        {},
        {"probs": {"probs": _P}, "logits": {"logits": _LOGIT}},
        0.4,
        D.ContinuousBernoulli,
    ),
    "RelaxedBernoulli": (
        {"temperature": 0.5},
        {"probs": {"probs": _P}, "logits": {"logits": _LOGIT}},
        0.4,
        D.RelaxedBernoulli,
    ),
    "Categorical": (
        {},
        {"probs": {"probs": _PROBS}, "logits": {"logits": _LOGITS}},
        1,
        D.Categorical,
    ),
    "OneHotCategorical": (
        {},
        {"probs": {"probs": _PROBS}, "logits": {"logits": _LOGITS}},
        (0, 1, 0),
        D.OneHotCategorical,
    ),
    "RelaxedOneHotCategorical": (
        {"temperature": 0.5},
        {"probs": {"probs": _PROBS}, "logits": {"logits": _LOGITS}},
        (0.2, 0.5, 0.3),
        D.RelaxedOneHotCategorical,
    ),
}

_PAIRS = [
    (family, alternative)
    for family, (_, alternatives, _, _) in sorted(_CASES.items())
    for alternative in sorted(alternatives)
]


def _tensor(value: Host) -> torch.Tensor:
    """A host value as a float64 tensor.

    Parameters
    ----------
    value : Host
        The value.

    Returns
    -------
    torch.Tensor
        The tensor.
    """
    return torch.tensor(value, dtype=torch.float64)


def test_every_alternative_parameterization_is_covered() -> None:
    covered = {
        family: {name for arguments in alternatives.values() for name in arguments}
        for family, (_, alternatives, _, _) in _CASES.items()
    }
    for record in FAMILIES.values():
        names = {parameter.name for parameter in record.parameters}
        for group in ({"probs", "logits"}, {"scale_tril", "covariance_matrix"}):
            if group <= names:
                assert record.name in covered, record.name
                assert names & covered[record.name] >= group, record.name


@pytest.mark.parametrize(("family", "alternative"), _PAIRS)
def test_each_alternative_scores_the_torch_density(
    family: str, alternative: str
) -> None:
    fixed, alternatives, value, oracle = _CASES[family]
    arguments = {**fixed, **alternatives[alternative]}
    expected = float(
        oracle(**{name: _tensor(item) for name, item in arguments.items()}).log_prob(
            _tensor(value)
        )
    )
    reference = ReferenceBackend().log_prob(family, arguments, value)
    assert reference == pytest.approx(expected, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("family", sorted(_CASES))
def test_each_alternative_draws_the_same_value_from_one_seed(family: str) -> None:
    fixed, alternatives, _, _ = _CASES[family]
    draws = [
        ReferenceBackend().sample(
            family, {**fixed, **alternatives[alternative]}, random.Random(7)
        )
        for alternative in sorted(alternatives)
    ]
    first = _tensor(cast(Host, draws[0]))
    for draw in draws[1:]:
        torch.testing.assert_close(
            _tensor(cast(Host, draw)), first, rtol=1e-9, atol=1e-9
        )


def test_a_wishart_scale_factor_sets_the_mean_it_names() -> None:
    rng = random.Random(11)
    backend = ReferenceBackend()
    draws = [
        _tensor(
            cast(Host, backend.sample("Wishart", {"df": 4.0, "scale_tril": _L}, rng))
        )
        for _ in range(4000)
    ]
    mean = torch.stack(draws).mean(dim=0)
    torch.testing.assert_close(mean, 4.0 * _tensor(_COVARIANCE), rtol=0.06, atol=0.06)


def test_a_one_hot_draw_holds_integers() -> None:
    draw = ReferenceBackend().sample(
        "OneHotCategorical", {"probs": _PROBS}, random.Random(3)
    )
    assert isinstance(draw, tuple)
    assert all(type(item) is int for item in draw)
    assert sorted(draw) == [0, 0, 1]
