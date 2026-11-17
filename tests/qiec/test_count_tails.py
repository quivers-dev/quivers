"""Scalar reference count tails against independent probability oracles."""

import math
import pytest
from scipy import stats
from quivers.qiec._count_tails import log_count_tails
from quivers.qiec.distributions import RuntimeDistribution
from quivers.qiec.reference_support import DistributionError


@pytest.mark.parametrize(
    "name,args,oracle",
    [
        ("Bernoulli", {"probs": 0.3}, stats.bernoulli(0.3)),
        ("Binomial", {"total_count": 10, "probs": 0.3}, stats.binom(10, 0.3)),
        (
            "NegativeBinomial",
            {"total_count": 2.0, "probs": 0.3},
            stats.nbinom(2.0, 0.7),
        ),
        ("Poisson", {"rate": 3.0}, stats.poisson(3.0)),
        ("Geometric", {"probs": 0.3}, stats.geom(0.3, loc=-1)),
        (
            "BetaBinomial",
            {"total_count": 20, "concentration1": 0.3, "concentration0": 2.0},
            stats.betabinom(20, 0.3, 2.0),
        ),
    ],
)
def test_reference_count_tails_against_scipy(name, args, oracle):
    for point in [-math.inf, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, math.inf]:
        lc, ls = log_count_tails(name, args, point)
        assert lc == pytest.approx(oracle.logcdf(point), abs=1e-11)
        assert ls == pytest.approx(oracle.logsf(point), abs=1e-11)


@pytest.mark.parametrize(
    "name,args",
    [
        ("Binomial", {"total_count": 8, "logits": 1000.0}),
        ("Binomial", {"total_count": 8, "logits": -1000.0}),
        ("NegativeBinomial", {"total_count": 2.0, "logits": 1000.0}),
        ("NegativeBinomial", {"total_count": 2.0, "logits": -1000.0}),
        ("Geometric", {"logits": -1000.0}),
        ("Geometric", {"logits": 1000.0}),
    ],
)
def test_reference_finite_logit_tail_coordinates_match_native(name, args):
    import torch
    from quivers.continuous._closed_probability import (
        ClosedBinomial,
        ClosedNegativeBinomial,
        ClosedGeometric,
    )

    cls = {
        "Binomial": ClosedBinomial,
        "NegativeBinomial": ClosedNegativeBinomial,
        "Geometric": ClosedGeometric,
    }[name]
    native = cls(**{k: torch.tensor(v, dtype=torch.float64) for k, v in args.items()})
    for point in [0.0, 1.0, 3.0]:
        lc, ls = log_count_tails(name, args, point)
        assert lc == pytest.approx(
            native.log_cdf(torch.tensor(point)).item(), abs=1e-11
        )
        assert ls == pytest.approx(
            native.log_survival_function(torch.tensor(point)).item(), abs=1e-11
        )


@pytest.mark.parametrize("args", [{"probs": 0.0}, {"logits": -math.inf}])
def test_reference_geometric_improper_limit_is_rejected(args):
    law = RuntimeDistribution(family="Geometric", arguments=args)
    with pytest.raises(DistributionError, match="proper success"):
        law.log_prob(0)
    with pytest.raises(DistributionError, match="proper success"):
        log_count_tails("Geometric", args, 0.0)


def test_reference_geometric_sampling_refuses_unrepresentable_count_without_loop():
    import random

    law = RuntimeDistribution(family="Geometric", arguments={"logits": -1000.0})
    with pytest.raises(DistributionError, match="capacity"):
        law.sample(random.Random(2))


def test_reference_beta_binomial_complete_tail_budget_is_explicit():
    with pytest.raises(DistributionError, match="capacity"):
        log_count_tails(
            "BetaBinomial",
            {"total_count": 10000, "concentration1": 2.0, "concentration0": 3.0},
            5000.0,
            max_terms=5,
        )
