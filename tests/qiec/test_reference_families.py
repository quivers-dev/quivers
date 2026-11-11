"""The reference backend scores every registry family as the host oracle does.

Each family is constructed with fixed parameters and scored at several
points by the reference backend and by torch, through the same
``distribution_class`` the transpile probes use, or by the torch runtime's
own definition where torch has no class. Samplers are checked by drawing
many values and comparing a moment against the oracle's, at a tolerance a
sample of that size warrants.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest
import torch
import torch.distributions as td

from quivers.continuous.families import _gauss_legendre_16
from quivers.qiec.distributions import (
    ReferenceBackend,
    RuntimeDistribution,
    seed_reference_rng,
)
from quivers.qiec.families import FAMILIES
from quivers.qiec.reference_families import RELAXATION_TEMPERATURE, TRANSFORMS
from quivers.transpile.family_meta import FAMILY_META


def _tensor(value: object) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def _torch_class(family: str, **arguments: object) -> td.Distribution:
    return FAMILY_META[family].distribution_class(
        **{name: _tensor(value) for name, value in arguments.items()}
    )


def _oracle_log_prob(family: str, arguments: dict[str, object], point: object) -> float:
    """The host's log density, from torch or the torch runtime's definition."""
    if family == "Kumaraswamy":
        distribution: td.Distribution = td.Kumaraswamy(
            _tensor(arguments["concentration1"]), _tensor(arguments["concentration0"])
        )
        return float(distribution.log_prob(_tensor(point)))
    if family == "RelaxedBernoulli":
        distribution = td.RelaxedBernoulli(
            _tensor(arguments.get("temperature", RELAXATION_TEMPERATURE)),
            probs=_tensor(arguments["probs"]),
        )
        return float(distribution.log_prob(_tensor(point)))
    if family == "RelaxedOneHotCategorical":
        distribution = td.RelaxedOneHotCategorical(
            _tensor(arguments.get("temperature", RELAXATION_TEMPERATURE)),
            probs=_tensor(arguments["probs"]),
        )
        return float(distribution.log_prob(_tensor(point)))
    if family == "LogisticNormal":
        distribution = td.LogisticNormal(
            _tensor(arguments["loc"]), _tensor(arguments["scale"])
        )
        return float(distribution.log_prob(_tensor(point)))
    if family == "LKJCholesky":
        distribution = td.LKJCholesky(len(point), _tensor(arguments["concentration"]))  # type: ignore[arg-type]
        return float(distribution.log_prob(_tensor(point)))
    if family == "LKJCorrelationFactor":
        factor = _tensor(point)
        size = factor.shape[-1]
        powers = (size - 1 - torch.arange(size, dtype=torch.float64)) + 2.0 * (
            float(arguments["concentration"]) - 1.0  # type: ignore[arg-type]
        )
        return float((powers * torch.diagonal(factor).log()).sum())
    if family == "LogitNormal":
        distribution = td.TransformedDistribution(
            td.Normal(_tensor(arguments["loc"]), _tensor(arguments["scale"])),
            [td.transforms.SigmoidTransform()],
        )
        return float(distribution.log_prob(_tensor(point)))
    if family == "LogGamma":
        # The log of a Gamma variable: the Gamma density at exp(y) times
        # the Jacobian exp(y), at points where the Gamma is well scaled.
        gamma = td.Gamma(
            _tensor(arguments["concentration"]), _tensor(arguments["rate"])
        )
        log_point = _tensor(point)
        return float(gamma.log_prob(log_point.exp()) + log_point)
    if family == "LogitBeta":
        # The logit of a Beta variable: the Beta density at sigmoid(x)
        # times the Jacobian sigmoid(x) (1 - sigmoid(x)).
        beta = td.Beta(
            _tensor(arguments["concentration1"]), _tensor(arguments["concentration0"])
        )
        logit = _tensor(point)
        return float(
            beta.log_prob(torch.sigmoid(logit))
            + torch.nn.functional.logsigmoid(logit)
            + torch.nn.functional.logsigmoid(-logit)
        )
    if family == "TruncatedNormal":
        loc, scale = _tensor(arguments["loc"]), _tensor(arguments["scale"])
        low, high = _tensor(arguments["low"]), _tensor(arguments["high"])
        normal = td.Normal(loc, scale)
        mass = normal.cdf(high) - normal.cdf(low)
        if not float(low) <= float(point) <= float(high):  # type: ignore[arg-type]
            return -math.inf
        return float(normal.log_prob(_tensor(point)) - mass.log())
    if family == "Logistic":
        loc, scale = _tensor(arguments["loc"]), _tensor(arguments["scale"])
        z = (_tensor(point) - loc) / scale
        return float(-z - 2 * torch.nn.functional.softplus(-z) - scale.log())
    if family == "HalfStudentT":
        distribution = td.StudentT(
            _tensor(arguments["df"]), _tensor(0.0), _tensor(arguments["scale"])
        )
        return float(distribution.log_prob(_tensor(point)) + math.log(2.0))
    if family in ("OrderedLogistic", "OrderedProbit"):
        eta, cutpoints = _tensor(arguments["eta"]), _tensor(arguments["cutpoints"])
        link = torch.sigmoid if family == "OrderedLogistic" else td.Normal(0.0, 1.0).cdf
        cumulative = link(cutpoints - eta)
        probabilities = torch.cat(
            [
                cumulative[:1],
                cumulative[1:] - cumulative[:-1],
                1.0 - cumulative[-1:],
            ]
        )
        return float(probabilities[int(point)].log())  # type: ignore[call-overload]
    if family == "Horseshoe":
        tau = float(arguments["scale"])  # type: ignore[arg-type]
        nodes, weights = _gauss_legendre_16(torch.device("cpu"), torch.float64)
        half_pi_t = 0.5 * math.pi * nodes
        lam = torch.tan(half_pi_t)
        jacobian = 0.5 * math.pi / torch.cos(half_pi_t) ** 2
        sigma = tau * lam
        log_norm = td.Normal(0.0, sigma).log_prob(_tensor(point))
        log_hc = math.log(2.0 / math.pi) - torch.log1p(lam * lam)
        return float(
            torch.logsumexp(log_norm + log_hc + jacobian.log() + weights.log(), 0)
        )
    if family == "InverseWishart":
        df, scale_tril = _tensor(arguments["df"]), _tensor(arguments["scale_tril"])
        sigma = _tensor(point)
        wishart = td.Wishart(df=df, scale_tril=scale_tril)
        return float(
            wishart.log_prob(torch.linalg.inv(sigma))
            - (sigma.shape[-1] + 1) * torch.logdet(sigma)
        )
    if family == "MatrixNormal":
        loc = _tensor(arguments["loc"])
        row_cov, col_cov = (
            _tensor(arguments["row_covariance"]),
            _tensor(arguments["col_covariance"]),
        )
        covariance = torch.kron(col_cov, row_cov)

        def flat(matrix: torch.Tensor) -> torch.Tensor:
            return matrix.transpose(-1, -2).reshape(-1)

        return float(
            td.MultivariateNormal(flat(loc), covariance).log_prob(flat(_tensor(point)))
        )
    if family == "GP":
        distribution = td.MultivariateNormal(
            _tensor(arguments["mean"]), _tensor(arguments["kernel"])
        )
        return float(distribution.log_prob(_tensor(point)))
    if family == "Mixture":
        component = arguments["component"]
        assert isinstance(component, RuntimeDistribution)
        weights = _tensor(arguments["weights"])
        loc = _tensor(component.arguments["loc"])
        scale = _tensor(component.arguments["scale"])
        distribution = td.MixtureSameFamily(
            td.Categorical(probs=weights), td.Normal(loc, scale)
        )
        return float(distribution.log_prob(_tensor(point)))
    if family == "Independent":
        base = arguments["base"]
        assert isinstance(base, RuntimeDistribution)
        distribution = td.Independent(
            td.Normal(_tensor(base.arguments["loc"]), _tensor(base.arguments["scale"])),
            1,
        )
        return float(distribution.log_prob(_tensor(point)))
    if family == "Transformed":
        base = arguments["base"]
        assert isinstance(base, RuntimeDistribution)
        distribution = td.TransformedDistribution(
            td.Normal(_tensor(base.arguments["loc"]), _tensor(base.arguments["scale"])),
            [td.transforms.ExpTransform(), td.transforms.SigmoidTransform()],
        )
        return float(distribution.log_prob(_tensor(point)))
    if family == "Truncated":
        base = arguments["base"]
        assert isinstance(base, RuntimeDistribution)
        normal = td.Normal(
            _tensor(base.arguments["loc"]), _tensor(base.arguments["scale"])
        )
        low, high = _tensor(arguments["low"]), _tensor(arguments["high"])
        mass = normal.cdf(high) - normal.cdf(low)
        return float(normal.log_prob(_tensor(point)) - mass.log())
    if family == "Restrict":
        base = arguments["base"]
        assert isinstance(base, RuntimeDistribution)
        normal = td.Normal(
            _tensor(base.arguments["loc"]), _tensor(base.arguments["scale"])
        )
        return float(normal.log_prob(_tensor(point)))
    if family == "Normalize":
        restricted = arguments["base"]
        assert isinstance(restricted, RuntimeDistribution)
        base = restricted.arguments["base"]
        assert isinstance(base, RuntimeDistribution)
        normal = td.Normal(
            _tensor(base.arguments["loc"]), _tensor(base.arguments["scale"])
        )
        low = _tensor(restricted.arguments["low"])
        high = _tensor(restricted.arguments["high"])
        mass = normal.cdf(high) - normal.cdf(low)
        return float(normal.log_prob(_tensor(point)) - mass.log())
    if family == "PointMass":
        return 0.0
    if family == "MixtureNormal":
        distribution = td.MixtureSameFamily(
            td.Categorical(probs=_tensor(arguments["weights"])),
            td.Normal(_tensor(arguments["loc"]), _tensor(arguments["scale"])),
        )
        return float(distribution.log_prob(_tensor(point)))
    return float(_torch_class(family, **arguments).log_prob(_tensor(point)))


def _component(loc: Sequence[float], scale: Sequence[float]) -> RuntimeDistribution:
    """A normal plated over the components of a mixture."""
    return RuntimeDistribution(
        "Normal",
        {"loc": tuple(loc), "scale": tuple(scale)},
        batch=(len(loc),),
        ranks={"loc": 0, "scale": 0},
    )


CASES: dict[str, tuple[dict[str, object], tuple[object, ...]]] = {
    "LogitNormal": ({"loc": 0.3, "scale": 0.8}, (0.2, 0.75)),
    "LogGamma": ({"concentration": 0.5, "rate": 2.0}, (-3.0, 0.4)),
    "LogitBeta": ({"concentration1": 0.5, "concentration0": 2.0}, (-4.0, 1.5)),
    "TruncatedNormal": (
        {"loc": 0.5, "scale": 1.5, "low": -1.0, "high": 2.0},
        (-0.5, 1.2, 3.0),
    ),
    "Gumbel": ({"loc": 0.5, "scale": 2.0}, (-1.0, 3.0)),
    "Chi2": ({"df": 3.0}, (0.5, 4.0)),
    "HalfCauchy": ({"scale": 1.5}, (0.2, 3.0)),
    "InverseGamma": ({"concentration": 3.0, "rate": 2.0}, (0.4, 1.5)),
    "Weibull": ({"scale": 2.0, "concentration": 1.5}, (0.5, 2.5)),
    "Pareto": ({"scale": 1.0, "alpha": 2.5}, (1.5, 4.0)),
    "Kumaraswamy": ({"concentration1": 2.0, "concentration0": 3.0}, (0.3, 0.8)),
    "ContinuousBernoulli": ({"probs": 0.3}, (0.2, 0.9)),
    "FisherSnedecor": ({"df1": 4.0, "df2": 6.0}, (0.5, 2.0)),
    "MultivariateNormal": (
        {"loc": (0.0, 1.0), "covariance_matrix": ((2.0, 0.5), (0.5, 1.0))},
        ((0.3, 0.8), (-1.0, 2.0)),
    ),
    "LowRankMVN": (
        {
            "loc": (0.0, 1.0, -1.0),
            "cov_factor": ((1.0,), (0.5,), (-0.5,)),
            "cov_diag": (0.5, 1.0, 1.5),
        },
        ((0.3, 0.8, -0.2), (-1.0, 2.0, 0.5)),
    ),
    "RelaxedBernoulli": ({"temperature": 0.7, "probs": 0.3}, (0.2, 0.9)),
    "RelaxedOneHotCategorical": (
        {"temperature": 1.3, "probs": (0.2, 0.3, 0.5)},
        ((0.2, 0.3, 0.5), (0.6, 0.1, 0.3)),
    ),
    "Wishart": (
        {"df": 4.0, "covariance_matrix": ((2.0, 0.5), (0.5, 1.0))},
        (((3.0, 0.2), (0.2, 1.5)), ((1.0, -0.3), (-0.3, 2.0))),
    ),
    "InverseWishart": (
        {"df": 4.0, "scale_tril": ((1.5, 0.0), (0.3, 1.0))},
        (((3.0, 0.2), (0.2, 1.5)), ((1.0, -0.3), (-0.3, 2.0))),
    ),
    "MatrixNormal": (
        {
            "loc": ((0.0, 1.0, 0.5), (1.0, 0.0, -0.5)),
            "row_covariance": ((2.0, 0.5), (0.5, 1.0)),
            "col_covariance": ((1.0, 0.2, 0.0), (0.2, 1.5, 0.3), (0.0, 0.3, 1.0)),
        },
        (((0.3, 0.8, 0.1), (-1.0, 2.0, 0.4)),),
    ),
    "GP": (
        {"mean": (0.0, 0.5), "kernel": ((1.0, 0.6), (0.6, 1.0))},
        ((0.3, 0.8), (-1.0, 2.0)),
    ),
    "Horseshoe": ({"scale": 1.5}, (0.0, 0.7, -2.0)),
    "NegativeBinomial": ({"total_count": 5.0, "probs": 0.4}, (0, 4)),
    "VonMises": ({"loc": 0.5, "concentration": 2.0}, (0.0, -2.0)),
    "LogisticNormal": (
        {"loc": (0.3, -0.2), "scale": (0.8, 1.2)},
        ((0.2, 0.3, 0.5), (0.6, 0.1, 0.3)),
    ),
    "OneHotCategorical": (
        {"probs": (0.2, 0.3, 0.5)},
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ),
    "LKJCholesky": (
        {"concentration": 2.0, "dimension": 2},
        (
            ((1.0, 0.0), (0.6, 0.8)),
            ((1.0, 0.0, 0.0), (0.6, 0.8, 0.0), (0.3, 0.4, math.sqrt(1 - 0.25))),
        ),
    ),
    "LKJCorrelationFactor": (
        {"concentration": 2.0, "dimension": 2},
        (((1.0, 0.0), (0.6, 0.8)),),
    ),
    "Mixture": (
        {"weights": (0.3, 0.7), "component": _component((0.0, 2.0), (1.0, 0.5))},
        (0.5, 2.5),
    ),
    "MixtureNormal": (
        {"weights": (0.3, 0.7), "loc": (0.0, 2.0), "scale": (1.0, 0.5)},
        (0.5, 2.5),
    ),
    "Independent": (
        {"base": _component((0.0, 2.0), (1.0, 0.5)), "reinterpreted_batch_ndims": 1},
        ((0.5, 2.5),),
    ),
    "Transformed": (
        {
            "base": RuntimeDistribution("Normal", {"loc": 0.2, "scale": 0.7}),
            "transforms": "exp, sigmoid",
        },
        (0.6, 0.9),
    ),
    "Truncated": (
        {
            "base": RuntimeDistribution("Normal", {"loc": 0.5, "scale": 1.5}),
            "low": -1.0,
            "high": 2.0,
        },
        (-0.5, 1.2),
    ),
    "BetaBinomial": (
        {"total_count": 10, "concentration1": 2.0, "concentration0": 3.0},
        (2, 7),
    ),
    "Restrict": (
        {
            "base": RuntimeDistribution("Normal", {"loc": 0.5, "scale": 1.5}),
            "low": -1.0,
            "high": 2.0,
        },
        (-0.5, 1.2),
    ),
    "Normalize": (
        {
            "base": RuntimeDistribution(
                "Restrict",
                {
                    "base": RuntimeDistribution("Normal", {"loc": 0.5, "scale": 1.5}),
                    "low": -1.0,
                    "high": 2.0,
                },
            )
        },
        (-0.5, 1.2),
    ),
    "PointMass": ({"value": 0.75}, (0.75,)),
    "ZeroInflatedPoisson": ({"zero_prob": 0.3, "rate": 2.5}, (0, 3)),
    "HurdlePoisson": ({"zero_prob": 0.3, "rate": 2.5}, (0, 3)),
    "ZeroOneInflatedBeta": (
        {"mu": 0.4, "phi": 5.0, "zoi": 0.2, "coi": 0.6},
        (0.0, 1.0, 0.35),
    ),
    "OrderedLogistic": ({"eta": 0.3, "cutpoints": (-1.0, 0.0, 1.0)}, (0, 2, 3)),
    "OrderedProbit": ({"eta": 0.3, "cutpoints": (-1.0, 0.0, 1.0)}, (0, 2, 3)),
    "Logistic": ({"loc": 0.5, "scale": 2.0}, (0.0, 3.0)),
    "HalfStudentT": ({"df": 4.0, "scale": 2.0}, (0.3, 3.0)),
}


def test_measure_algebra_masses() -> None:
    """A restriction's mass is its base's on the interval, a mixture's the
    weighted sum, and normalizing divides the density by the mass."""
    normal = RuntimeDistribution("Normal", {"loc": 0.5, "scale": 1.5})
    restricted = RuntimeDistribution(
        "Restrict", {"base": normal, "low": -1.0, "high": 2.0}
    )
    torch_normal = td.Normal(0.5, 1.5)
    expected = float(
        (torch_normal.cdf(_tensor(2.0)) - torch_normal.cdf(_tensor(-1.0))).log()
    )
    assert restricted.log_mass() == pytest.approx(expected, rel=1e-7)
    assert normal.log_mass() == 0.0
    counts = RuntimeDistribution("Poisson", {"rate": 2.0})
    bounded = RuntimeDistribution("Restrict", {"base": counts, "low": 1.0, "high": 3.0})
    pmf = td.Poisson(2.0).log_prob(torch.tensor([1.0, 2.0, 3.0])).exp().sum().log()
    assert bounded.log_mass() == pytest.approx(float(pmf), rel=1e-7)
    mixture = RuntimeDistribution(
        "Mixture", {"weights": (0.3, 0.7), "component": (restricted, normal)}
    )
    assert mixture.log_mass() == pytest.approx(math.log(0.3 * math.exp(expected) + 0.7))
    normalized = RuntimeDistribution("Normalize", {"base": mixture})
    assert normalized.log_mass() == 0.0
    point = 1.2
    density = float(torch_normal.log_prob(_tensor(point)))
    unnormalized = math.log(0.3 * math.exp(density) + 0.7 * math.exp(density))
    assert normalized.log_prob(point) == pytest.approx(
        unnormalized - mixture.log_mass(), rel=1e-7
    )
    atom = RuntimeDistribution("PointMass", {"value": 0.75})
    assert atom.log_prob(0.75) == 0.0
    assert atom.log_prob(0.8) == -math.inf
    assert atom.sample() == 0.75


def test_every_registry_family_is_covered() -> None:
    backend = ReferenceBackend()
    assert set(FAMILIES) <= backend.families
    core = set(FAMILIES) - set(CASES)
    assert core == {
        "Normal",
        "LogNormal",
        "HalfNormal",
        "Bernoulli",
        "Categorical",
        "Beta",
        "Gamma",
        "Exponential",
        "Uniform",
        "Poisson",
        "Binomial",
        "Geometric",
        "Laplace",
        "Cauchy",
        "StudentT",
        "Dirichlet",
    }


@pytest.mark.parametrize("family", sorted(CASES))
def test_reference_density_matches_the_oracle(family: str) -> None:
    arguments, points = CASES[family]
    backend = ReferenceBackend()
    for point in points:
        expected = _oracle_log_prob(family, arguments, point)
        actual = backend.log_prob(family, arguments, point)
        assert actual == pytest.approx(expected, rel=1e-7, abs=1e-9), (family, point)


@pytest.mark.parametrize("family", sorted(CASES))
def test_reference_samples_land_in_the_support(family: str) -> None:
    arguments, _ = CASES[family]
    backend = ReferenceBackend()
    rng = random.Random(3)
    for _ in range(50):
        draw = backend.sample(family, arguments, rng)
        assert backend.log_prob(family, arguments, draw) > -math.inf, (family, draw)


_MOMENTS: dict[str, tuple[float, float]] = {
    "LogitNormal": (0.5677, 0.02),
    # digamma(0.5) - log(2); sd sqrt(trigamma(0.5)) = 2.22 over 4000
    # draws gives a standard error of 0.035, and the tolerance is 4.5 of them.
    "LogGamma": (float(torch.digamma(torch.tensor(0.5))) - math.log(2.0), 0.16),
    # digamma(0.5) - digamma(2); sd 2.36 gives a standard error of 0.037.
    "LogitBeta": (
        float(torch.digamma(torch.tensor(0.5)) - torch.digamma(torch.tensor(2.0))),
        0.17,
    ),
    "TruncatedNormal": (0.5216, 0.03),
    "Gumbel": (0.5 + 2.0 * 0.5772156649, 0.08),
    "Chi2": (3.0, 0.08),
    "InverseGamma": (1.0, 0.05),
    "Weibull": (2.0 * math.gamma(1 + 1 / 1.5), 0.05),
    "Kumaraswamy": (3.0 * math.gamma(1.5) * math.gamma(3.0) / math.gamma(4.5), 0.02),
    "ContinuousBernoulli": (
        0.3 / (2 * 0.3 - 1) + 1 / (2 * math.atanh(1 - 2 * 0.3)),
        0.02,
    ),
    "FisherSnedecor": (6.0 / 4.0, 0.08),
    "NegativeBinomial": (5.0 * 0.4 / 0.6, 0.15),
    "BetaBinomial": (10 * 2.0 / 5.0, 0.12),
    "Logistic": (0.5, 0.15),
    "HalfStudentT": (
        2.0 * 2.0 * math.sqrt(4.0 / math.pi) * math.gamma(2.5) / math.gamma(2.0) / 3.0,
        0.15,
    ),
    "MixtureNormal": (0.3 * 0.0 + 0.7 * 2.0, 0.06),
    "Mixture": (1.4, 0.06),
}


@pytest.mark.parametrize("family", sorted(_MOMENTS))
def test_reference_sample_means_match_closed_forms(family: str) -> None:
    arguments, _ = CASES[family]
    backend = ReferenceBackend()
    rng = random.Random(11)
    draws = [float(backend.sample(family, arguments, rng)) for _ in range(4000)]  # type: ignore[arg-type]
    mean, tolerance = _MOMENTS[family]
    assert sum(draws) / len(draws) == pytest.approx(mean, abs=tolerance)


def test_lkj_samples_are_correlation_factors() -> None:
    backend = ReferenceBackend()
    rng = random.Random(5)
    for _ in range(20):
        factor = backend.sample(
            "LKJCholesky", {"concentration": 2.0, "dimension": 3}, rng
        )
        rows = [list(row) for row in factor]  # type: ignore[union-attr]
        for row in rows:
            assert sum(item * item for item in row) == pytest.approx(1.0)
        assert rows[0][1] == 0.0 and rows[0][2] == 0.0 and rows[1][2] == 0.0


def test_a_constructed_lkj_distribution_reads_its_dimension_from_the_type() -> None:
    seed_reference_rng(2)
    distribution = RuntimeDistribution(
        "LKJCholesky",
        {"concentration": 2.0},
        natural=(3, 3),
        ranks={"concentration": 0},
    )
    factor = distribution.sample()
    assert len(factor) == 3  # type: ignore[arg-type]
    assert distribution.log_prob(factor) > -math.inf


def test_transform_chains_are_the_documented_names() -> None:
    assert set(TRANSFORMS) == {
        "exp",
        "log",
        "sigmoid",
        "logit",
        "softplus",
        "tanh",
        "neg",
    }
    backend = ReferenceBackend()
    arguments = {
        "base": RuntimeDistribution("Normal", {"loc": 0.0, "scale": 1.0}),
        "transforms": "cube",
    }
    with pytest.raises(Exception, match="unknown transform"):
        backend.log_prob("Transformed", arguments, 0.5)
