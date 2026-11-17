"""An arbitrary user law shares checked formula fitting and posterior replay."""

import math
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypsecant
from scipy.stats import poisson
import torch
from torch.distributions import Distribution, constraints

from quivers.family_catalog import FamilyCatalog
from quivers.formulas import AuxParam, custom_family, fit, get_prior, links
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL, INT
from quivers.effects.observation_scoring import observation_scoring


class _HyperbolicSecant(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale):
        self.loc, self.scale = torch.broadcast_tensors(loc, scale)
        super().__init__(self.loc.shape, validate_args=False)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale.square() * (math.pi / 2) ** 2

    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        return math.log(2 / math.pi) - self.scale.log() - torch.logaddexp(z, -z)

    def rsample(self, sample_shape=torch.Size()):
        uniform = torch.rand(
            (*sample_shape, *self.batch_shape),
            dtype=self.loc.dtype,
            device=self.loc.device,
        )
        uniform = uniform.clamp(
            torch.finfo(uniform.dtype).eps, 1 - torch.finfo(uniform.dtype).eps
        )
        return self.loc + self.scale * (uniform * math.pi / 2).tan().log()

    def cdf(self, value):
        z = (value - self.loc) / self.scale
        tail = (2 / math.pi) * torch.atan(torch.exp(-z.abs()))
        return torch.where(z > 0, 1 - tail, tail)


def _catalog(*, shift=1):
    semantic = DistributionFamily(
        name="UserSecant",
        parameters=(
            FamilyParameter(name="center", constraint="real", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
            FamilyParameter(name="bias", constraint="real", rank=0),
        ),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
    )
    return FamilyCatalog().with_family(
        semantic,
        lambda parameters: _HyperbolicSecant(
            parameters["center"] + shift * parameters["bias"], parameters["scale"]
        ),
    )


def _family(catalog=None):
    return custom_family(
        "UserSecant",
        catalog=catalog or _catalog(),
        location="center",
        auxiliary=(
            AuxParam(name="scale", prior="LogNormal(0.0, 0.3)", link=links["log"]),
        ),
        data={"bias": "known_bias"},
    )


def _frame():
    return pd.DataFrame(
        {
            "y": [0.2, -0.3, 0.4, 1.1],
            "x": [-1.0, 0.0, 0.5, 1.0],
            "known_bias": [0.1, -0.2, 0.3, 0.0],
        }
    )


def test_custom_catalog_identity_survives_formula_value_semantics_and_snapshots():
    catalog = _catalog()
    family = _family(catalog)
    assert deepcopy(family) == family
    assert deepcopy(family).custom.catalog is catalog
    assert family != _family(_catalog(shift=2))
    assert hash(family) != hash(_family(_catalog(shift=2)))


def test_native_parameter_names_can_differ_from_distributional_predictor_names():
    family = custom_family(
        "UserSecant",
        catalog=_catalog(),
        location="center",
        auxiliary={
            "scale": AuxParam(
                name="sigma", prior="LogNormal(0.0, 0.3)", link=links["log"]
            )
        },
        data={"bias": "known_bias"},
    )
    fitted = fit(
        "y ~ x",
        data=_frame(),
        family=family,
        distributional={"sigma": "~x"},
        method="svi",
        num_steps=0,
    )
    assert set(fitted.draws(num_samples=3)) == {
        "intercept",
        "beta_x",
        "dpar_sigma_intercept",
        "dpar_sigma_beta_x",
    }
    assert fitted.posterior_epred(
        new_data=_frame().drop(columns="y"), num_samples=3
    ).shape == (1, 3, 4)


def test_mode_curvature_uses_the_authored_custom_law_through_native_joint_ad():
    frame = _frame().assign(
        x=[0.0, 0.0, 0.0, 0.0], known_bias=[0.0, 0.0, 0.0, 0.0], y=[0.0, 0.0, 0.0, 0.0]
    )
    fitted = fit(
        "y ~ 1",
        data=frame,
        family=_family(),
        fixed_parameters={"scale": 1.0},
        method="svi",
        num_steps=0,
    )

    def likelihood(value):
        observations = {**fitted.observations, "intercept": value}
        joint = fitted.program.log_joint(torch.zeros(1, 1), observations).sum()
        with observation_scoring(False):
            prior = fitted.program.log_joint(torch.zeros(1, 1), observations).sum()
        return joint - prior

    point = torch.tensor(0.0, requires_grad=True)
    gradient = torch.autograd.grad(likelihood(point), point, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient, point)[0]
    torch.testing.assert_close(gradient, torch.tensor(0.0))
    torch.testing.assert_close(curvature, torch.tensor(-4.0))


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_arbitrary_law_fits_and_predicts_without_another_fitting_engine(method):
    settings = (
        {"num_steps": 2}
        if method == "svi"
        else {"num_warmup": 2, "num_samples": 3, "num_chains": 1}
    )
    fitted = fit(
        "y ~ x", data=_frame(), family=_family(), method=method, seed=7, **settings
    )
    assert "bias" not in fitted.draws(num_samples=3)
    fresh = _frame().iloc[[3, 1]].copy()
    fresh["known_bias"] = [0.8, -0.7]
    means = fitted.posterior_epred(
        new_data=fresh.drop(columns="y"), num_samples=3, seed=13
    )
    scores = fitted.log_likelihood(new_data=fresh, num_samples=3, seed=13)["y"]
    draws = fitted.draws(num_samples=3, seed=13)
    location = (
        draws["intercept"][..., None]
        + draws["beta_x"][..., None] * torch.tensor(fresh.x.to_numpy())
        + torch.tensor(fresh.known_bias.to_numpy())
    )
    scale = draws["scale"][..., None]
    torch.testing.assert_close(means, location, check_dtype=False)
    expected = hypsecant.logpdf(
        fresh.y.to_numpy(), loc=location.detach().numpy(), scale=scale.detach().numpy()
    )
    np.testing.assert_allclose(scores, expected, rtol=2e-6, atol=2e-6)
    assert (
        fitted.posterior_predict(
            new_data=fresh.drop(columns="y"), num_samples=3, seed=13
        ).shape
        == means.shape
    )


def test_distributional_custom_joint_and_every_coefficient_gradient_match_scipy_and_torch_oracles():
    frame = _frame()
    fitted = fit(
        "y ~ x",
        data=frame,
        family=_family(),
        distributional={"scale": "~x"},
        method="svi",
        num_steps=0,
    )
    coefficients = torch.tensor([0.2, -0.3, 0.1, 0.4], requires_grad=True)
    names = ("intercept", "beta_x", "dpar_scale_intercept", "dpar_scale_beta_x")
    observations = {
        **fitted.observations,
        **dict(zip(names, coefficients, strict=True)),
    }
    domain = torch.zeros(1, 1)
    joint = fitted.program.log_joint(domain, observations).sum()
    with observation_scoring(False):
        prior = fitted.program.log_joint(domain, observations).sum()
    actual = joint - prior
    x = torch.tensor(frame.x.to_numpy())
    loc = (
        coefficients[0]
        + coefficients[1] * x
        + torch.tensor(frame.known_bias.to_numpy())
    )
    scale = (coefficients[2] + coefficients[3] * x).exp()
    z = (torch.tensor(frame.y.to_numpy()) - loc) / scale
    expected = (-math.log(math.pi) - scale.log() - z.cosh().log()).sum()
    torch.testing.assert_close(actual, expected, check_dtype=False)
    torch.testing.assert_close(
        torch.autograd.grad(actual, coefficients, retain_graph=True)[0],
        torch.autograd.grad(expected, coefficients)[0],
    )
    inspected = get_prior(
        "y ~ x", data=frame, family=_family(), distributional={"scale": "~x"}
    )
    assert "bias" not in str(inspected)


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "builtin", "link", "conflict"]
)
def test_bad_custom_roles_fail_before_compilation(failure):
    with pytest.raises((ValueError, TypeError)):
        if failure == "missing":
            custom_family("UserSecant", catalog=_catalog(), location="center")
        elif failure == "duplicate":
            custom_family(
                "UserSecant",
                catalog=_catalog(),
                location="center",
                data={"center": "x", "bias": "known_bias"},
            )
        elif failure == "builtin":
            custom_family("Normal", catalog=FamilyCatalog(), location="loc")
        elif failure == "link":
            custom_family(
                "UserSecant", catalog=_catalog(), location="center", link="unknown"
            )
        else:
            fit(
                "y ~ x",
                data=_frame(),
                family=_family(),
                fixed_parameters={"bias": 1.0},
                method="svi",
                num_steps=0,
            )


class _ShiftedPoisson(Distribution):
    arg_constraints = {"rate": constraints.positive}
    support = constraints.integer_interval(-math.inf, math.inf)

    def __init__(self, rate, shift):
        self.rate, self.shift = torch.broadcast_tensors(rate, shift)
        self.base = torch.distributions.Poisson(self.rate, validate_args=False)
        super().__init__(self.rate.shape, validate_args=False)

    @property
    def mean(self):
        return self.rate + self.shift

    @property
    def variance(self):
        return self.rate

    def sample(self, sample_shape=torch.Size()):
        return self.base.sample(sample_shape).to(torch.int64) + self.shift

    def log_prob(self, value):
        centered = value - self.shift
        safe = centered.clamp_min(0)
        return torch.where(centered >= 0, self.base.log_prob(safe), -math.inf)


def _count_family():
    semantic = DistributionFamily(
        name="UserShiftedPoisson",
        parameters=(
            FamilyParameter(name="rate", constraint="positive", rank=0),
            FamilyParameter(name="shift", constraint="integer", rank=0),
        ),
        support="integer",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
    )
    catalog = FamilyCatalog().with_family(
        semantic,
        lambda parameters: _ShiftedPoisson(parameters["rate"], parameters["shift"]),
        support=_ShiftedPoisson.support,
    )
    return custom_family(
        "UserShiftedPoisson",
        catalog=catalog,
        location="rate",
        link="log",
        data={"shift": "known_shift"},
    )


def test_custom_integer_response_and_typed_integer_extra_data_use_shared_carriers():
    frame = pd.DataFrame(
        {"y": [2, 4, 3, 5], "x": [-1.0, 0.0, 0.5, 1.0], "known_shift": [1, 2, 1, 3]}
    )
    fitted = fit(
        "y ~ x", data=frame, family=_count_family(), method="svi", num_steps=2, seed=7
    )
    assert fitted.observations["y"].dtype == torch.int64
    assert any(
        value.dtype == torch.int64
        for name, value in fitted.observations.items()
        if name.startswith("nlcov_")
    )
    fresh = frame.iloc[[3, 1]].copy()
    fresh["known_shift"] = [6, 7]
    fresh["y"] = [8, 10]
    draws = fitted.draws(num_samples=5, seed=11)
    rate = (
        draws["intercept"][..., None]
        + draws["beta_x"][..., None] * torch.tensor(fresh.x.to_numpy())
    ).exp()
    expected = rate + torch.tensor(fresh.known_shift.to_numpy())
    torch.testing.assert_close(
        fitted.posterior_epred(
            new_data=fresh.drop(columns="y"), num_samples=5, seed=11
        ),
        expected,
        check_dtype=False,
    )
    scores = fitted.log_likelihood(new_data=fresh, num_samples=5, seed=11)["y"]
    np.testing.assert_allclose(
        scores,
        poisson.logpmf(
            fresh.y.to_numpy() - fresh.known_shift.to_numpy(), rate.detach()
        ),
        atol=2e-6,
        rtol=2e-6,
    )
    replicated = fitted.posterior_predict(
        new_data=fresh.drop(columns="y"), num_samples=5, seed=11
    )
    assert (replicated >= torch.tensor(fresh.known_shift.to_numpy())).all()
    assert torch.equal(replicated, replicated.round())
    with pytest.raises(ValueError, match="carrier"):
        fitted.posterior_epred(new_data=fresh.assign(known_shift=[0.5, 1.5]))


def test_custom_weights_subsets_updates_and_refit_partitions_reuse_one_raw_likelihood():
    from quivers.diagnostics import FoldPlan, cross_validate

    frame = _frame().assign(w=[0.0, 0.5, 1.0, 2.0], keep=[True, False, True, True])
    fitted = fit(
        "y | weights(w) + subset(keep) ~ x",
        data=frame,
        family=_family(),
        method="svi",
        num_steps=2,
        seed=3,
    )
    updated = fitted.update(num_steps=0)
    assert updated.family.custom.catalog is fitted.family.custom.catalog
    callback = fitted.refit_callback(
        method_factory=lambda settings, seed: {**settings, "seed": seed}, num_samples=5
    )
    plan = FoldPlan(assignments=(0, 1, 2), seed=13)
    for index in range(3):
        split = plan.split(index)
        result = callback(split)
        row = [0, 2, 3][index]
        fresh = frame.iloc[[row]]
        actual = result.fit.log_likelihood(
            new_data=fresh, num_samples=5, seed=split.seed
        )["y"]
        torch.testing.assert_close(result.log_likelihood["y"], actual)
    assert cross_validate(plan, callback).complete


def test_custom_censoring_uses_declared_atomicity_and_the_existing_interval_operation():
    frame = _frame().assign(
        status=[-1, 0, 1, 2], upper=[math.nan, math.nan, math.nan, 1.8]
    )
    fitted = fit(
        "y | cens(status, upper) ~ x",
        data=frame,
        family=_family(),
        fixed_parameters={"scale": 1.1},
        method="svi",
        num_steps=0,
    )
    samples = fitted.draws(num_samples=7, seed=19)
    loc = (
        samples["intercept"][..., None]
        + samples["beta_x"][..., None] * torch.tensor(frame.x.to_numpy())
        + torch.tensor(frame.known_bias.to_numpy())
    )
    location = loc.detach().numpy()
    values = frame.y.to_numpy()
    expected = np.stack(
        (
            hypsecant.logcdf(values[0], loc=location[..., 0], scale=1.1),
            hypsecant.logpdf(values[1], loc=location[..., 1], scale=1.1),
            hypsecant.logsf(values[2], loc=location[..., 2], scale=1.1),
            np.log(
                hypsecant.cdf(1.8, loc=location[..., 3], scale=1.1)
                - hypsecant.cdf(values[3], loc=location[..., 3], scale=1.1)
            ),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(
        fitted.log_likelihood(num_samples=7, seed=19)["y"],
        expected,
        atol=3e-6,
        rtol=3e-6,
    )
