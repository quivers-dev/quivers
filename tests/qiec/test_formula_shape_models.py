"""Generated shape-family programs agree across both runtimes and scalar oracles."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import special, stats

from quivers.dsl import loads
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.formulas import FormulaToQVRModule, families, formula_from_data
from quivers.inference import trace
from quivers.qiec.program_runtime import run_program


_AUXILIARIES = {
    "skew_normal": {"sigma": 0.6, "alpha": -2.0},
    "shifted_lognormal": {"sigma": 0.6, "ndt": 0.03},
    "frechet": {"nu": 3.5},
    "gen_extreme_value": {"sigma": 0.6, "xi": 0.0},
}


def _frame():
    return pd.DataFrame(
        {"y": [0.4, 0.8, 1.4], "x": [-1.0, 0.0, 1.0], "z": [0.2, -0.4, 0.8]}
    )


def _likelihood(name, eta, auxiliary, y):
    if name == "skew_normal":
        alpha = auxiliary["alpha"]
        delta = alpha / np.sqrt(1 + alpha**2)
        scale = auxiliary["sigma"] / np.sqrt(1 - 2 * delta**2 / np.pi)
        loc = eta - scale * delta * np.sqrt(2 / np.pi)
        return stats.skewnorm.logpdf(y, alpha, loc=loc, scale=scale).sum()
    if name == "shifted_lognormal":
        return stats.lognorm.logpdf(
            y, auxiliary["sigma"], loc=auxiliary["ndt"], scale=np.exp(eta)
        ).sum()
    if name == "frechet":
        nu = auxiliary["nu"]
        return stats.invweibull.logpdf(
            y, nu, scale=np.exp(eta) / special.gamma(1 - 1 / nu)
        ).sum()
    return stats.genextreme.logpdf(
        y, -auxiliary["xi"], loc=eta, scale=auxiliary["sigma"]
    ).sum()


def _base_prior(sites, frame):
    return stats.norm.logpdf(sites["intercept"], scale=5) + stats.norm.logpdf(
        sites["beta_x"], scale=5 / np.sqrt(np.mean(np.square(frame.x)))
    )


def _scalar_prior(name, sites, frame):
    result = _base_prior(sites, frame)
    if "sigma" in sites:
        result += stats.halfcauchy.logpdf(sites["sigma"], scale=2)
    if name == "skew_normal":
        result += stats.norm.logpdf(sites["alpha"], scale=4)
    elif name == "shifted_lognormal":
        result += stats.uniform.logpdf(sites["ndt"], scale=frame.y.min())
    elif name == "frechet":
        # Gamma(2, rate=.1) survival at one has the independent elementary
        # form exp(-.1)*(1+.1), so no quadrature enters this oracle.
        log_survival = -0.1 + math.log1p(0.1)
        result += stats.gamma.logpdf(sites["nu"], 2, scale=10) - log_survival
    else:
        result += stats.norm.logpdf(sites["xi"], scale=0.3)
    return result


def _compile(name, *, priors=None, distributional=None, fixed_parameters=None):
    frame = _frame()
    formula = formula_from_data("y ~ x", frame)
    lens = FormulaToQVRModule(
        families[name],
        user_priors=priors,
        distributional=distributional,
        fixed_parameters=fixed_parameters,
    )
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    source = module_to_source(module)
    # Deliberately check the emitted source, rather than running an unchecked
    # expression or constructing a primitive family by hand.
    checked = lower_qvr_to_qiec(parse(source), file_path=f"{name}.qvr")
    native = loads(source).morphism
    columns = {
        key: tuple(value.double().tolist())
        for key, value in lens.fixed_column_observations(formula).items()
    }
    data = columns | {"y": tuple(frame.y)}
    return frame, checked, native, data


def _native_run(native, data, sites):
    tensors = {
        key: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for key, value in sites.items()
    }
    observations = {
        key: torch.tensor(value, dtype=torch.float64) for key, value in data.items()
    }
    return trace(native, torch.zeros(1, 1), observations | tensors), tensors


@pytest.mark.parametrize("name", _AUXILIARIES)
def test_complete_generated_model_default_priors_and_likelihood_match_reference(name):
    frame, checked, native, data = _compile(name)
    sites = {"intercept": 0.3, "beta_x": 0.1} | _AUXILIARIES[name]
    reference = run_program(checked, "model", data=data, sites=sites)
    recorded, _ = _native_run(native, data, sites)
    eta = sites["intercept"] + sites["beta_x"] * frame.x.to_numpy()
    expected = _scalar_prior(name, sites, frame) + _likelihood(
        name, eta, sites, frame.y.to_numpy()
    )
    assert reference.log_joint == pytest.approx(expected, abs=2e-10)
    assert recorded.log_joint.sum().item() == pytest.approx(expected, abs=3e-5)
    assert recorded.sites["y"].is_observed
    if name == "frechet":
        expected_prior = (
            stats.gamma.logpdf(sites["nu"], 2, scale=10) + 0.1 - math.log1p(0.1)
        )
        assert recorded.sites["nu"].log_prob.item() == pytest.approx(
            expected_prior, abs=1e-6
        )


@pytest.mark.parametrize("name", _AUXILIARIES)
def test_complete_model_latent_gradients_match_checked_reference_and_scipy(name):
    frame, checked, native, data = _compile(name)
    sites = {"intercept": 0.3, "beta_x": 0.1} | _AUXILIARIES[name]
    recorded, tensors = _native_run(native, data, sites)
    actual = torch.autograd.grad(recorded.log_joint.sum(), tuple(tensors.values()))
    step = 2e-5
    for (parameter, value), gradient in zip(sites.items(), actual, strict=True):
        reference_scores, oracle_scores = [], []
        for direction in (-1, 1):
            changed = sites | {parameter: value + direction * step}
            reference_scores.append(
                run_program(checked, "model", data=data, sites=changed).log_joint
            )
            eta = changed["intercept"] + changed["beta_x"] * frame.x.to_numpy()
            oracle_scores.append(
                _scalar_prior(name, changed, frame)
                + _likelihood(name, eta, changed, frame.y.to_numpy())
            )
        reference_gradient = (reference_scores[1] - reference_scores[0]) / (2 * step)
        oracle_gradient = (oracle_scores[1] - oracle_scores[0]) / (2 * step)
        assert reference_gradient == pytest.approx(oracle_gradient, rel=3e-7, abs=3e-7)
        assert gradient.item() == pytest.approx(oracle_gradient, rel=2e-5, abs=2e-5)


@pytest.mark.parametrize(
    "prior,distribution,low,high,nu",
    [
        ("Normal(0.0, 2.0)", stats.norm(scale=2), 1.0, math.inf, 3.5),
        ("Gamma(1.7, 0.9)", stats.gamma(1.7, scale=1 / 0.9), 1.0, math.inf, 3.5),
        (
            "Restrict(base=Gamma(concentration=2.0, rate=0.1), low=1.2, high=6.0)",
            stats.gamma(2, scale=10),
            1.2,
            6.0,
            3.5,
        ),
    ],
)
def test_generated_frechet_user_prior_is_normalized_in_both_runtimes(
    prior, distribution, low, high, nu
):
    frame, checked, native, data = _compile("frechet", priors={"nu": prior})
    sites = {"intercept": 0.3, "beta_x": 0.1, "nu": nu}
    eta = 0.3 + 0.1 * frame.x.to_numpy()
    log_mass = math.log(distribution.sf(low) - distribution.sf(high))
    expected_prior = distribution.logpdf(nu) - log_mass
    expected = (
        _base_prior(sites, frame)
        + expected_prior
        + _likelihood("frechet", eta, sites, frame.y.to_numpy())
    )
    reference = run_program(checked, "model", data=data, sites=sites)
    recorded, _ = _native_run(native, data, sites)
    assert reference.log_joint == pytest.approx(expected, abs=2e-9)
    assert recorded.log_joint.sum().item() == pytest.approx(expected, abs=3e-5)
    assert recorded.sites["nu"].log_prob.item() == pytest.approx(
        expected_prior, abs=2e-6
    )


@pytest.mark.parametrize(
    "name,parameter,intercept",
    [
        ("skew_normal", "alpha", -1.3),
        ("shifted_lognormal", "ndt", math.log(0.03)),
        ("frechet", "nu", math.log(2.5)),
        ("gen_extreme_value", "xi", 0.1),
    ],
)
def test_distributional_shape_predictors_preserve_links_in_checked_source(
    name, parameter, intercept
):
    auxiliary_formula = formula_from_data("y ~ z", _frame())
    fixed = {} if name == "frechet" else {"sigma": 0.6}
    frame, checked, native, data = _compile(
        name, distributional={parameter: auxiliary_formula}, fixed_parameters=fixed
    )
    sites = {
        "intercept": 0.3,
        "beta_x": 0.1,
        f"dpar_{parameter}_intercept": intercept,
        f"dpar_{parameter}_beta_z": 0.1,
    }

    def oracle(values):
        eta = values["intercept"] + values["beta_x"] * frame.x.to_numpy()
        linear = values[f"dpar_{parameter}_intercept"] + values[
            f"dpar_{parameter}_beta_z"
        ] * np.asarray(data[f"dpar_{parameter}_z"])
        if parameter == "nu":
            natural = 1 + np.exp(linear)
        elif parameter == "ndt":
            natural = np.exp(linear)
        else:
            natural = linear
        prior = _base_prior(values, frame)
        prior += stats.norm.logpdf(values[f"dpar_{parameter}_intercept"], scale=5)
        prior += stats.norm.logpdf(
            values[f"dpar_{parameter}_beta_z"],
            scale=5 / np.sqrt(np.mean(np.square(frame.z))),
        )
        return prior + _likelihood(
            name, eta, fixed | {parameter: natural}, frame.y.to_numpy()
        )

    expected = oracle(sites)
    reference = run_program(checked, "model", data=data, sites=sites)
    recorded, tensors = _native_run(native, data, sites)
    assert reference.log_joint == pytest.approx(expected, abs=2e-9)
    assert recorded.log_joint.sum().item() == pytest.approx(expected, abs=3e-5)
    gradients = torch.autograd.grad(recorded.log_joint.sum(), tuple(tensors.values()))
    step = 2e-5
    for (key, value), gradient in zip(sites.items(), gradients, strict=True):
        lower, upper = sites | {key: value - step}, sites | {key: value + step}
        expected_gradient = (oracle(upper) - oracle(lower)) / (2 * step)
        reference_gradient = (
            run_program(checked, "model", data=data, sites=upper).log_joint
            - run_program(checked, "model", data=data, sites=lower).log_joint
        ) / (2 * step)
        assert reference_gradient == pytest.approx(
            expected_gradient, rel=3e-7, abs=3e-7
        )
        assert gradient.item() == pytest.approx(expected_gradient, rel=2e-5, abs=2e-5)


@pytest.mark.parametrize("rate", [10.0, 100.0, 1000.0])
def test_frechet_conditioned_gamma_prior_preserves_small_survival_mass(rate):
    frame, checked, native, data = _compile(
        "frechet", priors={"nu": f"Gamma(2.0, {rate})"}
    )
    sites = {"intercept": 0.3, "beta_x": 0.1, "nu": 1.01}
    # For shape two, log survival at one is -rate + log(1+rate),
    # including tails whose probability underflows even in Float64.
    expected_prior = (
        2 * math.log(rate)
        + math.log(sites["nu"])
        - rate * sites["nu"]
        + rate
        - math.log1p(rate)
    )
    eta = 0.3 + 0.1 * frame.x.to_numpy()
    expected_joint = (
        _base_prior(sites, frame)
        + expected_prior
        + _likelihood("frechet", eta, sites, frame.y.to_numpy())
    )
    reference = run_program(checked, "model", data=data, sites=sites)
    recorded, _ = _native_run(native, data, sites)
    assert reference.log_joint == pytest.approx(expected_joint, abs=2e-9)
    assert recorded.log_joint.sum().item() == pytest.approx(expected_joint, abs=1e-4)
    assert recorded.sites["nu"].log_prob.item() == pytest.approx(
        expected_prior, abs=1e-4
    )


@pytest.mark.parametrize(
    "shape", [1e-10, 0.01, 0.1, 0.5, 1.0, 1.7, 2.0, 5.3, 100.0, 1000.0]
)
@pytest.mark.parametrize("point", [1e-10, 0.01, 0.5, 1.0, 10.0, 100.0, 1000.0])
def test_reference_gamma_regularized_log_tails_match_independent_scipy(shape, point):
    from quivers.qiec._gamma_tails import _regularized_gamma_logs

    lower, upper = _regularized_gamma_logs(shape, point)
    wanted_lower = stats.gamma.logcdf(point, shape)
    wanted_upper = stats.gamma.logsf(point, shape)
    if math.isfinite(wanted_lower):
        assert lower == pytest.approx(wanted_lower, rel=2e-12, abs=2e-12)
    if math.isfinite(wanted_upper):
        assert upper == pytest.approx(wanted_upper, rel=2e-12, abs=2e-12)
    assert np.logaddexp(lower, upper) == pytest.approx(0, abs=2e-15)


@pytest.mark.parametrize(
    "shape,rate,low,high",
    [
        (0.1, 0.3, 0.0, 1.0),
        (0.5, 2.0, 0.0, math.inf),
        (1.7, 0.9, 1.0, math.inf),
        (5.3, 1.0, 10.0, 11.0),
        (100.0, 0.1, 800.0, 1200.0),
        (1000.0, 1.0, 950.0, 1050.0),
        (2.0, 1000.0, 0.0, 0.001),
        (2.0, 1000.0, 1.0, 1.1),
    ],
)
def test_reference_gamma_interval_mass_matches_cdf_or_elementary_tail(
    shape, rate, low, high
):
    from quivers.qiec._gamma_tails import _gamma_interval_log_mass

    actual = _gamma_interval_log_mass(shape, rate, low, high)
    if shape == 2 and rate * low >= 1:
        log_low_sf = -rate * low + math.log1p(rate * low)
        log_high_sf = -rate * high + math.log1p(rate * high)
        expected = log_low_sf + math.log(-math.expm1(log_high_sf - log_low_sf))
    else:
        law = stats.gamma(shape, scale=1 / rate)
        expected = (
            math.log(law.cdf(high) - law.cdf(low))
            if law.cdf(high) < 0.5
            else math.log(law.sf(low) - law.sf(high))
        )
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)


def test_reference_gamma_tail_iteration_limit_is_explicit(monkeypatch):
    from quivers.qiec import _gamma_tails
    from quivers.qiec.reference_support import DistributionError

    monkeypatch.setattr(_gamma_tails, "_MAX_ITERATIONS", 1)
    with pytest.raises(DistributionError, match="did not converge"):
        _gamma_tails._regularized_gamma_logs(1.7, 0.5)


def test_explicit_truncated_gamma_reference_preserves_log_domain_tail():
    source = """object Out : Real 1
program model : Out -> Out
    sample value <- Truncated(base=Gamma(concentration=2.0, rate=1000.0), low=1.0, high=2.0)
    return value
export model
"""
    checked = lower_qvr_to_qiec(parse(source))
    result = run_program(checked, "model", data={"out": (0.0,)}, sites={"value": 1.01})
    log_mass = -1000 + math.log1p(1000)
    expected = 2 * math.log(1000) + math.log(1.01) - 1010 - log_mass
    assert result.log_joint == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize(
    "shape,point,expected",
    [
        (0.13, 1000.0, -1007.9888883440558162898072533354562644),
        (5.3, 1000.0, -973.9319834263428467865321583122627786),
        (1000.0, 100000.0, -94403.79784357095283538906779494503236),
        (1e-20, 0.1, -45.45126007738602743807656064079455),
    ],
)
def test_reference_gamma_far_tails_match_80_digit_incomplete_integrals(
    shape, point, expected
):
    from quivers.qiec._gamma_tails import _regularized_gamma_logs

    assert _regularized_gamma_logs(shape, point)[1] == pytest.approx(
        expected, rel=2e-15, abs=2e-12
    )


@pytest.mark.parametrize(
    "shape,rate,low,high,expected",
    [
        (0.5, 1.0, 1.0, 1.0 + 1e-12, -29.2032971622230799331906872172),
        (100.0, 1.0, 100.0, 100.0 + 1e-10, -26.2481900405457886616550049599),
    ],
)
def test_reference_gamma_narrow_interval_matches_80_digit_density_integral(
    shape, rate, low, high, expected
):
    from quivers.qiec._gamma_tails import _gamma_interval_log_mass

    assert _gamma_interval_log_mass(shape, rate, low, high) == pytest.approx(
        expected, abs=2e-12
    )
    from quivers.continuous._interval import log_interval_mass

    concentration = torch.tensor(shape, dtype=torch.float64, requires_grad=True)
    inverse_scale = torch.tensor(rate, dtype=torch.float64, requires_grad=True)
    native = log_interval_mass(
        torch.distributions.Gamma(concentration, inverse_scale),
        torch.tensor(low, dtype=torch.float64),
        torch.tensor(high, dtype=torch.float64),
    )
    assert native.item() == pytest.approx(expected, abs=2e-12)
    shape_gradient, rate_gradient = torch.autograd.grad(
        native, (concentration, inverse_scale)
    )
    midpoint = (low + high) / 2
    assert shape_gradient.item() == pytest.approx(
        math.log(rate * midpoint) - special.digamma(shape), abs=2e-9
    )
    assert rate_gradient.item() == pytest.approx(shape / rate - midpoint, abs=2e-9)
