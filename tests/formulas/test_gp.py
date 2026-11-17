"""Approximate brms ``gp()`` terms: bases, priors, densities, and workflow.

The oracle ``oracles/gp_brms.json`` is written by ``oracles/gp_brms.R`` with
brms 2.20.4 (``make_standata`` and ``get_prior``). It records, per formula,
brms's basis matrix ``Xgp`` on distinct inputs, the row index ``Jgp``, the
sub-GP rows ``Igp`` and contrasts ``Cgp``, the frequencies ``slambda``, and
the default ``sdgp`` and ``lscale`` priors.

Tolerances are fixed from floating-point error before comparison. The oracle
prints 15 significant digits and every basis entry is a product of at most
two sines of magnitude at most ``1 / sqrt(L) <= 1``, so ``1e-12`` absolute
error covers printing and evaluation-order rounding while any change of
scaling, centering, boundary, or order moves entries by more than ``1e-3``.
brms rounds its inverse-gamma parameters to six decimals, so they must match
exactly as strings.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.integrate as integrate
import scipy.special as special
import torch

from quivers.dsl import Compiler
from quivers.formulas import (
    FormulaToQVRModule,
    PriorSpec,
    conditional_smooths,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from quivers.formulas.formula import GaussianDesignTerm
from quivers.formulas._gp import (
    GaussianProcessBlock,
    lscale_prior,
    spectral_density,
)
from quivers.diagnostics.cross_validation import FoldPlan, cross_validate
from quivers.inference.trace import trace
from quivers.modeling import load_fit, refit_callback, save_fit

ORACLE = json.loads((Path(__file__).parent / "oracles" / "gp_brms.json").read_text())
FRAME = pd.DataFrame(ORACLE["data"]).assign(g=ORACLE["g"])


def _gp(block: GaussianDesignTerm) -> GaussianProcessBlock:
    """Return a design block's retained GP, which every block here has."""
    assert block.gp is not None
    return block.gp


def _brms_rows(standata: dict, suffix: str, rows: int) -> np.ndarray:
    """Expand brms's distinct-input basis to every response row.

    Parameters
    ----------
    standata : dict
        One case's GP standata.
    suffix : str
        ``_1`` or ``_1_j`` for a by contrast.
    rows : int
        Response rows.

    Returns
    -------
    numpy.ndarray
        Row-by-basis matrix, zero outside the sub-GP's rows.
    """
    basis = np.atleast_2d(np.asarray(standata["Xgp" + suffix], dtype=float))
    jgp = standata.get("Jgp" + suffix)
    igp = standata.get("Igp" + suffix)
    cgp = standata.get("Cgp" + suffix)
    member = np.arange(rows) if igp is None else np.asarray(igp, dtype=int) - 1
    values = basis if jgp is None else basis[np.asarray(jgp, dtype=int) - 1]
    if cgp is not None:
        values = values * np.asarray(cgp, dtype=float)[:, None]
    result = np.zeros((rows, values.shape[1]))
    result[member] = values
    return result


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["formula"])
def test_basis_frequencies_and_default_priors_equal_brms(case):
    formula = formula_from_data(case["formula"], FRAME)
    standata = case["standata"]
    blocks = formula.design_terms
    assert len(blocks) == standata["Kgp_1"]
    rows = len(FRAME)
    for j, block in enumerate(blocks):
        suffix = "_1" if len(blocks) == 1 else f"_1_{j + 1}"
        np.testing.assert_allclose(
            block.data, _brms_rows(standata, suffix, rows), rtol=0, atol=1e-12
        )
        frequencies = np.asarray(standata["slambda" + suffix], dtype=float)
        np.testing.assert_allclose(
            _gp(block).frequencies,
            frequencies.reshape(_gp(block).frequencies.shape),
            rtol=0,
            atol=1e-12,
        )
    expected = {
        (prior["class"], prior["coef"]): prior["prior"] for prior in case["priors"]
    }
    for block in blocks:
        for label, prior in zip(
            _gp(block).lscale_labels, _gp(block).lscale_priors, strict=True
        ):
            shape, scale = expected[("lscale", label)][len("inv_gamma(") : -1].split(
                ", "
            )
            assert prior == f"InverseGamma({float(shape)!r}, {float(scale)!r})"
        assert ("sdgp", _gp(block).label) in expected
    response = case["formula"].split(" ~")[0]
    scale = float(expected[("sdgp", "")].split(", ")[-1].rstrip(")"))
    priors = get_prior(case["formula"], data=FRAME)
    for block in blocks:
        assert (
            priors[f"sdgp_{_gp(block).label}"]
            == f"HalfStudentT(df=3.0, scale={scale!r})"
        )
    assert response in {"y", "y2"}


def test_prior_coefficients_follow_brms_labels():
    case = next(c for c in ORACLE["cases"] if "cmc = FALSE" in c["formula"])
    priors = get_prior(
        case["formula"],
        data=FRAME,
        priors=[
            PriorSpec(distribution="HalfNormal(0.3)", class_="lscale", coef="gpxgb"),
            PriorSpec(
                distribution="HalfNormal(2.0)", class_="sdgp", coef="gpxgIntercept"
            ),
        ],
    )
    assert priors["lscale_gpxgb"] == "HalfNormal(scale=0.3)"
    assert priors["sdgp_gpxgIntercept"] == "HalfNormal(scale=2.0)"
    assert priors["lscale_gpxgc"].startswith("InverseGamma(")
    labels = {
        prior["coef"]
        for prior in case["priors"]
        if prior["coef"] and prior["class"] == "sdgp"
    }
    assert {f"sdgp_{label}" for label in labels} <= set(priors)


def test_lscale_prior_places_one_percent_beyond_the_distance_bounds():
    points = np.array([[0.0], [0.1], [0.35], [1.0]])
    prior = lscale_prior(points)
    shape, scale = (float(v) for v in prior[len("InverseGamma(") : -1].split(", "))
    # P(l < lb) = Q(a, b / lb) and P(l > ub) = P(a, b / ub) for inverse gamma.
    assert special.gammaincc(shape, scale / 0.1) == pytest.approx(0.01, abs=1e-5)
    assert special.gammainc(shape, scale / 1.0) == pytest.approx(0.01, abs=1e-5)
    with pytest.raises(ValueError, match="one value"):
        lscale_prior(np.zeros((3, 1)))


def _kernel(kernel: str, distance: np.ndarray, lscale: float) -> np.ndarray:
    """Rasmussen and Williams's stationary kernels with unit variance."""
    r = np.abs(distance) / lscale
    if kernel == "exp_quad":
        return np.exp(-0.5 * r**2)
    if kernel == "exponential":
        return np.exp(-r)
    if kernel == "matern32":
        return (1 + math.sqrt(3) * r) * np.exp(-math.sqrt(3) * r)
    return (1 + math.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-math.sqrt(5) * r)


@pytest.mark.parametrize("kernel", ["exp_quad", "exponential", "matern32", "matern52"])
def test_spectral_density_is_the_fourier_transform_of_the_kernel(kernel):
    # S(w) = int k(t) exp(-i w t) dt, computed by adaptive quadrature.
    frequencies = np.array([[0.3], [1.1], [2.7], [6.0]])
    block = GaussianProcessBlock(
        label="gpx",
        lscale_labels=("gpx",),
        kernel=kernel,
        frequencies=frequencies,
        lscale_priors=("HalfNormal(1.0)",),
    )
    sdgp, lscale = 1.3, 0.7
    actual = spectral_density(
        block,
        torch.tensor(sdgp, dtype=torch.float64),
        (torch.tensor(lscale, dtype=torch.float64),),
    ).numpy()
    for value, w in zip(actual, frequencies[:, 0], strict=True):
        # The kernel is even and below 1e-20 beyond 60 length scales, so
        # S(w) = 2 int_0^{60 l} k(t) cos(w t) dt to quadrature accuracy.
        integral, _ = integrate.quad(
            lambda t: 2 * sdgp**2 * _kernel(kernel, np.array(t), lscale),
            0,
            60 * lscale,
            weight="cos",
            wvar=w,
            epsabs=1e-14,
            epsrel=1e-12,
            limit=500,
        )
        # QAWO's absolute error target is 1e-14 against densities above 1e-4.
        assert value == pytest.approx(integral, rel=1e-8)


def test_two_dimensional_densities_match_rasmussen_williams_closed_forms():
    # Matern: S(s) in cycles (R&W eq. 4.15) at s = w / (2 pi).
    frequencies = np.array([[0.4, 1.3], [2.0, 0.5]])
    norm2 = (frequencies**2).sum(1)
    sdgp, lscale = 0.8, 0.45
    for kernel, nu in (("matern32", 1.5), ("matern52", 2.5), ("exponential", 0.5)):
        block = GaussianProcessBlock(
            label="gpxz",
            lscale_labels=("gpxz",),
            kernel=kernel,
            frequencies=frequencies,
            lscale_priors=("HalfNormal(1.0)",),
        )
        actual = spectral_density(
            block,
            torch.tensor(sdgp, dtype=torch.float64),
            (torch.tensor(lscale, dtype=torch.float64),) * 2,
        ).numpy()
        s2 = norm2 / (2 * math.pi) ** 2
        expected = (
            sdgp**2
            * 2**2
            * math.pi
            * math.gamma(nu + 1)
            * (2 * nu) ** nu
            / (math.gamma(nu) * lscale ** (2 * nu))
            * (2 * nu / lscale**2 + 4 * math.pi**2 * s2) ** (-(nu + 1))
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
    # Squared exponential with per-axis length scales is a product of 1-D transforms.
    block = GaussianProcessBlock(
        label="gpxz",
        lscale_labels=("gpxzx", "gpxzz"),
        kernel="exp_quad",
        frequencies=frequencies,
        lscale_priors=("HalfNormal(1.0)",) * 2,
    )
    lengths = (0.3, 0.9)
    actual = spectral_density(
        block,
        torch.tensor(sdgp, dtype=torch.float64),
        tuple(torch.tensor(v, dtype=torch.float64) for v in lengths),
    ).numpy()
    expected = sdgp**2 * np.prod(
        [
            math.sqrt(2 * math.pi) * l * np.exp(-0.5 * l**2 * frequencies[:, d] ** 2)
            for d, l in enumerate(lengths)
        ],
        axis=0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


# Riutort-Mayol et al. (2023), eqs. (17)-(23): with m = a c / (l / S) basis
# functions and c >= b l / S, the approximate covariance's total variation
# error is below one percent of the exact covariance's area.
RELATIONS = {"exp_quad": (1.75, 3.2), "matern52": (2.65, 4.1), "matern32": (3.42, 4.5)}


@pytest.mark.parametrize("kernel", sorted(RELATIONS))
def test_approximate_covariance_meets_the_published_accuracy_and_converges(kernel):
    half_range, lscale = 1.0, 0.4
    a, b = RELATIONS[kernel]
    tau = np.linspace(0.0, half_range, 2001)
    exact = _kernel(kernel, tau, lscale)

    def error(c: float, m: int) -> np.ndarray:
        boundary = c * half_range
        frequencies = np.arange(1, m + 1, dtype=float) * math.pi / (2 * boundary)
        block = GaussianProcessBlock(
            label="gpx",
            lscale_labels=("gpx",),
            kernel=kernel,
            frequencies=frequencies[:, None],
            lscale_priors=("HalfNormal(1.0)",),
        )
        spd = spectral_density(
            block,
            torch.tensor(1.0, dtype=torch.float64),
            (torch.tensor(lscale, dtype=torch.float64),),
        ).numpy()

        def phi(x: np.ndarray) -> np.ndarray:
            return np.sin(frequencies * (x[:, None] + boundary)) / math.sqrt(boundary)

        return np.abs(exact - (phi(np.zeros(1)) * spd * phi(tau)).sum(-1))

    c = max(1.2, b * lscale / half_range)
    m = math.ceil(a * c / (lscale / half_range))
    # Eq. (17) at the published (m, c).
    area = integrate.trapezoid(exact, tau)
    assert integrate.trapezoid(error(c, m), tau) < 0.01 * area
    # Below the boundary-limited floor, more basis functions reduce the error.
    errors = [error(c, size).max() for size in (m // 2, m, 2 * m)]
    assert errors[0] > errors[1] > errors[2]
    # Widening the boundary with the basis drives the error toward zero.
    widened = [error(w, 4 * math.ceil(a * w / lscale)).max() for w in (1.5, 3.0)]
    assert widened[1] < widened[0] < errors[0]
    if kernel == "exp_quad":
        assert widened[1] < 1e-12


FORMULA = "y ~ gp(x, z, k = 3, cov = 'matern52', iso = FALSE) + gp(x, by = g, k = 4)"


def test_log_joint_and_derivatives_equal_an_independent_oracle():
    formula = formula_from_data(FORMULA, FRAME)
    lens = FormulaToQVRModule(families["gaussian"])
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    observed = lens.fixed_column_observations(formula)
    observed["y"] = torch.tensor(formula.response_values.copy(), dtype=torch.float64)
    blocks = formula.design_terms
    assert len(blocks) == 4
    widths = [block.data.shape[1] for block in blocks]
    scalars = sum(1 + len(_gp(block).lscale_labels) for block in blocks)
    size = 1 + scalars + sum(widths) + 1
    point = torch.linspace(0.2, 0.9, size, dtype=torch.float64).requires_grad_()
    priors = get_prior(FORMULA, data=FRAME)

    def split(v):
        out, at = {}, 1
        for block in blocks:
            out[f"sdgp_{_gp(block).label}"] = v[at]
            at += 1
            for label in _gp(block).lscale_labels:
                out[f"lscale_{label}"] = v[at]
                at += 1
        for block, width in zip(blocks, widths, strict=True):
            out[block.qvr_name + "_z"] = v[at : at + width] - 0.5
            at += width
        out["intercept"] = v[0]
        out["sigma"] = v[-1]
        return out

    def actual(v):
        return trace(
            program, v.new_zeros(1, 1), {**observed, **split(v)}
        ).log_joint.sum()

    def oracle(v):
        values = split(v)
        density = torch.distributions.Normal(0.0, 5.0).log_prob(values["intercept"])
        density = density + torch.distributions.HalfCauchy(2.0).log_prob(
            values["sigma"]
        )
        mu = values["intercept"].expand(len(FRAME))
        for block in blocks:
            sdgp = values[f"sdgp_{_gp(block).label}"]
            density = density + _student_half(sdgp, priors[f"sdgp_{_gp(block).label}"])
            lengths = []
            for label in _gp(block).lscale_labels:
                lscale = values[f"lscale_{label}"]
                shape, rate = _inverse_gamma(priors[f"lscale_{label}"])
                density = density + (
                    shape * math.log(rate)
                    - math.lgamma(shape)
                    - (shape + 1) * lscale.log()
                    - rate / lscale
                )
                lengths.append(lscale)
            if len(lengths) == 1:
                lengths = lengths * _gp(block).dimension
            w2 = v.new_tensor(_gp(block).frequencies ** 2)
            nu = {"matern52": 2.5, "exp_quad": None}[_gp(block).kernel]
            dim = _gp(block).dimension
            q = sum(l**2 * w2[:, d] for d, l in enumerate(lengths))
            if nu is None:
                spd = (
                    sdgp**2
                    * (2 * math.pi) ** (dim / 2)
                    * math.prod(lengths)
                    * torch.exp(-0.5 * q)
                )
            else:
                spd = (
                    sdgp**2
                    * 2**dim
                    * math.pi ** (dim / 2)
                    * math.gamma(nu + dim / 2)
                    * (2 * nu) ** nu
                    / math.gamma(nu)
                    * math.prod(lengths)
                    * (2 * nu + q) ** (-(nu + dim / 2))
                )
            raw = values[block.qvr_name + "_z"]
            density = density + torch.distributions.Normal(0.0, 1.0).log_prob(raw).sum()
            mu = mu + v.new_tensor(block.data.copy()) @ (spd.sqrt() * raw)
        return (
            density
            + torch.distributions.Normal(mu, values["sigma"])
            .log_prob(observed["y"])
            .sum()
        )

    torch.testing.assert_close(actual(point), oracle(point))
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(actual, point),
        torch.autograd.functional.jacobian(oracle, point),
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, point),
        torch.autograd.functional.hessian(oracle, point),
    )
    # A deliberately wrong kernel changes the density.
    wrong = formula_from_data(FORMULA.replace("matern52", "matern32"), FRAME)
    wrong_module, _ = FormulaToQVRModule(families["gaussian"]).forward(wrong)
    wrong_program = Compiler(wrong_module).compile().morphism
    wrong_value = trace(
        wrong_program, point.new_zeros(1, 1), {**observed, **split(point)}
    ).log_joint.sum()
    # Rounding error is below 1e-12 here; the kernels differ by far more.
    assert abs(float((wrong_value - actual(point)).detach())) > 1e-6


def _student_half(value: torch.Tensor, prior: str) -> torch.Tensor:
    """Half Student t log density from a ``HalfStudentT(df=..., scale=...)`` call."""
    df = float(prior.split("df=")[1].split(",")[0])
    scale = float(prior.split("scale=")[1].rstrip(")"))
    return math.log(2.0) + torch.distributions.StudentT(df, 0.0, scale).log_prob(value)


def _inverse_gamma(prior: str) -> tuple[float, float]:
    """Read ``InverseGamma(concentration=a, rate=b)``."""
    shape = float(prior.split("concentration=")[1].split(",")[0])
    rate = float(prior.split("rate=")[1].rstrip(")"))
    return shape, rate


def _fresh() -> pd.DataFrame:
    return pd.DataFrame(
        {"x": [-0.8, 0.4, 3.5], "z": [0.2, 1.1, 2.4], "g": ["b", "a", "c"]}
    )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_every_inference_method_predicts_fresh_rows(method):
    options = (
        {"num_steps": 3}
        if method == "svi"
        else {"num_warmup": 2, "num_samples": 2, "num_chains": 1}
    )
    fitted = fit(
        FORMULA, data=FRAME, family="gaussian", method=method, seed=2, **options
    )
    fresh = _fresh()
    for draws in (
        fitted.posterior_linpred(new_data=fresh, num_samples=2, seed=5),
        fitted.posterior_epred(new_data=fresh, num_samples=2, seed=5),
        fitted.posterior_predict(new_data=fresh, num_samples=2, seed=5),
        *fitted.log_likelihood(
            new_data=fresh.assign(y=[0.1, 0.4, 0.2]), num_samples=2, seed=5
        ).values(),
    ):
        assert draws.shape[-1] == 3 and torch.isfinite(draws).all()
    draws = fitted.draws(num_samples=2, seed=1)
    assert all(
        bool((draws[name] > 0).all())
        for name in draws
        if name.startswith(("sdgp_", "lscale_"))
    )
    with pytest.raises(ValueError, match="unseen levels"):
        fitted.posterior_epred(new_data=fresh.assign(g=["d", "a", "a"]), num_samples=1)


def test_fresh_rows_use_the_retained_scaling_centering_and_boundary():
    fitted = fit(FORMULA, data=FRAME, method="svi", num_steps=2, seed=3)
    torch.testing.assert_close(
        fitted.posterior_linpred(new_data=FRAME, num_samples=3, seed=4),
        fitted.posterior_linpred(num_samples=3, seed=4),
    )
    fresh = _fresh()
    effect = conditional_smooths(
        fitted, "gp(x, by=g, k=4)", fresh, num_samples=3, seed=2
    )
    draws = fitted.draws(num_samples=3, seed=2)
    expected = torch.zeros(effect.shape, dtype=torch.float64)
    for block in fitted.formula.design_terms:
        if block.term != "gp(x, by=g, k=4)":
            continue
        subterm = next(
            s
            for s in fitted.prediction_plan.designs["mu"]
            .common.terms[block.term]
            .components[0]
            .call.stateful_transform.state.subterms
            if s.block == block.gp
        )
        level = subterm.level
        # Rows of other levels contribute zero; the retained dmax, centers,
        # and boundary place fresh inputs on the training basis.
        centered = fresh["x"].to_numpy()[:, None] / subterm.dmax - subterm.centers
        half = subterm.boundary[0]
        orders = np.arange(1, 5)
        phi = np.sin(orders * math.pi / (2 * half) * (centered + half)) / math.sqrt(
            half
        )
        phi = phi * (fresh["g"].to_numpy() == level)[:, None]
        lengths = (draws[f"lscale_{_gp(block).label}"].double(),)
        spd = spectral_density(
            _gp(block), draws[f"sdgp_{_gp(block).label}"].double(), lengths
        )
        coefficients = spd.sqrt() * draws[block.qvr_name + "_z"].double()
        expected += torch.einsum("cdk,rk->cdr", coefficients, torch.as_tensor(phi))
    torch.testing.assert_close(effect, expected)


def test_update_kfold_and_artifact_round_trip(tmp_path, monkeypatch):
    import quivers.formulas._fit as api

    fitted = fit(FORMULA, data=FRAME, method="svi", num_steps=1, seed=7)
    refit = fitted.update(data=FRAME.iloc[:18].reset_index(drop=True))
    assert refit.formula.response_values.shape == (18,)
    callback = refit_callback(
        fitted.refit_recipe,
        lambda data, rows: data.iloc[list(rows)].copy(),
        lambda refitted, held, seed: refitted.log_likelihood(
            new_data=held, num_samples=2, seed=seed
        ),
        method_factory=lambda settings, seed: {**settings, "seed": seed},
    )
    result = cross_validate(
        FoldPlan(assignments=tuple(i % 2 for i in range(len(FRAME))), seed=1), callback
    )
    assert result.complete and result.log_predictive_density.shape == (len(FRAME),)
    fresh = _fresh()
    expected = fitted.posterior_epred(new_data=fresh, num_samples=2, seed=9)
    path = tmp_path / "gp.qfit"
    save_fit(fitted, path)

    def forbid(*args, **kwargs):
        raise AssertionError("restoring a gp fit must not run inference")

    monkeypatch.setattr(api, "_run_inference", forbid)
    restored = load_fit(path)
    assert restored.formula.design_terms == fitted.formula.design_terms
    torch.testing.assert_close(
        restored.posterior_epred(new_data=fresh, num_samples=2, seed=9), expected
    )


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("y ~ gp(x, k = 0)", "'k' must be positive"),
        ("y ~ gp(x, k = 4, cov = 'periodic')", "not a valid GP covariance kernel"),
        ("y ~ gp(x, z, k = 4, c = c(1.5, 2, 3))", "same length as the number"),
        ("y ~ gp(x, k = 4, c = -1)", "'c' must be positive"),
        ("y ~ gp(x, k = 4, nug = 0.1)", "not supported"),
        ("y ~ gp(g, k = 4)", "should be numeric vectors"),
        ("y ~ gp(x, k = 4):z", "standalone additive terms"),
        ("y ~ (gp(x, k = 4) | g)", "group-level"),
    ],
)
def test_invalid_specifications_are_rejected(formula, message):
    with pytest.raises(ValueError, match=message):
        formula_from_data(formula, FRAME)


def test_categorical_predictors_give_each_category_a_gp():
    frame = FRAME.assign(cat=np.array(["p", "q", "r"])[np.arange(len(FRAME)) % 3])
    priors = get_prior("cat ~ gp(x, k = 4)", data=frame, family="categorical")
    assert {"dpar_muq_lscale_gpx", "dpar_mur_lscale_gpx"} <= set(priors)
