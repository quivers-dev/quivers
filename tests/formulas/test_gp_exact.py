"""brms's exact ``gp()`` terms: inputs, densities, prior law, prediction, and workflow.

The oracle ``oracles/gp_brms.json`` (written by ``oracles/gp_brms.R`` with
brms 2.20.4) records, for each exact case, brms's distinct scaled inputs
``Xgp``, the row map ``Jgp``, the sub-GP rows ``Igp`` and contrasts ``Cgp``,
the counts ``Nsubgp``, and the default priors, plus flags read from brms's
Stan code (the ``1e-12`` diagonal and ``cholesky_decompose(cov) * zgp``).

Tolerances are fixed before comparison. Inputs are printed with 15
significant digits and lie in ``[0, 4]``, so ``1e-12`` absolute covers
printing. Densities, gradients, and Hessians are double-precision
evaluations of the same quantities in a different order on kernels with
condition numbers below 1e7, so they agree to ``1e-7`` relative. Monte Carlo
tolerances are six standard errors of the estimator, computed from the
estimator's known variance.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler
from quivers.formulas import (
    FormulaToQVRModule,
    PriorSpec,
    conditional_smooths,
    families,
    fit,
    formula_from_data,
    formula_to_qvr,
    get_prior,
)
from quivers.formulas.formula import GaussianDesignTerm
from quivers.formulas._gp import (
    GaussianProcessBlock,
    conditional_moments,
    exact_covariance,
    spectral_density,
)
from quivers.diagnostics.cross_validation import FoldPlan, cross_validate
from quivers.inference.trace import trace
from quivers.modeling import load_fit, refit_callback, save_fit

ORACLE = json.loads((Path(__file__).parent / "oracles" / "gp_brms.json").read_text())
FRAME = pd.DataFrame(ORACLE["data"]).assign(g=ORACLE["g"])
SMALL = FRAME.iloc[:10].reset_index(drop=True)


def _gp(block: GaussianDesignTerm) -> GaussianProcessBlock:
    """Return a design block's retained GP, which every block here has."""
    assert block.gp is not None
    return block.gp


def _brms_design(standata: dict, suffix: str, rows: int, gr: bool) -> np.ndarray:
    """Row-by-input design from brms's ``Igp``, ``Jgp``, and ``Cgp``."""
    count = len(np.asarray(standata["Xgp" + suffix], dtype=float))
    igp = standata.get("Igp" + suffix)
    member = np.arange(rows) if igp is None else np.asarray(igp, dtype=int) - 1
    jgp = standata.get("Jgp" + suffix)
    codes = np.asarray(jgp, dtype=int) - 1 if gr else np.arange(len(member))
    cgp = standata.get("Cgp" + suffix)
    weights = np.ones(len(member)) if cgp is None else np.asarray(cgp, dtype=float)
    design = np.zeros((rows, count))
    design[member, codes] = weights
    return design


@pytest.mark.parametrize(
    "case", ORACLE["exact_cases"], ids=lambda case: case["formula"]
)
def test_inputs_row_maps_and_default_priors_equal_brms(case):
    formula = formula_from_data(case["formula"], FRAME)
    standata = case["standata"]
    blocks = formula.design_terms
    assert len(blocks) == standata["Kgp_1"]
    gr = "gr = FALSE" not in case["formula"]
    for j, block in enumerate(blocks):
        suffix = "_1" if len(blocks) == 1 else f"_1_{j + 1}"
        inputs = np.asarray(standata["Xgp" + suffix], dtype=float)
        inputs = inputs.reshape(len(inputs), -1)
        np.testing.assert_allclose(_gp(block).inputs, inputs, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(
            block.data, _brms_design(standata, suffix, len(FRAME), gr)
        )
        if gr:
            counts = np.atleast_1d(standata["Nsubgp_1"])
            assert len(inputs) == counts[j if len(blocks) > 1 else 0]
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
    priors = get_prior(case["formula"], data=FRAME)
    scale = float(expected[("sdgp", "")].split(", ")[-1].rstrip(")"))
    for block in blocks:
        assert (
            priors[f"sdgp_{_gp(block).label}"]
            == f"HalfStudentT(df=3.0, scale={scale!r})"
        )


def test_the_program_is_brms_non_centered_construction():
    assert all(ORACLE["exact_stan"].values())
    text = formula_to_qvr("y ~ gp(x)", data=FRAME)
    assert "sample gp_gpx_z : gp_gpx_Coordinate <- Normal(loc=0.0, scale=1.0)" in text
    assert "let gp_gpx_factor = cholesky(gp_gpx_kernel)" in text
    assert "let gp_gpx_coefficients = matmul(gp_gpx_factor, gp_gpx_z)" in text
    formula = formula_from_data("y ~ gp(x)", FRAME)
    observed = FormulaToQVRModule(families["gaussian"]).fixed_column_observations(
        formula
    )
    torch.testing.assert_close(
        observed["gp_gpx_jitter"],
        1e-12
        * torch.eye(len(_gp(formula.design_terms[0]).inputs), dtype=torch.float64),
    )
    # k = NA is the exact GP, and the exact GP ignores c, as brms does.
    (same,) = formula_from_data("y ~ gp(x, k = NA, c = -1)", FRAME).design_terms
    (block,) = formula.design_terms
    assert same.gp == block.gp
    np.testing.assert_array_equal(same.data, block.data)


def _hand_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """Differentiable Cholesky-Banachiewicz recursion written out entry by entry."""
    size = matrix.shape[-1]
    rows: list[list[torch.Tensor]] = [[] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            partial = matrix[i, j] - sum(
                (rows[i][k] * rows[j][k] for k in range(j)),
                start=matrix.new_zeros(()),
            )
            rows[i].append(partial.sqrt() if i == j else partial / rows[j][j])
    zero = matrix.new_zeros(())
    return torch.stack([torch.stack(row + [zero] * (size - len(row))) for row in rows])


def _kernel(kernel: str, r: torch.Tensor, lscale: torch.Tensor) -> torch.Tensor:
    """Unit-variance kernels of brms's ``cov_gp_*`` (R/formula-gp.R)."""
    if kernel == "exp_quad":
        return torch.exp(-(r**2) / (2 * lscale**2))
    if kernel == "exponential":
        return torch.exp(-r / lscale)
    if kernel == "matern32":
        return (1 + math.sqrt(3) * r / lscale) * torch.exp(-math.sqrt(3) * r / lscale)
    return (1 + math.sqrt(5) * r / lscale + 5 * r**2 / (3 * lscale**2)) * torch.exp(
        -math.sqrt(5) * r / lscale
    )


def _student_half(value: torch.Tensor, prior: str) -> torch.Tensor:
    df = float(prior.split("df=")[1].split(",")[0])
    scale = float(prior.split("scale=")[1].rstrip(")"))
    return math.log(2.0) + torch.distributions.StudentT(df, 0.0, scale).log_prob(value)


def _inverse_gamma(value: torch.Tensor, prior: str) -> torch.Tensor:
    shape = float(prior.split("concentration=")[1].split(",")[0])
    rate = float(prior.split("rate=")[1].rstrip(")"))
    return (
        shape * math.log(rate)
        - math.lgamma(shape)
        - (shape + 1) * value.log()
        - rate / value
    )


def _subterms(frame: pd.DataFrame, columns: list[str], by: str | None, cmc: bool):
    """Independent brms ``data_gp``: per sub-GP rows, weights, inputs, and map."""
    x = frame[columns].to_numpy(dtype=float)
    if by is None:
        selections = [("", np.ones(len(frame)))]
    else:
        levels = sorted(frame[by].unique())
        selections = [] if cmc else [("Intercept", np.ones(len(frame)))]
        selections += [
            (level, (frame[by] == level).to_numpy(dtype=float))
            for level in (levels if cmc else levels[1:])
        ]
    result = []
    for _, weights in selections:
        member = np.flatnonzero(weights != 0)
        keys = [tuple(row) for row in x[member]]
        distinct = list(dict.fromkeys(keys))
        codes = np.array([distinct.index(key) for key in keys])
        points = np.array(distinct)
        dmax = math.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1).max())
        result.append((member, weights[member], points / dmax, codes))
    return result


@pytest.mark.parametrize("kernel", ["exp_quad", "exponential", "matern32", "matern52"])
def test_log_joint_and_derivatives_equal_an_independent_oracle(kernel):
    text = (
        f"y ~ gp(x, z, cov = '{kernel}', iso = FALSE) + gp(x, by = g, cov = '{kernel}')"
    )
    formula = formula_from_data(text, FRAME)
    lens = FormulaToQVRModule(families["gaussian"])
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    observed = lens.fixed_column_observations(formula)
    observed["y"] = torch.tensor(formula.response_values.copy(), dtype=torch.float64)
    blocks = formula.design_terms
    priors = get_prior(text, data=FRAME)
    terms = [_subterms(FRAME, ["x", "z"], None, True)[0]] + _subterms(
        FRAME, ["x"], "g", True
    )
    widths = [len(points) for _, _, points, _ in terms]
    assert widths == [block.data.shape[1] for block in blocks]
    scalars = sum(1 + len(_gp(block).lscale_labels) for block in blocks)
    size = 1 + scalars + sum(widths) + 1
    point = torch.linspace(0.3, 0.9, size, dtype=torch.float64).requires_grad_()

    def split(v):
        out, at = {}, 1
        for block in blocks:
            out[f"sdgp_{_gp(block).label}"] = v[at]
            at += 1
            for label in _gp(block).lscale_labels:
                out[f"lscale_{label}"] = v[at]
                at += 1
        for block, width in zip(blocks, widths, strict=True):
            out[block.qvr_name + "_z"] = 2 * v[at : at + width] - 1
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
        for block, (member, weights, points, codes) in zip(blocks, terms, strict=True):
            label = _gp(block).label
            sdgp = values[f"sdgp_{label}"]
            density = density + _student_half(sdgp, priors[f"sdgp_{label}"])
            lengths = [values[f"lscale_{name}"] for name in _gp(block).lscale_labels]
            for name, lscale in zip(_gp(block).lscale_labels, lengths, strict=True):
                density = density + _inverse_gamma(lscale, priors[f"lscale_{name}"])
            pts = torch.as_tensor(points)
            covariance = sdgp**2
            if len(lengths) == 1:
                r = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1).sqrt()
                covariance = covariance * _kernel(kernel, r, lengths[0])
            else:
                for d, lscale in enumerate(lengths):
                    r = (pts[:, None, d] - pts[None, :, d]).abs()
                    covariance = covariance * _kernel(kernel, r, lscale)
            covariance = covariance + 1e-12 * torch.eye(len(pts), dtype=torch.float64)
            raw = values[block.qvr_name + "_z"]
            density = density + torch.distributions.Normal(0.0, 1.0).log_prob(raw).sum()
            latent = _hand_cholesky(covariance) @ raw
            contribution = torch.zeros(len(FRAME), dtype=torch.float64)
            contribution = contribution.index_add(
                0,
                torch.as_tensor(member),
                torch.as_tensor(weights) * latent[torch.as_tensor(codes)],
            )
            mu = mu + contribution
        return (
            density
            + torch.distributions.Normal(mu, values["sigma"])
            .log_prob(observed["y"])
            .sum()
        )

    tolerance = {"rtol": 1e-7, "atol": 1e-7}
    torch.testing.assert_close(actual(point), oracle(point), **tolerance)
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(actual, point),
        torch.autograd.functional.jacobian(oracle, point),
        **tolerance,
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, point),
        torch.autograd.functional.hessian(oracle, point),
        **tolerance,
    )
    # A deliberately wrong kernel or isotropy changes the density.
    other = "matern52" if kernel != "matern52" else "matern32"
    for wrong_text in (
        text.replace(f"'{kernel}'", f"'{other}'", 1),
        text.replace("iso = FALSE", "iso = TRUE"),
    ):
        wrong = formula_from_data(wrong_text, FRAME)
        values = split(point)
        for block, wrong_block in zip(blocks, wrong.design_terms, strict=True):
            first = values[f"lscale_{_gp(block).lscale_labels[0]}"]
            for label in _gp(block).lscale_labels:
                if label not in _gp(wrong_block).lscale_labels:
                    values.pop(f"lscale_{label}")
            for label in _gp(wrong_block).lscale_labels:
                values.setdefault(f"lscale_{label}", first)
        wrong_module, _ = FormulaToQVRModule(families["gaussian"]).forward(wrong)
        wrong_program = Compiler(wrong_module).compile().morphism
        wrong_value = trace(
            wrong_program, point.new_zeros(1, 1), {**observed, **values}
        ).log_joint.sum()
        assert abs(float((wrong_value - actual(point)).detach())) > 1e-4


def test_prior_draws_of_the_latent_function_have_the_kernel_covariance():
    frame = FRAME.iloc[:5].reset_index(drop=True)
    formula = formula_from_data("y ~ gp(x)", frame)
    lens = FormulaToQVRModule(families["gaussian"])
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    observed = lens.fixed_column_observations(formula)
    sdgp, lscale, sigma = 1.3, 0.4, 1e-6
    runs = 4_000
    torch.manual_seed(11)
    sites = trace(
        program,
        torch.zeros(runs, 1),
        {
            **observed,
            "sdgp_gpx": torch.tensor(sdgp, dtype=torch.float64),
            "lscale_gpx": torch.tensor(lscale, dtype=torch.float64),
            "intercept": torch.tensor(0.0, dtype=torch.float64),
            "sigma": torch.tensor(sigma, dtype=torch.float64),
        },
    ).sites
    draws = sites["y"].value.double().reshape(runs, len(frame))
    x = frame["x"].to_numpy()
    distinct = np.unique(x)
    r = np.abs(x[:, None] - x[None, :]) / (distinct.max() - distinct.min())
    covariance = sdgp**2 * np.exp(-(r**2) / (2 * lscale**2)) + (
        1e-12 + sigma**2
    ) * np.eye(len(x))
    empirical = np.cov(draws.numpy(), rowvar=False)
    # Var(s_ij) = (C_ij^2 + C_ii C_jj) / runs for Gaussian draws.
    bound = 6 * np.sqrt(
        (covariance**2 + np.outer(np.diag(covariance), np.diag(covariance))) / runs
    )
    assert np.all(np.abs(empirical - covariance) <= bound)
    assert np.all(
        np.abs(draws.mean(0).numpy()) <= 6 * np.sqrt(np.diag(covariance) / runs)
    )
    # The kernel at twice the length scale is rejected by the same bound.
    wrong = sdgp**2 * np.exp(-(r**2) / (8 * lscale**2))
    assert not np.all(np.abs(empirical - wrong) <= bound)


def test_conditional_law_equals_the_closed_form():
    formula = formula_from_data("y ~ gp(x, cov = 'matern52')", SMALL)
    block = _gp(formula.design_terms[0])
    train = np.asarray(block.inputs)
    points = np.array([[0.05], [0.37], [0.9]])
    sdgp = torch.tensor([0.8, 1.4], dtype=torch.float64)
    lscale = torch.tensor([0.3, 0.2], dtype=torch.float64)
    values = torch.tensor(
        np.sin(np.arange(2 * len(train)).reshape(2, -1)), dtype=torch.float64
    )
    mean, factor = conditional_moments(block, values, sdgp, (lscale,), points)

    def k(a, b, s, length):
        r = np.abs(a[:, None, 0] - b[None, :, 0]) / length
        return s**2 * (1 + math.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-math.sqrt(5) * r)

    for i in range(2):
        s, length = float(sdgp[i]), float(lscale[i])
        inverse = np.linalg.inv(k(train, train, s, length) + 1e-12 * np.eye(len(train)))
        cross = k(train, points, s, length)
        expected_mean = cross.T @ inverse @ values[i].numpy()
        expected_cov = (
            k(points, points, s, length) - cross.T @ inverse @ cross + 1e-8 * np.eye(3)
        )
        np.testing.assert_allclose(mean[i].numpy(), expected_mean, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(
            (factor[i] @ factor[i].T).numpy(), expected_cov, rtol=1e-6, atol=1e-9
        )
    covariance = exact_covariance(block, train, train, sdgp[:1], (lscale[:1],))
    np.testing.assert_allclose(
        covariance[0].numpy(), k(train, train, 0.8, 0.3), rtol=1e-12, atol=1e-14
    )


def test_exact_and_approximate_covariances_agree_within_the_published_accuracy():
    exact = _gp(formula_from_data("y ~ gp(x)", FRAME).design_terms[0])
    approximate_term = formula_from_data(
        "y ~ gp(x, k = 40, c = 2)", FRAME
    ).design_terms[0]
    approximate = _gp(approximate_term)
    rows = formula_from_data("y ~ gp(x)", FRAME).design_terms[0].data
    sdgp = torch.tensor(1.0, dtype=torch.float64)
    lscale = torch.tensor(0.3, dtype=torch.float64)
    inputs = np.asarray(exact.inputs)
    exact_rows = (
        rows @ exact_covariance(exact, inputs, inputs, sdgp, (lscale,)).numpy() @ rows.T
    )
    spd = spectral_density(approximate, sdgp, (lscale,)).numpy()
    basis = approximate_term.data
    approximate_rows = basis @ np.diag(spd) @ basis.T
    # Riutort-Mayol et al. (2023) relations (19)-(20) give c >= 3.2 l / S and
    # m >= 1.75 c S / l for under one percent error; here S = 1/2, l = 0.3,
    # c = 2 >= 1.92, and m = 40 >= 6.
    assert np.abs(exact_rows - approximate_rows).max() < 1e-2


def _small_fit(method: str):
    options = (
        {"num_steps": 30}
        if method == "svi"
        else {"num_warmup": 5, "num_samples": 5, "num_chains": 1}
    )
    return fit(
        "y ~ gp(x, by = w)",
        data=SMALL,
        family="gaussian",
        method=method,
        seed=4,
        **options,
    )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_every_inference_method_fits_and_predicts_new_rows(method):
    fitted = _small_fit(method)
    fresh = pd.DataFrame({"x": [-0.8, 0.4, 3.5, 0.4], "w": [1.0, 0.7, 1.2, 0.7]})
    for draws in (
        fitted.posterior_linpred(new_data=fresh, num_samples=3, seed=5),
        fitted.posterior_epred(new_data=fresh, num_samples=3, seed=5),
        fitted.posterior_predict(new_data=fresh, num_samples=3, seed=5),
        *fitted.log_likelihood(
            new_data=fresh.assign(y=[0.1, 0.4, 0.2, 0.3]), num_samples=3, seed=5
        ).values(),
    ):
        assert draws.shape[-1] == 4 and torch.isfinite(draws).all()
    linpred = fitted.posterior_linpred(new_data=fresh, num_samples=3, seed=5)
    # Rows sharing an input and by value share one latent value.
    torch.testing.assert_close(linpred[..., 1], linpred[..., 3])


def test_training_and_new_rows_follow_the_replayed_latent_function():
    fitted = fit(
        "y ~ gp(x)",
        data=SMALL,
        method="hmc",
        num_warmup=10,
        num_samples=150,
        num_chains=1,
        seed=8,
    )
    draws = fitted.draws()
    sdgp = draws["sdgp_gpx"].double().reshape(-1).numpy()
    lscale = draws["lscale_gpx"].double().reshape(-1).numpy()
    z = draws["gp_gpx_z"].double().reshape(len(sdgp), -1).numpy()
    intercept = draws["intercept"].double().reshape(-1).numpy()
    x = SMALL["x"].to_numpy()
    keys = list(dict.fromkeys(x.tolist()))
    distinct = np.array(keys)
    dmax = distinct.max() - distinct.min()
    train = distinct / dmax
    fresh_x = 0.37
    linpred = fitted.posterior_linpred().double().reshape(len(sdgp), -1).numpy()
    fresh = fitted.posterior_linpred(
        new_data=pd.DataFrame({"x": [fresh_x, keys[2]]}), seed=3
    )
    fresh = fresh.double().reshape(len(sdgp), -1).numpy()
    standardized, replayed = [], []
    for i in range(len(sdgp)):

        def k(a, b):
            return sdgp[i] ** 2 * np.exp(
                -((a[:, None] - b[None, :]) ** 2) / (2 * lscale[i] ** 2)
            )

        covariance = k(train, train) + 1e-12 * np.eye(len(train))
        latent = np.linalg.cholesky(covariance) @ z[i]
        codes = [keys.index(value) for value in x]
        np.testing.assert_allclose(linpred[i] - intercept[i], latent[codes], atol=1e-8)
        cross = k(train, np.array([fresh_x / dmax]))
        mean = (cross.T @ np.linalg.solve(covariance, latent))[0]
        variance = (
            k(np.array([fresh_x / dmax]), np.array([fresh_x / dmax]))
            - cross.T @ np.linalg.solve(covariance, cross)
        )[0, 0] + 1e-8
        standardized.append((fresh[i, 0] - intercept[i] - mean) / math.sqrt(variance))
        replayed.append(fresh[i, 1] - intercept[i] - latent[2])
    standardized = np.array(standardized)
    count = len(standardized)
    # Each new-input draw is N(mean, variance) given its posterior draw.
    assert abs(standardized.mean()) <= 6 / math.sqrt(count)
    assert abs(standardized.var() - 1) <= 6 * math.sqrt(2 / count)
    # A training input is replayed up to brms's 1e-8 prediction nugget.
    assert np.abs(replayed).max() <= 6 * math.sqrt(2e-8)


def test_conditional_smooths_draw_the_latent_function_at_the_rows():
    fitted = fit("y ~ gp(x)", data=SMALL, method="svi", num_steps=20, seed=2)
    draws = fitted.draws(num_samples=4, seed=1)
    values = conditional_smooths(fitted, "gp(x)", SMALL, num_samples=4, seed=1)
    linpred = fitted.posterior_linpred(num_samples=4, seed=1)
    assert values.shape == linpred.shape
    torch.testing.assert_close(
        values,
        (linpred - draws["intercept"].double()[..., None]).double(),
        atol=1e-3,
        rtol=0,
    )


def test_update_kfold_prior_overrides_and_artifact_round_trip(tmp_path, monkeypatch):
    import quivers.formulas._fit as api

    text = "y ~ gp(x, z, iso = FALSE, cov = 'matern32')"
    priors = get_prior(
        text,
        data=FRAME,
        priors=[
            PriorSpec(distribution="HalfNormal(0.3)", class_="lscale", coef="gpxzz")
        ],
    )
    assert priors["lscale_gpxzz"] == "HalfNormal(scale=0.3)"
    fitted = fit(text, data=SMALL, method="svi", num_steps=2, seed=7)
    refit = fitted.update(data=SMALL.iloc[:8].reset_index(drop=True))
    assert refit.formula.response_values.shape == (8,)
    callback = refit_callback(
        fitted.refit_recipe,
        lambda data, rows: data.iloc[list(rows)].copy(),
        lambda refitted, held, seed: refitted.log_likelihood(
            new_data=held, num_samples=2, seed=seed
        ),
        method_factory=lambda settings, seed: {**settings, "seed": seed},
    )
    result = cross_validate(
        FoldPlan(assignments=tuple(i % 2 for i in range(len(SMALL))), seed=1), callback
    )
    assert result.complete and result.log_predictive_density.shape == (len(SMALL),)
    fresh = pd.DataFrame({"x": [0.1, 2.0], "z": [0.5, 1.5]})
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


def test_many_distinct_inputs_have_no_size_cap():
    frame = pd.DataFrame({"x": np.linspace(0, 1, 300)})
    frame["y"] = np.sin(5 * frame["x"])
    formula = formula_from_data("y ~ gp(x)", frame)
    lens = FormulaToQVRModule(families["gaussian"])
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    observed = lens.fixed_column_observations(formula)
    observed["y"] = torch.tensor(formula.response_values.copy(), dtype=torch.float64)
    value = trace(
        program,
        torch.zeros(1, 1),
        {
            **observed,
            "sdgp_gpx": torch.tensor(1.0, dtype=torch.float64),
            "lscale_gpx": torch.tensor(0.05, dtype=torch.float64),
            "gp_gpx_z": torch.zeros(300, dtype=torch.float64),
            "intercept": torch.tensor(0.0, dtype=torch.float64),
            "sigma": torch.tensor(1.0, dtype=torch.float64),
        },
    ).log_joint
    assert torch.isfinite(value).all()


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("y ~ gp(x, cov = 'periodic')", "not a valid GP covariance kernel"),
        ("y ~ gp(g)", "should be numeric vectors"),
        ("y ~ gp(x):z", "standalone additive terms"),
        ("y ~ (gp(x) | g)", "group-level"),
    ],
)
def test_invalid_specifications_are_rejected(formula, message):
    with pytest.raises(ValueError, match=message):
        formula_from_data(formula, FRAME)


def test_categorical_predictors_give_each_category_an_exact_gp():
    frame = FRAME.assign(cat=np.array(["p", "q", "r"])[np.arange(len(FRAME)) % 3])
    priors = get_prior("cat ~ gp(x)", data=frame, family="categorical")
    assert {"lscale_gpx", "sdgp_gpx"} <= set(priors)
    fitted = fit(
        "cat ~ gp(x)", data=frame, family="categorical", method="svi", num_steps=0
    )
    draws = fitted.draws(num_samples=1)
    assert draws["lscale_gpx"].shape[-1] == draws["sdgp_gpx"].shape[-1] == 2
