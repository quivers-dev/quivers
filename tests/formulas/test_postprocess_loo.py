"""brms's leave-one-out family on formula fits.

Two kinds of oracles pin the estimators. The R ``loo`` package
(``oracles/brms_postprocess.R``) computes PSIS-LOO, WAIC, comparisons,
weights, ``E_loo`` expectations and subsampled PSIS-LOO on exported pointwise
log-likelihoods, so the arithmetic is compared on identical numbers. A
conjugate Normal model gives exact leave-one-out and k-fold predictive
densities in closed form; fits holding exact independent posterior draws
then test PSIS, moment matching and subsampling within their Monte Carlo
error, and refits test exact refitting and k-fold cross-validation.
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd
import pytest
import torch
import arviz as az
from scipy import stats

from quivers.diagnostics.elpd import (
    ELPDEstimate,
    PSISWeights,
    compare_estimates,
    from_arviz,
    information_weights,
    loo_expectation,
    loo_r2_draws,
    psis_weights,
    stacking_weights,
    subsample_estimate,
    waic_estimate,
)
from quivers.diagnostics.posterior import PosteriorAnalysis
from quivers.formulas import (
    BayesianFit,
    fit,
    kfold_predict,
    loo_compare,
    loo_model_weights,
    model_weights,
)
from quivers.formulas._multivariate import bf
from quivers.formulas._postprocess_loo import kfold_plan, unconstrained_draws
from quivers.inference.mcmc.kernel import PotentialFn
from quivers.inference.mcmc.driver import MCMCResult

ORACLE = json.loads(
    (pathlib.Path(__file__).parent / "oracles" / "brms_postprocess.json").read_text()
)["loo"]
LOG_LIK = torch.from_numpy(
    np.asarray(
        json.loads(
            (
                pathlib.Path(__file__).parent / "oracles" / "brms_postprocess.json"
            ).read_text()
        )["gaussian"]["log_lik"],
        dtype=np.float64,
    )
).reshape(2, 150, 30)
# Same estimator on identical Float64 numbers: a few ulps of accumulated sums.
TOLERANCE = 1e-9

Y = np.array(
    [0.31, -1.24, 0.85, 1.93, -0.42, 0.13, 1.41, -0.73, 0.62, 2.21, -1.52, 6.5]
)
TAU = 2.0


def table(values: dict) -> np.ndarray:
    """Read an oracle matrix.

    Parameters
    ----------
    values : dict
        Oracle table.

    Returns
    -------
    numpy.ndarray
        Float64 matrix.
    """
    return np.array(values["values"], dtype=np.float64)


def arviz_loo(log_likelihood: torch.Tensor) -> ELPDEstimate:
    """Run ArviZ PSIS-LOO with relative efficiency one, as the oracle.

    Parameters
    ----------
    log_likelihood : torch.Tensor
        Scores ``(chain, draw, unit)``.

    Returns
    -------
    ELPDEstimate
        The converted estimate.
    """
    analysis = PosteriorAnalysis(samples={}, log_likelihood={"y": log_likelihood})
    result = az.loo(analysis.to_datatree(), var_name="y", pointwise=True, reff=1.0)
    return from_arviz(
        result, log_likelihood, criterion="loo", responses=("y",), hashed="h"
    )


def posterior(values: np.ndarray) -> tuple[float, float]:
    """Return the conjugate posterior of the mean with unit noise.

    Parameters
    ----------
    values : numpy.ndarray
        Observations.

    Returns
    -------
    tuple[float, float]
        Posterior mean and variance under the Normal(0, TAU) prior.
    """
    precision = 1 / TAU**2 + len(values)
    return float(values.sum() / precision), 1 / precision


def exact_loo() -> np.ndarray:
    """Return exact leave-one-out log predictive densities.

    Returns
    -------
    numpy.ndarray
        ``log p(y_i | y_-i)`` with predictive Normal(m_-i, 1 + v_-i).
    """
    terms = []
    for i in range(len(Y)):
        mean, var = posterior(np.delete(Y, i))
        terms.append(stats.norm.logpdf(Y[i], mean, math.sqrt(1 + var)))
    return np.array(terms)


@pytest.fixture(scope="module")
def conjugate() -> BayesianFit:
    """Return the conjugate model holding exact independent posterior draws.

    Returns
    -------
    BayesianFit
        A fit whose posterior is 2000 exact draws in four chains.
    """
    frame = pd.DataFrame({"y": Y})
    model = fit(
        "y ~ 1",
        data=frame,
        fixed_parameters={"sigma": 1.0},
        priors={"intercept": f"Normal(0.0, {TAU})"},
        method="svi",
        num_steps=5,
        seed=3,
    )
    mean, var = posterior(Y)
    generator = torch.Generator().manual_seed(11)
    draws = mean + math.sqrt(var) * torch.randn(4, 500, generator=generator)
    exact = MCMCResult(
        samples={"intercept": draws},
        log_densities=torch.zeros(4, 500),
        acceptance_rates=torch.ones(4),
        divergence_counts=torch.zeros(4),
        num_warmup=0,
        num_samples=500,
    )
    return model.with_(posterior=exact)


def importance_se(log_likelihood: torch.Tensor) -> np.ndarray:
    """Delta-method Monte Carlo SE of each self-normalized LOO estimate.

    Parameters
    ----------
    log_likelihood : torch.Tensor
        Scores at independent posterior draws ``(chain, draw, unit)``.

    Returns
    -------
    numpy.ndarray
        Standard error of ``log 1 / mean(1 / p_s)`` per unit.
    """
    ratios = torch.exp(-log_likelihood.flatten(0, 1).to(torch.float64))
    count = ratios.shape[0]
    return (ratios.std(0) / ratios.mean(0) / math.sqrt(count)).numpy()


def test_psis_loo_matches_loo_package() -> None:
    ours = arviz_loo(LOG_LIK)
    pointwise = table(ORACLE["loo"]["pointwise"])
    mcse = pointwise[:, 1]
    # ArviZ and loo implement the generalized Pareto tail fit differently, so
    # the two PSIS estimates agree within loo's own Monte Carlo standard error.
    np.testing.assert_array_less(
        np.abs(ours.pointwise_elpd.numpy() - pointwise[:, 0]), mcse
    )
    total = math.sqrt(float((mcse**2).sum()))
    estimates = table(ORACLE["loo"]["estimates"])
    ours_table = ours.estimates().values.numpy()
    assert abs(ours_table[0, 0] - estimates[0, 0]) < total
    assert abs(ours_table[1, 0] - estimates[1, 0]) < total
    assert abs(ours_table[2, 0] - estimates[2, 0]) < 2 * total
    assert abs(ours_table[0, 1] - estimates[0, 1]) < total
    k = ours.pareto_k.numpy()
    reference = np.array(ORACLE["loo"]["pareto_k"])
    np.testing.assert_array_less(np.abs(k - reference), 0.05)
    np.testing.assert_array_equal(k > 0.7, reference > 0.7)


def loo_from_pointwise(pointwise: dict) -> ELPDEstimate:
    """Build an estimate from loo's own pointwise terms.

    Parameters
    ----------
    pointwise : dict
        loo's pointwise table (elpd_loo, mcse_elpd_loo, p_loo, ...).

    Returns
    -------
    ELPDEstimate
        The estimate with loo's terms.
    """
    values = torch.from_numpy(table(pointwise))
    elpd, p = values[:, 0], values[:, 2]
    count = len(elpd)
    return ELPDEstimate(
        criterion="loo",
        elpd=float(elpd.sum()),
        se_elpd=math.sqrt(count * float(elpd.var())),
        p=float(p.sum()),
        se_p=math.sqrt(count * float(p.var())),
        pointwise_elpd=elpd,
        pointwise_p=p,
        num_samples=300,
        responses=("y",),
        response_hash="h",
    )


def test_waic_matches_loo_package() -> None:
    ours = waic_estimate(LOG_LIK, responses=("y",), hashed="h")
    np.testing.assert_allclose(
        ours.estimates().values.numpy(),
        table(ORACLE["waic"]["estimates"]),
        rtol=TOLERANCE,
    )
    pointwise = table(ORACLE["waic"]["pointwise"])
    np.testing.assert_allclose(
        ours.pointwise_elpd.numpy(), pointwise[:, 0], rtol=TOLERANCE
    )
    np.testing.assert_allclose(
        ours.pointwise_p.numpy(), pointwise[:, 1], rtol=TOLERANCE
    )


def shifted() -> torch.Tensor:
    """Return the oracle's second model, the first plus ``0.3 sin(i)``.

    Returns
    -------
    torch.Tensor
        Scores ``(chain, draw, unit)``.
    """
    return LOG_LIK + 0.3 * torch.sin(torch.arange(1, 31, dtype=torch.float64))


def test_compare_and_weights_match_loo_package() -> None:
    first = loo_from_pointwise(ORACLE["loo"]["pointwise"])
    second = loo_from_pointwise(ORACLE["loo2"]["pointwise"])
    np.testing.assert_allclose(
        first.estimates().values.numpy(),
        table(ORACLE["loo"]["estimates"]),
        rtol=TOLERANCE,
    )
    compared = compare_estimates({"m1": first, "m2": second})
    oracle = ORACLE["compare"]
    assert list(compared.rows) == oracle["rows"]
    assert list(compared.columns) == oracle["columns"]
    np.testing.assert_allclose(
        compared.values.numpy(), table(oracle), rtol=TOLERANCE, atol=1e-12
    )
    pseudo = stacking_weights(
        {"m1": first, "m2": second},
        method="pseudobma",
        bayesian_bootstrap=False,
        seed=0,
    )
    np.testing.assert_allclose(
        list(pseudo.values()), ORACLE["weights"]["pseudobma"], rtol=TOLERANCE
    )
    # Stacking maximizes a concave objective numerically; both optimizers
    # stop at their own tolerances (loo: constrOptim, ArviZ: SLSQP).
    stacked = stacking_weights(
        {"m1": first, "m2": second}, method="stacking", bayesian_bootstrap=False, seed=0
    )
    np.testing.assert_allclose(
        list(stacked.values()), ORACLE["weights"]["stacking"], atol=1e-4
    )
    akaike = information_weights({"m1": first, "m2": second})
    expected = np.exp(-np.array([first.ic, second.ic]) / 2)
    np.testing.assert_allclose(
        list(akaike.values()), expected / expected.sum(), rtol=1e-12
    )
    # ArviZ's own PSIS on the same scores gives the same ranking.
    assert compare_estimates(
        {"m1": arviz_loo(LOG_LIK), "m2": arviz_loo(shifted())}
    ).rows == (
        "m2",
        "m1",
    )


def test_loo_expectations_match_e_loo() -> None:
    draws = torch.from_numpy(
        np.asarray(
            json.loads(
                (
                    pathlib.Path(__file__).parent / "oracles" / "brms_postprocess.json"
                ).read_text()
            )["gaussian"]["predict_draws"],
            dtype=np.float64,
        )
    ).reshape(2, 150, 5)
    # loo's own smoothed weights isolate the expectation arithmetic.
    log_weights = torch.from_numpy(
        np.asarray(ORACLE["psis_log_weights_5"], dtype=np.float64)
    ).reshape(2, 150, 5)
    weights = PSISWeights(
        log_weights=log_weights,
        pareto_k=torch.tensor(ORACLE["psis_pareto_k_5"], dtype=torch.float64),
        r_eff=1.0,
        good_k=0.6,
    )
    oracle = ORACLE["e_loo"]
    mean = loo_expectation(draws, weights, kind="mean")
    np.testing.assert_allclose(
        mean.table.values[:, 0].numpy(), oracle["mean"], rtol=TOLERANCE
    )
    variance = loo_expectation(draws, weights, kind="var")
    # loo >= 2.7 (the version brms 2.23 installs) defines the weighted
    # predictive variance (E_w[x^2] - E_w[x]^2) / (1 - sum w^2). The installed
    # loo 2.6.0 instead returns sum(w^2 (x - E_w[x])^2), the Monte Carlo
    # variance of the mean, which the oracle pins to show the difference.
    w = torch.exp(log_weights).flatten(0, 1).numpy()
    x = draws.flatten(0, 1).numpy()
    m = (w * x).sum(0)
    predictive = ((w * x**2).sum(0) - m**2) / (1 - (w**2).sum(0))
    np.testing.assert_allclose(
        variance.table.values[:, 0].numpy(), predictive, rtol=1e-9
    )
    np.testing.assert_allclose((w**2 * (x - m) ** 2).sum(0), oracle["var"], rtol=1e-8)
    quantiles = loo_expectation(draws, weights, kind="quantile", probs=(0.1, 0.9))
    assert list(quantiles.table.columns) == ["q10", "q90"]
    np.testing.assert_allclose(
        quantiles.table.values.numpy(), np.array(oracle["quantile"]).T, rtol=1e-8
    )
    # Function-specific Pareto shapes come from ArviZ. The installed loo 2.6.0
    # predates loo's current E_loo diagnostic, so the shapes are not pinned
    # to it; each unit reports one finite shape.
    assert mean.pareto_k.shape == (5,) and bool(torch.isfinite(mean.pareto_k).all())


def test_subsample_estimator_matches_loo_subsample() -> None:
    oracle = ORACLE["subsample"]
    draws = torch.tensor(oracle["draws"], dtype=torch.float64)
    y = torch.tensor(oracle["y"], dtype=torch.float64)
    scores = (
        torch.distributions.Normal(draws[:, :1], draws[:, 1:]).log_prob(y).unsqueeze(0)
    )
    units = [int(i) - 1 for i in oracle["observations"]]
    weights = psis_weights(scores[..., units], r_eff=1.0)
    exact = torch.logsumexp(weights.log_weights + scores[..., units], dim=(0, 1))
    lpd = torch.logsumexp(scores.flatten(0, 1), 0) - math.log(scores.shape[1])
    point = torch.distributions.Normal(draws[:, 0].mean(), draws[:, 1].mean()).log_prob(
        y
    )
    for method, approximation in (("plpd", point), ("lpd", lpd)):
        estimate = subsample_estimate(
            exact,
            approximation,
            units,
            lpd[units],
            weights.pareto_k,
            num_samples=400,
            responses=("y",),
            hashed="h",
            method=method,
        )
        np.testing.assert_allclose(
            estimate.estimates().values.numpy(),
            table(oracle[method]["estimates"]),
            rtol=1e-8,
        )


def test_arviz_containers_carry_each_estimator_and_the_subsample() -> None:
    oracle = ORACLE["subsample"]
    draws = torch.tensor(oracle["draws"], dtype=torch.float64)
    y = torch.tensor(oracle["y"], dtype=torch.float64)
    scores = (
        torch.distributions.Normal(draws[:, :1], draws[:, 1:]).log_prob(y).unsqueeze(0)
    )
    units = [int(i) - 1 for i in oracle["observations"]]
    weights = psis_weights(scores[..., units], r_eff=1.0)
    exact = torch.logsumexp(weights.log_weights + scores[..., units], dim=(0, 1))
    lpd = torch.logsumexp(scores.flatten(0, 1), 0) - math.log(scores.shape[1])
    point = torch.distributions.Normal(draws[:, 0].mean(), draws[:, 1].mean()).log_prob(
        y
    )
    shift = 0.05 * torch.arange(len(units), dtype=torch.float64)
    first, second = (
        subsample_estimate(
            terms,
            approximation,
            units,
            lpd[units],
            weights.pareto_k,
            num_samples=400,
            responses=("y",),
            hashed="h",
            method=method,
        )
        for terms, approximation, method in (
            (exact, point, "plpd"),
            (exact - shift, lpd, "lpd"),
        )
    )
    container = first.to_arviz()
    assert container.kind == "loo"
    assert container.elpd == first.elpd and container.se == first.se_elpd
    assert container.n_data_points == len(y) and container.n_samples == 400
    assert getattr(container, "subsample_size") == len(units)
    assert getattr(container, "subsampling_se") == first.subsampling_se
    np.testing.assert_array_equal(
        getattr(container, "loo_subsample_observations"), np.asarray(units)
    )
    np.testing.assert_array_equal(
        getattr(container, "elpd_loo_approx").values, point.numpy()
    )
    np.testing.assert_array_equal(
        container.pareto_k.values[units], weights.pareto_k.numpy()
    )
    full = loo_from_pointwise(ORACLE["loo"]["pointwise"]).to_arviz()
    assert full.kind == "loo" and getattr(full, "subsample_size", None) is None
    # The difference estimator of subsampled PSIS-LOO over the shared units:
    # sum_i d_i + N mean_{j in S}(e_j - d_j), with d the difference of the
    # approximations and e the difference of the exact terms.
    compared = compare_estimates({"a": first, "b": second})
    best, other = (first, second) if first.elpd >= second.elpd else (second, first)
    assert best.approximation is not None and other.approximation is not None
    approximate = (other.approximation - best.approximation).numpy()
    sampled = (other.pointwise_elpd - best.pointwise_elpd).numpy()[units]
    expected = approximate.sum() + len(y) * (sampled - approximate[units]).mean()
    rows = list(compared.rows)
    np.testing.assert_allclose(
        compared.values.numpy()[rows.index("a" if other is first else "b"), 0],
        expected,
        rtol=1e-10,
    )
    assert compared.values.numpy()[0, 0] == 0.0


def test_psis_loo_recovers_exact_leave_one_out(conjugate: BayesianFit) -> None:
    estimate = conjugate.loo()
    exact = exact_loo()
    scores = conjugate.log_likelihood()["y"]
    tolerance = 5 * importance_se(scores)
    good = estimate.pareto_k.numpy() < 0.5
    assert good[:-1].all() and not good[-1]
    np.testing.assert_array_less(
        np.abs(estimate.pointwise_elpd.numpy() - exact)[good], tolerance[good]
    )
    # A wrong predictive variance (the noise only) is rejected.
    wrong = np.array(
        [
            stats.norm.logpdf(Y[i], posterior(np.delete(Y, i))[0], 1.0)
            for i in range(len(Y))
        ]
    )
    assert np.any(
        np.abs(estimate.pointwise_elpd.numpy() - wrong)[good] > tolerance[good]
    )


def test_moment_matching_repairs_the_outlier(conjugate: BayesianFit) -> None:
    plain = conjugate.loo()
    matched = conjugate.loo(moment_match=True, k_threshold=0.5)
    assert matched.moment_matched == (11,)
    assert matched.pareto_k[11] < 0.5 < plain.pareto_k[11]
    exact = exact_loo()
    # After moment matching the outlier's weights are well behaved, so the
    # independent-draw standard error bounds its error again.
    tolerance = 5 * importance_se(conjugate.log_likelihood()["y"])[:11].max()
    assert abs(float(matched.pointwise_elpd[11]) - exact[11]) < max(tolerance, 0.05)
    np.testing.assert_array_equal(
        matched.pointwise_elpd[:11].numpy(), plain.pointwise_elpd[:11].numpy()
    )
    direct = conjugate.loo_moment_match(k_threshold=0.5)
    assert direct.elpd == matched.elpd


def test_moment_matching_uses_the_exact_log_posterior(conjugate: BayesianFit) -> None:
    draws = conjugate.draws(num_samples=5)
    registry, upars = unconstrained_draws(conjugate, draws)
    potential = PotentialFn(
        conjugate.program,
        registry,
        torch.zeros(1, 1, dtype=torch.long),
        dict(conjugate.observations),
    )
    values = upars.flatten(0, 1)
    densities = torch.stack(
        [potential.log_density(row.float()) for row in values]
    ).double()
    mu = values[:, 0]
    exact = stats.norm.logpdf(mu.numpy(), 0, TAU) + stats.norm.logpdf(
        Y[None, :], mu.numpy()[:, None], 1.0
    ).sum(1)
    np.testing.assert_allclose(densities.numpy(), exact, rtol=1e-5)


def test_subsampled_loo_reuses_exact_terms(conjugate: BayesianFit) -> None:
    full = conjugate.loo()
    estimate = conjugate.loo_subsample(
        observations=[0, 2, 5, 7, 9, 11], approximation="lpd"
    )
    units = list(estimate.subsample)
    np.testing.assert_allclose(
        estimate.pointwise_elpd[units].numpy(),
        full.pointwise_elpd[units].numpy(),
        # Float32 replays of the training rows and of the same rows as new
        # data agree to Float32 rounding.
        rtol=1e-6,
    )
    lpd = conjugate.log_likelihood()["y"].to(torch.float64)
    lpd = torch.logsumexp(lpd.flatten(0, 1), 0) - math.log(2000)
    assert estimate.approximation is not None
    np.testing.assert_allclose(estimate.approximation.numpy(), lpd.numpy(), rtol=1e-6)
    assert abs(estimate.elpd - full.elpd) < 3 * estimate.subsampling_se
    plpd = conjugate.loo_subsample(observations=6, seed=4)
    assert plpd.approximation_method == "plpd" and len(plpd.subsample) == 6
    with pytest.raises(ValueError, match="subsample size"):
        conjugate.loo_subsample(observations=12)
    with pytest.raises(ValueError, match="distinct"):
        conjugate.loo_subsample(observations=[1, 1, 2])


def test_loo_predictions_and_r2(conjugate: BayesianFit) -> None:
    weights = conjugate.psis()
    expected = conjugate.posterior_epred()
    epred = conjugate.loo_epred()
    direct = loo_expectation(expected, weights, kind="mean")
    np.testing.assert_allclose(epred.table.values.numpy(), direct.table.values.numpy())
    # The leave-one-out mean of the intercept is the exact LOO posterior mean.
    for i in (0, 5):
        mean, var = posterior(np.delete(Y, i))
        assert abs(float(epred.table.values[i, 0]) - mean) < 5 * math.sqrt(var / 500)
    interval = conjugate.loo_predictive_interval(prob=0.8)
    assert list(interval.table.columns) == ["q10", "q90"]
    draws = conjugate.loo_r2(summary=False, seed=7)
    generator = torch.Generator().manual_seed(7)
    loo_mean = epred.table.values[:, 0]
    np.testing.assert_allclose(
        draws.numpy(),
        loo_r2_draws(torch.tensor(Y), loo_mean, generator, 2000).numpy(),
    )
    exponential = torch.empty(2000, 12, dtype=torch.float64).exponential_(
        generator=torch.Generator().manual_seed(7)
    )
    w = (exponential / exponential.sum(1, keepdim=True)).numpy()
    error = loo_mean.numpy() - Y
    var_y = 12 / 11 * ((w * Y**2).sum(1) - (w * Y).sum(1) ** 2)
    var_e = 12 / 11 * ((w * error**2).sum(1) - (w * error).sum(1) ** 2)
    np.testing.assert_allclose(
        draws.numpy(), np.clip(1 - var_e / var_y, -1, 1), rtol=1e-12
    )
    summary = conjugate.loo_r2()
    assert summary.rows == ("R2",)


def test_stored_criteria_drive_comparisons(conjugate: BayesianFit) -> None:
    stored = conjugate.add_criterion(["loo", "waic"])
    assert stored.loo() is stored.criteria["loo"]
    table = loo_compare({"a": stored, "b": stored})
    assert table.values[0, 0] == 0 and table.values[1, 0] == 0
    weights = model_weights({"a": stored, "b": stored}, weights="waic")
    assert weights == {"a": 0.5, "b": 0.5}
    assert set(loo_model_weights({"a": stored, "b": stored}, method="pseudobma")) == {
        "a",
        "b",
    }
    with pytest.raises(ValueError, match="add_criterion"):
        loo_compare({"a": conjugate, "b": stored})
    loo = stored.criteria["loo"]
    assert isinstance(loo, ELPDEstimate)
    other = loo.with_(response_hash="other")
    with pytest.raises(ValueError, match="same response"):
        loo_compare({"a": stored, "b": other})
    with pytest.raises(ValueError, match="weights must be"):
        model_weights({"a": stored}, weights="marginal")  # type: ignore[arg-type]


def test_exact_refit_replaces_the_problematic_unit(conjugate: BayesianFit) -> None:
    estimate = conjugate.loo(
        reloo=True,
        k_threshold=0.5,
        method_factory=lambda settings, seed: {
            **{key: value for key, value in settings.items() if key != "num_steps"},
            "method": "nuts",
            "num_samples": 400,
            "num_warmup": 300,
            "num_chains": 2,
            "seed": seed,
        },
    )
    assert estimate.refitted == (11,)
    exact = exact_loo()
    mean, var = posterior(np.delete(Y, 11))
    # MC SE of log mean p(y_11 | mu_s) over 800 correlated draws, taking an
    # effective size of half the draws.
    rng = np.random.default_rng(0)
    density = stats.norm.pdf(
        Y[11], mean + math.sqrt(var) * rng.standard_normal(100_000), 1.0
    )
    se = density.std() / density.mean() / math.sqrt(400)
    assert abs(float(estimate.pointwise_elpd[11]) - exact[11]) < 5 * se


def test_kfold_matches_exact_fold_predictions(conjugate: BayesianFit) -> None:
    folds = [0, 1, 2] * 4
    result = conjugate.kfold(
        folds=folds,
        method_factory=lambda settings, seed: {
            **{key: value for key, value in settings.items() if key != "num_steps"},
            "method": "nuts",
            "num_samples": 400,
            "num_warmup": 300,
            "num_chains": 2,
            "seed": seed,
        },
    )
    estimate = result.estimate
    exact = []
    for i in range(12):
        train = np.array([Y[j] for j in range(12) if folds[j] != folds[i]])
        mean, var = posterior(train)
        exact.append(stats.norm.logpdf(Y[i], mean, math.sqrt(1 + var)))
    rng = np.random.default_rng(1)
    tolerances = []
    for i in range(12):
        train = np.array([Y[j] for j in range(12) if folds[j] != folds[i]])
        mean, var = posterior(train)
        density = stats.norm.pdf(
            Y[i], mean + math.sqrt(var) * rng.standard_normal(100_000), 1.0
        )
        tolerances.append(5 * density.std() / density.mean() / math.sqrt(400))
    np.testing.assert_array_less(
        np.abs(estimate.pointwise_elpd.numpy() - np.array(exact)), np.array(tolerances)
    )
    lpd = conjugate.log_likelihood()["y"].to(torch.float64)
    lpd = torch.logsumexp(lpd.flatten(0, 1), 0) - math.log(2000)
    np.testing.assert_allclose(
        estimate.pointwise_p.numpy(),
        (lpd - estimate.pointwise_elpd).numpy(),
        rtol=1e-12,
    )
    assert estimate.folds == tuple(folds) and result.fold_type == "custom"


def test_kfold_partitions_follow_brms(conjugate: BayesianFit) -> None:
    groups = tuple("aabbccddeeff")
    plan, kind = kfold_plan(12, K=3, folds=None, groups=groups, seed=0)
    assert kind == "group" and plan.num_folds == 6
    plan, kind = kfold_plan(12, K=3, folds="grouped", groups=groups, seed=0)
    assignments = plan.assignments
    assert all(assignments[2 * k] == assignments[2 * k + 1] for k in range(6))
    plan, kind = kfold_plan(12, K=2, folds="stratified", groups=tuple("ab" * 6), seed=0)
    for fold in range(2):
        labels = [g for g, f in zip("ab" * 6, plan.assignments) if f == fold]
        assert labels.count("a") == labels.count("b") == 3
    plan, kind = kfold_plan(12, K=3, folds="loo", groups=groups, seed=0)
    split = plan.split(2)
    assert split.test == (2,) and split.omitted == (2, 3)
    with pytest.raises(ValueError, match="requires a group"):
        kfold_plan(12, K=3, folds="stratified", groups=None, seed=0)
    with pytest.raises(ValueError, match="one label per unit"):
        kfold_plan(12, K=3, folds=[0, 1], groups=None, seed=0)
    with pytest.raises(ValueError, match="joint='group' requires a group"):
        conjugate.kfold(K=2, joint="group")
    with pytest.raises(ValueError, match="Ksub"):
        conjugate.kfold(K=2, Ksub=[5])


def test_kfold_predict_with_saved_refits() -> None:
    frame = pd.DataFrame({"y": Y[:8], "x": np.linspace(-1, 1, 8)})
    model = fit("y ~ x", data=frame, method="svi", num_steps=30, seed=0)
    result = model.kfold(K=2, save_fits=True, joint="fold", num_samples=20)
    assert result.estimate.num_observations == 2 and result.estimate.joint == "fold"
    prediction = kfold_predict(result, num_samples=20)
    np.testing.assert_allclose(prediction.observed.numpy(), Y[:8])
    assert prediction.predicted.shape == (1, 20, 8)
    assert prediction.positions == tuple(range(8))
    with pytest.raises(ValueError, match="save_fits"):
        kfold_predict(model.kfold(K=2, num_samples=10))


def test_joint_fit_criteria() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {"y1": rng.normal(size=16), "y2": rng.normal(size=16), "x": rng.normal(size=16)}
    )
    joint = fit(
        bf("y1 ~ x") + bf("y2 ~ x"), data=frame, method="svi", num_steps=60, seed=1
    )
    estimate = joint.loo(num_samples=200)
    assert estimate.responses == ("y1", "y2") and estimate.num_observations == 16
    scores = joint.log_likelihood(num_samples=200)["y1__y2"]
    np.testing.assert_allclose(
        estimate.pointwise_elpd.numpy(),
        arviz_loo(scores).pointwise_elpd.numpy(),
        rtol=1e-12,
    )
    single = joint.loo(resp="y1", num_samples=200)
    assert single.responses == ("y1",)
    # Moment-match only the unit with the largest Pareto shape, a single
    # pseudo-chain of variational draws.
    worst = int(torch.argmax(estimate.pareto_k))
    threshold = float(estimate.pareto_k[worst]) - 1e-9
    matched = joint.loo(moment_match=True, k_threshold=threshold, num_samples=200)
    assert matched.moment_matched in ((), (worst,))
    assert matched.moment_matched == tuple(
        i for i in range(16) if float(matched.pareto_k[i]) < float(estimate.pareto_k[i])
    )
    waic = joint.waic(num_samples=200)
    assert waic.criterion == "waic"
    predictions = joint.loo_predict(num_samples=200)
    assert set(predictions) == {"y1", "y2"}
    r2 = joint.loo_r2(num_samples=200)
    assert r2.rows == ("R2y1", "R2y2")
    subsample = joint.loo_subsample(observations=6, num_samples=200)
    assert subsample.responses == ("y1", "y2")
    stored = joint.add_criterion("loo", num_samples=200)
    assert loo_compare({"a": stored, "b": stored}).rows == ("a", "b")


def test_variational_and_hmc_fits_run_the_criteria() -> None:
    frame = pd.DataFrame({"y": Y[:10], "x": np.linspace(-1, 1, 10)})
    for method, options in (
        ("svi", {"num_steps": 40}),
        ("hmc", {"num_samples": 60, "num_warmup": 60, "num_chains": 2}),
    ):
        model = fit("y ~ x", data=frame, method=method, seed=2, **options)
        loo = model.loo()
        waic = model.waic()
        assert loo.num_observations == waic.num_observations == 10
        assert np.isfinite(loo.elpd) and np.isfinite(waic.elpd)
        assert loo.pareto_k is not None and loo.pareto_k.shape == (10,)
