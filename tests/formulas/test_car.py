"""brms conditional autoregressive terms against brms and dense oracles.

brms values are pinned from ``tests/formulas/oracles/autocor_brms.R``. brms
adds ``max(diag(Q)) * sqrt(eps)`` to the Laplacian before inverting it, so its
BYM2 scaling factors carry a relative perturbation below 1e-7; the exact
values are compared at that tolerance. Densities use 1e-10.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal
import torch

from quivers.formulas import PriorSpec, fit, get_prior
from quivers.formulas._latent_effects import (
    ComponentSumToZero,
    ComponentSumToZeroTransform,
    SparseCAR,
    car_graph,
)
from quivers.modeling import load_fit, save_fit

RTOL = ATOL = 1e-10
BRMS_SCALE = {4: 0.57282194195281777, 5: 0.39999999310821294}


def _path(count, cycle=False):
    adjacency = np.zeros((count, count))
    for i in range(count - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
    if cycle:
        adjacency[0, -1] = adjacency[-1, 0] = 1.0
    return adjacency


def _labels(count):
    return tuple(f"L{i}" for i in range(count))


def _disconnected():
    adjacency = np.zeros((6, 6))
    adjacency[:3, :3] = _path(3)
    adjacency[3:5, 3:5] = _path(2)
    return adjacency


@pytest.mark.parametrize("count,cycle", [(4, False), (5, True)])
def test_bym2_scaling_factor_matches_brms(count, cycle):
    graph = car_graph(_path(count, cycle), _labels(count), "bym2")
    np.testing.assert_allclose(
        1 / graph.inverse_scale, BRMS_SCALE[count], rtol=1e-7, atol=0
    )


def test_adjacency_validation_follows_brms():
    with pytest.raises(ValueError, match="symmetric"):
        car_graph(np.triu(_path(3)), _labels(3), "escar")
    with pytest.raises(ValueError, match="zero diagonal"):
        car_graph(_path(3) + np.eye(3), _labels(3), "icar")
    with pytest.raises(ValueError, match="at least one neighbor"):
        car_graph(_disconnected(), _labels(6), "escar")
    graph = car_graph(_disconnected(), _labels(6), "bym2")
    assert graph.components == ((0, 1, 2), (3, 4), (5,))
    assert graph.rank == 3 and graph.inverse_scale[5] == 0.0


def test_component_constraint_transform_round_trips():
    support = ComponentSumToZero(((0, 1, 2), (3, 4), (5,)), 6)
    transform = ComponentSumToZeroTransform(support)
    free = torch.tensor([[0.3, -1.2, 0.7], [1.0, 2.0, -0.5]], dtype=torch.float64)
    full = transform(free)
    assert full.shape == (2, 6)
    assert bool(support.check(full).all())
    torch.testing.assert_close(transform.inv(full), free)
    assert full[:, 5].abs().max() == 0


@pytest.mark.parametrize("alpha", [0.0, 0.35, 0.9])
def test_proper_car_density_matches_dense_precision(alpha):
    adjacency = _path(5, cycle=True)
    adjacency[0, 2] = adjacency[2, 0] = 1.0
    graph = car_graph(adjacency, _labels(5), "escar")
    scale = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    alpha_t = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
    value = torch.tensor([0.3, -0.4, 1.1, 0.2, -0.9], dtype=torch.float64)
    score = SparseCAR(graph, scale, alpha_t).log_prob(value)
    precision = (np.diag(adjacency.sum(1)) - alpha * adjacency) / 0.49
    expected = multivariate_normal.logpdf(value.numpy(), cov=np.linalg.inv(precision))
    np.testing.assert_allclose(float(score), expected, rtol=RTOL, atol=ATOL)
    grad = torch.autograd.grad(score, alpha_t)[0]
    step = 1e-6

    def at(a):
        q = (np.diag(adjacency.sum(1)) - a * adjacency) / 0.49
        return multivariate_normal.logpdf(value.numpy(), cov=np.linalg.inv(q))

    numeric = (at(alpha + step) - at(alpha - step)) / (2 * step)
    np.testing.assert_allclose(float(grad), numeric, rtol=1e-6, atol=1e-7)
    assert SparseCAR(graph, scale, torch.tensor(1.0)).log_prob(value) == -math.inf


def test_intrinsic_car_scores_free_coordinates_of_each_component():
    adjacency = _disconnected()
    graph = car_graph(adjacency, _labels(6), "icar")
    transform = ComponentSumToZeroTransform(ComponentSumToZero(graph.components, 6))
    free = torch.tensor([0.4, -0.3, 0.8], dtype=torch.float64)
    value = transform(free)
    score = SparseCAR(graph, torch.tensor(1.3, dtype=torch.float64), None).log_prob(
        value
    )
    expansion = transform(torch.eye(3, dtype=torch.float64)).numpy().T
    laplacian = np.diag(adjacency.sum(1)) - adjacency
    precision = expansion.T @ laplacian @ expansion / 1.69
    expected = multivariate_normal.logpdf(free.numpy(), cov=np.linalg.inv(precision))
    np.testing.assert_allclose(float(score), expected, rtol=RTOL, atol=ATOL)


def test_prior_draws_respect_the_constraint_and_proper_covariance():
    torch.manual_seed(3)
    adjacency = _path(4)
    intrinsic = SparseCAR(
        car_graph(adjacency, _labels(4), "icar"),
        torch.tensor(1.0, dtype=torch.float64),
        None,
    )
    draws = intrinsic.sample(torch.Size((20000,)))
    assert draws.sum(-1).abs().max() < 1e-12
    graph = car_graph(adjacency, _labels(4), "bym2")
    covariance = np.cov(draws.numpy().T)
    # The marginal variances are the generalized inverse diagonal.
    np.testing.assert_allclose(
        np.exp(np.mean(np.log(np.diag(covariance)))),
        1 / graph.inverse_scale[0],
        rtol=0.05,
    )
    proper = SparseCAR(
        car_graph(adjacency, _labels(4), "escar"),
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(0.5, dtype=torch.float64),
    )
    sample = proper.sample(torch.Size((20000,))).numpy()
    expected = np.linalg.inv(np.diag(adjacency.sum(1)) - 0.5 * adjacency)
    np.testing.assert_allclose(np.cov(sample.T), expected, atol=0.06)


def _spatial_frame(seed=0):
    rng = np.random.default_rng(seed)
    labels = _labels(6)
    frame = pd.DataFrame({"loc": np.repeat(labels, 3), "x": rng.normal(size=18)})
    frame["y"] = rng.poisson(np.exp(0.3 + 0.2 * frame.x))
    adjacency = _path(6, cycle=True)
    return frame, pd.DataFrame(adjacency, index=labels, columns=labels)


@pytest.mark.parametrize("car_type", ["escar", "esicar", "icar", "bym2"])
def test_car_formula_classes_and_replayed_predictor(car_type):
    frame, adjacency = _spatial_frame()
    formula = f"y ~ x + car(M, gr = loc, type = '{car_type}')"
    prior = get_prior(formula, data=frame, data2={"M": adjacency}, family="poisson")
    assert "sdcar" in prior
    assert ("car" in prior) == (car_type == "escar")
    assert ("rhocar" in prior) == (car_type == "bym2")
    fitted = fit(
        formula,
        data=frame,
        data2={"M": adjacency},
        family="poisson",
        method="svi",
        num_steps=20,
        seed=1,
    )
    draws = fitted.draws(num_samples=3, seed=1)
    eta = fitted.posterior_linpred(num_samples=3, seed=1)
    codes = np.array([int(label[1:]) for label in frame["loc"]])
    graph = car_graph(adjacency.to_numpy(), _labels(6), car_type)
    for draw in range(3):
        value = {k: v[0, draw].double().numpy() for k, v in draws.items()}
        if car_type in {"escar", "esicar"}:
            effect = value["rcar"]
        elif car_type == "icar":
            effect = value["zcar"] * value["sdcar"]
        else:
            effect = (
                np.sqrt(1 - value["rhocar"]) * value["nszcar"]
                + np.sqrt(value["rhocar"] * graph.inverse_scale) * value["zcar"]
            ) * value["sdcar"]
        if car_type != "escar":
            spatial = value["rcar"] if car_type == "esicar" else value["zcar"]
            assert abs(spatial.sum()) < 1e-5
        expected = (
            value["intercept"] + value["beta_x"] * frame.x.to_numpy() + effect[codes]
        )
        np.testing.assert_allclose(
            eta[0, draw].double().numpy(), expected, rtol=2e-5, atol=2e-5
        )


@pytest.mark.parametrize("method", ["hmc", "nuts"])
def test_car_runs_with_mcmc_and_gaussian_responses(method):
    frame, adjacency = _spatial_frame(2)
    fitted = fit(
        "y ~ x + car(M, gr = loc, type = 'icar')",
        data=frame.assign(y=frame.y.astype(float)),
        data2={"M": adjacency},
        method=method,
        num_warmup=4,
        num_samples=3,
        num_chains=1,
        seed=2,
    )
    draws = fitted.draws()["zcar"]
    assert draws.sum(-1).abs().max() < 1e-4
    assert torch.isfinite(fitted.log_likelihood()["y"]).all()


def test_car_prediction_refit_and_artifacts(tmp_path):
    frame, adjacency = _spatial_frame(4)
    fitted = fit(
        "y ~ x + car(M, gr = loc, type = 'bym2')",
        data=frame,
        data2={"M": adjacency},
        family="poisson",
        method="svi",
        num_steps=10,
        priors=PriorSpec(distribution="Beta(2.0, 2.0)", class_="rhocar"),
        seed=3,
    )
    fresh = pd.DataFrame({"loc": ["L5", "L0"], "x": [0.1, -0.2]})
    assert fitted.posterior_predict(new_data=fresh, num_samples=2).shape == (1, 2, 2)
    with pytest.raises(ValueError, match="new locations"):
        fitted.posterior_epred(new_data=fresh.assign(loc=["L9", "L0"]), num_samples=1)
    assert fitted.update(num_steps=3).posterior_epred(num_samples=1).shape == (1, 1, 18)
    path = tmp_path / "car.pt"
    save_fit(fitted, path)
    torch.testing.assert_close(
        load_fit(path).posterior_epred(new_data=fresh, num_samples=2, seed=1),
        fitted.posterior_epred(new_data=fresh, num_samples=2, seed=1),
    )


def test_car_rejections_and_combination_with_a_time_term():
    frame, adjacency = _spatial_frame()
    with pytest.raises(ValueError, match="Row names"):
        fit(
            "y ~ x + car(M, gr = loc)",
            data=frame,
            data2={"M": adjacency.to_numpy()},
            family="poisson",
            method="svi",
            num_steps=1,
        )
    ungrouped = fit(
        "y ~ x + car(M, type = 'icar')",
        data=frame.iloc[:6],
        data2={"M": _path(6)},
        family="poisson",
        method="svi",
        num_steps=2,
    )
    with pytest.raises(ValueError, match="Without a grouping factor"):
        ungrouped.posterior_epred(new_data=frame.iloc[:2], num_samples=1)
    with pytest.raises(ValueError, match="one spatial term"):
        fit(
            "y ~ x + car(M, gr = loc) + sar(M)",
            data=frame,
            data2={"M": adjacency},
            method="svi",
            num_steps=1,
        )
    series = frame.assign(t=np.tile(np.arange(3), 6), y=frame.y.astype(float))
    combined = fit(
        "y ~ x + ar(time = t, gr = loc) + car(M, gr = loc, type = 'esicar')",
        data=series,
        data2={"M": adjacency},
        method="svi",
        num_steps=3,
    )
    assert "ar_1" in combined.qvr_source and "rcar" in combined.qvr_source


def _count_series(seed=5):
    rng = np.random.default_rng(seed)
    rows = [{"g": g, "t": t, "x": rng.normal()} for g in ("a", "b") for t in range(5)]
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=seed)
    frame["y"] = rng.poisson(2.0, size=len(frame))
    return frame.reset_index(drop=True)


@pytest.mark.parametrize(
    "term,kind",
    [
        ("ar(time = t, gr = g, p = 2)", "ar"),
        ("arma(time = t, gr = g, cov = TRUE)", "arma"),
        ("cosy(time = t, gr = g)", "cosy"),
    ],
)
def test_latent_residual_law_matches_brms_construction(term, kind):
    from quivers.formulas._autocor import latent_residual_term, parse_autocor_call

    frame = _count_series()
    latent = latent_residual_term(
        parse_autocor_call(term),
        data=frame,
        caller="test",
    )
    values = {"ar_1": 0.4, "ar_2": -0.2, "ma_1": 0.3, "cosy": 0.35}
    sites = [site for site, *_ in latent.coefficients]
    law = latent.catalog.build(
        "LatentResidual",
        {
            "scale": torch.tensor(0.6, dtype=torch.float64),
            **{site: torch.tensor(values[site], dtype=torch.float64) for site in sites},
            "anchor": torch.zeros(len(frame), dtype=torch.float64),
        },
    )
    err = torch.tensor(np.linspace(-1, 1, len(frame)), dtype=torch.float64)
    expected = 0.0
    layout = latent.layout
    assert layout is not None
    for start, length in zip(layout.starts, layout.lengths, strict=True):
        rows = list(layout.order[start : start + length])
        if kind == "ar":
            # brms: u_n = err_n + sum_k ar_k err_(n-k), err iid N(0, sderr).
            operator = np.eye(length)
            for lag, coefficient in ((1, 0.4), (2, -0.2)):
                operator += coefficient * np.eye(length, k=-lag)
            cov = 0.36 * operator @ operator.T
        elif kind == "arma":
            g0 = (1 + 0.09 + 2 * 0.4 * 0.3) / (1 - 0.16)
            g1 = (1 + 0.12) * 0.7 / (1 - 0.16)
            cov = 0.36 * np.array(
                [
                    [
                        g0 if i == j else g1 * 0.4 ** (abs(i - j) - 1)
                        for j in range(length)
                    ]
                    for i in range(length)
                ]
            )
        else:
            cov = 0.36 * (0.65 * np.eye(length) + 0.35)
        expected += multivariate_normal.logpdf(err.numpy()[rows], cov=cov)
    np.testing.assert_allclose(float(law.log_prob(err)), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_latent_residuals_fit_replay_and_predict(method):
    frame = _count_series()
    settings = (
        {"method": "svi", "num_steps": 10}
        if method == "svi"
        else {"method": method, "num_warmup": 3, "num_samples": 2, "num_chains": 1}
    )
    fitted = fit(
        "y ~ x + cosy(time = t, gr = g)",
        data=frame,
        family="poisson",
        seed=4,
        **settings,
    )
    prior = get_prior("y ~ x + cosy(time = t, gr = g)", data=frame, family="poisson")
    assert "sderr" in prior and "cosy" in prior
    draws = fitted.draws(num_samples=2, seed=1)
    eta = fitted.posterior_linpred(num_samples=2, seed=1)
    plain = fitted.posterior_linpred(num_samples=2, seed=1, incl_autocor=False)
    for draw in range(2):
        value = {k: v[0, draw].double() for k, v in draws.items()}
        base = value["intercept"] + value["beta_x"] * torch.tensor(frame.x.to_numpy())
        torch.testing.assert_close(
            eta[0, draw].double(), base + value["err"], rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(plain[0, draw].double(), base, rtol=2e-5, atol=2e-5)
    fresh = pd.DataFrame({"g": "z", "t": [0, 1, 2, 3], "x": [0.1, -0.2, 0.3, 0.0]})
    assert fitted.posterior_predict(new_data=fresh, num_samples=2).shape == (1, 2, 4)


def test_latent_residual_refusals_follow_brms():
    frame = _count_series()
    conditional = fit(
        "y ~ x + ar(time = t, gr = g)",
        data=frame,
        family="poisson",
        method="svi",
        num_steps=2,
    )
    with pytest.raises(ValueError, match="cov = FALSE"):
        conditional.posterior_epred(new_data=frame.iloc[:3], num_samples=1)
    for formula, message in [
        ("y ~ x + ma(time = t, gr = g)", "MA structures"),
        ("y ~ x + sar(M)", "SAR terms are not implemented"),
    ]:
        with pytest.raises(ValueError, match=message):
            fit(
                formula,
                data=frame,
                data2={"M": np.eye(len(frame))},
                family="poisson",
                method="svi",
                num_steps=1,
            )
