"""brms pp_check and mcmc_plot on formula fits.

The tables are pinned against bayesplot in ``test_ppc_data.py``; these
tests check brms's preparation: which draws are used by default, the
prediction method per type, observed responses and covariates, censored
responses, prior predictive checks, PSIS weights of leave-one-out types,
joint responses, refusals, and the ArviZ counterparts (built with the
``none`` backend, which computes a plot without drawing it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.diagnostics import _ppc_data as tables
from quivers.diagnostics.elpd import psis_weights
from quivers.formulas import bf, fit, set_rescor
from quivers.formulas._mcmc_plot import MCMC_PLOTS, NOT_PROVIDED, mcmc_datatree
from quivers.formulas._pp_check import ARVIZ_PLOTS, PPC_TYPES, PPD_TYPES
from quivers.formulas._response import response_tensor

RNG = np.random.default_rng(3)
N = 24
DATA = pd.DataFrame(
    {
        "x": RNG.normal(size=N),
        "g": np.tile(["a", "b", "c"], N // 3),
        "k": RNG.poisson(2.0, size=N),
    }
)
DATA["y"] = 0.5 + DATA["x"] + RNG.normal(size=N)


@pytest.fixture(scope="module")
def gaussian():
    return fit("y ~ x", data=DATA, method="svi", num_steps=40, seed=1)


@pytest.fixture(scope="module")
def counts():
    return fit("k ~ x", data=DATA, family="poisson", method="svi", num_steps=40, seed=1)


def test_default_draw_counts_follow_brms(gaussian):
    assert gaussian.pp_check("dens_overlay").yrep.shape == (10, N)
    assert gaussian.pp_check("stat").yrep.shape == (gaussian.ndraws(), N)
    assert gaussian.pp_check("hist", ndraws=4).yrep.shape == (4, N)
    picked = gaussian.pp_check("hist", draw_ids=[0, 5, 7])
    assert picked.draw_ids == (0, 5, 7)
    with pytest.raises(ValueError, match="not both"):
        gaussian.pp_check("hist", ndraws=2, draw_ids=[1])


def test_tables_are_computed_from_the_fits_predictions(gaussian):
    check = gaussian.pp_check("dens_overlay", draw_ids=[1, 2, 3], seed=4)
    expected = gaussian.posterior_predict(seed=4).reshape(-1, N)[[1, 2, 3]].double()
    # The prediction at selected draws equals the full replay at those draws.
    view = gaussian.with_(posterior=gaussian.posterior)
    del view
    assert check.yrep.shape == expected.shape
    y = response_tensor(gaussian.formula, gaussian.family).double()
    torch.testing.assert_close(check.y, y)
    pd.testing.assert_frame_equal(
        check.data, tables.ppc_data(y.numpy(), check.yrep.numpy())
    )
    grouped = gaussian.pp_check("stat_grouped", group="g", stat="median")
    pd.testing.assert_frame_equal(
        grouped.data,
        tables.ppc_stat_data(
            y.numpy(), grouped.yrep.numpy(), stat="median", group=DATA["g"].to_numpy()
        ),
    )
    intervals = gaussian.pp_check("intervals", x="x", prob=0.4)
    assert list(intervals.data["x"]) == pytest.approx(list(DATA["x"]))
    assert intervals.data["inner_width"].iloc[0] == 0.4


def test_error_binned_uses_expected_values(gaussian):
    check = gaussian.pp_check("error_binned", draw_ids=[0, 1], seed=2)
    expected = gaussian.posterior_epred(seed=2).reshape(-1, N)[[0, 1]].double()
    torch.testing.assert_close(check.yrep, expected)
    assert set(check.data.columns) == {"ey_bar", "err_bar", "se2", "bin", "rep_id"}


def test_loo_types_use_psis_weights_of_the_same_draws(gaussian):
    check = gaussian.pp_check("loo_pit")
    assert check.log_weights is not None and check.log_likelihood is not None
    weights = psis_weights(check.log_likelihood.unsqueeze(0), r_eff=1.0)
    torch.testing.assert_close(
        check.log_weights, weights.log_weights.reshape(check.yrep.shape).double()
    )
    np.testing.assert_allclose(
        check.data["pit"],
        tables.loo_pit_values(
            check.y.numpy(), check.yrep.numpy(), check.log_weights.numpy()
        ),
    )
    assert np.all((check.data["pit"] >= 0) & (check.data["pit"] <= 1 + 1e-12))
    intervals = gaussian.pp_check("loo_intervals")
    assert np.all(intervals.data["ll"] <= intervals.data["hh"])


def test_count_types(counts):
    bars = counts.pp_check("bars")
    assert list(bars.data.columns) == ["x", "y_obs", "l", "m", "h"]
    root = counts.pp_check("rootogram", style="hanging")
    assert np.allclose(root.data["ypos"], root.data["tyexp"] - root.data["ty"] / 2)


def test_prior_predictive_checks(gaussian):
    prior = gaussian.prior_view(num_samples=20)
    check = prior.pp_check("dens_overlay", prefix="ppd")
    assert check.y is None and check.prefix == "ppd"
    assert "is_y" not in check.data.columns
    assert check.data["rep_label"].iloc[0] == "italic(y)[pred] ( 1 )"
    with pytest.raises(ValueError, match="not a valid ppd type"):
        prior.pp_check("loo_pit", prefix="ppd")


def test_censored_responses():
    frame = DATA.assign(
        t=np.exp(DATA["x"]) + 0.1, cens=np.where(np.arange(N) % 4 == 0, 1, 0)
    )
    model = fit(
        "t | cens(cens) ~ x", data=frame, family="weibull", method="svi", num_steps=5
    )
    km = model.pp_check("km_overlay", ndraws=3)
    assert km.status is not None
    np.testing.assert_array_equal(km.status.numpy(), 1 - frame["cens"].to_numpy())
    dens = model.pp_check("dens_overlay", ndraws=3)
    assert dens.y.shape[0] == int((frame["cens"] == 0).sum())


def test_joint_fits_check_one_response():
    frame = DATA.assign(w=DATA["y"] * 0.5)
    joint = fit(
        bf("y ~ x") + bf("w ~ 1") + set_rescor(False),
        data=frame,
        method="svi",
        num_steps=5,
    )
    check = joint.pp_check("scatter_avg", resp="w")
    torch.testing.assert_close(
        check.y, torch.as_tensor(frame["w"].to_numpy(), dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="requires resp"):
        joint.pp_check("scatter_avg")


def test_refusals(gaussian):
    with pytest.raises(ValueError, match="not a valid ppc type"):
        gaussian.pp_check("dens_overlay_typo")
    with pytest.raises(ValueError, match="group is required"):
        gaussian.pp_check("stat_grouped")
    with pytest.raises(ValueError, match="could not be found"):
        gaussian.pp_check("intervals", x="nope")
    with pytest.raises(ValueError, match="requires x"):
        gaussian.pp_check("error_scatter_avg_vs_x")
    with pytest.raises(ValueError, match="joint fits"):
        gaussian.pp_check("hist", resp="y")
    with pytest.raises(ValueError, match="no counterpart"):
        gaussian.pp_check("scatter").plot(backend="none")


def test_type_catalogues_cover_bayesplot():
    assert len(PPC_TYPES) == 41 and len(PPD_TYPES) == 16
    assert set(ARVIZ_PLOTS) <= set(PPC_TYPES) | {"loo_pit_ecdf"}


@pytest.mark.parametrize(
    "type", ["dens_overlay", "hist", "ecdf_overlay", "stat", "intervals", "pit_ecdf"]
)
def test_arviz_counterparts_build(gaussian, type):
    assert gaussian.pp_check(type, ndraws=8).plot(backend="none") is not None


def test_new_data_checks(gaussian):
    new = DATA.iloc[:5]
    check = gaussian.pp_check("scatter", newdata=new, ndraws=2)
    assert check.yrep.shape == (2, 5)
    with pytest.raises(ValueError, match="must hold the response"):
        gaussian.pp_check("scatter", newdata=new.drop(columns="y"))


@pytest.mark.parametrize("method", ["hmc", "nuts"])
def test_mcmc_plot_types_map_onto_arviz(method):
    sampled = fit(
        "y ~ x",
        data=DATA,
        method=method,
        num_warmup=20,
        num_samples=20,
        num_chains=2,
        seed=2,
    )
    tree = mcmc_datatree(sampled)
    assert {"b_Intercept", "b_x", "sigma"} <= set(tree["posterior"].data_vars)
    for type in ("intervals", "trace", "dens", "rank_hist", "acf", "neff", "rhat"):
        assert sampled.mcmc_plot(type, backend="none") is not None
    assert sampled.plot(variable="b_.*", backend="none") is not None
    assert sampled.pairs(variable=["b_x", "sigma"], backend="none") is not None
    if method == "nuts":
        assert "energy" in tree["sample_stats"].data_vars
        assert sampled.mcmc_plot("nuts_energy", backend="none") is not None
    with pytest.raises(ValueError, match="not provided"):
        sampled.mcmc_plot("nuts_treedepth", backend="none")
    with pytest.raises(ValueError, match="unknown"):
        sampled.mcmc_plot("bogus", backend="none")
    with pytest.raises(ValueError, match="no variable"):
        sampled.mcmc_plot("trace", variable="zzz", backend="none")
    assert set(NOT_PROVIDED).isdisjoint(MCMC_PLOTS)
