"""Tests for :mod:`quivers.diagnostics`: the ArviZ adapter.

The adapter is glue between :class:`~quivers.inference.MCMCResult`
records and ArviZ's :class:`xarray.DataTree`-based data model.  The
tests exercise the conversion round-trip plus the user-facing
:func:`compare` and :func:`posterior_predictive_check` wrappers on
synthetic data.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import arviz as az
import numpy as np
import pytest
import torch
import xarray as xr

from quivers.diagnostics import (
    compare,
    loo,
    loo_pit,
    posterior_predictive_check,
    to_datatree,
)
from quivers.inference.mcmc.driver import (
    MCMCResult,
    _ess_1d,
    _split_rhat_1d,
)


def test_arviz_dependency_floor_matches_datatree_api() -> None:
    metadata = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )["project"]["optional-dependencies"]
    assert "arviz>=1.0" in metadata["diagnostics"]
    assert "arviz>=1.0" in metadata["formulas"]


def _make_result(samples: dict[str, torch.Tensor]) -> MCMCResult:
    """Construct an MCMCResult with consistent dimensions for the
    supplied per-site samples."""
    first = next(iter(samples.values()))
    n_chains, n_draws = first.shape[0], first.shape[1]
    return MCMCResult(
        samples=samples,
        log_densities=torch.randn(n_chains, n_draws),
        acceptance_rates=torch.tensor([0.85] * n_chains),
        divergence_counts=torch.tensor([0] * n_chains),
        r_hat={k: torch.tensor(1.003) for k in samples},
        ess={k: torch.tensor(950.0) for k in samples},
        num_warmup=500,
        num_samples=n_draws,
    )


def _make_fit(
    log_likelihood: dict[str, torch.Tensor],
    *,
    seed: int = 0,
) -> xr.DataTree:
    """Build a fit whose posterior and log likelihood have matching draws."""
    first = next(iter(log_likelihood.values()))
    n_chains, n_draws = first.shape[:2]
    generator = torch.Generator().manual_seed(seed)
    samples = {"theta": torch.randn(n_chains, n_draws, generator=generator)}
    observed = {
        name: torch.zeros(values.shape[2:]) for name, values in log_likelihood.items()
    }
    return to_datatree(
        _make_result(samples),
        log_likelihood=log_likelihood,
        observed_data=observed,
    )


def _quivers_convergence(draws: np.ndarray) -> tuple[float, float]:
    tensor = torch.as_tensor(draws, dtype=torch.float64)
    return float(_split_rhat_1d(tensor)), float(_ess_1d(tensor))


class TestMCMCConvergenceDiagnostics:
    @pytest.mark.parametrize("n_draws", [200, 201])
    def test_iid_chains_match_arviz_for_odd_and_even_draws(self, n_draws):
        draws = np.random.default_rng(7).normal(size=(4, n_draws))

        r_hat, ess = _quivers_convergence(draws)

        assert r_hat == pytest.approx(float(az.rhat(draws, method="rank")), rel=1e-10)
        assert ess == pytest.approx(float(az.ess(draws, method="bulk")), rel=1e-10)

    def test_separated_locations_match_arviz_and_reduce_bulk_ess(self):
        generator = np.random.default_rng(8)
        draws = generator.normal(size=(4, 500))
        draws[2:] += 2.0

        r_hat, ess = _quivers_convergence(draws)

        assert r_hat == pytest.approx(float(az.rhat(draws, method="rank")), rel=1e-10)
        assert ess == pytest.approx(float(az.ess(draws, method="bulk")), rel=1e-10)
        assert ess < 20

    def test_separated_scales_use_conservative_folded_rhat(self):
        generator = np.random.default_rng(9)
        draws = generator.normal(size=(4, 500))
        draws[2:] *= 4.0

        r_hat, ess = _quivers_convergence(draws)
        rank = float(az.rhat(draws, method="z_scale"))
        folded = float(az.rhat(draws, method="folded"))

        assert folded > rank
        assert r_hat == pytest.approx(max(rank, folded), rel=1e-10)
        assert r_hat > 1.1
        assert ess == pytest.approx(float(az.ess(draws, method="bulk")), rel=1e-10)

    def test_constant_coordinates_match_arviz_contract(self):
        draws = np.ones((4, 100))
        r_hat, ess = _quivers_convergence(draws)

        assert np.isnan(r_hat)
        assert np.isnan(float(az.rhat(draws, method="rank")))
        assert ess == float(az.ess(draws, method="bulk")) == draws.size

    def test_insufficient_draws_return_nan(self):
        r_hat, ess = _quivers_convergence(np.ones((4, 3)))

        assert np.isnan(r_hat)
        assert np.isnan(ess)


class TestToDataTree:
    def test_returns_datatree(self):
        samples = {"theta": torch.distributions.Beta(80.0, 20.0).sample((4, 1000))}
        dt = to_datatree(_make_result(samples))
        assert isinstance(dt, xr.DataTree)
        assert "posterior" in dt.children
        assert "sample_stats" in dt.children

    def test_posterior_shapes(self):
        samples = {
            "theta": torch.randn(4, 1000),
            "beta": torch.randn(4, 1000, 3),  # per-coordinate site
        }
        dt = to_datatree(_make_result(samples))
        assert dt["posterior"]["theta"].shape == (4, 1000)
        assert dt["posterior"]["beta"].shape == (4, 1000, 3)

    def test_sample_stats_populated(self):
        samples = {"x": torch.randn(4, 100)}
        dt = to_datatree(_make_result(samples))
        assert "lp" in dt["sample_stats"].data_vars
        assert dt["sample_stats"]["lp"].shape == (4, 100)

    def test_observed_data_group(self):
        samples = {"theta": torch.randn(4, 100)}
        observed = {"y": torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)}
        dt = to_datatree(_make_result(samples), observed_data=observed)
        assert "observed_data" in dt.children
        assert dt["observed_data"]["y"].shape == (5,)

    def test_log_likelihood_group(self):
        samples = {"theta": torch.randn(4, 100)}
        loglik = {"y": torch.randn(4, 100, 50)}
        dt = to_datatree(_make_result(samples), log_likelihood=loglik)
        assert "log_likelihood" in dt.children
        assert dt["log_likelihood"]["y"].shape == (4, 100, 50)

    def test_coords_dims_propagate(self):
        samples = {"beta": torch.randn(4, 100, 3)}
        dt = to_datatree(
            _make_result(samples),
            coords={"Verb": ["eat", "drink", "run"]},
            dims={"beta": ["Verb"]},
        )
        assert "Verb" in dt["posterior"]["beta"].coords
        assert list(dt["posterior"]["beta"]["Verb"].values) == [
            "eat",
            "drink",
            "run",
        ]


class TestCompare:
    def test_ranks_known_better_model_first(self):
        torch.manual_seed(0)
        n_chains, n_draws, n_obs = 4, 500, 100
        y = torch.randn(n_obs) * 0.5 + 1.0  # true: N(1, 0.5)

        # Model A: high-likelihood; sample a posterior over the
        # location whose draws hover near the truth so the per-draw
        # log-likelihood has nontrivial spread (PSIS-LOO requires
        # variation across draws to fit the Pareto tail).
        loc_a = torch.randn(n_chains, n_draws) * 0.1 + 1.0
        ll_a = -0.5 * ((y[None, None, :] - loc_a[..., None]) / 0.5) ** 2 - 0.5 * np.log(
            2 * np.pi * 0.5**2
        )
        # Model B: low-likelihood; draws around a wrong centre.
        loc_b = torch.randn(n_chains, n_draws) * 0.2 + 5.0
        ll_b = -0.5 * ((y[None, None, :] - loc_b[..., None]) / 3.0) ** 2 - 0.5 * np.log(
            2 * np.pi * 3.0**2
        )

        samples = {"theta": torch.randn(n_chains, n_draws)}
        dt_a = to_datatree(
            _make_result(samples), log_likelihood={"y": ll_a}, observed_data={"y": y}
        )
        dt_b = to_datatree(
            _make_result(samples), log_likelihood={"y": ll_b}, observed_data={"y": y}
        )
        comp = compare({"good": dt_a, "bad": dt_b})
        assert comp.index[0] == "good"
        assert comp.index[1] == "bad"

    def test_none_rounding_preserves_weight_precision(self):
        generator = torch.Generator().manual_seed(11)
        log_likelihood = torch.randn(4, 200, 12, generator=generator)
        fit = _make_fit({"y": log_likelihood})

        comp = compare(
            {"a": fit, "b": fit, "c": fit},
            var_name="y",
            round_to="none",
        )

        assert float(comp["weight"].sum()) == pytest.approx(1.0, abs=1e-12)
        assert all(weight != 0.33 for weight in comp["weight"])

    @pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
    def test_rounding_composes_with_methods_var_name_and_reference(self, method):
        generator = torch.Generator().manual_seed(12)
        fits = {}
        for index, name in enumerate(("first", "middle", "last")):
            log_likelihood = (
                torch.randn(4, 200, 10, generator=generator) * 0.2 - index * 0.1
            )
            fits[name] = _make_fit({"y": log_likelihood}, seed=index)

        comp = compare(
            fits,
            method=method,
            var_name="y",
            reference="middle",
            round_to="none",
        )

        assert float(comp.loc["middle", "elpd_diff"]) == 0.0
        assert float(comp["weight"].sum()) == pytest.approx(1.0, abs=1e-10)

    def test_integer_rounding_retains_arviz_semantics(self):
        generator = torch.Generator().manual_seed(13)
        fits = {
            name: _make_fit(
                {"y": torch.randn(4, 200, 10, generator=generator) - offset},
                seed=index,
            )
            for index, (name, offset) in enumerate((("first", 0.0), ("second", 0.2)))
        }

        raw = compare(fits, round_to="none")
        rounded = compare(fits, round_to=3)

        for column in ("elpd", "p", "elpd_diff", "weight", "se", "dse"):
            np.testing.assert_allclose(
                rounded[column].to_numpy(),
                raw[column].round(3).to_numpy(),
            )


class TestLoo:
    def test_pointwise_result_uses_log_scale_and_observation_shape(self):
        generator = torch.Generator().manual_seed(14)
        fit = _make_fit({"y": torch.randn(4, 300, 9, generator=generator)})

        result = loo(fit, pointwise=True, var_name="y")

        assert result.scale == "log"
        assert result.elpd_i.shape == (9,)
        assert result.pareto_k.shape == (9,)

    def test_result_preserves_nonfinite_pareto_values(self, monkeypatch):
        sentinel = type(
            "FakeELPDData",
            (),
            {"pareto_k": xr.DataArray([np.inf, -np.inf, np.nan])},
        )()

        def fake_loo(fit, *, pointwise, var_name):
            assert pointwise is True
            assert var_name == "y"
            return sentinel

        fit = _make_fit({"y": torch.randn(2, 20, 3)})
        monkeypatch.setattr(az, "loo", fake_loo)

        result = loo(fit, pointwise=True, var_name="y")

        assert result is sentinel
        assert np.isposinf(result.pareto_k[0])
        assert np.isneginf(result.pareto_k[1])
        assert np.isnan(result.pareto_k[2])

    def test_missing_log_likelihood_names_the_group(self):
        fit = to_datatree(_make_result({"theta": torch.randn(2, 20)}))

        with pytest.raises(ValueError, match="missing log_likelihood group"):
            loo(fit)

    def test_ambiguous_log_likelihood_names_variables(self):
        fit = _make_fit(
            {
                "y": torch.randn(2, 20, 3),
                "z": torch.randn(2, 20, 4),
            }
        )

        with pytest.raises(ValueError, match="var_name.*multiple variables.*'y'.*'z'"):
            loo(fit)

    def test_unknown_log_likelihood_variable_is_named(self):
        fit = _make_fit({"y": torch.randn(2, 20, 3)})

        with pytest.raises(ValueError, match="'z'.*not found.*'y'"):
            loo(fit, var_name="z")

    def test_total_elpd_agrees_with_unrounded_comparison(self):
        generator = torch.Generator().manual_seed(15)
        fits = {
            "first": _make_fit({"y": torch.randn(4, 200, 8, generator=generator)}),
            "second": _make_fit(
                {"y": torch.randn(4, 200, 8, generator=generator) - 0.2},
                seed=1,
            ),
        }

        loo_result = loo(fits["first"], pointwise=True, var_name="y")
        comp = compare(fits, var_name="y", round_to="none")

        assert float(loo_result.elpd) == pytest.approx(float(comp.loc["first", "elpd"]))


class TestPosteriorPredictiveCheck:
    def test_well_specified_ppp_central(self):
        # Posterior predictive identical to observed-data distribution
        # → PPP-value near 0.5 for the mean statistic.
        torch.manual_seed(0)
        y = torch.randn(100) * 2.0 + 3.0
        n_chains, n_draws = 4, 500
        pp = torch.randn(n_chains, n_draws, 100) * 2.0 + 3.0
        samples = {"theta": torch.randn(n_chains, n_draws)}
        dt = to_datatree(
            _make_result(samples),
            observed_data={"y": y},
            posterior_predictive={"y": pp},
        )
        result = posterior_predictive_check(dt, observed_name="y", statistic="mean")
        assert 0.2 < float(result["ppp"]) < 0.8

    def test_mis_specified_ppp_extreme(self):
        # Posterior predictive far from observed → PPP-value near 0 or 1.
        torch.manual_seed(0)
        y = torch.randn(100) * 0.5 + 10.0
        n_chains, n_draws = 4, 500
        pp = torch.randn(n_chains, n_draws, 100) * 0.5
        samples = {"theta": torch.randn(n_chains, n_draws)}
        dt = to_datatree(
            _make_result(samples),
            observed_data={"y": y},
            posterior_predictive={"y": pp},
        )
        result = posterior_predictive_check(dt, observed_name="y", statistic="mean")
        ppp = float(result["ppp"])
        assert ppp < 0.05 or ppp > 0.95

    def test_user_statistic(self):
        torch.manual_seed(0)
        y = torch.randn(100)
        n_chains, n_draws = 2, 100
        pp = torch.randn(n_chains, n_draws, 100)
        samples = {"theta": torch.randn(n_chains, n_draws)}
        dt = to_datatree(
            _make_result(samples),
            observed_data={"y": y},
            posterior_predictive={"y": pp},
        )

        def kurtosis(arr):
            arr = np.asarray(arr)
            m = arr.mean()
            return float(((arr - m) ** 4).mean() / (((arr - m) ** 2).mean()) ** 2)

        result = posterior_predictive_check(dt, observed_name="y", statistic=kurtosis)
        assert result["statistic"] == "kurtosis"
        assert "ppp" in result

    def test_unknown_statistic_raises(self):
        torch.manual_seed(0)
        y = torch.randn(10)
        pp = torch.randn(2, 5, 10)
        samples = {"theta": torch.randn(2, 5)}
        dt = to_datatree(
            _make_result(samples),
            observed_data={"y": y},
            posterior_predictive={"y": pp},
        )
        with pytest.raises(ValueError, match="unknown statistic"):
            posterior_predictive_check(dt, observed_name="y", statistic="nonexistent")


class TestLooPit:
    def test_pit_values_lie_in_unit_interval(self):
        torch.manual_seed(0)
        y = torch.randn(20)
        n_chains, n_draws = 2, 200
        pp = torch.randn(n_chains, n_draws, 20)
        log_lik = -0.5 * (y - torch.randn(n_chains, n_draws, 1)) ** 2
        samples = {"theta": torch.randn(n_chains, n_draws)}
        dt = to_datatree(
            _make_result(samples),
            observed_data={"y": y},
            posterior_predictive={"y": pp},
            log_likelihood={"y": log_lik},
        )
        result = loo_pit(dt, observed_name="y")
        assert isinstance(result, xr.Dataset)
        values = np.asarray(result["y"])
        assert values.shape == (20,)
        assert np.all((values >= 0.0) & (values <= 1.0))
