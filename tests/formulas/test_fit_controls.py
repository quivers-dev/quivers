"""brms ``brm`` fitting controls mapped onto the shared inference drivers."""

from __future__ import annotations

from pathlib import Path
import tempfile

import didactic.api as dx
import pandas as pd
import pytest
import torch

from quivers.formulas import FormulaToQVRModule, bf, families, fit, formula_from_data
from quivers.formulas._fit_controls import (
    FixedParamKernel,
    SamplerControl,
    chain_initialization,
    mcmc_kernel,
    resolve_algorithm,
)
from quivers.dsl import Compiler
from quivers.inference import (
    MCMC,
    AutoLaplaceApproximation,
    AutoMultivariateNormalGuide,
    AutoNormalGuide,
    NUTSKernel,
)

FRAME = pd.DataFrame(
    {
        "y": [0.31, -1.24, 0.85, 1.93, -0.42, 0.13, 1.41, -0.73, 0.62, 2.21],
        "y2": [1.2, 0.4, -0.3, 2.2, 0.9, 1.1, -0.8, 0.2, 1.7, 0.6],
        "x": [0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8],
    }
)
STARTS = {"intercept": 0.0, "beta_x": 0.0, "sigma": 1.0}


def _program():
    formula = formula_from_data("y ~ x", FRAME)
    lens = FormulaToQVRModule(families["gaussian"])
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    observations = lens.fixed_column_observations(formula) | {
        "y": torch.tensor(FRAME.y.to_numpy(), dtype=torch.float32)
    }
    return program, observations


def test_thin_keeps_every_thinned_transition_of_the_same_chain():
    program, observations = _program()
    x = torch.zeros(1, 1, dtype=torch.long)
    full = MCMC(NUTSKernel(), num_warmup=15, num_samples=12, num_chains=1, seed=9).run(
        program, x, observations
    )
    thinned = MCMC(
        NUTSKernel(), num_warmup=15, num_samples=4, num_chains=1, seed=9, thin=3
    ).run(program, x, observations)
    for name, value in thinned.samples.items():
        assert value.shape == (1, 4)
        torch.testing.assert_close(value, full.samples[name][:, 2::3])
    with pytest.raises(ValueError, match="thin must be an integer"):
        MCMC(NUTSKernel(), num_warmup=1, num_samples=1, thin=0)


def test_control_configures_the_kernel():
    kernel = mcmc_kernel(
        "nuts",
        SamplerControl(
            adapt_delta=0.95, max_treedepth=6, stepsize=0.3, metric="dense_e"
        ),
    )
    configuration = kernel.configuration()
    assert configuration["target_accept"] == 0.95
    assert configuration["max_tree_depth"] == 6
    assert configuration["mass_matrix"] == "dense"
    assert isinstance(mcmc_kernel("fixed_param", None), FixedParamKernel)
    with pytest.raises(ValueError, match="max_treedepth applies to NUTS"):
        mcmc_kernel("hmc", SamplerControl(max_treedepth=4))
    with pytest.raises((ValueError, dx.ValidationError), match="adapt_delta"):
        SamplerControl(adapt_delta=1.0)
    fitted = fit(
        "y ~ x",
        data=FRAME,
        num_warmup=20,
        num_samples=6,
        num_chains=1,
        control={"adapt_delta": 0.9, "max_treedepth": 3},
        seed=4,
    )
    assert float(fitted.posterior.sample_stats["tree_depth"].max()) <= 3
    with pytest.raises(ValueError, match="unknown control settings"):
        fit("y ~ x", data=FRAME, control={"delta": 0.9}, num_samples=1)


def test_init_strategies_follow_brms():
    assert chain_initialization(None, STARTS, 2) == ("jitter", STARTS)
    assert chain_initialization("random", STARTS, 2)[0] == "uniform"
    assert chain_initialization(0, STARTS, 2)[0] == "zero"
    assert chain_initialization("0", STARTS, 2)[0] == "zero"
    strategy, values = chain_initialization(
        lambda chain: {"beta_x": float(chain)}, STARTS, 3
    )
    assert strategy == "uniform"
    assert [v["beta_x"] for v in values] == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="numeric init must be 0"):
        chain_initialization(0.5, STARTS, 1)
    with pytest.raises(ValueError, match="one mapping per chain"):
        chain_initialization([{"beta_x": 0.1}], STARTS, 2)
    # fixed_param keeps each chain at its starting values.
    fitted = fit(
        "y ~ x",
        data=FRAME,
        algorithm="fixed_param",
        num_samples=3,
        num_chains=2,
        init=[{**STARTS, "beta_x": 0.3}, {**STARTS, "beta_x": -0.7, "sigma": 2.0}],
    )
    torch.testing.assert_close(
        fitted.posterior.samples["beta_x"],
        torch.tensor([[0.3] * 3, [-0.7] * 3]),
    )
    # init = "random" draws every unconstrained coordinate inside (-init_r, init_r).
    random = fit(
        "y ~ x",
        data=FRAME,
        algorithm="fixed_param",
        num_samples=1,
        num_chains=6,
        init="random",
        init_r=0.25,
        seed=12,
    ).posterior.samples
    assert torch.all(random["beta_x"].abs() < 0.25)
    assert torch.all(random["sigma"].log().abs() < 0.25)
    assert len(set(random["beta_x"].flatten().tolist())) == 6


def test_algorithm_maps_onto_methods_and_guides():
    assert resolve_algorithm("sampling", "nuts", None) == ("nuts", None)
    assert resolve_algorithm("meanfield", "nuts", None) == ("svi", AutoNormalGuide)
    assert resolve_algorithm("fullrank", "nuts", None) == (
        "svi",
        AutoMultivariateNormalGuide,
    )
    assert resolve_algorithm("laplace", "svi", None) == (
        "svi",
        AutoLaplaceApproximation,
    )
    assert resolve_algorithm("pathfinder", "nuts", None) == ("pathfinder", None)
    with pytest.raises(ValueError, match="omit method"):
        resolve_algorithm("fullrank", "hmc", None)
    fitted = fit("y ~ x", data=FRAME, algorithm="fullrank", num_steps=15, seed=2)
    assert isinstance(fitted.posterior, AutoMultivariateNormalGuide)
    assert fitted.posterior_epred(new_data=FRAME.iloc[:2], num_samples=3).shape == (
        1,
        3,
        2,
    )
    with pytest.raises(ValueError, match="configure MCMC"):
        fit("y ~ x", data=FRAME, method="svi", num_steps=2, thin=2)


def test_file_refit_policies_reuse_or_refresh_the_artifact():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fit"
        first = fit("y ~ x", data=FRAME, method="svi", num_steps=4, seed=1, file=path)
        assert path.exists()
        cached = fit("y ~ x", data=FRAME, method="svi", num_steps=9, file=path)
        assert cached.losses == first.losses  # "never": the saved fit
        changed = fit(
            "y ~ x",
            data=FRAME,
            method="svi",
            num_steps=9,
            seed=1,
            file=path,
            file_refit="on_change",
        )
        assert len(changed.losses) == 9
        same = fit(
            "y ~ x",
            data=FRAME,
            method="svi",
            num_steps=9,
            seed=1,
            file=path,
            file_refit="on_change",
        )
        assert same.losses == changed.losses
        moved = fit(
            "y ~ x",
            data=FRAME.assign(y=FRAME.y + 1.0),
            method="svi",
            num_steps=9,
            seed=1,
            file=path,
            file_refit="on_change",
        )
        assert moved.losses != same.losses
        again = fit(
            "y ~ x",
            data=FRAME,
            method="svi",
            num_steps=9,
            seed=1,
            file=path,
            file_refit="always",
        )
        assert again.losses == changed.losses
        with pytest.raises(ValueError, match="file_refit must be"):
            fit("y ~ x", data=FRAME, file=path, file_refit="sometimes")


def test_joint_formulas_take_the_same_controls():
    spec = bf("y ~ x", center=True) + bf("y2 ~ x", decomp="QR")
    fitted = fit(
        spec,
        data=FRAME,
        default_priors="brms",
        num_warmup=20,
        num_samples=3,
        num_chains=2,
        thin=2,
        control={"adapt_delta": 0.9},
        init="random",
        init_r=0.5,
        seed=8,
    )
    draws = fitted.draws()
    assert draws["response_y__intercept"].shape == (2, 3)
    torch.testing.assert_close(
        draws["response_y__intercept"],
        draws["response_y__intercept_centered"]
        - float(FRAME.x.mean()) * draws["response_y__beta_x"],
    )
    assert "response_y2__beta_x_qr" in fitted.posterior.samples
    assert "response_y2__beta_x" in draws
    with pytest.raises(ValueError, match="carry the improper flat prior"):
        fit(
            spec,
            data=FRAME,
            default_priors="brms",
            sample_prior="yes",
            num_samples=1,
            num_warmup=1,
            num_chains=1,
        )
