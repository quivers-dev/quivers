"""brm_multiple-style fitting over imputed datasets and combine_models pooling."""

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import (
    MultipleFit,
    bf,
    combine_models,
    fit,
    fit_multiple,
    set_rescor,
)


def _datasets():
    generator = np.random.default_rng(5)
    x = generator.normal(size=10)
    base = pd.DataFrame({"x": x, "y": 0.4 + 0.9 * x + 0.2 * generator.normal(size=10)})
    # Two imputations differing only in previously missing predictor values.
    first, second = base.copy(), base.copy()
    first.loc[[2, 7], "x"] = [0.1, -0.3]
    second.loc[[2, 7], "x"] = [0.5, 0.2]
    return first, second


def _budget():
    return {"method": "nuts", "num_warmup": 20, "num_samples": 15, "num_chains": 2}


def test_fit_multiple_pools_every_dataset_chain_and_reports_per_dataset_rhat():
    datasets = _datasets()
    result = fit_multiple("y ~ x", data=datasets, seed=3, **_budget())
    assert isinstance(result, MultipleFit) and len(result.fits) == 2
    pooled = result.pooled.draws()
    for name, value in pooled.items():
        torch.testing.assert_close(
            value, torch.cat([fitted.draws()[name] for fitted in result.fits], dim=0)
        )
    assert pooled["beta_x"].shape[:2] == (4, 15)
    rhats = result.rhats()
    assert len(rhats) == 2 and set(rhats[0]) == set(pooled)
    # Each dataset is the ordinary fit with the shared seed, as in brms.
    alone = fit("y ~ x", data=datasets[1], seed=3, **_budget())
    torch.testing.assert_close(
        result.fits[1].draws()["beta_x"], alone.draws()["beta_x"]
    )
    means = result.pooled.posterior_epred(new_data=pd.DataFrame({"x": [0.0, 1.0]}))
    assert means.shape == (4, 15, 2)


def test_combine_models_pools_joint_fits_and_refuses_guides_and_other_models():
    first, second = _datasets()
    spec = bf("y ~ x") + bf("x ~ 1") + set_rescor(False)
    a = fit(spec, data=first, seed=1, **_budget())
    b = fit(spec, data=second, seed=2, **_budget())
    pooled = combine_models(a, b)
    assert pooled.draws()["response_y__beta_x"].shape[:2] == (4, 15)
    with pytest.raises(ValueError, match="latent sites differ"):
        combine_models(
            fit("y ~ x", data=first, seed=1, **_budget()),
            fit("y ~ 1", data=first, seed=1, **_budget()),
        )
    with pytest.raises(TypeError, match="variational guide"):
        combine_models(fit("y ~ x", data=first, method="svi", num_steps=1))
    with pytest.raises(ValueError, match="nuts"):
        fit_multiple("y ~ x", data=[first], method="svi", num_steps=1)


def test_update_refits_every_dataset_through_the_first_fits_recipe():
    datasets = _datasets()
    result = fit_multiple("y ~ x", data=datasets, seed=3, **_budget())
    # brms requires the datasets again: update(<brmsfit_multiple>) stops with
    # "'newdata' is required when updating a 'brmsfit_multiple' object."
    with pytest.raises(ValueError, match="is required when updating"):
        result.update()
    with pytest.raises(ValueError, match="sequence of dataframes"):
        result.update(datasets[0])
    with pytest.raises(ValueError, match="same length as 'data'"):
        result.update(datasets, data2=[{}])
    with pytest.raises(ValueError, match="recorded draws"):
        result.update(datasets, method="svi", num_steps=2)
    # Unchanged arguments reproduce the original pooled draws.
    same = result.update(datasets)
    for name, value in result.pooled.draws().items():
        torch.testing.assert_close(same.pooled.draws()[name], value)
    # Overrides reach every dataset, and the dataset count may change.
    third = datasets[0].assign(x=lambda rows: rows.x + 0.05)
    updated = result.update((*datasets, third), formula="y ~ 1", num_samples=10, seed=5)
    assert isinstance(updated, MultipleFit) and len(updated.fits) == 3
    pooled = updated.pooled.draws()
    assert "beta_x" not in pooled and pooled["intercept"].shape[:2] == (6, 10)
    alone = fit("y ~ 1", data=third, seed=5, **{**_budget(), "num_samples": 10})
    torch.testing.assert_close(
        updated.fits[2].draws()["intercept"], alone.draws()["intercept"]
    )
    assert len(updated.rhats()) == 3


def test_update_of_joint_multiple_fits_takes_one_data2_per_dataset():
    first, second = _datasets()
    spec = bf("y ~ x") + bf("x ~ 1") + set_rescor(False)
    result = fit_multiple(spec, data=(first, second), seed=1, **_budget())
    updated = result.update((second, first), data2=[{}, {}], num_samples=8)
    assert updated.pooled.draws()["response_y__beta_x"].shape[:2] == (4, 8)
    torch.testing.assert_close(
        updated.fits[0].draws()["response_y__beta_x"],
        fit(spec, data=second, seed=1, **{**_budget(), "num_samples": 8}).draws()[
            "response_y__beta_x"
        ],
    )
