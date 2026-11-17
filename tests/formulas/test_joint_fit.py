"""Joint formula execution delegates all inference and response replay once."""

import pandas as pd
import torch

from quivers.formulas import bf, fit
from quivers.formulas._joint_fit import fit_joint


def test_heterogeneous_named_response_carriers_and_joint_prior_streams():
    specification = bf("y~1") + bf("choice~1", family="bernoulli")
    data = pd.DataFrame({"y": [0.3, -0.4, 1.2], "choice": [1, 0, 1]})
    fitted = fit_joint(
        specification,
        data,
        method="svi",
        num_steps=2,
        sample_prior="yes",
        num_prior_samples=4,
        seed=3,
    )
    assert fitted.prior is not None
    assert fitted.response_names == ("y", "choice")
    predictions = fitted.posterior_predict(num_samples=4, seed=6)
    alone = fit(
        "choice~1", data=data, family="bernoulli", method="svi", num_steps=2, seed=3
    )
    assert (
        predictions["choice"].dtype
        == alone.posterior_predict(num_samples=4, seed=6).dtype
    )
    assert set(predictions["choice"].unique().tolist()) <= {0.0, 1.0}
    assert predictions["y"].dtype.is_floating_point
    assert predictions["choice"].shape == predictions["y"].shape == (1, 4, 3)
    expected = (
        fitted.draws(num_samples=4, seed=6)["response_choice__intercept"]
        .sigmoid()
        .unsqueeze(-1)
        .expand(1, 4, 3)
    )
    torch.testing.assert_close(
        fitted.posterior_epred(num_samples=4, seed=6)["choice"], expected
    )
    prior = fitted.prior_view().posterior_epred(num_samples=4, seed=6)
    assert prior["choice"].shape == (1, 4, 3)
