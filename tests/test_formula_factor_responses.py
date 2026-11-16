"""Factor responses retain scalar codes, declared levels, and prediction labels."""

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import FormulaToQVRModule, families, fit, formula_from_data


def test_factor_codes_retain_contrast_levels_without_becoming_vector_events():
    frame = pd.DataFrame({"y": ["b", "a", "c", "b"], "x": [0.0, 1.0, 2.0, 3.0]})
    parsed = formula_from_data("y~x", frame)
    assert parsed.response_layout.category_levels == ("a", "b", "c")
    assert parsed.response_layout.component_names == ()
    assert parsed.response_layout.event_shape == ()
    np.testing.assert_array_equal(parsed.response_values, [1, 0, 2, 1])
    lens = FormulaToQVRModule(families["categorical"])
    module, complement = lens.forward(parsed)
    assert lens.backward(module, complement) == parsed
    with pytest.raises(ValueError, match="numeric response"):
        FormulaToQVRModule(families["gaussian"]).forward(parsed)


def test_ordered_factor_uses_declared_levels_including_an_unobserved_category():
    frame = pd.DataFrame(
        {
            "y": pd.Categorical(
                ["medium", "high", "medium", "high"],
                categories=["low", "medium", "high"],
                ordered=True,
            )
        }
    )
    parsed = formula_from_data("y~1", frame)
    assert parsed.response_layout.category_levels == ("low", "medium", "high")
    assert parsed.response_layout.ordered
    np.testing.assert_array_equal(parsed.response_values, [1, 2, 1, 2])
    assert FormulaToQVRModule._category_count(parsed, family="cumulative") == 3
    FormulaToQVRModule(families["cumulative"]).forward(parsed)


@pytest.mark.parametrize("family", ["categorical", "cumulative", "bernoulli"])
def test_binary_factor_prediction_retains_level_labels_and_event_shape(family):
    frame = pd.DataFrame(
        {
            "y": pd.Categorical(
                ["no", "yes", "yes", "no", "yes", "no"],
                categories=["no", "yes"],
                ordered=family == "cumulative",
            ),
            "x": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        }
    )
    result = fit("y~x", data=frame, family=family, method="svi", num_steps=3, seed=11)
    assert result.category_labels == ("no", "yes")
    expected = result.posterior_epred(num_samples=2, seed=4)
    assert (
        expected.shape == (1, 2, 6, 2)
        if family != "bernoulli"
        else expected.shape == (1, 2, 6)
    )
    assert torch.isfinite(expected).all()
    if family != "bernoulli":
        torch.testing.assert_close(expected.sum(-1), torch.ones_like(expected[..., 0]))
    predicted = result.posterior_predict(num_samples=2, seed=4)
    assert predicted.shape == (1, 2, 6)
    assert set(predicted.unique().tolist()) <= {0, 1}
    new = result.posterior_epred(
        new_data=pd.DataFrame({"x": [0.2, 0.8]}), num_samples=2, seed=4
    )
    assert new.shape[-2:] == (2, 2) if family != "bernoulli" else new.shape[-1:] == (2,)


def test_three_factor_levels_cannot_be_used_as_bernoulli_codes():
    frame = pd.DataFrame({"y": ["a", "b", "c"]})
    parsed = formula_from_data("y~1", frame)
    with pytest.raises(ValueError, match="exactly two"):
        FormulaToQVRModule(families["bernoulli"]).forward(parsed)
