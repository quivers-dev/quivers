"""Vector response events preserve rows, component order, and count carriers."""

import numpy as np
import pandas as pd
import pytest

from quivers.formulas import (
    FormulaToQVRModule,
    ResponseLayout,
    families,
    formula_from_data,
)


def test_vector_counts_preserve_exact_integer_events_and_row_weights():
    frame = pd.DataFrame(
        {"a": [2**53 + 1, 0], "b": [0, 3], "x": [0.5, -0.5], "w": [0.2, 2.0]}
    )
    parsed = formula_from_data("cbind(b, a) ~ x", frame, weights="w")
    assert parsed.response_layout == ResponseLayout(
        component_names=("b", "a"), carrier="integer"
    )
    assert parsed.response_layout.event_shape == (2,)
    assert parsed.response_values.shape == (2, 2)
    assert parsed.response_values.dtype == np.int64
    assert parsed.response_values[0, 1] == 2**53 + 1
    np.testing.assert_array_equal(parsed.modifiers.weights.data, [0.2, 2.0])
    assert not parsed.response_values.flags.writeable
    with pytest.raises(ValueError, match="scalar response event"):
        FormulaToQVRModule(families["poisson"]).forward(parsed)


def test_transformed_vector_component_labels_and_real_carrier_are_retained():
    frame = pd.DataFrame({"a": [0.3, 0.2], "b": [0.7, 0.8]})
    parsed = formula_from_data("cbind(a, 1 - a) ~ 1", frame)
    assert parsed.response_layout.component_names == ("a", "1 - a")
    assert parsed.response_layout.carrier == "real"
    np.testing.assert_allclose(parsed.response_values, frame[["a", "b"]])
    assert parsed.fixed_columns[0].data.shape == (2,)


@pytest.mark.parametrize("formula", ["cbind(a) ~ 1", "cbind(a, a) ~ 1"])
def test_vector_labels_cannot_be_ambiguous(formula):
    with pytest.raises(ValueError, match="distinct component labels"):
        formula_from_data(formula, pd.DataFrame({"a": [1, 2]}))


def test_scalar_event_layout_survives_formula_lens_roundtrip():
    parsed = formula_from_data("a ~ 1", pd.DataFrame({"a": [1.0, 2.0]}))
    assert parsed.response_values.shape == (2,)
    assert parsed.response_layout.event_shape == ()
    lens = FormulaToQVRModule(families["gaussian"])
    module, complement = lens.forward(parsed)
    assert lens.backward(module, complement) == parsed
