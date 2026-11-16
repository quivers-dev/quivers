"""Role-checked response data share one immutable modifier record."""

import numpy as np
import pandas as pd
import pytest

from quivers.formulas import (
    FormulaToQVRModule,
    ResponseModifiers,
    families,
    formula_from_data,
)


@pytest.fixture
def frame():
    return pd.DataFrame(
        dict(
            rt=[0.3, 0.4, 0.6],
            choice=[0, 1, 0],
            label=["lower", "upper", "lower"],
            w=[0.0, 0.5, 2.0],
            x=[-0.3, 0.0, 0.4],
        )
    )


def test_keyword_and_surface_modifiers_retain_identical_typed_columns(frame):
    surface = formula_from_data("rt | dec(choice) + weights(w) ~ x", frame)
    keyword = formula_from_data("rt ~ x", frame, decision="choice", weights="w")
    assert isinstance(surface.modifiers, ResponseModifiers)
    assert surface.modifiers == keyword.modifiers
    np.testing.assert_array_equal(surface.response_values, frame.rt)
    assert surface.modifiers.decision.data.dtype == np.int64
    assert not surface.modifiers.decision.data.flags.writeable
    assert not surface.modifiers.weights.data.flags.writeable
    frame.loc[0, "choice"] = 1
    assert surface.modifiers.decision.data[0] == 0


def test_named_labels_and_numeric_decisions_have_same_normalized_values(frame):
    labeled = formula_from_data("rt ~ x", frame, decision="label")
    numeric = formula_from_data("rt ~ x", frame, decision="choice")
    np.testing.assert_array_equal(
        labeled.modifiers.decision.data, numeric.modifiers.decision.data
    )
    assert labeled.modifiers.decision.qvr_name == "formula_decision"


@pytest.mark.parametrize(
    "bad",
    [
        [-1, 0, 1],
        [0, 2, 1],
        [0, 0.5, 1],
        [0, np.inf, 1],
        [0, np.nan, 1],
        [False, True, False],
        ["low", "upper", "lower"],
        ["lower", None, "upper"],
    ],
)
def test_decision_domains_reject_ambiguous_or_missing_codes(frame, bad):
    frame["bad"] = bad
    with pytest.raises(ValueError, match="decisions"):
        formula_from_data("rt ~ x", frame, decision="bad")


@pytest.mark.parametrize(
    "source,kwargs",
    [
        ("rt | dec(choice) ~ x", dict(decision="label")),
        ("rt | weights(w) ~ x", dict(weights="x")),
        ("rt | unknown(choice) ~ x", {}),
        ("rt | dec(choice,x) ~ x", {}),
        ("rt | dec(choice+1) ~ x", {}),
        ('rt | dec("choice") ~ x', {}),
    ],
)
def test_modifiers_reject_conflicts_and_unevaluated_expression_arguments(
    frame, source, kwargs
):
    with pytest.raises(ValueError, match="modifier|requires|conflicting"):
        formula_from_data(source, frame, **kwargs)


def test_matching_duplicate_declarations_are_allowed(frame):
    formula = formula_from_data(
        "rt | dec(choice) + weights(w) ~ x", frame, decision="choice", weights="w"
    )
    assert len(formula.modifiers.columns()) == 2


def test_decision_is_not_silently_ignored_by_other_families(frame):
    formula = formula_from_data("rt | dec(choice) ~ x", frame)
    with pytest.raises(ValueError, match="requires the wiener family"):
        FormulaToQVRModule(families["gaussian"]).forward(formula)


def test_wiener_requires_decisions_and_uses_visible_compiler_owned_input_aliases(frame):
    with pytest.raises(ValueError, match="decision column"):
        FormulaToQVRModule(families["wiener"]).forward(
            formula_from_data("rt ~ x", frame)
        )
    formula = formula_from_data("rt | dec(choice) + weights(w) ~ x", frame)
    lens = FormulaToQVRModule(
        families["wiener"], fixed_parameters={"bs": 1.0, "ndt": 0.1, "bias": 0.5}
    )
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    assert complement.modifiers == formula.modifiers
    from quivers.dsl import module_to_source, loads

    source = module_to_source(module)
    assert "formula_decision : Tensor[Bool]([Resp])" in source
    assert "formula_response : Tensor[Real]([Resp])" in source
    assert "let rt = formula_response * (2.0 * real(formula_decision) - 1.0)" in source
    assert "weights=formula_weights" in source
    loads(source)
