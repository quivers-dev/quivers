"""Response-indexed metadata and the existing prior-selector semantics."""

import pickle

import didactic.api as dx
import pandas as pd
import pytest

from quivers.formulas import formula_to_qvr
from quivers.formulas._multivariate import MultivariateFormula, ResponseModel
from quivers.formulas._priors import PriorSet, PriorSpec, PriorTarget, resolve_priors


def test_response_specs_preserve_authored_axes_order_and_immutable_options():
    options = {"sigma": "~ 1+x"}
    first = ResponseModel(formula="accuracy~x", family="bernoulli")
    second = ResponseModel(formula="latency~x", distributional=options)
    model = MultivariateFormula(responses={"choice": first, "time": second})
    options["sigma"] = "~0"
    assert model.response_names == ("choice", "time")
    assert model.responses["time"].distributional["sigma"] == "~ 1+x"
    with pytest.raises(TypeError):
        model.responses["third"] = first
    assert pickle.loads(pickle.dumps(model)) == model
    assert hash(pickle.loads(pickle.dumps(model))) == hash(model)
    assert model != MultivariateFormula(responses={"time": second, "choice": first})


@pytest.mark.parametrize(
    "responses",
    [
        {},
        {"a-b": ResponseModel(formula="a~1"), "a b": ResponseModel(formula="b~1")},
    ],
)
def test_invalid_response_coordinate_system_fails(responses):
    # One response is a valid joint formula (used by mi()/me() models).
    with pytest.raises((ValueError, dx.ValidationError)):
        MultivariateFormula(responses=responses)


@pytest.mark.parametrize("family", ["gaussian", "student_t"])
def test_residual_correlation_keeps_response_and_native_component_axes_distinct(family):
    spec = MultivariateFormula(
        responses={
            "first": ResponseModel(formula="y~x", family=family),
            "second": ResponseModel(formula="z~x", family=family),
        },
        rescor=True,
    )
    assert spec.rescor
    assert spec.response_names == ("first", "second")


def test_residual_correlation_refuses_heterogeneous_and_nonelliptical_families():
    frame = pd.DataFrame({"y": [0.1, 0.4, -0.2, 0.3], "z": [1, 0, 1, 1], "x": [0.0, 1.0, 2.0, 3.0]})
    for second in ("student_t", "bernoulli"):
        spec = MultivariateFormula(
            responses={
                "a": ResponseModel(formula="y~x"),
                "b": ResponseModel(formula="z~x", family=second),
            },
            rescor=True,
        )
        # A plain refusal when the program is built, not a validation error.
        with pytest.raises(ValueError, match="homogeneous") as refused:
            formula_to_qvr(spec, data=frame)
        assert not isinstance(refused.value, dx.ValidationError)


def test_response_prior_selectors_refine_the_existing_catalogue():
    targets = tuple(
        PriorTarget(site=name + "_beta", class_="b", coef="x", response=name)
        for name in ("time", "choice")
    )
    broad = PriorSpec(distribution="Normal(0,2)")
    time = PriorSpec(distribution="Normal(0,1)", response="time")
    resolved = resolve_priors(PriorSet(rules=(broad, time)), targets)
    assert resolved["time_beta"] == time
    assert resolved["choice_beta"] == broad
    with pytest.raises(ValueError, match="unused"):
        resolve_priors(
            PriorSet(
                rules=(PriorSpec(distribution="Normal(0,1)", response="missing"),)
            ),
            targets,
        )
    assert pickle.loads(pickle.dumps(targets)) == targets
