"""Typed retained contrasts share QVR arithmetic, nominal axes, and joint streams."""

from unittest.mock import patch

import pytest
import torch

from quivers.diagnostics import (
    ContrastExpression,
    PosteriorAnalysis,
    hypothesis_summary,
)
from quivers.diagnostics.contrast import _contrast_plan, _evaluate_contrast
from quivers.dsl import loads
from quivers.inference.trace import trace


def _analysis(samples, kind="mcmc"):
    return PosteriorAnalysis(samples=samples, log_likelihood={}, kind=kind)


def test_scalar_contrast_matches_native_reference_and_independent_arithmetic():
    a = torch.linspace(-0.4, 0.8, 12, dtype=torch.double).reshape(2, 6)
    b = a.square() - 0.2
    result = _analysis({"a": a, "b": b}).hypothesis("exp(a)-2*b", name="effect")
    expected = torch.exp(a) - 2 * b
    oracle = hypothesis_summary(expected)
    torch.testing.assert_close(result.estimate, oracle.estimate)
    torch.testing.assert_close(result.quantiles, oracle.quantiles)
    assert result.sample_shape == (2, 6) and result.kind == "mcmc"
    assert result.name == "effect" and not result.estimate.requires_grad
    source = loads("""object Out:Real 1
program contrast(a:Real,b:Real):Out->Out
    let value=exp(a)-2*b
    return value
export contrast
""")
    for row, column in ((0, 0), (1, 3)):
        data = {"a": a[row, column], "b": b[row, column]}
        native = trace(source.morphism, torch.zeros(1, 1), data)
        torch.testing.assert_close(
            native.sites["value"].value.squeeze(), expected[row, column]
        )
        reference = source.run(
            "contrast",
            data={key: value.item() for key, value in data.items()} | {"out": (0.0,)},
            sites={},
        )
        assert reference.value == pytest.approx(expected[row, column].item(), abs=2e-14)


def test_axes_and_original_labels_survive_arithmetic_and_reduction():
    matrix = torch.arange(2 * 5 * 3 * 4, dtype=torch.double).reshape(2, 5, 3, 4) / 100
    analysis = _analysis({"a": matrix, "b": matrix / 2})
    dims = {"a": ("group", "component"), "b": ("group", "component")}
    coords = {
        "group": ("first", "second", "third"),
        "component": (2, 7, 9, 11),
        "chain": ("left", "right"),
    }
    result = analysis.hypothesis("sum(a-b)", dims=dims, coords=coords)
    torch.testing.assert_close(result.estimate, (matrix / 2).sum(-1).mean((0, 1)))
    assert result.dimension_names == ("group",)
    assert result.coordinate_labels == (("first", "second", "third"),)
    assert result.chain_labels == ("left", "right")


def test_equal_extents_do_not_identify_distinct_event_axes():
    values = torch.ones(1, 4, 3)
    with pytest.raises(ValueError, match="differing shapes"):
        _analysis({"a": values, "b": values}).hypothesis("a-b")
    with pytest.raises(ValueError, match="differing shapes"):
        _analysis({"a": values, "b": values}).hypothesis(
            "a-b", dims={"a": ("Item",), "b": ("Component",)}
        )
    result = _analysis({"a": values, "b": values}).hypothesis(
        "a-b", dims={"a": ("Item",), "b": ("Item",)}
    )
    assert result.dimension_names == ("Item",)


def test_aliases_keep_authored_labels_and_index_only_event_axes():
    x = torch.arange(24, dtype=torch.double).reshape(2, 4, 3)
    expression = ContrastExpression(
        expression="a[2]-a[0]", bindings={"a": "condition[x]"}
    )
    result = _analysis({"condition[x]": x}).hypothesis(expression)
    torch.testing.assert_close(result.estimate, (x[..., 2] - x[..., 0]).mean())
    assert result.sample_shape == (2, 4) and result.dimension_names == ()


def test_prior_streams_are_evaluated_jointly_with_one_compilation():
    a = torch.tensor([[-1.0, 2.0, -3.0, 4.0]])
    b = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    prior_a = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    posterior = _analysis({"a": a, "b": b})
    prior = _analysis({"a": prior_a, "b": prior_a + 0.5}, "prior")
    with patch(
        "quivers.diagnostics.contrast._contrast_plan", wraps=_contrast_plan
    ) as compiled:
        result = posterior.hypothesis("a-b", prior=prior)
    assert compiled.call_count == 1
    assert result.prior_sample_shape == (1, 5)
    assert result.below.prior_probability.item() == 1
    assert torch.isnan(result.below.log_bayes_factor)
    assert result.prior_equal_probability.item() == 0


@pytest.mark.parametrize(
    "expression,error",
    [
        ("unknown+1", "unknown contrast"),
        (
            ContrastExpression(expression="a", bindings={"unused": "a"}),
            "unused contrast",
        ),
        (
            ContrastExpression(expression="a", bindings={"a": "missing"}),
            "unknown contrast",
        ),
        ("a>0", "predicate"),
        ("not_a_function(a)", "unknown|unavailable|not.*declared"),
        ("a; return a", "one nonempty"),
        ("a[0]", "index|tensor"),
    ],
)
def test_invalid_expressions_fail_before_evaluation(expression, error):
    with pytest.raises((ValueError, TypeError), match=error):
        _analysis({"a": torch.ones(1, 4)}).hypothesis(expression)


def test_nonfinite_capacity_alignment_and_prior_shape_checks():
    with pytest.raises(ValueError, match="finite"):
        _analysis({"a": torch.tensor([[float("nan")]])}).hypothesis("a")
    with pytest.raises(ValueError, match=r"2\*\*53"):
        _analysis(
            {"a": torch.tensor([[2**53 + 1]], dtype=torch.int64)}, "prior"
        ).hypothesis("a")
    with pytest.raises(ValueError, match="event shapes"):
        _analysis({"a": torch.ones(1, 3, 2)}).hypothesis(
            "sum(a)", prior=_analysis({"a": torch.ones(1, 3, 3)}, "prior")
        )
    with pytest.raises(ValueError, match="finite"):
        _analysis({"a": torch.full((1, 3), 1000.0)}).hypothesis("exp(a)")


def test_constant_expression_is_repeated_over_retained_sample_axes():
    result = _analysis({"a": torch.zeros(2, 4)}).hypothesis("3.0")
    assert result.estimate.item() == 3 and result.est_error.item() == 0
    assert result.sample_shape == (2, 4)


def test_plan_evaluation_retains_full_event_layout_without_summary_reduction():
    values = torch.arange(24, dtype=torch.double).reshape(2, 3, 4)
    samples = {"x": values}
    plan = _contrast_plan(
        ContrastExpression(expression="softmax(x)"),
        samples,
        {"x": ("Component",)},
        None,
    )
    actual = _evaluate_contrast(plan, samples)
    torch.testing.assert_close(actual, values.softmax(-1))
    assert plan.dimensions == ("Component",)
