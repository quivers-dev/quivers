"""Native constructor boundaries preserve carriers and reject structural errors."""

import math

import pytest
import torch

from quivers.continuous.inline import (
    _build_operator_value,
    _literal_scalar,
    _raw_distribution_value,
    make_fixed_categorical,
)
from quivers.continuous.measure import Independent
from quivers.core.objects import FinSet


def _operator(name, values):
    return _build_operator_value(
        name, tuple(values.values()), {}, bound=True, parameter_names=tuple(values)
    )


@pytest.mark.parametrize(
    "family,values",
    [
        ("Restrict", {"base": 0.0, "low": -1.0}),
        ("Truncated", {"base": 0.0, "low": -1.0, "high": 1.0}),
        ("Normalize", {"base": 0.0}),
        ("Pushforward", {"base": 0.0, "bijector": "Exp"}),
        ("Independent", {"base": 0.0, "reinterpreted_batch_ndims": 0}),
    ],
)
def test_compositional_bases_require_real_native_distributions(family, values):
    with pytest.raises(ValueError, match="Distribution"):
        _raw_distribution_value(family, values)


@pytest.mark.parametrize("flag", [1, 0.5, "true"])
def test_endpoint_flags_require_boolean_values(flag):
    with pytest.raises(TypeError, match="Boolean"):
        _operator(
            "Restrict",
            {
                "base": torch.distributions.Normal(0.0, 1.0),
                "low": -1.0,
                "lower_closed": flag,
            },
        )


def test_mixture_component_list_is_validated_at_shared_distribution_boundary():
    base = torch.distributions.Normal(0.0, 1.0)
    with pytest.raises(ValueError, match="Distribution"):
        _operator("Mixture", {"weights": [0.5, 0.5], "components": [base, 3.0]})
    actual = _operator(
        "Mixture",
        {
            "weights": [0.3, 0.7],
            "components": [base, torch.distributions.Normal(2.0, 1.0)],
        },
    )
    value = torch.tensor(0.4)
    expected = torch.logaddexp(
        math.log(0.3) + base.log_prob(value),
        math.log(0.7) + torch.distributions.Normal(2.0, 1.0).log_prob(value),
    )
    torch.testing.assert_close(actual.log_prob(value), expected)


@pytest.mark.parametrize("count", [0, 1, torch.tensor(0), torch.tensor(1)])
def test_independent_tensor_counts_retain_the_existing_scalar_integer_contract(count):
    base = torch.distributions.Normal(torch.tensor([0.0, 1.0]), torch.ones(2))
    actual = Independent(base, count)
    expected = torch.distributions.Independent(base, int(count))
    value = torch.tensor([0.3, 0.7])
    torch.testing.assert_close(actual.log_prob(value), expected.log_prob(value))
    assert actual.reinterpreted_batch_ndims == int(count)


@pytest.mark.parametrize(
    "count", [True, 0.5, torch.tensor(True), torch.tensor(0.5), torch.tensor([1])]
)
def test_independent_invalid_counts_are_not_numerically_coerced(count):
    base = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
    with pytest.raises((TypeError, ValueError), match="integer"):
        _operator("Independent", {"base": base, "reinterpreted_batch_ndims": count})


@pytest.mark.parametrize("family", ["Restrict", "Truncated"])
def test_literal_restriction_endpoints_keep_double_precision_and_parameter_gradients(
    family,
):
    loc = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    lower = -0.7
    upper = 1.3
    base = torch.distributions.Normal(loc, scale)
    actual = _raw_distribution_value(
        family, {"base": base, "low": lower, "high": upper}
    )
    value = torch.tensor(0.4, dtype=torch.float64)
    expected = base.log_prob(value)
    if family == "Truncated":
        expected = (
            expected
            - (
                base.cdf(torch.tensor(upper, dtype=torch.float64))
                - base.cdf(torch.tensor(lower, dtype=torch.float64))
            ).log()
        )
    torch.testing.assert_close(actual.log_prob(value), expected, rtol=2e-12, atol=2e-12)
    actual_gradient = torch.autograd.grad(
        actual.log_prob(value), (loc, scale), retain_graph=True
    )
    expected_gradient = torch.autograd.grad(expected, (loc, scale))
    for measured, oracle in zip(actual_gradient, expected_gradient, strict=True):
        torch.testing.assert_close(measured, oracle, rtol=2e-11, atol=2e-11)


def test_literal_tensor_endpoints_keep_autograd_identity():
    lower = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
    upper = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    base = torch.distributions.Normal(
        torch.tensor(0.2, dtype=torch.float64), torch.tensor(0.7, dtype=torch.float64)
    )
    actual = _raw_distribution_value(
        "Truncated", {"base": base, "low": lower, "high": upper}
    )
    value = torch.tensor(0.4, dtype=torch.float64)
    expected = base.log_prob(value) - (base.cdf(upper) - base.cdf(lower)).log()
    for measured, oracle in zip(
        torch.autograd.grad(actual.log_prob(value), (lower, upper), retain_graph=True),
        torch.autograd.grad(expected, (lower, upper)),
        strict=True,
    ):
        torch.testing.assert_close(measured, oracle, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("values", [0.5, [1, 3], (1.0, 3.0), torch.tensor(0.5)])
def test_fixed_categorical_scalar_and_vector_literals_keep_native_law(values):
    actual = make_fixed_categorical(
        values, FinSet(name="Class", cardinality=2), parameterization="probs"
    )
    value = torch.tensor([1])
    expected = math.log(0.75) if isinstance(values, (list, tuple)) else math.log(0.5)
    torch.testing.assert_close(
        actual.log_prob(torch.zeros(1, 1), value), torch.tensor([expected])
    )


@pytest.mark.parametrize(
    "value", [None, "0.5", torch.ones(2), torch.distributions.Normal(0.0, 1.0)]
)
def test_fixed_categorical_refuses_non_scalar_literal_leaves(value):
    with pytest.raises(TypeError, match="real scalar literal"):
        make_fixed_categorical(
            value, FinSet(name="Class", cardinality=2), parameterization="probs"
        )


def test_scalar_boundary_keeps_integer_lexemes_exact_until_family_validation():
    integer = 2**53 + 1
    assert _literal_scalar(integer, "test") == integer
    assert isinstance(_literal_scalar(integer, "test"), int)


def test_real_numeric_scalar_protocol_preserves_optional_array_backend_inputs():
    np = pytest.importorskip("numpy")
    for scalar in (np.float32(0.5), np.float64(0.5), np.int64(1)):
        literal = _literal_scalar(scalar, "test")
        assert literal == scalar
        actual = make_fixed_categorical(
            [scalar, scalar],
            FinSet(name="Class", cardinality=2),
            parameterization="probs",
        )
        torch.testing.assert_close(
            actual.log_prob(torch.zeros(1, 1), torch.tensor([1])),
            torch.tensor([math.log(0.5)]),
        )
