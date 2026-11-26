"""Independent domain, tail, and differentiation checks for log probabilities."""

import math

import numpy as np
import pytest
import torch
from scipy.special import log_ndtr

from quivers.dsl import loads
from quivers.dsl.pure_builtins import EAGER_BUILTINS
from quivers.inference.trace import trace
from quivers.qiec.numbers import NumberDomain
from quivers.qiec.primitives import IMPLEMENTATIONS, primitive
from quivers.qiec.types import REAL


def _machine(name: str, *arguments: float | bool) -> float:
    """Apply a primitive of the reference machine to host numbers.

    Parameters
    ----------
    name : str
        The primitive.
    *arguments : float | bool
        Host numbers, made ``float64`` tensors (a truth value a ``bool``
        tensor) as the machine holds them.

    Returns
    -------
    float
        The result.
    """
    tensors = [
        torch.tensor(argument)
        if isinstance(argument, bool)
        else torch.tensor(argument, dtype=torch.float64)
        for argument in arguments
    ]
    return float(IMPLEMENTATIONS[name](NumberDomain(), *tensors))


@pytest.mark.parametrize(
    "name", ("log_probability", "log1mexp", "log_ndtr", "positive_parameter")
)
def test_probability_primitives_have_checked_scalar_signatures(name):
    signature = primitive(name)
    assert signature.parameters == (REAL,)
    assert signature.result == REAL
    assert signature.capability == "probability"


@pytest.mark.parametrize(
    "x",
    (
        -1e5,
        -1000.0,
        -100.0,
        -40.0,
        -11.0,
        -10.0,
        -8.0,
        -1.0,
        0.0,
        1.0,
        8.0,
        10.0,
        40.0,
        1000.0,
    ),
)
def test_normal_logcdf_standard_library_reference_matches_scipy(x):
    expected = float(log_ndtr(x))
    assert _machine("log_ndtr", x) == pytest.approx(expected, rel=4e-15, abs=2e-16)


@pytest.mark.parametrize(
    "name,values,expected",
    (
        (
            "log_probability",
            [0.0, 0.1, 0.8, 1.0],
            [-math.inf, math.log(0.1), math.log(0.8), 0.0],
        ),
        (
            "log1mexp",
            [-math.inf, -1000.0, -1.0, -1e-15, 0.0],
            [
                0.0,
                0.0,
                math.log1p(-math.exp(-1)),
                math.log(-math.expm1(-1e-15)),
                -math.inf,
            ],
        ),
        ("log_ndtr", [-math.inf, math.inf], [-math.inf, 0.0]),
    ),
)
def test_reference_and_eager_endpoint_values(name, values, expected):
    np.testing.assert_allclose(
        [_machine(name, x) for x in values], expected, rtol=1e-14
    )
    actual = EAGER_BUILTINS[name](torch.tensor(values, dtype=torch.double))
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


@pytest.mark.parametrize(
    "name,value",
    (
        ("log_probability", -0.1),
        ("log_probability", 1.1),
        ("log_probability", math.inf),
        ("log_probability", math.nan),
        ("log_probability", True),
        ("log1mexp", 0.1),
        ("log1mexp", math.inf),
        ("log1mexp", math.nan),
        ("log1mexp", True),
        ("log_ndtr", math.nan),
        ("log_ndtr", True),
    ),
)
def test_domains_match_in_reference_eager_scalar_and_tensor(name, value):
    for argument in (value, torch.tensor(value)):
        with pytest.raises(ValueError):
            EAGER_BUILTINS[name](argument)
    with pytest.raises(ValueError):
        _machine(name, value)


@pytest.mark.parametrize(
    "name,values",
    (
        ("log_probability", [0.01, 0.4, 0.9]),
        ("log1mexp", [-10.0, -1.0, -0.1]),
        ("log_ndtr", [-40.0, -0.3, 10.0]),
    ),
)
def test_native_interior_derivatives_and_hessian(name, values):
    x = torch.tensor(values, dtype=torch.double, requires_grad=True)
    fn = EAGER_BUILTINS[name]
    assert torch.autograd.gradcheck(fn, (x,), check_forward_ad=True)
    assert torch.autograd.gradgradcheck(fn, (x,))


@pytest.mark.parametrize(
    "name,values,mask",
    (
        ("log_probability", [0.0, 0.4], [False, True]),
        ("log1mexp", [0.0, -math.inf, -0.4], [False, False, True]),
        ("log_ndtr", [-math.inf, math.inf, -0.4], [False, False, True]),
    ),
)
def test_singular_inactive_branches_do_not_poison_interior_gradients(
    name, values, mask
):
    x = torch.tensor(values, dtype=torch.double, requires_grad=True)
    output = EAGER_BUILTINS[name](x)
    loss = torch.where(torch.tensor(mask), output, torch.zeros_like(output)).sum()
    (gradient,) = torch.autograd.grad(loss, (x,))
    assert torch.isfinite(gradient).all()
    assert (gradient[~torch.tensor(mask)] == 0).all()


@pytest.mark.parametrize(
    "name,value", (("log_probability", 0.4), ("log1mexp", -0.4), ("log_ndtr", -40.0))
)
def test_authored_checked_computation_executes_on_reference_and_native(name, value):
    source = f"""object One : FinSet 1
object Out : Real 1
define f(x : Real) : Real ! {{}} =
    return {name}(x)

program model : One -> Out
    let answer <- f({value})
    return answer
export model
"""
    compiled = loads(source)
    result = compiled.run("f", value)
    expected = _machine(name, value)
    assert result.value == pytest.approx(expected)
    recorded = trace(compiled.morphism, torch.zeros(1, 1, dtype=torch.double))
    actual = recorded.output
    assert actual.item() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "value", (False, True, 0.0, -0.0, -0.1, -math.inf, math.inf, math.nan)
)
def test_positive_parameter_has_strict_finite_real_domain(value):
    with pytest.raises(ValueError):
        _machine("positive_parameter", value)
    with pytest.raises(ValueError):
        EAGER_BUILTINS["positive_parameter"](value)
    with pytest.raises(ValueError):
        EAGER_BUILTINS["positive_parameter"](torch.tensor(value))


def test_positive_parameter_retains_tensor_identity_and_derivatives():
    value = torch.tensor([0.01, 1.0, 10.0], dtype=torch.double, requires_grad=True)
    assert EAGER_BUILTINS["positive_parameter"](value) is value
    for item in value.detach().tolist():
        assert _machine("positive_parameter", item) == item
    fn = EAGER_BUILTINS["positive_parameter"]
    assert torch.autograd.gradcheck(fn, (value,), check_forward_ad=True)
    assert torch.autograd.gradgradcheck(fn, (value,))
