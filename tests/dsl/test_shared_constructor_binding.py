"""Program constructor paths share keyword binding and preserve located failures."""

from unittest.mock import patch

import pytest
import torch

from quivers.dsl import loads, parse
from quivers.dsl.draw_args import bind_family_arguments
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.inference.trace import trace


@pytest.fixture(autouse=True)
def double_literals():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _source(constructor, *, initializer=""):
    return f"""object Out : Real 1
{initializer}program model : Out -> Out
    sample b <- {constructor}
    return b
export model
"""


@pytest.mark.parametrize(
    ("constructor", "initializer", "minimum_bindings"),
    [
        ("Normal(scale=2.0,loc=0.25)", "", 1),
        ("Restrict(high=1.0,base=Normal(scale=2.0,loc=0.25),low=-1.0)", "", 2),
        (
            "Restrict(high=1.0,base=base,low=-1.0)",
            "morphism base : Out -> Out ~ Normal(scale=2.0,loc=0.25)\n",
            2,
        ),
    ],
)
def test_draw_nested_and_morphism_values_share_binding_and_density(
    constructor, initializer, minimum_bindings
):
    source = _source(constructor, initializer=initializer)
    with patch(
        "quivers.dsl.program_elaboration.bind_family_arguments",
        wraps=bind_family_arguments,
    ) as bind:
        lower_qvr_to_qiec(parse(source))
    assert bind.call_count >= minimum_bindings
    program = loads(source)
    point = torch.tensor(0.4, dtype=torch.double)
    normal = torch.distributions.Normal(point.new_tensor(0.25), point.new_tensor(2.0))
    expected = normal.log_prob(point)
    if constructor.startswith("Restrict"):
        expected -= (
            normal.cdf(point.new_tensor(1.0)) - normal.cdf(point.new_tensor(-1.0))
        ).log()
    actual = trace(
        program.morphism, point.new_zeros(1, 1), {"b": point}
    ).log_joint.sum()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert program.run(
        "model", data={"out": (0.0,)}, sites={"b": 0.4}
    ).log_joint == pytest.approx(expected.item(), abs=1e-12)


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        ("Normal(loc=0.0,loc=1.0,scale=2.0)", "parameter 'loc' twice"),
        (
            "Restrict(base=Normal(loc=0.0,loc=1.0,scale=2.0),low=-1.0,high=1.0)",
            "parameter 'loc' twice",
        ),
        (
            "Restrict(base=Normal(loc=0.0),low=-1.0,high=1.0)",
            "complete parameter schema",
        ),
        ("Restrict(base=Normal(),low=-1.0,high=1.0)", "complete parameter schema"),
        ("Normal(0.0,1.0)", "keyword-only"),
    ],
)
def test_constructor_binding_errors_keep_source_location_and_qiec_category(
    constructor, message
):
    source = _source(constructor)
    with pytest.raises(QiecDiagnosticError, match=message) as caught:
        lower_qvr_to_qiec(parse(source))
    assert caught.value.code == "qiec-program"
    assert caught.value.line == 3
    assert caught.value.column == 4


def test_default_policy_remains_explicit_at_shared_binding_boundary():
    assert bind_family_arguments("Normal", ()) == ()
    with pytest.raises(TypeError, match="complete parameter schema"):
        bind_family_arguments("Normal", (), allow_defaults=False)


def test_morphism_value_binding_failure_keeps_the_consuming_source_location():
    source = _source(
        "Restrict(base=base,low=-1.0,high=1.0)",
        initializer="morphism base : Out -> Out ~ Normal(loc=0.0,loc=1.0,scale=2.0)\n",
    )
    with pytest.raises(QiecDiagnosticError, match="parameter 'loc' twice") as caught:
        lower_qvr_to_qiec(parse(source))
    assert caught.value.code == "qiec-program"
    assert (caught.value.line, caught.value.column) == (4, 4)
